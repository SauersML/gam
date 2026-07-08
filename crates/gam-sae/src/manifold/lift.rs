//! Lifted linear solvers for curved SAE atoms — *curvature is linear structure
//! one polynomial degree up*.
//!
//! # The pattern
//!
//! A curved atom fit by the dense per-row Newton over latent coordinates `t` is
//! nonconvex and basin-plagued. But every curved atom in this crate is, by
//! construction, a *linear* map applied to a fixed nonlinear feature map `Φ(t)`
//! (harmonic phasors, degree-2 monomials, …). Fitting the linear block is a
//! *convex* problem — the **lifted fit** — and recovering the underlying spike
//! parameters `{(a_j, t_j)}` from the fitted linear block is a closed-form
//! algebraic descent. [`crate::super_resolution`] is exactly this pattern for the
//! **circle**: a degree-`H` harmonic circle is a linear map on
//! `(cos t, …, cos Ht, sin t, …, sin Ht)`, and the matrix-pencil / Prony descent
//! un-superposes the fitted Fourier block into point masses. This module
//! generalises the descent to the two remaining curved topologies the term
//! carries ([`crate::manifold::SaeAtomBasisKind::Sphere`],
//! [`crate::manifold::SaeAtomBasisKind::Torus`]).
//!
//! * **Sphere — the Veronese lift.** A mixture of `m` points `v_1..v_m ∈ S^{d-1}`
//!   with amplitudes `a_j > 0` lifts to the PSD matrix `M = Σ_j a_j v_j v_jᵀ`
//!   (the degree-2 Veronese / symmetric-outer-product feature block). The descent
//!   is a symmetric eigendecomposition: [`recover_sphere_spikes`].
//! * **Torus — the Kronecker pencil.** A spike at `(θ, φ) ∈ T²` with per-axis
//!   harmonic degrees `(H₁, H₂)` lifts to the Kronecker product of the two
//!   harmonic phasor vectors; `m` spikes give a sum of `m` Kronecker-rank-1 terms
//!   sampled on the `H₁ × H₂` grid. The descent is 2-D harmonic retrieval by an
//!   enhanced matrix pencil with *auto-paired* axes: [`recover_torus_spikes`].
//!
//! Both descents are exact only in the noiseless limit; [`polish_spikes`] runs a
//! few damped Gauss–Newton steps on the *original* nonconvex objective
//! `‖z − Σ_j a_j Φ(t_j)‖²` given a caller-supplied basis evaluation, and reports
//! the final residual so a caller can gate acceptance.

use faer::Side;
use faer::prelude::*;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array1, Array2, ArrayView2};
use std::f64::consts::TAU;

// ============================================================================
// Shared order selection (mirrors `super_resolution`'s derived thresholds)
// ============================================================================

/// Gavish–Donoho (2014) optimal-hard-threshold coefficient `λ(β)` for a known
/// noise level, `β ∈ (0, 1]` the matrix aspect ratio (short/long dimension). The
/// same closed form [`crate::super_resolution`] uses; duplicated here so the two
/// lifted descents stay module-independent.
fn optimal_hard_threshold_coefficient(beta: f64) -> f64 {
    (2.0 * (beta + 1.0)
        + 8.0 * beta / ((beta + 1.0) + (beta * beta + 14.0 * beta + 1.0).sqrt()))
    .sqrt()
}

/// Model order from a descending singular spectrum, by the same derived rule as
/// [`crate::super_resolution::recover_spikes`]: the Gavish–Donoho optimal hard
/// threshold `λ(β)·σ_entry·√n` when `sigma > 0` (`σ_entry = √2·sigma` for a
/// complex entry), else the numerical-rank floor `σ₁·max(rows,cols)·ε`; refined
/// by an unambiguous relative collapse `σ_{k+1}/σ_k < √ε_f32` (the geometric mean
/// between the f32 quantisation shelf and unity, so an f32-quantised code's
/// round-off shelf never over-selects). Clamped to `max_order`.
fn singular_value_order(
    singular_values: &[f64],
    rows: usize,
    cols: usize,
    sigma: f64,
    max_order: usize,
) -> usize {
    let sigma_1 = singular_values.first().copied().unwrap_or(0.0);
    let n_big = rows.max(cols) as f64;
    let numerical_floor = sigma_1 * n_big * f64::EPSILON;
    let threshold = if sigma > 0.0 {
        let beta = rows.min(cols) as f64 / n_big;
        let sigma_entry = std::f64::consts::SQRT_2 * sigma;
        (optimal_hard_threshold_coefficient(beta) * sigma_entry * n_big.sqrt())
            .max(numerical_floor)
    } else {
        numerical_floor
    };
    let threshold_order = singular_values
        .iter()
        .filter(|&&s| s > threshold)
        .count()
        .min(max_order);

    let dramatic_gap = (f32::EPSILON as f64).sqrt();
    let mut gap_order = None;
    for k in 1..max_order.min(singular_values.len()) {
        if singular_values[k - 1] <= 0.0 {
            break;
        }
        if singular_values[k] / singular_values[k - 1] < dramatic_gap {
            gap_order = Some(k);
            break;
        }
    }
    gap_order.unwrap_or(threshold_order)
}

// ============================================================================
// Sphere — the Veronese lift
// ============================================================================

/// A single recovered point mass on the sphere `S^{d-1}`.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereSpike {
    /// Canonical unit direction `v ∈ S^{d-1}` (length `d`). The lift `v vᵀ` is
    /// invariant under the antipodal flip `v ↦ −v`, so the reported vector is the
    /// canonical representative of the `{v, −v}` gauge orbit: its
    /// largest-magnitude component is non-negative (see [`canonicalize_direction`]).
    pub direction: Vec<f64>,
    /// Amplitude `a > 0` of the spike (the corresponding eigenvalue of the lift).
    pub amplitude: f64,
}

/// The full result of a Veronese-lift recovery.
#[derive(Clone, Debug)]
pub struct SphereRecovery {
    /// Recovered spikes, sorted by amplitude descending.
    pub spikes: Vec<SphereSpike>,
    /// Selected model order `m` (number of point masses), from the count of
    /// eigenvalues above the noise-derived floor.
    pub model_order: usize,
    /// Frobenius norm of `M̂ − Σ_j a_j v_j v_jᵀ` for the recovered model.
    pub residual: f64,
    /// Eigenvalues of the symmetrised lift, descending — the spectrum the order
    /// selection thresholded.
    pub eigenvalues: Vec<f64>,
}

/// Forward map: lift a spike measure `{(a_j, v_j)}` to its degree-2 Veronese
/// code `M = Σ_j a_j v_j v_jᵀ` (symmetric `d × d`, `trace M = Σ_j a_j`). The
/// directions need not be unit vectors here — the amplitude absorbs `‖v_j‖²` —
/// but [`recover_sphere_spikes`] reports unit directions, so round-trip tests
/// should feed unit `v_j`. `d` is the ambient dimension; every direction must
/// have length `d`.
pub fn sphere_lift(spikes: &[SphereSpike], d: usize) -> Result<Array2<f64>, String> {
    if d == 0 {
        return Err("sphere_lift: ambient dimension must be positive".into());
    }
    let mut m = Array2::<f64>::zeros((d, d));
    for (idx, spike) in spikes.iter().enumerate() {
        if spike.direction.len() != d {
            return Err(format!(
                "sphere_lift: spike {idx} direction has length {}, expected {d}",
                spike.direction.len()
            ));
        }
        let a = spike.amplitude;
        for i in 0..d {
            for j in 0..d {
                m[[i, j]] += a * spike.direction[i] * spike.direction[j];
            }
        }
    }
    Ok(m)
}

/// Canonicalise a direction's antipodal gauge in place: the lift `v vᵀ` is
/// invariant under `v ↦ −v`, so we fix the sign deterministically by requiring
/// the component of largest magnitude to be non-negative (ties broken by the
/// lowest index, since the first maximal-magnitude entry is the pivot). A vector
/// whose largest-magnitude entry is exactly zero is the zero vector and is left
/// unchanged. This makes the recovered representative a deterministic function of
/// the lift alone, independent of the sign convention the eigensolver happened to
/// return.
pub fn canonicalize_direction(v: &mut [f64]) {
    let mut pivot = 0usize;
    let mut best = 0.0_f64;
    for (i, &x) in v.iter().enumerate() {
        if x.abs() > best {
            best = x.abs();
            pivot = i;
        }
    }
    if v.get(pivot).is_some_and(|&x| x < 0.0) {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

/// Eigenvalue noise floor for the Veronese lift's order selection.
///
/// The noisy code `M̂ = M + E` carries a symmetric perturbation `E` whose entries
/// have per-entry standard deviation `sigma`. The empirical spectrum of such a
/// `d × d` symmetric random matrix follows the semicircle law on
/// `[−2σ√d, +2σ√d]`, so its top eigenvalue concentrates at the edge `2σ√d` with
/// Tracy–Widom fluctuations of width `O(σ d^{−1/6})`. Placing the cut at *twice*
/// the edge, `4σ√d`, puts it several fluctuation-widths above the entire noise
/// spectrum for the small `d` of a Veronese lift while remaining far below any
/// `O(1)` signal eigenvalue — so genuine point masses are kept and noise
/// eigenvalues are rejected. When `sigma ≤ 0` the floor is the numerical rank
/// bound `λ_max·d·ε` (the noiseless path).
fn sphere_eigenvalue_floor(lambda_max: f64, d: usize, sigma: f64) -> f64 {
    /// Multiple of the `2σ√d` semicircle edge at which the order cut is placed;
    /// `2` keeps the cut several Tracy–Widom widths above the noise bulk for
    /// small `d`.
    const EDGE_SAFETY: f64 = 2.0;
    let numerical_floor = lambda_max.abs() * (d as f64) * f64::EPSILON;
    if sigma > 0.0 {
        (EDGE_SAFETY * 2.0 * sigma * (d as f64).sqrt()).max(numerical_floor)
    } else {
        numerical_floor
    }
}

/// Recover the `m` point masses `{(a_j, v_j)}` underlying a Veronese-lift code
/// `M̂` by symmetric eigendecomposition.
///
/// `m_hat` is the (possibly noisy) `d × d` lifted code — the linear fit over the
/// degree-2 Veronese feature block. `sigma` is the per-entry standard deviation
/// of the additive noise on `M̂`; pass `sigma ≤ 0` for the noiseless numerical
/// path. The lift is first symmetrised (`½(M̂ + M̂ᵀ)`), then eigendecomposed; the
/// top-`m` non-negative eigenpairs are the recovered amplitudes and directions.
///
/// # Identifiability
///
/// A PSD matrix has a *unique* decomposition into **orthogonal** rank-1 terms
/// (its spectral decomposition) but infinitely many non-orthogonal PSD rank-1
/// decompositions. So the eigendecomposition recovers the planted spikes exactly
/// iff the generating directions `v_j` are mutually orthogonal — the identifiable
/// regime for a symmetric-matrix lift. For `m = 1` this is automatic; for
/// `m ≥ 2` the caller must supply (near-)orthogonal directions for the recovered
/// directions to match the planted ones. The returned directions are always the
/// canonical orthogonal eigenbasis of `M̂` regardless.
pub fn recover_sphere_spikes(
    m_hat: ArrayView2<'_, f64>,
    sigma: f64,
) -> Result<SphereRecovery, String> {
    let d = m_hat.nrows();
    if d == 0 || m_hat.ncols() != d {
        return Err(format!(
            "recover_sphere_spikes: lift must be square and non-empty; got {:?}",
            m_hat.dim()
        ));
    }
    if m_hat.iter().any(|x| !x.is_finite()) {
        return Err("recover_sphere_spikes: lift has non-finite entries".into());
    }

    // Symmetrise: the lift is symmetric by construction, but a noisy linear fit
    // need not be, and the eigenpath needs a self-adjoint operator.
    let mut sym = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            sym[[i, j]] = 0.5 * (m_hat[[i, j]] + m_hat[[j, i]]);
        }
    }

    let (evals, evecs) = sym
        .eigh(Side::Lower)
        .map_err(|e| format!("recover_sphere_spikes: eigendecomposition failed: {e:?}"))?;
    // `eigh` returns ascending eigenvalues; walk them descending.
    let order_desc: Vec<usize> = {
        let mut idx: Vec<usize> = (0..d).collect();
        idx.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));
        idx
    };
    let eigenvalues: Vec<f64> = order_desc.iter().map(|&i| evals[i]).collect();

    let lambda_max = eigenvalues.first().copied().unwrap_or(0.0);
    let floor = sphere_eigenvalue_floor(lambda_max, d, sigma);
    let model_order = eigenvalues
        .iter()
        .filter(|&&lam| lam > floor)
        .count()
        .min(d);

    let mut spikes: Vec<SphereSpike> = Vec::with_capacity(model_order);
    for k in 0..model_order {
        let col = order_desc[k];
        let mut direction: Vec<f64> = (0..d).map(|r| evecs[[r, col]]).collect();
        // Normalise (eigenvectors are unit already, but noise repair may perturb)
        // then fix the antipodal gauge.
        let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for x in direction.iter_mut() {
                *x /= norm;
            }
        }
        canonicalize_direction(&mut direction);
        spikes.push(SphereSpike {
            direction,
            amplitude: eigenvalues[k],
        });
    }

    // Residual: ‖M̂ − Σ_j a_j v_j v_jᵀ‖_F over the recovered model.
    let mut residual_sq = 0.0;
    for i in 0..d {
        for j in 0..d {
            let mut fit = 0.0;
            for spike in &spikes {
                fit += spike.amplitude * spike.direction[i] * spike.direction[j];
            }
            let diff = m_hat[[i, j]] - fit;
            residual_sq += diff * diff;
        }
    }

    Ok(SphereRecovery {
        spikes,
        model_order,
        residual: residual_sq.sqrt(),
        eigenvalues,
    })
}

// ============================================================================
// Torus — the Kronecker pencil (2-D harmonic retrieval, auto-paired)
// ============================================================================

/// A single recovered point mass on the torus `T² = S¹ × S¹`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorusSpike {
    /// Axis-1 position `θ ∈ [0, 1)` (fraction of a full turn); the phasor is
    /// `e^{2πi θ}`.
    pub theta: f64,
    /// Axis-2 position `φ ∈ [0, 1)`; the phasor is `e^{2πi φ}`.
    pub phi: f64,
    /// Amplitude `a > 0` (real part of the least-squares Kronecker-Vandermonde
    /// coefficient).
    pub amplitude: f64,
}

/// The full result of a Kronecker-pencil recovery.
#[derive(Clone, Debug)]
pub struct TorusRecovery {
    /// Recovered spikes, sorted by `(θ, φ)` ascending.
    pub spikes: Vec<TorusSpike>,
    /// Selected model order `m`.
    pub model_order: usize,
    /// Frobenius norm of `Ŷ − Σ_j a_j (z_j^{h₁} w_j^{h₂})` for the recovered
    /// model over the `H₁ × H₂` grid.
    pub residual: f64,
    /// Singular values of the enhanced (block-Hankel) matrix, descending.
    pub enhanced_singular_values: Vec<f64>,
}

/// Forward map: sample the lift of a torus spike measure on the `H₁ × H₂`
/// harmonic grid, `Ŷ[h₁, h₂] = Σ_j a_j e^{2πi(h₁ θ_j + h₂ φ_j)}`,
/// `h₁ = 0..H₁−1`, `h₂ = 0..H₂−1`. This is the sum of Kronecker-rank-1 phasor
/// grids the descent inverts.
pub fn torus_lift(spikes: &[TorusSpike], h1: usize, h2: usize) -> Result<Mat<c64>, String> {
    if h1 == 0 || h2 == 0 {
        return Err("torus_lift: both harmonic counts must be positive".into());
    }
    let grid = Mat::<c64>::from_fn(h1, h2, |a, b| {
        let mut acc = c64::new(0.0, 0.0);
        for spike in spikes {
            let phase = TAU * (a as f64 * spike.theta + b as f64 * spike.phi);
            acc += c64::new(spike.amplitude, 0.0) * c64::new(phase.cos(), phase.sin());
        }
        acc
    });
    Ok(grid)
}

/// Small dense complex matrix product `A · B` (m is tiny in the pencil).
fn matmul(a: &Mat<c64>, b: &Mat<c64>) -> Mat<c64> {
    let (n, k, p) = (a.nrows(), a.ncols(), b.ncols());
    Mat::<c64>::from_fn(n, p, |i, j| {
        let mut acc = c64::new(0.0, 0.0);
        for t in 0..k {
            acc += a[(i, t)] * b[(t, j)];
        }
        acc
    })
}

/// Recover the `m` spikes `{(a_j, θ_j, φ_j)}` underlying a torus lift sampled on
/// the `H₁ × H₂` grid, by the enhanced matrix pencil (2-D harmonic retrieval).
///
/// `grid[h₁, h₂]` is the complex lifted sample (as built by [`torus_lift`]);
/// `sigma` is the per-component noise standard deviation (each of `Re`, `Im` is
/// `N(0, sigma²)`), with `sigma ≤ 0` selecting the noiseless numerical-rank path.
///
/// # Method (auto-paired 2-D pencil)
///
/// Build the block-Hankel *enhanced* matrix `Xe[(k,p),(l,q)] = Ŷ[p+q, k+l]` with
/// pencil windows `P` on axis 1 and `K` on axis 2 (Hua's Matrix Enhancement
/// Matrix Pencil, 1992). One SVD gives the model order and the signal subspace
/// `Es` (its columns span the enhanced steering vectors, each a Kronecker product
/// `w_vec(K) ⊗ z_vec(P)`). Within `Es` there are **two** shift invariances that
/// share the *same* similarity `T`: the intra-block `p`-shift gives
/// `Ψ_z = T⁻¹ diag(z) T` and the inter-block `k`-shift gives
/// `Ψ_w = T⁻¹ diag(w) T`. Because both are diagonalised by the same `T`, the
/// eigenvectors `E` of a generic complex combination `Ψ_z + γ Ψ_w` recover `T⁻¹`
/// and the pairing is automatic: `z_j = (E⁻¹ Ψ_z E)_{jj}` and
/// `w_j = (E⁻¹ Ψ_w E)_{jj}` for the *same* index `j` belong to the *same* source.
/// This is what defeats the classic 2-D failure of mispairing `(θ₁, φ₂)`:
/// estimating the two axes independently and re-pairing by sorting produces ghost
/// spikes, whereas the joint eigenvector basis never separates the axes. `γ` is
/// complex (irrational imaginary part) so the combined pencil has distinct
/// eigenvalues even when one axis carries a repeated frequency (two spikes
/// sharing `θ`), the degenerate case that breaks a single-axis eigensolve.
/// Amplitudes are a final least-squares Kronecker-Vandermonde solve.
pub fn recover_torus_spikes(grid: &Mat<c64>, sigma: f64) -> Result<TorusRecovery, String> {
    let h1 = grid.nrows();
    let h2 = grid.ncols();
    if h1 < 2 || h2 < 2 {
        return Err(format!(
            "recover_torus_spikes: need at least a 2×2 grid; got {h1}×{h2}"
        ));
    }
    for a in 0..h1 {
        for b in 0..h2 {
            if !grid[(a, b)].re.is_finite() || !grid[(a, b)].im.is_finite() {
                return Err(format!("recover_torus_spikes: grid[{a},{b}] is not finite"));
            }
        }
    }

    // Pencil windows: balanced (≈ half each axis) with `+1` so both the axis-1
    // shift `(P−1)·K` and the axis-2 shift `(K−1)·P` retain enough rows. Clamped
    // to a valid window `[2, H]`.
    let p = (h1 / 2 + 1).clamp(2, h1);
    let k = (h2 / 2 + 1).clamp(2, h2);
    let rows = k * p;
    let cols = (h2 - k + 1) * (h1 - p + 1);
    // Identifiable order: bounded by the enhanced matrix rank and by both shift
    // subselections having at least `m` rows.
    let max_order = rows
        .min(cols)
        .min((p - 1) * k)
        .min((k - 1) * p);
    if max_order == 0 {
        return Err(format!(
            "recover_torus_spikes: grid {h1}×{h2} too small to resolve any spike"
        ));
    }

    // Enhanced (2-level block-Hankel) matrix Xe[(k,p),(l,q)] = Ŷ[p+q, k+l].
    let xe = Mat::<c64>::from_fn(rows, cols, |r, c| {
        let (bk, pp) = (r / p, r % p);
        let (bl, qq) = (c / (h1 - p + 1), c % (h1 - p + 1));
        grid[(pp + qq, bk + bl)]
    });

    let svd = xe
        .thin_svd()
        .map_err(|e| format!("recover_torus_spikes: enhanced SVD failed: {e:?}"))?;
    let singular_values: Vec<f64> = svd.S().column_vector().iter().map(|c| c.re).collect();
    let model_order = singular_value_order(&singular_values, rows, cols, sigma, max_order);

    if model_order == 0 {
        let mut residual_sq = 0.0;
        for a in 0..h1 {
            for b in 0..h2 {
                residual_sq += grid[(a, b)].norm_sqr();
            }
        }
        return Ok(TorusRecovery {
            spikes: Vec::new(),
            model_order: 0,
            residual: residual_sq.sqrt(),
            enhanced_singular_values: singular_values,
        });
    }

    // Signal subspace Es = first `m` left singular vectors (KP × m).
    let u = svd.U();
    let es = Mat::<c64>::from_fn(rows, model_order, |r, c| u[(r, c)]);

    // Axis-1 (z) shift: intra-block p-shift. Both selections are (P−1)K × m.
    let z_rows = (p - 1) * k;
    let es_zup = Mat::<c64>::from_fn(z_rows, model_order, |r, c| {
        let (bk, pp) = (r / (p - 1), r % (p - 1));
        es[(bk * p + pp, c)]
    });
    let es_zdn = Mat::<c64>::from_fn(z_rows, model_order, |r, c| {
        let (bk, pp) = (r / (p - 1), r % (p - 1));
        es[(bk * p + pp + 1, c)]
    });
    // Ψ_z = Es_zup⁺ Es_zdn = T⁻¹ diag(z) T.
    let psi_z = es_zup.qr().solve_lstsq(&es_zdn);

    // Axis-2 (w) shift: inter-block k-shift. Both selections are (K−1)P × m and
    // are contiguous row slices of Es.
    let w_rows = (k - 1) * p;
    let es_wup = Mat::<c64>::from_fn(w_rows, model_order, |r, c| es[(r, c)]);
    let es_wdn = Mat::<c64>::from_fn(w_rows, model_order, |r, c| es[(r + p, c)]);
    // Ψ_w = Es_wup⁺ Es_wdn = T⁻¹ diag(w) T.
    let psi_w = es_wup.qr().solve_lstsq(&es_wdn);

    // Joint diagonalisation: eigenvectors E of a generic complex combination
    // Ψ_z + γ Ψ_w recover T⁻¹, and reading each shift's diagonal in that basis
    // auto-pairs the axes. The golden-ratio imaginary part is irrational, so the
    // combination has distinct eigenvalues even on a repeated-frequency axis.
    let gamma = c64::new(1.0, 0.618_033_988_749_895);
    let psi_comb = Mat::<c64>::from_fn(model_order, model_order, |i, j| {
        psi_z[(i, j)] + gamma * psi_w[(i, j)]
    });
    let eig = psi_comb
        .eigen()
        .map_err(|e| format!("recover_torus_spikes: joint eigenproblem failed: {e:?}"))?;
    let e_vecs = eig.U().to_owned();

    // z_j = diag(E⁻¹ Ψ_z E), w_j = diag(E⁻¹ Ψ_w E), same index ⇒ same source.
    let bz = matmul(&psi_z, &e_vecs);
    let dz = e_vecs.qr().solve_lstsq(&bz);
    let bw = matmul(&psi_w, &e_vecs);
    let dw = e_vecs.qr().solve_lstsq(&bw);

    let mut z_phasors: Vec<c64> = Vec::with_capacity(model_order);
    let mut w_phasors: Vec<c64> = Vec::with_capacity(model_order);
    for j in 0..model_order {
        let z = dz[(j, j)];
        let w = dw[(j, j)];
        let zn = z.norm();
        let wn = w.norm();
        z_phasors.push(if zn > 0.0 { z / zn } else { c64::new(1.0, 0.0) });
        w_phasors.push(if wn > 0.0 { w / wn } else { c64::new(1.0, 0.0) });
    }

    // Amplitudes: least-squares Kronecker-Vandermonde solve over the full grid,
    // Σ_j a_j z_j^{h₁} w_j^{h₂} = Ŷ[h₁, h₂].
    let vander = Mat::<c64>::from_fn(h1 * h2, model_order, |r, j| {
        let (a, b) = (r / h2, r % h2);
        z_phasors[j].powu(a as u32) * w_phasors[j].powu(b as u32)
    });
    let rhs = Mat::<c64>::from_fn(h1 * h2, 1, |r, _| {
        let (a, b) = (r / h2, r % h2);
        grid[(a, b)]
    });
    let amps = vander.qr().solve_lstsq(&rhs);

    let mut spikes: Vec<TorusSpike> = (0..model_order)
        .map(|j| {
            let theta = {
                let t = z_phasors[j].arg() / TAU;
                if t < 0.0 { t + 1.0 } else { t }
            };
            let phi = {
                let t = w_phasors[j].arg() / TAU;
                if t < 0.0 { t + 1.0 } else { t }
            };
            TorusSpike {
                theta,
                phi,
                amplitude: amps[(j, 0)].re,
            }
        })
        .collect();
    spikes.sort_by(|a, b| a.theta.total_cmp(&b.theta).then(a.phi.total_cmp(&b.phi)));

    // Residual of the recovered physical model.
    let mut residual_sq = 0.0;
    for a in 0..h1 {
        for b in 0..h2 {
            let mut fit = c64::new(0.0, 0.0);
            for spike in &spikes {
                let zp = c64::new((TAU * spike.theta).cos(), (TAU * spike.theta).sin());
                let wp = c64::new((TAU * spike.phi).cos(), (TAU * spike.phi).sin());
                fit += c64::new(spike.amplitude, 0.0) * zp.powu(a as u32) * wp.powu(b as u32);
            }
            residual_sq += (grid[(a, b)] - fit).norm_sqr();
        }
    }

    Ok(TorusRecovery {
        spikes,
        model_order,
        residual: residual_sq.sqrt(),
        enhanced_singular_values: singular_values,
    })
}

// ============================================================================
// Polish — damped Gauss–Newton on the original nonconvex objective
// ============================================================================

/// Tuning for [`polish_spikes`].
#[derive(Clone, Debug)]
pub struct PolishOptions {
    /// Maximum outer Gauss–Newton iterations.
    pub max_iters: usize,
    /// Initial Levenberg–Marquardt damping `μ` (added as `μ‖δ‖²`). Small so the
    /// first step is nearly a pure Gauss–Newton step from a good pencil seed.
    pub initial_damping: f64,
    /// Stop when the residual norm improves by less than this between outer
    /// iterations (a local minimum has been reached to working precision).
    pub residual_tol: f64,
}

impl Default for PolishOptions {
    fn default() -> Self {
        Self {
            max_iters: 64,
            initial_damping: 1e-6,
            residual_tol: 1e-12,
        }
    }
}

/// Spike parameters in the original (un-lifted) coordinates: per-spike amplitude
/// and latent coordinate `t_j ∈ ℝ^d`.
#[derive(Clone, Debug)]
pub struct PolishState {
    /// Amplitudes `a_j`, one per spike.
    pub amplitudes: Vec<f64>,
    /// Latent coordinates `t_j`, `coords[j]` of length `d`.
    pub coords: Vec<Vec<f64>>,
}

/// Outcome of [`polish_spikes`].
#[derive(Clone, Debug)]
pub struct PolishResult {
    /// Polished parameters.
    pub state: PolishState,
    /// Final residual `‖z − Σ_j a_j Φ(t_j)‖₂`.
    pub residual: f64,
    /// Outer iterations actually taken.
    pub iterations: usize,
    /// `true` if the loop stopped on the residual-improvement tolerance or a
    /// damped step could no longer improve the residual (local optimum), `false`
    /// if it exhausted `max_iters`.
    pub converged: bool,
}

/// Damped Gauss–Newton (Levenberg–Marquardt) polish of a spike model on the
/// *original* nonconvex objective `‖z − Σ_j a_j Φ(t_j)‖²`.
///
/// The pencil / eigen descents above are exact only in the noiseless limit; this
/// refines their output against the true per-atom objective. `observed` is the
/// atom's within-block code `z ∈ ℝ^D`; `phi` evaluates the atom's basis at a
/// single latent point, returning `(Φ(t), ∂Φ/∂t)` with `Φ(t)` of length `D` and
/// the Jacobian of shape `(D, d)`. The closure interface keeps this file free of
/// the heavy fit drivers: any atom that can evaluate its basis and Jacobian can
/// be polished. `init` seeds the parameters (typically from a lifted descent).
///
/// Each iteration forms the model Jacobian `J` (columns `∂/∂a_j = Φ(t_j)` and
/// `∂/∂t_{j,c} = a_j ∂Φ/∂t_c`), solves the damped normal equations
/// `min_δ ‖J δ − r‖² + μ‖δ‖²` by stacked QR, and accepts the step only if it
/// lowers the residual — backtracking on `μ` otherwise. Reports the final
/// residual so the caller can gate acceptance.
pub fn polish_spikes<F>(
    observed: &[f64],
    phi: F,
    init: PolishState,
    opts: &PolishOptions,
) -> Result<PolishResult, String>
where
    F: Fn(&[f64]) -> (Array1<f64>, Array2<f64>),
{
    let big_d = observed.len();
    let m = init.amplitudes.len();
    if m == 0 || init.coords.len() != m {
        return Err(format!(
            "polish_spikes: amplitudes ({}) and coords ({}) must have equal positive length",
            m,
            init.coords.len()
        ));
    }
    if big_d == 0 {
        return Err("polish_spikes: observed code must be non-empty".into());
    }
    let d = init.coords[0].len();
    if d == 0 || init.coords.iter().any(|c| c.len() != d) {
        return Err("polish_spikes: all latent coords must share one positive dimension".into());
    }
    if observed.iter().any(|x| !x.is_finite()) {
        return Err("polish_spikes: observed code has non-finite entries".into());
    }

    let n_params = m * (d + 1);

    // Evaluate the model residual `r = z − Σ_j a_j Φ(t_j)` and its norm, caching
    // the per-spike `(Φ_j, J_j)` used to build the Jacobian.
    let eval = |state: &PolishState| -> Result<(Vec<f64>, f64, Vec<(Array1<f64>, Array2<f64>)>), String> {
        let mut fit = vec![0.0_f64; big_d];
        let mut cache = Vec::with_capacity(m);
        for j in 0..m {
            let (phi_j, jac_j) = phi(&state.coords[j]);
            if phi_j.len() != big_d || jac_j.dim() != (big_d, d) {
                return Err(format!(
                    "polish_spikes: phi returned Φ len {} / J {:?}, expected {big_d} / ({big_d}, {d})",
                    phi_j.len(),
                    jac_j.dim()
                ));
            }
            let a = state.amplitudes[j];
            for i in 0..big_d {
                fit[i] += a * phi_j[i];
            }
            cache.push((phi_j, jac_j));
        }
        let mut r = vec![0.0_f64; big_d];
        let mut nsq = 0.0;
        for i in 0..big_d {
            r[i] = observed[i] - fit[i];
            nsq += r[i] * r[i];
        }
        Ok((r, nsq.sqrt(), cache))
    };

    let mut state = init;
    let (mut r, mut res_norm, mut cache) = eval(&state)?;
    let mut mu = opts.initial_damping;
    let mut converged = false;
    let mut iterations = 0;

    for _ in 0..opts.max_iters {
        iterations += 1;

        // Jacobian J (D × n_params): column j*(d+1) is Φ_j (∂/∂a_j); columns
        // j*(d+1)+1+c are a_j·∂Φ/∂t_c.
        let mut jdata = vec![0.0_f64; big_d * n_params];
        for j in 0..m {
            let (phi_j, jac_j) = &cache[j];
            let a = state.amplitudes[j];
            let base = j * (d + 1);
            for i in 0..big_d {
                jdata[i * n_params + base] = phi_j[i];
                for c in 0..d {
                    jdata[i * n_params + base + 1 + c] = a * jac_j[[i, c]];
                }
            }
        }

        // Damped step with backtracking on μ. Solve the stacked least-squares
        // problem [J; √μ I] δ = [r; 0].
        let mut stepped = false;
        for _ in 0..24 {
            let sqrt_mu = mu.sqrt();
            let aug = Mat::<f64>::from_fn(big_d + n_params, n_params, |row, col| {
                if row < big_d {
                    jdata[row * n_params + col]
                } else if row - big_d == col {
                    sqrt_mu
                } else {
                    0.0
                }
            });
            let rhs = Mat::<f64>::from_fn(big_d + n_params, 1, |row, _| {
                if row < big_d { r[row] } else { 0.0 }
            });
            let delta = aug.qr().solve_lstsq(&rhs);

            let mut trial = state.clone();
            for j in 0..m {
                let base = j * (d + 1);
                trial.amplitudes[j] += delta[(base, 0)];
                for c in 0..d {
                    trial.coords[j][c] += delta[(base + 1 + c, 0)];
                }
            }
            let (r_new, res_new, cache_new) = eval(&trial)?;
            if res_new < res_norm {
                let improvement = res_norm - res_new;
                state = trial;
                r = r_new;
                cache = cache_new;
                res_norm = res_new;
                mu = (mu * 0.5).max(1e-12);
                stepped = true;
                if improvement < opts.residual_tol {
                    converged = true;
                }
                break;
            }
            mu *= 4.0;
        }

        if !stepped {
            // No damped step improved the residual: a local optimum to working
            // precision.
            converged = true;
        }
        if converged {
            break;
        }
    }

    Ok(PolishResult {
        state,
        residual: res_norm,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt as _;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn gaussian(rng: &mut StdRng) -> f64 {
        let u1 = rng.random::<f64>().max(1e-16);
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    // ---- Sphere ----------------------------------------------------------

    /// Standard-basis direction `e_i` in dimension `d`.
    fn basis_dir(i: usize, d: usize) -> Vec<f64> {
        let mut v = vec![0.0; d];
        v[i] = 1.0;
        v
    }

    /// Chordal distance between two unit vectors, antipodal-aware.
    fn dir_dist(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        (1.0 - dot.abs()).max(0.0)
    }

    #[test]
    fn sphere_noiseless_roundtrip() {
        // m = 3 orthogonal directions in d = 4, distinct amplitudes.
        let d = 4;
        let planted = vec![
            SphereSpike { direction: basis_dir(0, d), amplitude: 2.0 },
            SphereSpike { direction: basis_dir(1, d), amplitude: 1.3 },
            SphereSpike { direction: basis_dir(2, d), amplitude: 0.7 },
        ];
        let m = sphere_lift(&planted, d).expect("lift");
        let rec = recover_sphere_spikes(m.view(), 0.0).expect("recover");
        assert_eq!(rec.model_order, 3, "order from clean spectrum");
        // Spikes come back sorted by amplitude descending == planted order.
        for (r, p) in rec.spikes.iter().zip(planted.iter()) {
            assert!(dir_dist(&r.direction, &p.direction) < 1e-8, "direction");
            assert!((r.amplitude - p.amplitude).abs() < 1e-8, "amplitude");
        }
        assert!(rec.residual < 1e-8, "residual {:.3e}", rec.residual);
    }

    #[test]
    fn sphere_antipodal_gauge_deterministic() {
        // v and −v produce the SAME lift and MUST canonicalise identically.
        let d = 3;
        let raw = vec![0.6, -0.8, 0.0];
        let neg = vec![-0.6, 0.8, 0.0];
        let m_pos = sphere_lift(&[SphereSpike { direction: raw.clone(), amplitude: 1.5 }], d).unwrap();
        let m_neg = sphere_lift(&[SphereSpike { direction: neg, amplitude: 1.5 }], d).unwrap();
        let rec_pos = recover_sphere_spikes(m_pos.view(), 0.0).unwrap();
        let rec_neg = recover_sphere_spikes(m_neg.view(), 0.0).unwrap();
        assert_eq!(rec_pos.spikes.len(), 1);
        assert_eq!(rec_neg.spikes.len(), 1);
        let mut canon = raw.clone();
        canonicalize_direction(&mut canon);
        for k in 0..d {
            assert!((rec_pos.spikes[0].direction[k] - canon[k]).abs() < 1e-10);
            assert!((rec_neg.spikes[0].direction[k] - canon[k]).abs() < 1e-10);
        }
    }

    #[test]
    fn sphere_multiplicity_detection() {
        // Exact m ∈ {1, 2, 3} on orthogonal, well-separated amplitudes.
        let d = 5;
        for m in 1..=3 {
            let amps = [2.5, 1.7, 1.0];
            let planted: Vec<SphereSpike> = (0..m)
                .map(|i| SphereSpike { direction: basis_dir(i, d), amplitude: amps[i] })
                .collect();
            let mut lift = sphere_lift(&planted, d).unwrap();
            // Tiny noise well below the amplitude scale.
            let mut rng = StdRng::seed_from_u64(100 + m as u64);
            let sigma = 1e-3;
            for i in 0..d {
                for j in i..d {
                    let e = sigma * gaussian(&mut rng);
                    lift[[i, j]] += e;
                    if i != j {
                        lift[[j, i]] += e;
                    }
                }
            }
            let rec = recover_sphere_spikes(lift.view(), sigma).unwrap();
            assert_eq!(rec.model_order, m, "multiplicity m={m}");
        }
    }

    #[test]
    fn sphere_noise_recovery() {
        // m = 2 orthogonal directions, d = 3, sigma = 0.05. Tolerances are
        // perturbation-theory order: Weyl bounds an eigenvalue shift by the noise
        // operator norm ‖E‖ ≈ 2σ√d ≈ 0.17, and a well-separated eigenvector's
        // first-order tilt is ‖E‖/gap ≈ 0.17/1.5 ≈ 0.11 rad, so the chordal
        // distance 1−|⟨v̂,v⟩| ≈ ‖tilt‖²/2 ≈ 0.006. The asserted bounds sit a few×
        // above these with a safety margin.
        let d = 3;
        let sigma = 0.05;
        let planted = vec![
            SphereSpike { direction: basis_dir(0, d), amplitude: 3.0 },
            SphereSpike { direction: basis_dir(1, d), amplitude: 1.5 },
        ];
        let mut lift = sphere_lift(&planted, d).unwrap();
        let mut rng = StdRng::seed_from_u64(2024);
        for i in 0..d {
            for j in i..d {
                let e = sigma * gaussian(&mut rng);
                lift[[i, j]] += e;
                if i != j {
                    lift[[j, i]] += e;
                }
            }
        }
        let rec = recover_sphere_spikes(lift.view(), sigma).unwrap();
        assert_eq!(rec.model_order, 2, "order under moderate noise");
        for (r, p) in rec.spikes.iter().zip(planted.iter()) {
            assert!(dir_dist(&r.direction, &p.direction) < 0.05, "direction {:.3e}", dir_dist(&r.direction, &p.direction));
            assert!((r.amplitude - p.amplitude).abs() < 0.5, "amplitude {:.3e}", (r.amplitude - p.amplitude).abs());
        }
    }

    // ---- Torus -----------------------------------------------------------

    fn torus_dist(a: f64, b: f64) -> f64 {
        let d = (a - b).abs();
        d.min(1.0 - d)
    }

    /// Greedy match recovered→planted by nearest (θ,φ); returns max position and
    /// amplitude error. Equal counts required.
    fn torus_match_error(rec: &[TorusSpike], planted: &[TorusSpike]) -> (f64, f64) {
        assert_eq!(rec.len(), planted.len(), "spike-count mismatch");
        let mut max_pos = 0.0_f64;
        let mut max_amp = 0.0_f64;
        let mut used = vec![false; planted.len()];
        for r in rec {
            let mut best = usize::MAX;
            let mut best_d = f64::INFINITY;
            for (j, p) in planted.iter().enumerate() {
                if used[j] {
                    continue;
                }
                let dd = torus_dist(r.theta, p.theta).max(torus_dist(r.phi, p.phi));
                if dd < best_d {
                    best_d = dd;
                    best = j;
                }
            }
            used[best] = true;
            max_pos = max_pos.max(best_d);
            max_amp = max_amp.max((r.amplitude - planted[best].amplitude).abs());
        }
        (max_pos, max_amp)
    }

    #[test]
    fn torus_noiseless_roundtrip_m1_m2() {
        let (h1, h2) = (6, 6);
        for planted in [
            vec![TorusSpike { theta: 0.23, phi: 0.61, amplitude: 1.4 }],
            vec![
                TorusSpike { theta: 0.15, phi: 0.72, amplitude: 1.0 },
                TorusSpike { theta: 0.63, phi: 0.28, amplitude: 0.8 },
            ],
        ] {
            let grid = torus_lift(&planted, h1, h2).unwrap();
            let rec = recover_torus_spikes(&grid, 0.0).unwrap();
            assert_eq!(rec.model_order, planted.len(), "order");
            let (pos_err, amp_err) = torus_match_error(&rec.spikes, &planted);
            assert!(pos_err < 1e-8, "position error {pos_err:.3e}");
            assert!(amp_err < 1e-8, "amplitude error {amp_err:.3e}");
            assert!(rec.residual < 1e-8, "residual {:.3e}", rec.residual);
        }
    }

    #[test]
    fn torus_multiplicity_detection() {
        let (h1, h2) = (6, 6);
        let candidates = [
            vec![TorusSpike { theta: 0.30, phi: 0.40, amplitude: 1.2 }],
            vec![
                TorusSpike { theta: 0.12, phi: 0.70, amplitude: 1.1 },
                TorusSpike { theta: 0.55, phi: 0.20, amplitude: 0.9 },
            ],
            vec![
                TorusSpike { theta: 0.10, phi: 0.15, amplitude: 1.3 },
                TorusSpike { theta: 0.45, phi: 0.62, amplitude: 1.0 },
                TorusSpike { theta: 0.80, phi: 0.35, amplitude: 0.8 },
            ],
        ];
        for planted in candidates {
            let grid = torus_lift(&planted, h1, h2).unwrap();
            let rec = recover_torus_spikes(&grid, 0.0).unwrap();
            assert_eq!(rec.model_order, planted.len(), "multiplicity");
            let (pos_err, _) = torus_match_error(&rec.spikes, &planted);
            assert!(pos_err < 1e-7, "position error {pos_err:.3e}");
        }
    }

    #[test]
    fn torus_adversarial_pairing() {
        // Adversarial fixture: θ₁ < θ₂ but φ₁ > φ₂. Estimating each axis
        // independently and re-pairing by sorting (θ ascending with φ ascending)
        // yields the GHOST spikes (θ₁,φ₂) and (θ₂,φ₁). The joint-diagonalisation
        // pencil must instead return the TRUE pairing (θ₁,φ₁), (θ₂,φ₂).
        let (h1, h2) = (7, 7);
        let planted = vec![
            TorusSpike { theta: 0.20, phi: 0.75, amplitude: 1.0 },
            TorusSpike { theta: 0.60, phi: 0.25, amplitude: 0.9 },
        ];
        let grid = torus_lift(&planted, h1, h2).unwrap();
        let rec = recover_torus_spikes(&grid, 0.0).unwrap();
        assert_eq!(rec.model_order, 2, "order");
        let (pos_err, _) = torus_match_error(&rec.spikes, &planted);
        assert!(pos_err < 1e-7, "correct-pairing position error {pos_err:.3e}");
        // Assert the GHOST pairing is NOT what came back: the ghost set has a
        // spike near (0.20, 0.25), which no true spike is close to.
        let ghost = [
            TorusSpike { theta: 0.20, phi: 0.25, amplitude: 1.0 },
            TorusSpike { theta: 0.60, phi: 0.75, amplitude: 0.9 },
        ];
        let (ghost_err, _) = torus_match_error(&rec.spikes, &ghost);
        assert!(ghost_err > 0.3, "recovery must not be the ghost pairing (err {ghost_err:.3e})");
    }

    #[test]
    fn torus_noise_recovery() {
        // Well-separated m = 2, sigma = 0.05. The 2-D pencil frequency error is
        // CRB-order: for an H×H grid the single-tone frequency CRB scales as
        // σ/(a·H^{3/2}) rad, i.e. ≈ 0.05/6^{1.5} ≈ 3e-3 rad ⇒ ≈ 5e-4 of a period;
        // the (non-efficient) pencil sits a small constant above this, so 0.03 of
        // a period is a safe several-× bound.
        let (h1, h2) = (6, 6);
        let sigma = 0.05;
        let planted = vec![
            TorusSpike { theta: 0.18, phi: 0.66, amplitude: 1.2 },
            TorusSpike { theta: 0.62, phi: 0.24, amplitude: 1.0 },
        ];
        let mut grid = torus_lift(&planted, h1, h2).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        for a in 0..h1 {
            for b in 0..h2 {
                let re = grid[(a, b)].re + sigma * gaussian(&mut rng);
                let im = grid[(a, b)].im + sigma * gaussian(&mut rng);
                grid[(a, b)] = c64::new(re, im);
            }
        }
        let rec = recover_torus_spikes(&grid, sigma).unwrap();
        assert_eq!(rec.model_order, 2, "order under noise");
        let (pos_err, amp_err) = torus_match_error(&rec.spikes, &planted);
        assert!(pos_err < 0.03, "position error {pos_err:.3e}");
        assert!(amp_err < 0.3, "amplitude error {amp_err:.3e}");
    }

    // ---- Polish ----------------------------------------------------------

    /// Harmonic-circle basis Φ(t) ∈ ℝ^{2H}: for h=1..H the pair
    /// (cos 2πht, sin 2πht), with Jacobian in the single latent coordinate. This
    /// is exactly the original per-atom objective the circle lift is convex-for.
    fn circle_phi(h_max: usize) -> impl Fn(&[f64]) -> (Array1<f64>, Array2<f64>) {
        move |t: &[f64]| {
            let t0 = t[0];
            let d = 2 * h_max;
            let mut phi = Array1::<f64>::zeros(d);
            let mut jac = Array2::<f64>::zeros((d, 1));
            for h in 1..=h_max {
                let w = TAU * h as f64;
                let (s, c) = (w * t0).sin_cos();
                phi[2 * (h - 1)] = c;
                phi[2 * (h - 1) + 1] = s;
                jac[[2 * (h - 1), 0]] = -w * s;
                jac[[2 * (h - 1) + 1, 0]] = w * c;
            }
            (phi, jac)
        }
    }

    fn circle_code(spikes: &[(f64, f64)], h_max: usize) -> Vec<f64> {
        let d = 2 * h_max;
        let mut z = vec![0.0; d];
        for &(t, a) in spikes {
            for h in 1..=h_max {
                let w = TAU * h as f64;
                z[2 * (h - 1)] += a * (w * t).cos();
                z[2 * (h - 1) + 1] += a * (w * t).sin();
            }
        }
        z
    }

    #[test]
    fn polish_converges_from_perturbed_seed() {
        let h_max = 6;
        let true_spikes = [(0.22, 1.1), (0.61, 0.8)];
        let z = circle_code(&true_spikes, h_max);
        // Seed the polish off the truth (as a noisy descent would land).
        let init = PolishState {
            amplitudes: vec![1.1 + 0.12, 0.8 - 0.1],
            coords: vec![vec![0.22 + 0.02], vec![0.61 - 0.025]],
        };
        let (_, seed_res, _) = {
            // Residual at the seed for a sanity floor.
            let phi = circle_phi(h_max);
            let mut fit = vec![0.0; z.len()];
            for j in 0..init.amplitudes.len() {
                let (p, _) = phi(&init.coords[j]);
                for i in 0..z.len() {
                    fit[i] += init.amplitudes[j] * p[i];
                }
            }
            let nsq: f64 = z.iter().zip(&fit).map(|(a, b)| (a - b) * (a - b)).sum();
            ((), nsq.sqrt(), ())
        };
        let res = polish_spikes(&z, circle_phi(h_max), init, &PolishOptions::default())
            .expect("polish");
        assert!(res.converged, "should converge");
        assert!(res.residual < 1e-6, "residual {:.3e}", res.residual);
        assert!(res.residual < seed_res, "polish must reduce the residual");
    }

    #[test]
    fn polish_rejects_shape_mismatch() {
        let h_max = 4;
        let z = circle_code(&[(0.3, 1.0)], h_max);
        // phi returns the wrong length ⇒ must error, not panic.
        let bad = |_t: &[f64]| (Array1::<f64>::zeros(3), Array2::<f64>::zeros((3, 1)));
        let init = PolishState { amplitudes: vec![1.0], coords: vec![vec![0.3]] };
        assert!(polish_spikes(&z, bad, init, &PolishOptions::default()).is_err());
    }
}
