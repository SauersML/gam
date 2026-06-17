//! Order-3 (quadratic-annihilating, r = 3) measure-jet energy via the two-pass frame trick.
//!
//! The affine-jet (r = 2) energy in [`super::measure_jet_smooth`] penalizes the
//! local residual of a weighted **degree-<2 (affine)** fit: it annihilates affine
//! functions and charges everything with curvature. To carry a *trend across a
//! gap* — the "peak in the gap" capability — the energy must instead annihilate
//! **quadratics** so that a smooth quadratic ridge running through a sampling gap
//! is paid for only by its *cubic* departure, not by its curvature. That is the
//! r = 3 energy: the local residual of a weighted **degree-<3** polynomial fit.
//!
//! ## The ambient-d⁴ obstruction and the two-pass cure
//!
//! Done naively in the ambient `d` coordinates, a degree-<3 fit needs the full
//! quadratic monomial design — `1 + d + d(d+1)/2` columns. Its normal equations
//! require moments through degree 4, because products of quadratic basis
//! functions have degree up to 4. For ambient `d` in the tens that is costly and
//! ill-conditioned (the data sits on a thin low-dimensional set, so many
//! quadratic monomials are degenerate noise).
//!
//! The cure is a **two-pass local fit** that exploits the same empirical-measure
//! geometry the r = 2 energy already learns:
//!
//! 1. **Pass 1 — frames from order-2 local moments.** At each outer center we
//!    form the *same* local weighted covariance `G = Φ̃ᵀWΦ̃/q` that the affine
//!    energy uses, eigendecompose it, and keep the top `q ≤ Q_MAX` eigenvectors
//!    `U` (`d × q`) as an orthonormal **intrinsic frame**. The data's local
//!    variation lives in `range(U)` up to the roundoff-floor directions, so the
//!    frame captures the tangent (and, where the set bends, the dominant normal)
//!    geometry without ever naming an ambient axis.
//!
//! 2. **Pass 2 — moments of frame-projected coordinates to degree 4.** We project
//!    the centered scaled features into the frame, `Y = Φ̃·U` (`ml × q`), and run
//!    the degree-<3 polynomial fit *in the frame coordinates*: the design columns
//!    are `[1, y_a, y_a·y_b (a ≤ b)]`, dimension
//!    `p₃ = 1 + q + q(q+1)/2 ≤ 1 + 8 + 36 = 45`. The local roughness is the
//!    weighted residual of *this* fit. This annihilates quadratics in the
//!    retained frame coordinates. It annihilates arbitrary ambient quadratics
//!    only when the retained frame spans the local affine hull; otherwise
//!    quadratic terms in dropped directions can leak. No ambient high-order
//!    object is built — the degree-4 moments live in the `q`-dimensional frame.
//!
//! The residual operator is the weighted-least-squares projector complement
//!
//! ```text
//!   R = W − W·P·(PᵀWP)⁺·PᵀW          (ml × ml, PSD, W = diag(local kernel)),
//! ```
//!
//! with `P` the degree-<3 frame design and `(·)⁺` the rank-revealing
//! pseudo-inverse (degenerate frame directions and aliased monomials carry zero
//! penalty rather than being ridged). Each scale's local `R` is scattered into the
//! `m × m` center-space energy with the same Mellin outer weight
//! `log_step · ε^{−η} · m_i · q^{1−2α}`, `η = 2s + d(2−2α)`, as the affine energy,
//! so the two energies
//! are the *same functional at a different polynomial degree* and compose cleanly.
//!
//! The first design column is the constant, so `R·1 = 0` exactly: the energy
//! annihilates constants (a fortiori affines and quadratics) to machine
//! precision. The whole construction is deterministic — no RNG, no neighbor graph
//! — so freeze→replay is bit-stable.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

use faer::Side;

use crate::linalg::faer_ndarray::FaerEigh;

use super::{BasisError, MeasureJetBand};

/// Truncation radius of the Gaussian profile in units of the scale ε. Mirrors
/// the affine energy: weights past `3ε` are below `e^{-4.5}` of the peak and are
/// dropped from both the local fit and the `q^{1−2α}` outer weight so the
/// truncated `q` stays self-consistent across both passes.
pub(crate) const MEASURE_JET3_PROFILE_CUTOFF: f64 = 3.0;

/// Relative eigenvalue threshold for the rank-revealing pseudo-inverses (frame
/// extraction and the degree-<3 normal-equation solve). Directions at the
/// roundoff floor are unresolved and excluded — never ridged.
pub(crate) const MEASURE_JET3_PSEUDOINVERSE_RTOL: f64 = 64.0 * f64::EPSILON;

/// Maximum intrinsic frame dimension `q` retained from the local order-2 moment
/// spectrum. The degree-<3 design then has at most `1 + 8 + 36 = 45` columns, so
/// the quadratic local fit stays bounded regardless of the ambient `d`. Data on a
/// set whose local affine hull has dimension `p ≤ q` is captured exactly; on richer sets the
/// frame keeps the `q` most energetic order-2 directions (magic by default: the
/// cap is derived from the polynomial budget, not exposed as a dial).
pub(crate) const MEASURE_JET3_FRAME_MAX: usize = 8;

/// Memory budget (in f64 entries) above which the per-scale assembly stops
/// parallelizing over scales — the partials cost `L · m²` doubles; past this the
/// scales run sequentially (identical numbers: the per-scale loop and the ordered
/// cross-scale sum are deterministic either way).
pub(crate) const MEASURE_JET3_PARALLEL_BUDGET_DOUBLES: usize = 1 << 26;

/// Pairwise squared distances `‖a_i − a_j‖²`, clamped at zero for roundoff.
pub(crate) fn pairwise_sq_dists(a: ArrayView2<'_, f64>) -> Array2<f64> {
    let norms: Vec<f64> = a.outer_iter().map(|r| r.dot(&r)).collect();
    let mut g = a.dot(&a.t());
    g.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (norms[i] + norms[j] - 2.0 * *v).max(0.0);
            }
        });
    g
}

/// Rank-revealing symmetric pseudo-inverse via the symmetric eigendecomposition,
/// with eigenvalues clamped at zero and a `RTOL · n · λ_max` rank cutoff. Mirrors
/// the affine energy so both layers treat unresolved directions identically.
pub(crate) fn symmetric_pseudoinverse(
    a: &Array2<f64>,
    label: &str,
) -> Result<Array2<f64>, BasisError> {
    let n = a.nrows();
    if a.ncols() != n {
        crate::bail_dim_basis!(
            "measure-jet r=3 pseudo-inverse `{label}` needs a square matrix, got {:?}",
            a.dim()
        );
    }
    if n == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let (evals, evecs) = a.eigh(Side::Lower).map_err(|e| {
        BasisError::InvalidInput(format!(
            "measure-jet r=3 pseudo-inverse `{label}` eigendecomposition failed: {e}"
        ))
    })?;
    let lam_max = evals.iter().fold(0.0_f64, |acc, v| acc.max((*v).max(0.0)));
    let rank_tol = MEASURE_JET3_PSEUDOINVERSE_RTOL * (n as f64) * lam_max;
    let mut scaled = evecs.clone();
    for (k, mut col) in scaled.axis_iter_mut(Axis(1)).enumerate() {
        let lam = evals[k].max(0.0);
        let inv = if lam > rank_tol { 1.0 / lam } else { 0.0 };
        col.mapv_inplace(|v| v * inv);
    }
    Ok(scaled.dot(&evecs.t()))
}

/// The order-3 (cubic) multiscale measure-jet energy `Q` (`m × m`, symmetric
/// PSD) on the center set. Each `(scale, outer-net center)` contributes the
/// weighted residual of a **degree-<3** polynomial fit in the local intrinsic
/// frame — annihilating ambient quadratics so smooth quadratic trends bridge
/// sampling gaps and only the cubic departure is penalized.
///
/// `centers` is `m × d`, `masses` length `m`. `band`, `order_s`, `alpha`, `tau0`
/// match the affine [`super::measure_jet_energy_form`] contract; `order_s` lives
/// in `(0, 3)` here because the r = 3 jet is pointwise-defined for intrinsic
/// `p ≤ 4`. The τ slot is retained for layout parity and does not enter the
/// energy (the projection is exact).
pub fn measure_jet_order3_energy_form(
    centers: ArrayView2<'_, f64>,
    masses: ArrayView1<'_, f64>,
    band: &MeasureJetBand,
    order_s: f64,
    alpha: f64,
    tau0: f64,
) -> Result<Array2<f64>, BasisError> {
    let m = centers.nrows();
    let d = centers.ncols();
    if masses.len() != m {
        crate::bail_dim_basis!(
            "measure-jet r=3 mass/center mismatch: {} masses for {} centers",
            masses.len(),
            m
        );
    }
    if band.eps.is_empty() || band.eps.iter().any(|e| !(e.is_finite() && *e > 0.0)) {
        crate::bail_invalid_basis!("measure-jet r=3 energy needs a nonempty positive scale band");
    }
    if !(order_s.is_finite() && order_s > 0.0 && order_s < 3.0) {
        crate::bail_invalid_basis!(
            "measure-jet r=3 order s must lie in (0, 3) for the cubic-jet energy; got {order_s}"
        );
    }
    if !(alpha.is_finite() && tau0.is_finite() && tau0 >= 0.0) {
        crate::bail_invalid_basis!(
            "measure-jet r=3 energy needs finite alpha and finite tau0 >= 0; got alpha={alpha}, tau0={tau0}"
        );
    }
    if masses.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
        crate::bail_invalid_basis!("measure-jet r=3 energy needs finite nonnegative center masses");
    }
    if m == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }

    let dist2 = pairwise_sq_dists(centers);

    // One `m × m` accumulator per scale. Each scale's center loop is sequential
    // and the cross-scale sum runs in band order, so the result is
    // bit-deterministic whether or not the scales themselves run in parallel.
    let assemble_scale = |eps: f64| -> Result<Array2<f64>, BasisError> {
        let mut out = Array2::<f64>::zeros((m, m));
        let cutoff2 = (MEASURE_JET3_PROFILE_CUTOFF * eps) * (MEASURE_JET3_PROFILE_CUTOFF * eps);
        let inv_two_eps2 = 1.0 / (2.0 * eps * eps);
        let eta = 2.0 * order_s + (d as f64) * (2.0 - 2.0 * alpha);
        let scale_weight = band.log_step * eps.powf(-eta);

        // Outer-quadrature coarsening: greedy ε/2-net over the centers in fixed
        // index order (deterministic), every center's mass aggregated to its
        // nearest net member (lowest-index tie break). The inner local fit still
        // uses the full center set, so the local residual identities (exact
        // constant annihilation, PSD) are untouched.
        let net_radius2 = 0.25 * eps * eps;
        let mut outer: Vec<usize> = Vec::new();
        for i in 0..m {
            if masses[i] <= 0.0 {
                continue;
            }
            let covered = outer.iter().any(|&o| dist2[(i, o)] <= net_radius2);
            if !covered {
                outer.push(i);
            }
        }
        let mut net_mass = vec![0.0_f64; m];
        for i in 0..m {
            if masses[i] <= 0.0 {
                continue;
            }
            let mut best = f64::INFINITY;
            let mut best_o = usize::MAX;
            for &o in &outer {
                if dist2[(i, o)] < best {
                    best = dist2[(i, o)];
                    best_o = o;
                }
            }
            if best_o != usize::MAX {
                net_mass[best_o] += masses[i];
            }
        }

        for &i in &outer {
            // Local neighbor set (always includes i itself).
            let mut idx: Vec<usize> = Vec::new();
            for j in 0..m {
                if dist2[(i, j)] <= cutoff2 {
                    idx.push(j);
                }
            }
            let ml = idx.len();
            // Kernel weights W (diagonal) and truncated mass q.
            let mut w = Array1::<f64>::zeros(ml);
            let mut q = 0.0_f64;
            for (a, &j) in idx.iter().enumerate() {
                let wj = masses[j] * (-dist2[(i, j)] * inv_two_eps2).exp();
                w[a] = wj;
                q += wj;
            }
            if !(q > 0.0) {
                continue;
            }

            // Scaled local features Φ (ml × d) and weighted column mean a.
            let mut phi = Array2::<f64>::zeros((ml, d));
            for (a, &j) in idx.iter().enumerate() {
                for k in 0..d {
                    phi[(a, k)] = (centers[(j, k)] - centers[(i, k)]) / eps;
                }
            }
            let a_mean = phi.t().dot(&w) / q;
            // Weighted-centered features Φ̃ = Φ − 1·aᵀ (ml × d).
            let mut phi_c = phi.clone();
            for mut row in phi_c.outer_iter_mut() {
                for k in 0..d {
                    row[k] -= a_mean[k];
                }
            }

            // ---- PASS 1: intrinsic frame from order-2 moments ----
            // G = Φ̃ᵀWΦ̃ / q (d × d, the same centered Gram the affine energy
            // uses). Its top eigenvectors span the local intrinsic geometry.
            let mut wphic = phi_c.clone();
            for (a, mut row) in wphic.outer_iter_mut().enumerate() {
                row.mapv_inplace(|v| v * w[a]);
            }
            let mut g = phi_c.t().dot(&wphic);
            g.mapv_inplace(|v| v / q);
            let (g_evals, g_evecs) = g.eigh(Side::Lower).map_err(|e| {
                BasisError::InvalidInput(format!(
                    "measure-jet r=3 frame eigendecomposition failed: {e}"
                ))
            })?;
            // eigh returns ascending eigenvalues; keep the top up-to-Q_MAX
            // directions above the roundoff floor.
            let lam_max = g_evals
                .iter()
                .fold(0.0_f64, |acc, v| acc.max((*v).max(0.0)));
            if !(lam_max > 0.0) {
                continue;
            }
            let frame_tol = MEASURE_JET3_PSEUDOINVERSE_RTOL * (d.max(1) as f64) * lam_max;
            let mut keep: Vec<usize> = (0..g_evals.len())
                .rev()
                .filter(|&k| g_evals[k] > frame_tol)
                .collect();
            keep.truncate(MEASURE_JET3_FRAME_MAX);
            let qdim = keep.len();
            if qdim == 0 {
                continue;
            }
            // Frame U (d × qdim) and frame coordinates Y = Φ̃·U (ml × qdim).
            let mut u = Array2::<f64>::zeros((d, qdim));
            for (col, &k) in keep.iter().enumerate() {
                for r in 0..d {
                    u[(r, col)] = g_evecs[(r, k)];
                }
            }
            let y = phi_c.dot(&u);

            // ---- PASS 2: degree-<3 fit in the q-dim frame ----
            // Design columns: [1, y_a (a<qdim), y_a·y_b (a<=b)].
            // p3 = 1 + qdim + qdim(qdim+1)/2 ≤ 45.
            let n_quad = qdim * (qdim + 1) / 2;
            let p3 = 1 + qdim + n_quad;
            let mut pmat = Array2::<f64>::zeros((ml, p3));
            for a in 0..ml {
                pmat[(a, 0)] = 1.0;
                for c in 0..qdim {
                    pmat[(a, 1 + c)] = y[(a, c)];
                }
                let mut col = 1 + qdim;
                for c0 in 0..qdim {
                    for c1 in c0..qdim {
                        pmat[(a, col)] = y[(a, c0)] * y[(a, c1)];
                        col += 1;
                    }
                }
            }

            // Weighted least-squares residual operator
            //   R = W − W·P·(PᵀWP)⁺·PᵀW   (ml × ml, PSD).
            // The constant column makes R·1 = 0 exactly. Form WP, the normal
            // matrix PᵀWP, its pseudo-inverse, and the projector complement.
            let mut wp = pmat.clone();
            for (a, mut row) in wp.outer_iter_mut().enumerate() {
                row.mapv_inplace(|v| v * w[a]);
            }
            let ptwp = pmat.t().dot(&wp); // (PᵀW)P = PᵀWP, symmetric p3×p3
            let ptwp_pinv = symmetric_pseudoinverse(&ptwp, "degree-<3 frame normal matrix")?;
            // H = WP·(PᵀWP)⁺·(WP)ᵀ  (ml × ml), the weighted projection onto
            // range(P) in the W inner product. R = diag(w) − H.
            let tmp = wp.dot(&ptwp_pinv); // ml × p3
            let h = tmp.dot(&wp.t()); // ml × ml

            let base = scale_weight * net_mass[i] * q.powf(1.0 - 2.0 * alpha);

            // Scatter Σ_{a,c} base · R[a,c] into the center-space energy.
            for (a, &ja) in idx.iter().enumerate() {
                for (c, &jc) in idx.iter().enumerate() {
                    let mut r_ac = -h[(a, c)];
                    if a == c {
                        r_ac += w[a];
                    }
                    out[(ja, jc)] += base * r_ac;
                }
            }
        }
        Ok(out)
    };

    let n_scales = band.eps.len();
    let parallel_ok =
        m.saturating_mul(m).saturating_mul(n_scales) <= MEASURE_JET3_PARALLEL_BUDGET_DOUBLES;
    let per_scale: Vec<Array2<f64>> = if parallel_ok {
        band.eps
            .par_iter()
            .map(|&eps| assemble_scale(eps))
            .collect::<Result<Vec<_>, BasisError>>()?
    } else {
        band.eps
            .iter()
            .map(|&eps| assemble_scale(eps))
            .collect::<Result<Vec<_>, BasisError>>()?
    };

    let mut total = Array2::<f64>::zeros((m, m));
    for part in per_scale {
        total += &part;
    }
    // Numerical symmetrization (the analytic form is symmetric).
    Ok((&total + &total.t()) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terms::basis::measure_jet_band;
    use ndarray::array;

    /// Deterministic, well-spread 2D center cloud (no RNG). A 5×5 jittered grid
    /// gives a generic spread so the local degree-<3 design is full rank and the
    /// quadratic-annihilation test is non-trivial.
    pub(crate) fn grid_centers() -> Array2<f64> {
        let n = 5usize;
        let mut v: Vec<f64> = Vec::with_capacity(n * n * 2);
        for i in 0..n {
            for j in 0..n {
                // Deterministic jitter from a closed-form trig pattern.
                let xi = i as f64 / (n as f64 - 1.0);
                let yj = j as f64 / (n as f64 - 1.0);
                let jx = 0.04 * (3.0 * xi + 1.7 * yj).sin();
                let jy = 0.04 * (2.1 * yj - 1.3 * xi).cos();
                v.push(xi + jx);
                v.push(yj + jy);
            }
        }
        Array2::from_shape_vec((n * n, 2), v).expect("grid centers")
    }

    pub(crate) fn uniform_masses(m: usize) -> Array1<f64> {
        Array1::from_elem(m, 1.0 / m as f64)
    }

    pub(crate) fn band_for(centers: &Array2<f64>) -> MeasureJetBand {
        measure_jet_band(centers.view(), 0).expect("auto band")
    }

    /// Quadratic energy of a coefficient vector evaluated at the centers under
    /// an energy form: `f(centers)ᵀ Q f(centers)`.
    pub(crate) fn energy_of(q: &Array2<f64>, f: &Array1<f64>) -> f64 {
        f.dot(&q.dot(f))
    }

    /// Sample an ambient quadratic `f(x) = a + bᵀx + xᵀMx` at the centers.
    pub(crate) fn sample_quadratic(
        centers: &Array2<f64>,
        a: f64,
        b: [f64; 2],
        m: [[f64; 2]; 2],
    ) -> Array1<f64> {
        let n = centers.nrows();
        Array1::from_shape_fn(n, |i| {
            let x0 = centers[(i, 0)];
            let x1 = centers[(i, 1)];
            a + b[0] * x0
                + b[1] * x1
                + m[0][0] * x0 * x0
                + (m[0][1] + m[1][0]) * x0 * x1
                + m[1][1] * x1 * x1
        })
    }

    /// ORACLE 1 — the capability gain. The r=3 energy annihilates ambient
    /// QUADRATICS (peak-in-the-gap carrier), while the r=2 affine energy does
    /// NOT. We assert the r=3 quadratic energy is at the roundoff floor relative
    /// to a "rough" reference, AND that the r=2 energy of the *same* quadratic is
    /// strictly, substantially larger — proving the degree gain is real.
    #[test]
    pub(crate) fn order3_annihilates_ambient_quadratics_r2_does_not() {
        let centers = grid_centers();
        let masses = uniform_masses(centers.nrows());
        let band = band_for(&centers);

        let q3 =
            measure_jet_order3_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 1e-3)
                .expect("r=3 energy");
        let q2 = crate::terms::basis::measure_jet_energy_form(
            centers.view(),
            masses.view(),
            &band,
            1.5,
            1.0,
            1e-3,
        )
        .expect("r=2 energy");

        // A genuine quadratic with curvature in every channel.
        let fq = sample_quadratic(&centers, 0.3, [-0.7, 1.1], [[0.9, 0.4], [0.4, -0.6]]);

        // Roughness scale: the r=3 energy of a clearly cubic field. The cubic
        // carries energy that the quadratic must not, so it is a fair "rough"
        // yardstick for the relative-zero assertion.
        let n = centers.nrows();
        let fcubic = Array1::from_shape_fn(n, |i| {
            let x0 = centers[(i, 0)];
            let x1 = centers[(i, 1)];
            x0 * x0 * x0 - 1.4 * x1 * x1 * x1 + 0.8 * x0 * x0 * x1
        });
        let rough3 = energy_of(&q3, &fcubic).abs().max(1e-30);

        let e3_quad = energy_of(&q3, &fq);
        // (1) r=3 annihilates the quadratic to the roundoff floor.
        assert!(
            e3_quad.abs() <= 1e-8 * rough3,
            "r=3 quadratic energy {e3_quad:.3e} not annihilated (rough3 = {rough3:.3e})"
        );

        // (2) r=2 charges the SAME quadratic substantially — capability gain.
        let e2_quad = energy_of(&q2, &fq);
        assert!(
            e2_quad > 1e-6 * rough3,
            "r=2 quadratic energy {e2_quad:.3e} unexpectedly small; expected the affine \
             energy to penalize curvature (rough3 = {rough3:.3e})"
        );
        // The gain is not marginal: r=2 must dwarf the annihilated r=3 value.
        assert!(
            e2_quad > 1e6 * e3_quad.abs().max(f64::MIN_POSITIVE),
            "expected r=2 quadratic energy ({e2_quad:.3e}) >> r=3 ({e3_quad:.3e})"
        );
    }

    /// ORACLE 2 — the form is PSD. Deterministic test vectors (closed-form,
    /// no RNG) must all give nonnegative energy.
    #[test]
    pub(crate) fn order3_energy_form_is_psd() {
        let centers = grid_centers();
        let masses = uniform_masses(centers.nrows());
        let band = band_for(&centers);
        let q =
            measure_jet_order3_energy_form(centers.view(), masses.view(), &band, 1.5, 1.0, 1e-3)
                .expect("r=3 energy");
        let m = q.nrows();
        let scale = q.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(scale > 0.0, "r=3 energy form is identically zero");
        for trial in 0..8usize {
            let v = Array1::from_shape_fn(m, |i| ((i * 13 + trial * 29) % 23) as f64 / 23.0 - 0.5);
            let e = energy_of(&q, &v);
            assert!(
                e >= -1e-9 * scale,
                "vᵀQv = {e:.3e} < 0 on trial {trial} (scale {scale:.3e})"
            );
        }
        // Constants are annihilated exactly (a fortiori PSD-consistent).
        let ones = Array1::ones(m);
        let qv = q.dot(&ones);
        let leak = qv.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            leak <= 1e-10 * scale,
            "Q·1 leak {leak:.3e} vs scale {scale:.3e}"
        );
    }

    /// ORACLE 3 — scale concentration. A single fine sinusoid deposits its
    /// energy at the FINE scales of the band: the highest-frequency sinusoid's
    /// per-scale energy must peak at a finer scale than a slow sinusoid's.
    /// (Coarse scales see a high-frequency sinusoid as cubic-irreducible noise
    /// averaged toward zero, while fine scales resolve its local curvature
    /// departure.) Deterministic, no RNG.
    #[test]
    pub(crate) fn order3_sinusoid_energy_concentrates_at_fine_scales() {
        let centers = grid_centers();
        let masses = uniform_masses(centers.nrows());
        let band = band_for(&centers);
        let n = centers.nrows();

        // Per-scale energy of a field f: assemble the band one scale at a time.
        let per_scale_energy = |f: &Array1<f64>| -> Vec<f64> {
            band.eps
                .iter()
                .map(|&e| {
                    let single = MeasureJetBand {
                        eps: vec![e],
                        log_step: band.log_step,
                    };
                    let q = measure_jet_order3_energy_form(
                        centers.view(),
                        masses.view(),
                        &single,
                        1.5,
                        1.0,
                        1e-3,
                    )
                    .expect("single-scale r=3 energy");
                    energy_of(&q, f).max(0.0)
                })
                .collect()
        };

        let sinusoid = |freq: f64| -> Array1<f64> {
            Array1::from_shape_fn(n, |i| {
                let x0 = centers[(i, 0)];
                let x1 = centers[(i, 1)];
                (freq * (x0 + 0.5 * x1)).sin()
            })
        };

        // Argmax (finest = index 0, coarsest = last) of the per-scale energy.
        let weighted_scale_index = |es: &[f64]| -> f64 {
            let tot: f64 = es.iter().sum::<f64>().max(1e-300);
            es.iter()
                .enumerate()
                .map(|(k, e)| k as f64 * e)
                .sum::<f64>()
                / tot
        };

        let f_fast = sinusoid(14.0);
        let f_slow = sinusoid(3.0);
        let es_fast = per_scale_energy(&f_fast);
        let es_slow = per_scale_energy(&f_slow);

        // band.eps is ascending (fine → coarse), so a smaller weighted index
        // means the energy sits at finer scales. The fast sinusoid must
        // concentrate strictly finer than the slow one.
        let idx_fast = weighted_scale_index(&es_fast);
        let idx_slow = weighted_scale_index(&es_slow);
        assert!(
            idx_fast < idx_slow,
            "fine sinusoid did not concentrate at finer scales: \
             idx_fast {idx_fast:.3} should be < idx_slow {idx_slow:.3} \
             (es_fast = {es_fast:?}, es_slow = {es_slow:?})"
        );
    }
}
