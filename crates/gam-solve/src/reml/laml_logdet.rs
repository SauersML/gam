//! `log|H|` for the LAML/REML criterion, priced from a ROOT of
//! `H = XᵀWX + S_λ + δI` instead of from the assembled matrix's spectrum.
//!
//! # Why the assembled matrix cannot answer this
//!
//! `H` is a sum of squares. Every backward-stable factorization of the
//! ASSEMBLED matrix — `eigh` and Cholesky alike — perturbs it by `O(ε‖H‖)`,
//! which moves `log|H| = Σ log σ_i` by `O(ε‖H‖·tr(H⁻¹)) = O(ε·κ(H))`. The
//! outer smoothing search drives `κ(H)` to whatever `λ_max/λ_min` it needs: on
//! `y ~ s(pc1,k=5) + s(pc2,k=5)` (binomial/logit, `bench/datasets/prostate.csv`)
//! it reaches `σ_max = 6.2e11` against `σ_min = 0.163`, and `log|H|` then
//! scatters by `6.6e-4` across evaluations at BIT-IDENTICAL `rho` — `σ_max` is
//! constant to every printed digit while `σ_min` moves by `6.2e-4` relative,
//! which is `ε·κ` exactly. The outer criterion's own relative cost floor there
//! is `1.8e-5`, so the line search is asked to resolve a decrease 15x smaller
//! than the noise on the function, reports `StepSizeTooSmall`, and the fit is
//! refused for non-stationarity at an interior PSD minimum (#2644).
//!
//! Perturbing a ROOT instead is a different error law. For `H = BᵀB`, a
//! backward error `‖ΔB‖ = O(ε‖B‖)` moves each singular value by
//! `Δσ_i(B) = O(ε·σ_max(B))`, so
//! `Δ log|H| = 2 Σ Δσ_i/σ_i = O(ε·σ_max(B)/σ_min(B)) = O(ε·√κ(H))`.
//! On the fixture above that is `2.2e-16 · 1.6e6 / 0.4 ≈ 9e-10` instead of
//! `6.6e-4`.
//!
//! # Why the root has to come from the rows
//!
//! `G = XᵀWX` cannot be recovered from `H` by subtracting the (exactly known)
//! penalty: `H` was already rounded when it was assembled, so `H − S_λ − δI`
//! returns `G` with `O(ε‖H‖)` ABSOLUTE error — `1.4e-4` against a `‖G‖` of
//! `105` on the fixture above, i.e. `1.3e-6` relative, which reproduces the
//! same `O(ε·κ)` in the log-determinant. The information is destroyed at
//! assembly. So the data half of the root is built from `√W·X` directly, and
//! the penalty half from the per-component roots the penalties already carry.
//!
//! # What this refuses to do
//!
//! Everything here is a strict upgrade or nothing: every branch that cannot
//! establish that it is pricing the SAME `H` returns `None` and the caller
//! keeps the assembled value. In particular it declines on
//!
//! * a spectral floor that is actually regularizing (`log|H|` is then
//!   `Σ log r_ε(σ)`, which is not a log-determinant of anything),
//! * any negative weight (the observed information of a non-canonical link has
//!   no real root),
//! * a reconstruction `‖(G + S_λ + δI) − H‖` above roundoff (a Firth term, an
//!   active-constraint projection, or a frame mismatch — anything that means
//!   the caller's `H` is not the matrix assembled here),
//! * a disagreement with the assembled log-determinant larger than the
//!   assembled route's OWN error bound `p·ε·κ(H)` (the correction must be
//!   explicable as that route's error, not as a different quantity).
//!
//! The gate that decides whether to pay for it at all is the same
//! `√EPSILON` envelope the outer value-agreement audit is derived from
//! (`rho_optimizer::run::outer_value_agreement_bound`): pay when the assembled
//! route's error bound `p·ε·κ(H)` exceeds `√ε·(1+|log|H||)`, i.e. exactly when
//! the criterion cannot be reproduced to the tolerance its own consumers apply.

use gam_linalg::faer_ndarray::FaerSvd;
use gam_linalg::matrix::DesignMatrix;
use ndarray::{Array1, Array2, ArrayView1};

use gam_terms::construction::CanonicalPenalty;

/// The ingredients of `H = XᵀWX + Σ_k λ_k S_k + δI`, in ONE frame.
pub(crate) struct HessianRootInputs<'a> {
    pub design: &'a DesignMatrix,
    pub weights: ArrayView1<'a, f64>,
    pub penalties: &'a [CanonicalPenalty],
    pub lambdas: &'a [f64],
    pub delta: f64,
}

/// How far `Σ log σ_i` off the assembled spectrum can be from the truth.
///
/// `eigh` perturbs `H` by `O(p·ε·‖H‖)`, and `Δ log|H| = tr(H⁻¹ΔH)`, so the
/// bound is `p·ε·σ_max/σ_min` summed over the modes — `p·ε·κ` up to the same
/// constant. This is the quantity the whole module is about, so it is derived
/// here once and used both to decide whether to pay and to bound the accepted
/// correction.
pub(crate) fn assembled_logdet_error_bound(spectrum: &[f64]) -> f64 {
    let p = spectrum.len();
    if p == 0 {
        return 0.0;
    }
    let mut max = 0.0_f64;
    let mut min = f64::INFINITY;
    for &s in spectrum {
        if !s.is_finite() || s <= 0.0 {
            return f64::INFINITY;
        }
        max = max.max(s);
        min = min.min(s);
    }
    (p as f64) * f64::EPSILON * (max / min)
}

/// `true` when the assembled spectrum resolves `log|H|` more tightly than the
/// `√EPSILON` envelope its consumers compare criterion values against, so the
/// root-scale route would be pure cost.
pub(crate) fn assembled_logdet_is_resolved(spectrum: &[f64], assembled_logdet: f64) -> bool {
    let envelope = f64::EPSILON.sqrt() * (1.0 + assembled_logdet.abs());
    assembled_logdet_error_bound(spectrum) <= envelope
}

/// Rows of a root `R` with `RᵀR = S` for a symmetric PSD `S`, taken from `S`'s
/// OWN eigensystem and truncated at `S`'s own relative noise floor.
///
/// Used only on the UNPENALIZED Gram, whose spectrum is the data curvature and
/// therefore well-scaled — this eigendecomposition is an `O(ε)` operation, not
/// an `O(ε·κ(H))` one.
fn psd_root_rows(s: &Array2<f64>) -> Result<Vec<Array1<f64>>, String> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;
    let dim = s.nrows();
    if dim == 0 {
        return Ok(Vec::new());
    }
    let (evals, evecs) = s
        .eigh(Side::Lower)
        .map_err(|e| format!("rooting the unpenalized Gram failed: {e}"))?;
    let max = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    // The same relative floor the penalty side uses: `100 · p · ε · max|e|`.
    let threshold = 100.0 * (dim as f64) * f64::EPSILON * max;
    let mut rows = Vec::new();
    for i in 0..dim {
        let ev = evals[i];
        if !(ev.is_finite()) {
            return Err(format!("the unpenalized Gram has eigenvalue {ev}"));
        }
        if ev <= threshold {
            continue;
        }
        let scale = ev.sqrt();
        let mut row = Array1::<f64>::zeros(dim);
        for c in 0..dim {
            row[c] = scale * evecs[[c, i]];
        }
        rows.push(row);
    }
    Ok(rows)
}

/// The eigenvalues of a symmetric matrix, for callers whose operator does not
/// keep a spectrum (the Cholesky lane). `None` when the decomposition fails.
///
/// This costs one `O(p³)` eigendecomposition and is only ever called on the
/// branch the resolution gate has already selected, i.e. where the assembled
/// log-determinant provably cannot be trusted.
pub(crate) fn symmetric_spectrum(h: &Array2<f64>) -> Option<Vec<f64>> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;
    h.eigh(Side::Lower).ok().map(|(evals, _)| evals.to_vec())
}

/// `log|H|` from the singular values of `B = [√W·X ; √λ_k R_k ; √δ I]`.
///
/// Returns `None` — keep the assembled value — in every case listed in the
/// module header.
pub(crate) fn root_scale_hessian_logdet(
    inputs: &HessianRootInputs<'_>,
    h_assembled: &Array2<f64>,
    spectrum: &[f64],
    assembled_logdet: f64,
) -> Option<f64> {
    match root_scale_hessian_logdet_inner(inputs, h_assembled, spectrum, assembled_logdet) {
        Ok(value) => Some(value),
        Err(reason) => {
            log::debug!("[2644-logdet] declined: {reason}");
            None
        }
    }
}

fn root_scale_hessian_logdet_inner(
    inputs: &HessianRootInputs<'_>,
    h_assembled: &Array2<f64>,
    spectrum: &[f64],
    assembled_logdet: f64,
) -> Result<f64, String> {
    let p = h_assembled.nrows();
    if p == 0 || h_assembled.ncols() != p || spectrum.len() != p {
        return Err(format!(
            "shape mismatch: p={p}, H is {}x{}, spectrum has {} entries",
            h_assembled.nrows(), h_assembled.ncols(), spectrum.len()
        ));
    }
    if !(inputs.delta.is_finite() && inputs.delta >= 0.0) {
        return Err(format!("stabilization ridge is {}", inputs.delta));
    }
    if inputs.design.ncols() != p || inputs.design.nrows() != inputs.weights.len() {
        return Err(format!(
            "design is {}x{} against p={p} and {} weights",
            inputs.design.nrows(), inputs.design.ncols(), inputs.weights.len()
        ));
    }
    if inputs.lambdas.len() != inputs.penalties.len() {
        return Err(format!(
            "{} lambdas against {} penalties",
            inputs.lambdas.len(), inputs.penalties.len()
        ));
    }
    // A negative weight makes `XᵀWX` indefinite, so its root is complex and
    // `B` does not exist. `psd_root_rows` would silently drop those directions.
    // Decline instead.
    if inputs.weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
        return Err(
            "the Hessian weights are not all finite and non-negative, so H has no real root"
                .to_string(),
        );
    }

    // ── The data half.
    //
    // `G = XᵀWX` is formed on its OWN, before any penalty is added, and rooted
    // from its own eigensystem. That is the whole trick: `G`'s scale is the
    // data curvature (`‖G‖ ≈ 1e2` on the fixture in the module header) while
    // `‖H‖ ≈ 6e11`, so a Gram of the rows loses nothing that matters, whereas
    // recovering `G` from the assembled `H` by subtracting the penalty would
    // hand back `O(ε‖H‖)` of absolute error and reproduce the very defect this
    // module exists to remove. The design is never densified: `xt_diag_x_signed`
    // streams it, so a lazy or operator-backed design works too.
    let gram = gam_linalg::matrix::xt_diag_x_signed(
        inputs.design,
        gam_linalg::matrix::FiniteSignedWeightsView::try_from_array(&inputs.weights.to_owned())
            .map_err(|e| format!("Hessian weights are not a valid signed-weight view: {e}"))?,
    )
    .map_err(|e| format!("forming the unpenalized Gram XtWX failed: {e}"))?
    .to_dense();
    if gram.nrows() != p || gram.ncols() != p {
        return Err(format!(
            "the unpenalized Gram is {}x{} against p={p}",
            gram.nrows(),
            gram.ncols()
        ));
    }
    let mut rows: Vec<Array1<f64>> = Vec::with_capacity(3 * p);
    for row in psd_root_rows(&gram)? {
        rows.push(row);
    }

    // ── The penalty half. `CanonicalPenalty::root` is `rank × block_dim` with
    //    `S_k = rootᵀroot`, so `√λ_k · root` is exactly the block of `B` that
    //    contributes `λ_k S_k`.
    for (k, penalty) in inputs.penalties.iter().enumerate() {
        let lambda = inputs.lambdas[k];
        if !(lambda.is_finite() && lambda >= 0.0) {
            return Err(format!("penalty {k} carries lambda={lambda}"));
        }
        if lambda == 0.0 {
            continue;
        }
        let start = penalty.col_range.start;
        let end = penalty.col_range.end;
        if end > p || penalty.root.ncols() != end - start {
            return Err(format!(
                "penalty {k} root is {}x{} against col_range {start}..{end} in p={p}",
                penalty.root.nrows(), penalty.root.ncols()
            ));
        }
        let scale = lambda.sqrt();
        for r in 0..penalty.root.nrows() {
            let mut row = Array1::<f64>::zeros(p);
            for (local, global) in (start..end).enumerate() {
                row[global] = scale * penalty.root[[r, local]];
            }
            rows.push(row);
        }
    }

    // ── The stabilization ridge.
    if inputs.delta > 0.0 {
        let scale = inputs.delta.sqrt();
        for i in 0..p {
            let mut row = Array1::<f64>::zeros(p);
            row[i] = scale;
            rows.push(row);
        }
    }
    if rows.len() < p {
        // Fewer rows than columns: `BᵀB` is singular and `log|H|` is `-inf`
        // for this root, which disagrees with any finite assembled value.
        return Err(format!("the root has {} rows for p={p} columns", rows.len()));
    }

    let mut stacked = Array2::<f64>::zeros((rows.len(), p));
    for (i, row) in rows.iter().enumerate() {
        stacked.row_mut(i).assign(row);
    }

    // ── Self-verification: the root must reproduce the caller's `H`.
    //
    // This is what makes the substitution safe without knowing which inner
    // solver produced `H`, which weights it used, or whether a Firth term or an
    // active-constraint projection is in play. `BᵀB` is formed once, at the
    // same `O(n·p²)` the data root already cost.
    let reconstructed = stacked.t().dot(&stacked);
    let mut worst = 0.0_f64;
    let mut scale = 0.0_f64;
    for i in 0..p {
        for j in 0..p {
            worst = worst.max((reconstructed[[i, j]] - h_assembled[[i, j]]).abs());
            scale = scale.max(h_assembled[[i, j]].abs());
        }
    }
    // `BᵀB` and the caller's `H` are two assemblies of the same sum, so they
    // may differ by their own accumulation roundoff and nothing more.
    let reconstruction_tolerance = 64.0 * (p as f64) * f64::EPSILON * scale.max(1.0);
    if !(worst <= reconstruction_tolerance) {
        return Err(format!(
            "the root reproduces H only to {worst:.3e} against a roundoff tolerance of \
             {reconstruction_tolerance:.3e} (max|H|={scale:.3e}); the caller's Hessian is not \
             XtWX + sum(lambda S) + delta I"
        ));
    }

    let (_, singular, _) = stacked
        .svd(false, false)
        .map_err(|_| "the stacked-root SVD did not converge".to_string())?;
    if singular.len() < p {
        return Err(format!("SVD returned {} singular values for p={p}", singular.len()));
    }
    let mut logdet = 0.0_f64;
    for i in 0..p {
        let sigma = singular[i];
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(format!("singular value {i} of the root is {sigma:.3e}"));
        }
        logdet += 2.0 * sigma.ln();
    }
    if !logdet.is_finite() {
        return Err(format!("the root log-determinant is {logdet}"));
    }

    // ── The correction must be explicable as the assembled route's own error.
    let bound = assembled_logdet_error_bound(spectrum);
    let allowed = 16.0 * bound.max(f64::EPSILON.sqrt() * (1.0 + assembled_logdet.abs()));
    if (logdet - assembled_logdet).abs() > allowed {
        return Err(format!(
            "root={logdet:.9e} vs assembled={assembled_logdet:.9e}, gap {:.3e} exceeds the \
             assembled route's own error budget {allowed:.3e}; that is a different quantity, \
             not a sharper one",
            (logdet - assembled_logdet).abs(),
        ));
    }
    log::debug!(
        "[2644-logdet] installed: {logdet:.12e} (assembled {assembled_logdet:.12e}, correction \
         {:.3e}, assembled error bound {bound:.3e})",
        logdet - assembled_logdet,
    );
    Ok(logdet)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_terms::construction::CanonicalPenalty;
    use ndarray::Array2;

    /// A fixed orthogonal matrix with no axis-aligned column, from a Givens
    /// product. Deterministic, no RNG. The rotation is the point: on an
    /// axis-aligned fixture the assembled routes are exact and the test would
    /// measure nothing.
    fn dense_orthogonal(p: usize) -> Array2<f64> {
        let mut q = Array2::<f64>::eye(p);
        for i in 0..p {
            for j in (i + 1)..p {
                let theta = 0.29 + 0.23 * (i as f64) + 0.07 * (j as f64);
                let (sin, cos) = theta.sin_cos();
                for row in 0..p {
                    let a = q[[row, i]];
                    let b = q[[row, j]];
                    q[[row, i]] = cos * a - sin * b;
                    q[[row, j]] = sin * a + cos * b;
                }
            }
        }
        q
    }

    /// #2644. `log|H|` must be priced from a ROOT of `XᵀWX + λS + δI`, not from
    /// the assembled matrix's spectrum, once `κ(H)` passes what `ε·κ` can
    /// resolve.
    ///
    /// The fixture reproduces the shape the `prostate` fit reaches: a rank-3
    /// penalty at `λ = 6.2e11` on three of six directions, against a data
    /// curvature of `O(1e2)` on all six. `H` and the penalty share one
    /// eigenbasis, so the exact `log|H|` is closed-form and the assertion is
    /// against arithmetic rather than against another route.
    ///
    /// `κ(H) = 3.8e12` here, so `p·ε·κ ≈ 9e-3`: the assembled spectrum's own
    /// error budget is four hundred times the `1.8e-5` relative cost floor the
    /// outer line search is asked to resolve on that fit.
    #[test]
    fn root_scale_hessian_logdet_beats_the_assembled_spectrum_at_prostate_conditioning() {
        let p = 6usize;
        let q = dense_orthogonal(p);
        let rotate = |d: &[f64]| -> Array2<f64> {
            let mut m = Array2::<f64>::zeros((p, p));
            for (i, &di) in d.iter().enumerate() {
                if di == 0.0 {
                    continue;
                }
                for r in 0..p {
                    for c in 0..p {
                        m[[r, c]] += di * q[[r, i]] * q[[c, i]];
                    }
                }
            }
            let mt = m.t().to_owned();
            m += &mt;
            m *= 0.5;
            m
        };

        let d_pen: [f64; 6] = [1.0, 0.7, 0.44, 0.0, 0.0, 0.0];
        let d_data: [f64; 6] = [105.0, 15.5, 8.1, 3.3, 1.03, 0.1627];
        let lambda = 6.193e11_f64;
        let delta = 1.0e-8_f64;

        // The "design": `X = diag(√d_data)·Qᵀ` with unit weights reproduces
        // `XᵀWX = Q diag(d_data) Qᵀ` exactly, which is what a root-scale route
        // must be able to use and an assembled-`H` route cannot recover.
        let mut x = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            let scale = d_data[i].sqrt();
            for c in 0..p {
                x[[i, c]] = scale * q[[c, i]];
            }
        }
        let weights = Array1::<f64>::ones(p);

        let mut penalty_root = Array2::<f64>::zeros((3, p));
        for i in 0..3 {
            let scale = d_pen[i].sqrt();
            for c in 0..p {
                penalty_root[[i, c]] = scale * q[[c, i]];
            }
        }
        let penalty = CanonicalPenalty::from_dense_root(penalty_root.clone(), p);

        let h = rotate(&d_data) + rotate(&d_pen).mapv(|v| v * lambda) + Array2::<f64>::eye(p) * delta;
        let spectrum: Vec<f64> = (0..p).map(|i| d_data[i] + lambda * d_pen[i] + delta).collect();
        let exact: f64 = spectrum.iter().map(|s| s.ln()).sum();

        // What the assembled route reports, from its own eigendecomposition of
        // the very matrix the fitter would hand it.
        let assembled: f64 = symmetric_spectrum(&h)
            .expect("SPD fixture")
            .iter()
            .map(|s| s.ln())
            .sum();

        let design = gam_linalg::matrix::DesignMatrix::from(x);
        let inputs = HessianRootInputs {
            design: &design,
            weights: weights.view(),
            penalties: std::slice::from_ref(&penalty),
            lambdas: &[lambda],
            delta,
        };
        let sorted_spectrum = {
            let mut s = spectrum.clone();
            s.sort_by(f64::total_cmp);
            s
        };
        let from_root = root_scale_hessian_logdet(&inputs, &h, &sorted_spectrum, assembled)
            .expect("the root reproduces H, so the upgrade must be taken");

        assert!(
            (from_root - exact).abs() <= 1.0e-9 * exact.abs().max(1.0),
            "root-scale log|H|: got {from_root}, exact {exact}, assembled {assembled}"
        );
        // The bound this route is judged against — and the reason it exists.
        let budget = assembled_logdet_error_bound(&sorted_spectrum);
        assert!(
            budget > 1.0e-3,
            "the fixture must be in the regime where the assembled spectrum cannot \
             resolve the criterion; its error budget is {budget:.3e}"
        );
        assert!(
            !assembled_logdet_is_resolved(&sorted_spectrum, assembled),
            "the resolution gate must SELECT this fixture, or the upgrade never runs"
        );
    }

    /// The gate must decline a well-conditioned fit outright: the assembled
    /// spectrum resolves it, and the root-scale route would be pure cost.
    #[test]
    fn well_conditioned_hessians_are_declined_by_the_resolution_gate() {
        let spectrum = [1.0_f64, 2.0, 3.5, 10.0];
        let logdet: f64 = spectrum.iter().map(|s| s.ln()).sum();
        assert!(assembled_logdet_is_resolved(&spectrum, logdet));
        assert!(assembled_logdet_error_bound(&spectrum) < 1.0e-14);
    }
}
