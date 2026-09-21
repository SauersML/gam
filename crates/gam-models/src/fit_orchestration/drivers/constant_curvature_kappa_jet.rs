// #2458 — the constant-curvature ψ profile's exact derivative jet.
// #2747 — ψ is TWO coordinates, not one.
//
// Extracted from `spatial_optimization.rs` rather than added to it: that file
// is 9231 lines at the point this landed, and a file that crosses 10,000 goes on
// permanent probation requiring ≤7000, which is a 30% cut rather than a trim.
// `include!`d into `drivers/mod.rs` exactly like the sibling files, so the flat
// module namespace and every private-item reference are unchanged.

/// Value, exact gradient AND exact Hessian of the λ-profiled Gaussian REML
/// negative log evidence in the constant-curvature smooth's TWO outer
/// coordinates `ψ = (κ, η)`, `η = ln ℓ`.
///
/// # Why two coordinates (#2747)
///
/// `exp(−d_κ/ℓ)` has a curvature and a range in one exponent, and they are
/// strongly confounded: to first order `d_κ = d_0·(1 + κ·a(x,y))`, so the MEAN
/// of `a` acts exactly like a rescaling of `ℓ`. Optimizing κ at a PINNED `ℓ`
/// therefore does not estimate curvature — κ absorbs the range error, which is
/// monotone in one direction, and `V_p(κ)` rails. Measured: with `ℓ` pinned the
/// criterion recovers a planted κ⋆ only when the truth's own radial length
/// scale is the heuristic's; at half or twice it rails, inverts the sign, or
/// invents an interior `κ̂ = ∓0.94` from flat data.
///
/// # Why exact second derivatives (#2458)
///
/// The outer stationarity certificate accepts a point when no computation could
/// distinguish it from a stationary one — `|Pg| ≤ √(2·h·τ)`, with `h` the
/// curvature along the projected gradient and `τ` the criterion's own forward
/// resolution. Every route that cannot supply `h` was instead held to a raw
/// gradient band with no derivation, which is how one subsystem ended up with
/// stationarity bounds spanning five orders **tiered by which derivative
/// machinery each route happens to implement**. This function supplies `h`, so
/// that route stops being judged by its implementation.
///
/// # The expression
///
/// Write `F(ψ, ρ)` for the REML score at FIXED `ρ = log λ`, with the design
/// `X(ψ)` and penalty `S(ψ)`; the profile is `V(ψ) = F(ψ, ρ̂(ψ))`. The envelope
/// theorem gives `∇V = F_ψ`, but it does NOT carry to second order —
/// differentiating again picks up the ρ̂ response:
///
/// ```text
///   ∇²V = F_ψψ − F_ψρ F_ψρᵀ / F_ρρ
/// ```
///
/// the Schur complement of the λ block. `F_ρρ` is already computed by the
/// forward fit (`reml_hess_rho`); `F_ψψ` and `F_ψρ` are computed here.
///
/// # The chart
///
/// The score is fully explicit, so no second-order adjoint of the REML solver is
/// needed — only exact matrix calculus in the reduced `p×p` chart, where `p` is
/// the coefficient count (a handful of centers), not `n`:
///
/// ```text
///   A = XᵀX      b = Xᵀy      H = A + λS      β = H⁻¹b
///   dp = yᵀy − bᵀβ            (the PENALIZED deviance: RSS + λβᵀSβ)
///   F  = ½[log|H| − log|S|₊ − rank·ρ] + ½ν[1 + log(2π·dp/ν)]
/// ```
///
/// Every ψ-derivative below is the exact derivative of that expression, driven
/// by the design/penalty jets in both coordinates. **No finite differences and
/// no autodiff participate** (SPEC 1/2); the FD gates that validate this live
/// in tests.
///
/// The recomputed value is checked against the forward fit's own `reml_score`
/// before any derivative is returned: a chart that does not reproduce the
/// shipped objective cannot be differentiating it.
#[derive(Clone, Debug)]
struct ProfiledRemlPsiJet {
    /// `V(ψ)` — the λ-profiled REML negative log evidence.
    value: f64,
    /// `∇V` in the coordinate order `(κ, η)`.
    gradient: [f64; 2],
    /// `∇²V`, symmetric, in the same order.
    hessian: [[f64; 2]; 2],
}

impl ProfiledRemlPsiJet {
    /// The `κ` slice: value, `∂V/∂κ`, `∂²V/∂κ²` at FIXED `η`.
    fn kappa_slice(&self) -> (f64, f64, f64) {
        (self.value, self.gradient[0], self.hessian[0][0])
    }

    /// The η-PROFILED κ jet: `V_p(κ) = min_η V(κ, η)`, read off the jet at an
    /// interior η̂ with `V_ηη > 0`.
    ///
    /// The envelope theorem `V_p′ = V_κ` holds only where `V_η = 0` exactly, and
    /// the inner solve does not stop there: its certificate bounds the Newton
    /// decrement `V_η²/V_ηη` by the criterion's statistical resolution `1/(2n)`,
    /// which leaves the minimizer `δ = −V_η/V_ηη` away. `V_κ` at η̂ then misses
    /// `V_p′` by `V_κη·δ` — FIRST order in the residual, and not small where κ
    /// and η are coupled: measured `V_η = −9.0e-3`, `V_ηη = 8.96`,
    /// `V_κη = 14.9` gives a slope error of `1.5e-2` on `V_p′ = −131`, with a
    /// decrement of `9e-6` the certificate rightly accepts (gam#3426).
    ///
    /// So the profile is taken at the Newton minimizer of the jet's own
    /// quadratic in η, `η̂ + δ`, rather than at η̂:
    ///
    /// ```text
    /// V_p  = V − ½·V_η²/V_ηη          (error O(δ³))
    /// V_p′ = V_κ − V_κη·V_η/V_ηη       (error O(δ²))
    /// V_p″ = V_κκ − V_κη²/V_ηη         (error O(δ), the Schur complement)
    /// ```
    ///
    /// Each line is the exact κ-derivative of the one above up to the order it
    /// carries, so the triple is the jet of one function wherever the inner
    /// solve stopped inside its band — and at `V_η = 0` it is the plain envelope
    /// and Schur reduction, the same one this file applies to the ρ̂ response.
    ///
    /// Refuses rather than substituting a number when `V_ηη` cannot identify the
    /// reduction: at a non-positive η-curvature the inner problem is not at a
    /// minimum and neither the Newton minimizer nor the profile's second
    /// derivative is defined there.
    fn eta_profiled_kappa_jet(&self) -> Result<(f64, f64, f64), EstimationError> {
        let v_ee = self.hessian[1][1];
        if !(v_ee.is_finite() && v_ee > 0.0) {
            crate::bail_invalid_estim!(
                "constant-curvature ψ jet cannot profile η out: the range curvature is {v_ee:.3e}, \
                 so the inner minimum in η is not identified"
            );
        }
        let v_e = self.gradient[1];
        let v_ke = self.hessian[0][1];
        Ok((
            self.value - 0.5 * v_e * v_e / v_ee,
            self.gradient[0] - v_ke * v_e / v_ee,
            self.hessian[0][0] - v_ke * v_ke / v_ee,
        ))
    }
}

/// One coordinate's design/penalty first derivatives, plus the second
/// derivatives it shares with a partner coordinate. Grouping them keeps the
/// argument list of the jet below a list of MATRICES rather than a list of
/// positional `Array2`s whose order a caller could permute silently.
struct PsiCoordinateBlocks<'a> {
    design_first: [&'a Array2<f64>; 2],
    design_second: [&'a Array2<f64>; 3],
    penalty_first: [&'a Array2<f64>; 2],
    penalty_second: [&'a Array2<f64>; 3],
}

fn profiled_gaussian_reml_psi_jet(
    design: &Array2<f64>,
    penalty: &Array2<f64>,
    blocks: &PsiCoordinateBlocks<'_>,
    response: ArrayView1<'_, f64>,
) -> Result<ProfiledRemlPsiJet, EstimationError> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;

    let (n, p) = design.dim();
    let design_shape_ok = blocks
        .design_first
        .iter()
        .chain(blocks.design_second.iter())
        .all(|m| m.dim() == (n, p));
    let penalty_shape_ok = blocks
        .penalty_first
        .iter()
        .chain(blocks.penalty_second.iter())
        .all(|m| m.dim() == (p, p));
    if !design_shape_ok || !penalty_shape_ok || penalty.dim() != (p, p) || response.len() != n {
        crate::bail_invalid_estim!("constant-curvature profile ψ-jet shape mismatch");
    }

    let response_2d = response.insert_axis(ndarray::Axis(1));
    let fit = gam_solve::gaussian_reml::gaussian_reml_multi_closed_form(
        design.view(),
        response_2d.view(),
        penalty.view(),
        None,
        None,
    )?;
    let lambda = fit.lambda;
    let nullity = fit.cache.nullity;
    let rank = p.saturating_sub(nullity);
    if rank == 0 {
        crate::bail_invalid_estim!(
            "constant-curvature profile ψ-jet needs a penalty of positive rank; got nullity {nullity} of {p}"
        );
    }
    let nu = n as f64 - nullity as f64;
    if nu <= 0.0 {
        crate::bail_invalid_estim!(
            "constant-curvature profile ψ-jet needs positive residual degrees of freedom"
        );
    }

    // Reduced ψ-jets. `A = XᵀX`, `b = Xᵀy` and the residual `r = y − Xβ` with its
    // design images are the only places `n` appears; everything else is p×p. Index
    // order is (κ, η) for first derivatives and (κκ, κη, ηη) for seconds.
    let sym = |m: Array2<f64>| -> Array2<f64> { (&m + &m.t()) * 0.5 };
    let pair = |a: usize| -> (usize, usize) {
        match a {
            0 => (0, 0),
            1 => (0, 1),
            _ => (1, 1),
        }
    };
    let a0 = sym(fast_atb(&design.view(), &design.view()));
    let a1: Vec<Array2<f64>> = (0..2)
        .map(|a| sym(fast_atb(&blocks.design_first[a].view(), &design.view()) * 2.0))
        .collect();
    let a2: Vec<Array2<f64>> = (0..3)
        .map(|s| {
            let (a, b) = pair(s);
            sym(
                fast_atb(&blocks.design_second[s].view(), &design.view()) * 2.0
                    + fast_atb(
                        &blocks.design_first[a].view(),
                        &blocks.design_first[b].view(),
                    ) * 2.0,
            )
        })
        .collect();
    let b0 = fast_atv(&design.view(), &response);

    let s0 = sym(penalty.clone());
    let s1: Vec<Array2<f64>> = (0..2).map(|a| sym(blocks.penalty_first[a].clone())).collect();
    let s2: Vec<Array2<f64>> = (0..3)
        .map(|s| sym(blocks.penalty_second[s].clone()))
        .collect();

    // H = A + λS and its ψ-jets at FIXED ρ.
    let h0 = &a0 + &(&s0 * lambda);
    let h1: Vec<Array2<f64>> = (0..2).map(|a| &a1[a] + &(&s1[a] * lambda)).collect();
    let h2: Vec<Array2<f64>> = (0..3).map(|s| &a2[s] + &(&s2[s] * lambda)).collect();
    let chol = h0
        .cholesky(Side::Lower)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let beta0 = chol.solvevec(&b0);
    // #2280: the deviance and every one of its jets are formed from the residual
    // `r = y − Xβ` and the design images `X_a·β`, never as `yᵀy − bᵀβ`-shaped
    // differences. Those equal the forms below only at the exact solve, and they
    // round against `yᵀy` and `‖X_a‖·‖β‖·‖y‖` rather than against the residual they
    // come to, so a residual below that scale came back as roundoff.
    let residual = &response - &design.dot(&beta0);
    let design_image1: Vec<Array1<f64>> =
        (0..2).map(|a| blocks.design_first[a].dot(&beta0)).collect();
    let design_image2: Vec<Array1<f64>> =
        (0..3).map(|s| blocks.design_second[s].dot(&beta0)).collect();
    // `∂β/∂ψ_a = H⁻¹v_a`, with `v_a = X_aᵀr − XᵀX_aβ − λS_aβ` the ψ_a-derivative of
    // the normal equations' residual `Xᵀ(y − Xβ) − λSβ` at fixed `β`.
    let normal_residual1: Vec<Array1<f64>> = (0..2)
        .map(|a| {
            fast_atv(&blocks.design_first[a].view(), &residual)
                - fast_atv(&design.view(), &design_image1[a])
                - s1[a].dot(&beta0) * lambda
        })
        .collect();
    let beta1: Vec<Array1<f64>> = (0..2)
        .map(|a| chol.solvevec(&normal_residual1[a]))
        .collect();

    // log|H| and its ψ-jets. `g_a = H⁻¹H_a` is formed once and reused.
    let logdet_h = {
        let diag = chol.diag();
        2.0 * diag.iter().map(|value| value.ln()).sum::<f64>()
    };
    let trace = |m: &Array2<f64>| -> f64 { (0..m.nrows()).map(|i| m[(i, i)]).sum::<f64>() };
    let trace_product = |left: &Array2<f64>, right: &Array2<f64>| -> f64 {
        left.iter()
            .zip(right.t().iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
    };
    let g1: Vec<Array2<f64>> = (0..2).map(|a| chol.solve_mat(&h1[a])).collect();
    let g2: Vec<Array2<f64>> = (0..3).map(|s| chol.solve_mat(&h2[s])).collect();
    let logdet_h_1: Vec<f64> = (0..2).map(|a| trace(&g1[a])).collect();
    let logdet_h_2: Vec<f64> = (0..3)
        .map(|s| {
            let (a, b) = pair(s);
            trace(&g2[s]) - trace_product(&g1[a], &g1[b])
        })
        .collect();

    // log|S|₊ and its ψ-jets, with the positive/null split held fixed.
    let logdet_s = positive_pseudo_logdet_psi_jet(&s0, &s1, &s2, rank)?;
    let logdet_s_positive = logdet_s.value;
    let logdet_s_1 = logdet_s.first;
    let logdet_s_2 = logdet_s.second;

    // Penalized deviance `dp = ‖y − Xβ‖² + λβᵀSβ` and its ψ-jets. `β` minimizes the
    // penalized sum of squares, so by the envelope theorem its ψ response does not
    // enter the first derivative, and enters the second only through `β_a = H⁻¹v_a`:
    //   dp_a  = −2(X_aβ)ᵀr + λβᵀS_aβ
    //   dp_ab = 2(X_aβ)ᵀ(X_bβ) − 2(X_abβ)ᵀr + λβᵀS_abβ − 2·v_aᵀβ_b
    let penalty_quadratic = |m: &Array2<f64>| -> f64 { beta0.dot(&m.dot(&beta0)) };
    let dp = residual.dot(&residual) + lambda * penalty_quadratic(&s0);
    if !(dp > 0.0) {
        crate::bail_invalid_estim!(
            "constant-curvature profile ψ-jet found a non-positive profiled deviance {dp:.6e}"
        );
    }
    let dp_1: Vec<f64> = (0..2)
        .map(|a| -2.0 * design_image1[a].dot(&residual) + lambda * penalty_quadratic(&s1[a]))
        .collect();
    let dp_2: Vec<f64> = (0..3)
        .map(|s| {
            let (a, b) = pair(s);
            2.0 * design_image1[a].dot(&design_image1[b])
                - 2.0 * design_image2[s].dot(&residual)
                + lambda * penalty_quadratic(&s2[s])
                - 2.0 * normal_residual1[a].dot(&beta1[b])
        })
        .collect();

    // The value, recomputed in this chart, must be the shipped objective.
    let rho = fit.rho;
    let value = 0.5 * (logdet_h - (logdet_s_positive + rank as f64 * rho))
        + 0.5 * nu * (1.0 + (2.0 * std::f64::consts::PI * dp / nu).ln());
    if !value.is_finite() || (value - fit.reml_score).abs() > 1.0e-7 * (1.0 + fit.reml_score.abs())
    {
        crate::bail_invalid_estim!(
            "constant-curvature profile ψ-jet chart does not reproduce the REML score it differentiates: chart {value:.9e} vs forward {:.9e}",
            fit.reml_score
        );
    }

    let f_1: Vec<f64> = (0..2)
        .map(|a| 0.5 * (logdet_h_1[a] - logdet_s_1[a]) + 0.5 * nu * dp_1[a] / dp)
        .collect();
    let f_2: Vec<f64> = (0..3)
        .map(|s| {
            let (a, b) = pair(s);
            0.5 * (logdet_h_2[s] - logdet_s_2[s])
                + 0.5 * nu * (dp_2[s] / dp - (dp_1[a] / dp) * (dp_1[b] / dp))
        })
        .collect();

    // Mixed ∂²F/∂ψ_a∂ρ. Only H and β carry ρ: ∂H/∂ρ = λS, ∂H_a/∂ρ = λS_a, and
    // `∂β/∂ρ = −λH⁻¹Sβ`. The deviance's ρ-jets follow from the same envelope as
    // its ψ-jets: `dp_ρ = λβᵀSβ` and `dp_aρ = λβᵀS_aβ + 2λ·β_aᵀSβ`.
    let lambda_s0 = &s0 * lambda;
    let dp_rho = lambda * penalty_quadratic(&s0);
    let s0_beta0 = s0.dot(&beta0);
    let f_1rho: Vec<f64> = (0..2)
        .map(|a| {
            let lambda_s1 = &s1[a] * lambda;
            let logdet_h_1rho = trace(&chol.solve_mat(&lambda_s1))
                - trace_product(&chol.solve_mat(&lambda_s0), &g1[a]);
            let dp_1rho =
                lambda * penalty_quadratic(&s1[a]) + 2.0 * lambda * beta1[a].dot(&s0_beta0);
            0.5 * logdet_h_1rho + 0.5 * nu * (dp_1rho / dp - dp_1[a] * dp_rho / (dp * dp))
        })
        .collect();

    // The λ̂ response is the derivative of an INTERIOR stationary root only. At a
    // railed ρ̂ the selection is locally constant, so dλ̂/dψ = 0 and the Schur
    // term is absent — the same premise the backward VJP already gates on. When
    // the ρ curvature is unusable the profile's second derivative is not
    // identified and this refuses rather than substituting a number.
    let hess_rho = fit.reml_hess_rho;
    // The closed-form selector evaluates both walls exactly and starts from the
    // better one, so a railed ρ̂ is the wall value itself.
    let rho_at_bound = rho == fit.rho_domain.0 || rho == fit.rho_domain.1;
    let schur = if rho_at_bound {
        0.0
    } else if hess_rho.is_finite() && hess_rho != 0.0 {
        // Any finite nonzero curvature inverts. A Schur term that overflows is
        // refused by the non-finite check on the assembled Hessian below.
        1.0 / hess_rho
    } else {
        crate::bail_invalid_estim!(
            "constant-curvature profile ψ-jet cannot identify ∇²V: the ρ curvature is {hess_rho:.3e} at an interior ρ̂"
        );
    };
    let mut hessian = [[0.0_f64; 2]; 2];
    for s in 0..3 {
        let (a, b) = pair(s);
        let entry = f_2[s] - f_1rho[a] * f_1rho[b] * schur;
        hessian[a][b] = entry;
        hessian[b][a] = entry;
    }
    let gradient = [f_1[0], f_1[1]];
    if !gradient.iter().all(|v| v.is_finite())
        || !hessian.iter().flatten().all(|v| v.is_finite())
    {
        crate::bail_invalid_estim!(
            "constant-curvature profile ψ-jet produced a non-finite derivative"
        );
    }
    Ok(ProfiledRemlPsiJet {
        value: fit.reml_score,
        gradient,
        hessian,
    })
}

/// `log|S|₊ = Σ_{i∈pos} ln λ_i` and its exact ψ-jets. Index order is (κ, η) for
/// first derivatives and (κκ, κη, ηη) for seconds, as in
/// [`profiled_gaussian_reml_psi_jet`].
struct PseudoLogdetPsiJet {
    value: f64,
    first: [f64; 2],
    second: [f64; 3],
}

/// `log|S(ψ)|₊` over the `rank` largest eigenvalues of `S`, differentiated with
/// the positive/null classification held fixed. That is the function the
/// forward fit evaluates at every ψ: it classifies the spectrum with a
/// tolerance, and a null eigenvalue that moves by roundoff stays null.
///
/// Write `S = Σ λ_i v_i v_iᵀ` with eigenvalues in descending order, `P` for the
/// first `rank` eigenvectors (λ_i > 0) and `N` for the rest (eigenvalues μ_j),
/// and `S_a`, `S_ab` for the ψ-derivatives expressed in that eigenbasis.
/// Hellmann–Feynman and second-order Rayleigh–Schrödinger perturbation theory
/// give
///
/// ```text
///   λ_i,a  = (S_a)_ii
///   λ_i,ab = (S_ab)_ii + 2 Σ_{j≠i} (S_a)_ij (S_b)_ij / (λ_i − λ_j)
/// ```
///
/// Summing `λ_i,a/λ_i` and `λ_i,ab/λ_i − λ_i,a λ_i,b/λ_i²` over i∈pos, the
/// positive–positive pairs of the sum collapse to `−(S_a)_ij (S_b)_ij/(λ_iλ_j)`,
/// which needs no gap between positive eigenvalues:
///
/// ```text
///   ∂_a  log|S|₊ = Σ_{i∈pos} (S_a)_ii / λ_i
///   ∂_ab log|S|₊ = Σ_{i∈pos} (S_ab)_ii / λ_i
///                − Σ_{i,j∈pos} (S_a)_ij (S_b)_ij / (λ_i λ_j)
///                + 2 Σ_{i∈pos, j∈null} (S_a)_ij (S_b)_ij / (λ_i (λ_i − μ_j))
/// ```
///
/// The first two terms of the second derivative are `tr(R⁻¹R_ab) − tr(R⁻¹R_aR⁻¹R_b)`
/// with `R = PᵀSP`, the formula restricted to the positive subspace. That
/// formula alone is exact only if every `S_a` annihilates the null frame. The
/// last term is the positive/null coupling it leaves out. It is present whenever
/// ψ rotates the null space. For example, a whitening chart that re-realizes
/// the identifiable subspace at each ψ moves the null directions of `TᵀST` even
/// though the penalty's structural null space is fixed. The first derivative
/// has no such term, so the gradient was exact either way.
///
/// The only premise is that the split is resolved. A backward-stable symmetric
/// eigensolver returns the spectrum of `S + E` with `‖E‖₂` inside
/// [`gam_linalg::roundoff::symmetric_spectrum_rounding_band`] (`band`). By Weyl,
/// each computed eigenvalue is within `band` of an exact one. So the smallest
/// positive eigenvalue is certified positive only above `band`, and a computed
/// gap `λ_rank − μ_max` certifies an open gap only above `2·band`. Below that,
/// the coupling denominators are not measurements, and the jet refuses.
fn positive_pseudo_logdet_psi_jet(
    s0: &Array2<f64>,
    s1: &[Array2<f64>],
    s2: &[Array2<f64>],
    rank: usize,
) -> Result<PseudoLogdetPsiJet, EstimationError> {
    use faer::Side;
    use gam_linalg::faer_ndarray::strict_symmetric_eigh;

    let p = s0.nrows();
    let shapes_ok = s0.ncols() == p
        && s1.len() == 2
        && s2.len() == 3
        && s1.iter().chain(s2.iter()).all(|m| m.dim() == (p, p));
    if !shapes_ok || rank == 0 || rank > p {
        crate::bail_invalid_estim!(
            "log|S|₊ ψ-jet shape mismatch (p {p}, rank {rank}, {} first and {} second blocks)",
            s1.len(),
            s2.len()
        );
    }
    let (eigenvalues, eigenvectors) = strict_symmetric_eigh(
        s0,
        // `sym` forms `(S + Sᵀ)/2` with commutative additions: mirrored.
        gam_linalg::roundoff::SymmetricAssembly::Mirrored,
        Side::Lower,
    )
    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&i, &j| {
        eigenvalues[j]
            .partial_cmp(&eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let values: Vec<f64> = order.iter().map(|&i| eigenvalues[i]).collect();
    let band = gam_linalg::roundoff::symmetric_spectrum_rounding_band(&values);
    let smallest_positive = values[rank - 1];
    if !(smallest_positive > band) {
        crate::bail_invalid_estim!(
            "log|S|₊ ψ-jet: the smallest of {rank} positive eigenvalues, {smallest_positive:.3e}, is not resolved from zero (rounding band {band:.3e})"
        );
    }
    if rank < p {
        let gap = smallest_positive - values[rank];
        if !(gap > 2.0 * band) {
            crate::bail_invalid_estim!(
                "log|S|₊ ψ-jet: the positive/null eigengap {gap:.3e} is not resolved (twice the rounding band is {:.3e})",
                2.0 * band
            );
        }
    }
    let mut basis = Array2::<f64>::zeros((p, p));
    for (column, &index) in order.iter().enumerate() {
        basis.column_mut(column).assign(&eigenvectors.column(index));
    }
    let project = |m: &Array2<f64>| -> Array2<f64> {
        let projected = basis.t().dot(&m.dot(&basis));
        (&projected + &projected.t()) * 0.5
    };
    let first_blocks: Vec<Array2<f64>> = s1.iter().map(|m| project(m)).collect();
    let second_blocks: Vec<Array2<f64>> = s2.iter().map(|m| project(m)).collect();

    let value = values[..rank].iter().map(|v| v.ln()).sum::<f64>();
    let first: [f64; 2] = std::array::from_fn(|a| {
        (0..rank)
            .map(|i| first_blocks[a][(i, i)] / values[i])
            .sum::<f64>()
    });
    let second: [f64; 3] = std::array::from_fn(|s| {
        let (a, b) = match s {
            0 => (0, 0),
            1 => (0, 1),
            _ => (1, 1),
        };
        let (block_a, block_b) = (&first_blocks[a], &first_blocks[b]);
        let mut total = 0.0_f64;
        for i in 0..rank {
            total += second_blocks[s][(i, i)] / values[i];
            for j in 0..rank {
                total -= block_a[(i, j)] * block_b[(i, j)] / (values[i] * values[j]);
            }
            for j in rank..p {
                total += 2.0 * block_a[(i, j)] * block_b[(i, j)]
                    / (values[i] * (values[i] - values[j]));
            }
        }
        total
    });
    Ok(PseudoLogdetPsiJet {
        value,
        first,
        second,
    })
}

#[cfg(test)]
mod positive_pseudo_logdet_psi_jet_tests {
    use super::*;

    fn unit(p: usize, i: usize, j: usize) -> Array2<f64> {
        let mut m = Array2::<f64>::zeros((p, p));
        m[(i, j)] = 1.0;
        m
    }

    fn rotation_generator(p: usize, i: usize, j: usize) -> Array2<f64> {
        unit(p, i, j) - unit(p, j, i)
    }

    fn commutator(left: &Array2<f64>, right: &Array2<f64>) -> Array2<f64> {
        left.dot(right) - right.dot(left)
    }

    /// The exact ψ-jets at ψ = 0 of `S(ψ) = e^X D(ψ) e^{−X}` with
    /// `X = ψ_κ G_κ + ψ_η G_η` and antisymmetric `G_a`. The spectrum of `S(ψ)` is
    /// that of `D(ψ)`, so log|S|₊ is known in closed form, and every nonzero
    /// `G_a` moves the null space.
    fn rotated_jets(
        generators: [&Array2<f64>; 2],
        d0: &Array2<f64>,
        d1: [&Array2<f64>; 2],
        d2: [&Array2<f64>; 3],
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let first = (0..2)
            .map(|a| d1[a] + &commutator(generators[a], d0))
            .collect();
        let second = (0..3)
            .map(|s| {
                let (a, b) = match s {
                    0 => (0, 0),
                    1 => (0, 1),
                    _ => (1, 1),
                };
                d2[s]
                    + &commutator(generators[a], d1[b])
                    + &commutator(generators[b], d1[a])
                    + &((commutator(generators[a], &commutator(generators[b], d0))
                        + commutator(generators[b], &commutator(generators[a], d0)))
                        * 0.5)
            })
            .collect();
        (first, second)
    }

    // p = 3 with O(1) entries and gaps ≥ 1: each derivative is a sum of at most
    // p² = 9 products of a few rounded operations each. An error above 1e-12
    // (about 4500 ulps of 1) is therefore not roundoff.
    const ARITHMETIC_BAND: f64 = 1.0e-12;

    #[test]
    fn a_rotating_null_space_keeps_log_pseudo_determinant_flat_2902() {
        // D = diag(2, 1, 0). G_κ rotates (e₂, e₃), mixing λ = 1 into the null
        // direction, and G_η rotates (e₁, e₃), mixing λ = 2 into it. The
        // spectrum never moves, so log|S|₊ = ln 2 with zero gradient and Hessian.
        let p = 3;
        let mut d0 = Array2::<f64>::zeros((p, p));
        d0[(0, 0)] = 2.0;
        d0[(1, 1)] = 1.0;
        let zero = Array2::<f64>::zeros((p, p));
        let g_kappa = rotation_generator(p, 1, 2);
        let g_eta = rotation_generator(p, 0, 2);
        let (s1, s2) = rotated_jets(
            [&g_kappa, &g_eta],
            &d0,
            [&zero, &zero],
            [&zero, &zero, &zero],
        );
        let jet = positive_pseudo_logdet_psi_jet(&d0, &s1, &s2, 2).expect("resolved split");
        assert!((jet.value - 2.0_f64.ln()).abs() <= ARITHMETIC_BAND);
        for (a, g) in jet.first.iter().enumerate() {
            assert!(g.abs() <= ARITHMETIC_BAND, "gradient[{a}] = {g:e}");
        }
        for (s, h) in jet.second.iter().enumerate() {
            assert!(h.abs() <= ARITHMETIC_BAND, "hessian[{s}] = {h:e}");
        }

        // Positive control: the positive-subspace formula alone,
        // tr(R⁻¹R_aa) − tr((R⁻¹R_a)²) with R = diag(2, 1), reads −2 on both
        // diagonal entries. That −2 is what the coupling term cancels, and it
        // is the defect the old null-frame leak check refused instead of
        // computing.
        for (s, a) in [(0usize, 0usize), (2, 1)] {
            let restricted = (0..2)
                .map(|i| {
                    s2[s][(i, i)] / d0[(i, i)]
                        - (0..2)
                            .map(|j| s1[a][(i, j)].powi(2) / (d0[(i, i)] * d0[(j, j)]))
                            .sum::<f64>()
                })
                .sum::<f64>();
            assert!(
                (restricted + 2.0).abs() <= ARITHMETIC_BAND,
                "restricted formula entry {s}: {restricted:e}"
            );
        }
    }

    #[test]
    fn a_moving_spectrum_under_rotation_has_its_closed_form_jet_2902() {
        // D(ψ) = diag(2e^{ψ_κ}, 1 + ψ_η², 0), rotated by generators that each
        // couple every pair of directions. log|S|₊ = ln 2 + ψ_κ + ln(1 + ψ_η²):
        // gradient (1, 0) and Hessian diag(0, 2) at ψ = 0.
        let p = 3;
        let mut d0 = Array2::<f64>::zeros((p, p));
        d0[(0, 0)] = 2.0;
        d0[(1, 1)] = 1.0;
        let mut d_kappa = Array2::<f64>::zeros((p, p));
        d_kappa[(0, 0)] = 2.0;
        let mut d_eta_eta = Array2::<f64>::zeros((p, p));
        d_eta_eta[(1, 1)] = 2.0;
        let zero = Array2::<f64>::zeros((p, p));
        let g_kappa = rotation_generator(p, 1, 2) + rotation_generator(p, 0, 1) * 0.3;
        let g_eta = rotation_generator(p, 0, 2) + rotation_generator(p, 0, 1) * 0.5
            - rotation_generator(p, 1, 2) * 0.7;
        let (s1, s2) = rotated_jets(
            [&g_kappa, &g_eta],
            &d0,
            [&d_kappa, &zero],
            [&d_kappa, &zero, &d_eta_eta],
        );
        // Conjugate by a fixed rotation so the null direction is not a
        // coordinate axis. The jet must not depend on the basis.
        let q = {
            let (c, s) = (0.6_f64, 0.8_f64);
            let mut q = Array2::<f64>::eye(p);
            q[(0, 0)] = c;
            q[(0, 2)] = -s;
            q[(2, 0)] = s;
            q[(2, 2)] = c;
            q
        };
        let turn = |m: &Array2<f64>| {
            let turned = q.dot(&m.dot(&q.t()));
            (&turned + &turned.t()) * 0.5
        };
        let s0 = turn(&d0);
        let s1: Vec<Array2<f64>> = s1.iter().map(|m| turn(m)).collect();
        let s2: Vec<Array2<f64>> = s2.iter().map(|m| turn(m)).collect();
        let jet = positive_pseudo_logdet_psi_jet(&s0, &s1, &s2, 2).expect("resolved split");
        let expected_first = [1.0, 0.0];
        let expected_second = [0.0, 0.0, 2.0];
        assert!((jet.value - 2.0_f64.ln()).abs() <= ARITHMETIC_BAND);
        for a in 0..2 {
            assert!(
                (jet.first[a] - expected_first[a]).abs() <= ARITHMETIC_BAND,
                "gradient[{a}] = {:e}",
                jet.first[a]
            );
        }
        for s in 0..3 {
            assert!(
                (jet.second[s] - expected_second[s]).abs() <= ARITHMETIC_BAND,
                "hessian[{s}] = {:e} against {:e}",
                jet.second[s],
                expected_second[s]
            );
        }
    }

    #[test]
    fn an_unresolved_positive_null_split_is_refused_2902() {
        // λ₂ = 1e-17 sits inside 3·ε·2 of the null eigenvalue, so no
        // decomposition can tell it is positive, and the coupling denominator
        // λ₂ − μ is not a measurement.
        let p = 3;
        let mut d0 = Array2::<f64>::zeros((p, p));
        d0[(0, 0)] = 2.0;
        d0[(1, 1)] = 1.0e-17;
        let zero = Array2::<f64>::zeros((p, p));
        let s1 = vec![zero.clone(), zero.clone()];
        let s2 = vec![zero.clone(), zero.clone(), zero];
        assert!(positive_pseudo_logdet_psi_jet(&d0, &s1, &s2, 2).is_err());
    }
}
