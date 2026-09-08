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
    /// Did the inner `ρ̂` land on a bound?
    ///
    /// `V` is only a PROFILE where it does not: at a railed `ρ̂` the value is a
    /// constrained minimum over a truncated λ range, and constrained and
    /// unconstrained minima are not comparable to one another. That distinction
    /// is a corner case in the κ direction, where the box is derived from the
    /// geometry; it is decisive in the RANGE direction. Measured: `ρ̂ ≈ const −
    /// ln ℓ` — each ×100 in `ℓ` costs 4.6 in `ρ̂`, which is `ln 100` exactly,
    /// because the realized design scales like `1/ℓ` and λ has to follow it. So
    /// a range box wide enough to be a pure floating-point wall ALWAYS walks
    /// `ρ̂` into `RHO_LOWER`, and past that point the value moves spuriously by
    /// tens of nats while the true criterion is flat.
    rho_at_bound: bool,
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

    /// The η-PROFILED κ jet: `V_p(κ) = min_η V(κ, η)`, valid only AT an interior
    /// η̂ where `V_η = 0`. The envelope theorem gives `V_p′ = V_κ`, and one more
    /// differentiation the Schur complement `V_p″ = V_κκ − V_κη²/V_ηη` — the
    /// same reduction this file already applies to the ρ̂ response.
    ///
    /// Refuses rather than substituting a number when `V_ηη` cannot identify the
    /// reduction: at a non-positive η-curvature the inner problem is not at a
    /// minimum and the profile's second derivative is not defined there.
    fn eta_profiled_kappa_jet(&self) -> Result<(f64, f64, f64), EstimationError> {
        let v_ee = self.hessian[1][1];
        if !(v_ee.is_finite() && v_ee > 0.0) {
            crate::bail_invalid_estim!(
                "constant-curvature ψ jet cannot profile η out: the range curvature is {v_ee:.3e}, \
                 so the inner minimum in η is not identified"
            );
        }
        let v_ke = self.hessian[0][1];
        Ok((
            self.value,
            self.gradient[0],
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
    use gam_linalg::faer_ndarray::{FaerCholesky, strict_symmetric_eigh};

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

    // Reduced ψ-jets. `A = XᵀX` and `b = Xᵀy` are the only places `n` appears;
    // everything after this is p×p. Index order is (κ, η) for first derivatives
    // and (κκ, κη, ηη) for seconds.
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
    let b1: Vec<Array1<f64>> = (0..2)
        .map(|a| fast_atv(&blocks.design_first[a].view(), &response))
        .collect();
    let b2: Vec<Array1<f64>> = (0..3)
        .map(|s| fast_atv(&blocks.design_second[s].view(), &response))
        .collect();
    let yty = response.dot(&response);

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
    let beta1: Vec<Array1<f64>> = (0..2)
        .map(|a| chol.solvevec(&(&b1[a] - &h1[a].dot(&beta0))))
        .collect();
    let beta2: Vec<Array1<f64>> = (0..3)
        .map(|s| {
            let (a, b) = pair(s);
            chol.solvevec(
                &(&b2[s] - &h1[a].dot(&beta1[b]) - &h1[b].dot(&beta1[a]) - &h2[s].dot(&beta0)),
            )
        })
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

    // log|S|₊ and its ψ-jets on the penalty's POSITIVE subspace. The subspace is
    // ψ-fixed (the smooth's null directions are structural — the unpenalized
    // parametric coordinates — and carry no ψ dependence), which is what makes
    // the pseudo-determinant differentiable at all; it is verified rather than
    // assumed: every S′ and S″ must annihilate the null frame.
    let (s_eigenvalues, s_eigenvectors) = strict_symmetric_eigh(&s0, Side::Lower)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&i, &j| {
        s_eigenvalues[j]
            .partial_cmp(&s_eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut positive_frame = Array2::<f64>::zeros((p, rank));
    for (column, &index) in order.iter().take(rank).enumerate() {
        positive_frame
            .column_mut(column)
            .assign(&s_eigenvectors.column(index));
    }
    if nullity > 0 {
        let mut null_frame = Array2::<f64>::zeros((p, nullity));
        for (column, &index) in order.iter().skip(rank).enumerate() {
            null_frame
                .column_mut(column)
                .assign(&s_eigenvectors.column(index));
        }
        let s0_scale = s0.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let labelled = [
            ("∂S/∂κ", &s1[0]),
            ("∂S/∂η", &s1[1]),
            ("∂²S/∂κ²", &s2[0]),
            ("∂²S/∂κ∂η", &s2[1]),
            ("∂²S/∂η²", &s2[2]),
        ];
        for (label, block) in labelled {
            let leak = block.dot(&null_frame);
            let leak_norm = leak.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            if leak_norm > 1.0e-8 * (1.0 + s0_scale) {
                crate::bail_invalid_estim!(
                    "constant-curvature profile ψ-jet requires a ψ-fixed penalty null space, but {label} moves it by {leak_norm:.3e}"
                );
            }
        }
    }
    let restrict =
        |m: &Array2<f64>| -> Array2<f64> { sym(positive_frame.t().dot(&m.dot(&positive_frame))) };
    let r0 = restrict(&s0);
    let r_chol = r0
        .cholesky(Side::Lower)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let logdet_s_positive = {
        let diag = r_chol.diag();
        2.0 * diag.iter().map(|value| value.ln()).sum::<f64>()
    };
    let m1: Vec<Array2<f64>> = (0..2)
        .map(|a| r_chol.solve_mat(&restrict(&s1[a])))
        .collect();
    let m2: Vec<Array2<f64>> = (0..3)
        .map(|s| r_chol.solve_mat(&restrict(&s2[s])))
        .collect();
    let logdet_s_1: Vec<f64> = (0..2).map(|a| trace(&m1[a])).collect();
    let logdet_s_2: Vec<f64> = (0..3)
        .map(|s| {
            let (a, b) = pair(s);
            trace(&m2[s]) - trace_product(&m1[a], &m1[b])
        })
        .collect();

    // Penalized deviance `dp = yᵀy − bᵀβ` and its ψ-jets.
    let dp = yty - b0.dot(&beta0);
    if !(dp > 0.0) {
        crate::bail_invalid_estim!(
            "constant-curvature profile ψ-jet found a non-positive profiled deviance {dp:.6e}"
        );
    }
    let dp_1: Vec<f64> = (0..2)
        .map(|a| -b1[a].dot(&beta0) - b0.dot(&beta1[a]))
        .collect();
    let dp_2: Vec<f64> = (0..3)
        .map(|s| {
            let (a, b) = pair(s);
            -b2[s].dot(&beta0) - b1[a].dot(&beta1[b]) - b1[b].dot(&beta1[a]) - b0.dot(&beta2[s])
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

    // Mixed ∂²F/∂ψ_a∂ρ. Only H and β carry ρ: ∂H/∂ρ = λS, ∂H_a/∂ρ = λS_a.
    let lambda_s0 = &s0 * lambda;
    let beta0_rho = chol.solvevec(&lambda_s0.dot(&beta0)) * -1.0;
    let f_1rho: Vec<f64> = (0..2)
        .map(|a| {
            let lambda_s1 = &s1[a] * lambda;
            let beta1_rho = {
                let inner =
                    lambda_s1.dot(&beta0) * -1.0 + h1[a].dot(&chol.solvevec(&lambda_s0.dot(&beta0)));
                chol.solvevec(&(&inner - &lambda_s0.dot(&beta1[a])))
            };
            let logdet_h_1rho = trace(&chol.solve_mat(&lambda_s1))
                - trace_product(&chol.solve_mat(&lambda_s0), &g1[a]);
            let dp_rho = -b0.dot(&beta0_rho);
            let dp_1rho = -b1[a].dot(&beta0_rho) - b0.dot(&beta1_rho);
            0.5 * logdet_h_1rho + 0.5 * nu * (dp_1rho / dp - dp_1[a] * dp_rho / (dp * dp))
        })
        .collect();

    // The λ̂ response is the derivative of an INTERIOR stationary root only. At a
    // railed ρ̂ the selection is locally constant, so dλ̂/dψ = 0 and the Schur
    // term is absent — the same premise the backward VJP already gates on. When
    // the ρ curvature is unusable the profile's second derivative is not
    // identified and this refuses rather than substituting a number.
    let hess_rho = fit.reml_hess_rho;
    let rho_at_bound =
        (rho - fit.rho_domain.0).abs() <= 1.0e-9 || (rho - fit.rho_domain.1).abs() <= 1.0e-9;
    let schur = if rho_at_bound {
        0.0
    } else if hess_rho.is_finite() && hess_rho.abs() > 1.0e-14 {
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
        rho_at_bound,
        gradient,
        hessian,
    })
}
