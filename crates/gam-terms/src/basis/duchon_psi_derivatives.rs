use super::*;

pub(crate) fn duchon_coeff_exponents(p_order: usize, s_order: usize, m_or_n: usize) -> f64 {
    // In the partial fractions
    //   1 / (z^p (z + kappa^2)^s)
    // = Σ a_m(kappa) / z^m + Σ b_n(kappa) / (z + kappa^2)^n,
    // both a_m and b_n are pure powers of kappa:
    //   c(kappa) = C * kappa^{-2(p+s-index)}.
    // With psi = log(kappa), that gives c_psi = alpha c and
    // c_psipsi = alpha^2 c with alpha below. This is the exact coefficient
    // derivative rule from the Duchon spectral factorization.
    -2.0 * (p_order + s_order - m_or_n) as f64
}

#[inline(always)]
pub(crate) fn duchon_scaling_exponent(p_order: usize, s_order: usize, k_dim: usize) -> f64 {
    k_dim as f64 - 2.0 * (p_order + s_order) as f64
}

#[derive(Clone, Copy)]
pub(crate) struct DuchonMaternDerivativeTerm {
    pub(crate) coeff: f64,
    pub(crate) kappa_power: usize,
    pub(crate) r_power: f64,
    pub(crate) bessel_order: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PsiTriplet {
    pub(crate) value: f64,
    pub(crate) psi: f64,
    pub(crate) psi_psi: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DuchonRadialCore {
    pub(crate) phi: PsiTriplet,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DuchonRadialJets {
    pub(crate) phi: f64,
    pub(crate) phi_r: f64,
    pub(crate) phi_rr: f64,
    pub(crate) phi_rrr: f64,
    pub(crate) q: f64,
    pub(crate) q_r: f64,
    pub(crate) q_rr: f64,
    pub(crate) lap: f64,
    pub(crate) lap_r: f64,
    pub(crate) lap_rr: f64,
    /// R-operator radial scalar: t = R²φ = (φ'' - q) / r² = q' / r.
    /// At collision (r = 0): t = φ''''(0) / 3, computed via assembled
    /// fourth-derivative collision limits of the partial-fraction blocks.
    pub(crate) t: f64,
    /// First radial derivative of t:
    ///   t_r = dt/dr = (q_rr - t) / r  for r > 0.
    /// At collision, the exact radial limit is t_r(0) = 0.
    pub(crate) t_r: f64,
    /// Second radial derivative of t:
    ///   t_rr = d²t/dr² = [lap_rr + 2 t - (d + 4) q_rr] / r²  for r > 0,
    /// using Delta phi = d q + r² t.
    ///
    /// At collision, the exact radial limit is
    ///   t_rr(0) = φ⁽⁶⁾(0) / 15.
    pub(crate) t_rr: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DuchonRegularizedOperatorCore {
    pub(crate) q: f64,
    pub(crate) t: f64,
    pub(crate) t_r: f64,
    pub(crate) t_rr: f64,
}

#[inline(always)]
pub(crate) fn duchon_operator_jets_from_primary_core(
    core: DuchonRegularizedOperatorCore,
    r: f64,
    d: f64,
) -> DuchonRadialJets {
    let r2 = r * r;
    let mut out = DuchonRadialJets {
        q: core.q,
        t: core.t,
        t_r: core.t_r,
        t_rr: core.t_rr,
        ..DuchonRadialJets::default()
    };
    out.q_r = r * out.t;
    out.q_rr = out.t + r * out.t_r;
    out.lap = d * out.q + r2 * out.t;
    out.lap_r = (d + 2.0) * r * out.t + r2 * out.t_r;
    out.lap_rr = (d + 2.0) * out.t + (d + 4.0) * r * out.t_r + r2 * out.t_rr;
    out.phi_r = r * out.q;
    out.phi_rr = out.q + r2 * out.t;
    out.phi_rrr = 3.0 * r * out.t + r2 * out.t_r;

    assert!(
        ((out.phi_rr - (out.q + r * out.q_r)).abs()) <= 1e-10 * out.phi_rr.abs().max(1.0),
        "radial scalar identity failed: phi_rr != q + r*q_r, phi_rr={}, q={}, r={}, q_r={}",
        out.phi_rr,
        out.q,
        r,
        out.q_r
    );
    assert!(
        ((out.phi_rr - (out.q + r2 * out.t)).abs()) <= 1e-10 * out.phi_rr.abs().max(1.0),
        "radial scalar identity failed: phi_rr != q + r2*t, phi_rr={}, q={}, r2={}, t={}",
        out.phi_rr,
        out.q,
        r2,
        out.t
    );
    assert!(
        ((out.lap - (d * out.q + r2 * out.t)).abs()) <= 1e-10 * out.lap.abs().max(1.0),
        "radial scalar identity failed: lap != d*q + r2*t, lap={}, d={}, q={}, r2={}, t={}",
        out.lap,
        d,
        out.q,
        r2,
        out.t
    );

    out
}

#[inline(always)]
pub(crate) fn scaled_log_kappa_derivatives(
    value: f64,
    radial_first: f64,
    radialsecond: f64,
    exponent: f64,
    r: f64,
) -> (f64, f64) {
    // Scaling-law differentiation template
    // For any radial quantity of the form
    //   F(r; kappa) = kappa^a G(kappa r),
    // with psi = log(kappa), one has d/dpsi = kappa d/dkappa.
    //
    // Writing t = kappa r,
    //   F_psi
    //   = kappa d/dkappa [kappa^a G(t)]
    //   = a kappa^a G(t) + kappa^a (kappa r) G'(t)
    //   = a F + r F_r.
    //
    // Differentiating again,
    //   F_psipsi
    //   = d/dpsi [a F + r F_r]
    //   = a F_psi + r (F_r)_psi
    //   = a (a F + r F_r) + r d/dr(F_psi)
    //   = a^2 F + (2a + 1) r F_r + r^2 F_rr.
    //
    // This helper is the common exact formula used for:
    //   - phi            with exponent delta
    //   - q = phi_r / r  with exponent delta + 2
    //   - Delta phi      with exponent delta + 2.
    let first = exponent * value + r * radial_first;
    let second = exponent * exponent * value
        + (2.0 * exponent + 1.0) * r * radial_first
        + r * r * radialsecond;
    (first, second)
}

#[inline(always)]
pub(crate) fn duchon_q_psi_triplet_from_jets(
    jets: &DuchonRadialJets,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    r: f64,
) -> (f64, f64) {
    scaled_log_kappa_derivatives(
        jets.q,
        jets.q_r,
        jets.q_rr,
        duchon_operator_scaling_exponent(p_order, s_order, k_dim),
        r,
    )
}

#[inline(always)]
pub(crate) fn duchon_operator_scaling_exponent(
    p_order: usize,
    s_order: usize,
    k_dim: usize,
) -> f64 {
    // For the hybrid Duchon spectrum
    //   1 / (|w|^(2p) (kappa^2 + |w|^2)^s),
    // the spatial kernel scales as
    //   phi(r; kappa) = kappa^delta H(kappa r),
    // where
    //   delta = d - 2p - 2s.
    //
    // A first spatial derivative contributes one extra factor of kappa, so
    // phi_r scales like kappa^(delta + 1). Dividing by r gives
    //   q(r; kappa) = phi_r / r = kappa^(delta + 2) Q(kappa r).
    //
    // The Laplacian also contributes two spatial derivatives, so
    //   Delta phi(r; kappa) = kappa^(delta + 2) L(kappa r).
    //
    // Thus both Duchon operator scalars use exponent delta + 2.
    duchon_scaling_exponent(p_order, s_order, k_dim) + 2.0
}

pub(crate) fn duchon_regularized_operator_core(
    r_eval: f64,
    kappa: f64,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRegularizedOperatorCore, BasisError> {
    // Assemble the operator scalars with compensated summation because the
    // partial-fraction coefficients can alternate in sign and span many orders
    // of magnitude in higher dimensions.
    let mut q_sum = KahanSum::default();
    let mut t_sum = KahanSum::default();
    let mut t_r_sum = KahanSum::default();
    let mut t_rr_sum = KahanSum::default();

    for (m, coeff) in coeffs.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (q, t, t_r, t_rr) = duchon_polyharmonic_operator_block_jets(r_eval, m as f64, k_dim)?;
        q_sum.add(coeff * q);
        t_sum.add(coeff * t);
        t_r_sum.add(coeff * t_r);
        t_rr_sum.add(coeff * t_rr);
    }
    // One Bessel-K ladder at z = κ·r serves every Matérn block and every
    // term of their derivative lattices (see [`BesselKLadder`]); the old
    // per-term Bessel calls restarted the seed+recurrence hundreds of times
    // per evaluation point.
    let max_ladder_steps = coeffs
        .b
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, coeff)| **coeff != 0.0)
        .map(|(n, _)| duchon_matern_block_max_ladder_steps(n, k_dim))
        .max();
    if let Some(max_ladder_steps) = max_ladder_steps {
        let ladder =
            BesselKLadder::build(kappa * r_eval, !k_dim.is_multiple_of(2), max_ladder_steps);
        for (n, coeff) in coeffs.b.iter().enumerate().skip(1) {
            if *coeff == 0.0 {
                continue;
            }
            let (q, t, t_r, t_rr) =
                duchon_matern_operator_block_jets_with_ladder(r_eval, kappa, n, k_dim, &ladder)?;
            q_sum.add(coeff * q);
            t_sum.add(coeff * t);
            t_r_sum.add(coeff * t_r);
            t_rr_sum.add(coeff * t_rr);
        }
    }
    Ok(DuchonRegularizedOperatorCore {
        q: q_sum.sum(),
        t: t_sum.sum(),
        t_r: t_r_sum.sum(),
        t_rr: t_rr_sum.sum(),
    })
}

#[inline(always)]
pub(crate) fn duchon_collision_taylor_operator_core(
    r: f64,
    phi_rr_collision: f64,
    t_collision: f64,
    t_rr_collision: f64,
) -> DuchonRegularizedOperatorCore {
    let r2 = r * r;
    let r4 = r2 * r2;
    DuchonRegularizedOperatorCore {
        q: phi_rr_collision + 0.5 * t_collision * r2 + 0.125 * t_rr_collision * r4,
        t: t_collision + 0.5 * t_rr_collision * r2,
        t_r: t_rr_collision * r,
        t_rr: t_rr_collision,
    }
}

pub(crate) fn duchon_radial_jets(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRadialJets, BasisError> {
    let kappa = 1.0 / length_scale.max(1e-300);
    let r_floor = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale.max(1e-8);
    let collision_taylor_radius = DUCHON_COLLISION_TAYLOR_REL * length_scale.max(1e-8);
    let r_eval = r.max(r_floor);
    let d = k_dim as f64;

    // Value path keeps the intrinsic diagonal convention used by the actual basis.
    let phi = duchon_matern_kernel_general_from_distance(
        r,
        Some(length_scale),
        p_order,
        s_order,
        k_dim,
        Some(coeffs),
    )?;
    if !phi.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Duchon radial kernel value at r={r}, length_scale={length_scale}, p={p_order}, s={s_order}, dim={k_dim}"
        );
    }

    // Assemble the operator scalars. The partial-fraction operator core
    //   q = Σ a_m q_m + Σ b_n q_n,  t = Σ … (`duchon_regularized_operator_core`)
    // is a sign-alternating sum whose blocks individually scale like
    // r^{2m-d}; in high dimensions each block is ~1e3 while the true operator
    // scalar is ~1e-13, so f64 loses every digit (gam#1424 / gam#1453). For the
    // genuine Matérn-blend orders, evaluate `(q, t, t_r, t_rr)` via the same
    // cancellation-free single integral as the kernel value, differentiated
    // under the integral sign — each w-slice is one well-conditioned
    // r^a K_ν(c r) term with no cross-block cancellation. The complementary
    // orders (s = 0 pure polyharmonic, or 2p ≥ d at low d) keep the direct
    // partial-fraction core, which has no meaningful cancellation there.
    let operator_core = if duchon_hybrid_stable_integral_applies(p_order, s_order, k_dim) {
        duchon_hybrid_operator_stable_integral(r_eval, kappa, p_order, s_order, k_dim)?
    } else {
        duchon_regularized_operator_core(r_eval, kappa, k_dim, coeffs)?
    };
    let generic_jets = duchon_operator_jets_from_primary_core(operator_core, r_eval, d);
    let mut out = DuchonRadialJets {
        phi,
        ..generic_jets
    };

    // Smoothness check: the collision Taylor expansion requires analytic
    // collision limits (t(0) = φ''''(0)/3, etc.) which only exist when the
    // kernel is sufficiently smooth at the origin: 2(p+s) > d + 2j.
    // For the borderline case (2(p+s) == d+4), φ''''(0) diverges
    // logarithmically and the Taylor carrier cannot represent t(r) accurately.
    // In that regime, keep the generic-path values at r_eval = r_floor.
    let smoothness_order = 2 * (p_order + s_order);
    let collision_q_exists = smoothness_order > k_dim + 2;
    let collision_t_exists = smoothness_order > k_dim + 4;
    let collision_t_rr_exists = smoothness_order > k_dim + 6;

    if r <= collision_taylor_radius.max(r_floor) && collision_t_exists {
        // Tier 2+: full collision Taylor expansion using φ''(0), φ''''(0)/3,
        // and optionally φ⁽⁶⁾(0)/15.  Replaces the generic r_floor path for
        // all radial scalars in the near-origin region.
        let (analytic_phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        let analytic_t_collision =
            duchon_phi_rrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)? / 3.0;
        let analytic_t_rr_collision = if collision_t_rr_exists {
            duchon_phi_rrrrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)? / 15.0
        } else {
            // t_rr(0) does not exist as a finite limit for this smoothness
            // order, so the smooth-origin carrier must stop at the quadratic
            // term in t(r) and the quartic term in q(r), phi_r(r), phi_rr(r).
            0.0
        };
        let collision_jets = duchon_operator_jets_from_primary_core(
            duchon_collision_taylor_operator_core(
                r,
                analytic_phi_rr,
                analytic_t_collision,
                analytic_t_rr_collision,
            ),
            r,
            d,
        );
        out = DuchonRadialJets {
            phi: out.phi,
            ..collision_jets
        };
    } else if r < r_floor && collision_q_exists {
        // Tier 1: only lower-order collision identities exist.  φ''(0) is
        // finite but φ''''(0) diverges logarithmically at this smoothness
        // order.  Override phi_r, phi_rr, q, q_r, lap, lap_r with exact
        // values; leave t, t_r, t_rr, q_rr, lap_rr at their generic-path
        // values from r_eval = r_floor (best available for the divergent tier).
        let (analytic_phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        out.phi_r = analytic_phi_rr * r;
        out.phi_rr = analytic_phi_rr;
        out.q = analytic_phi_rr;
        out.q_r = 0.0;
        out.lap = d * analytic_phi_rr;
        out.lap_r = 0.0;
    }
    if !out.phi_r.is_finite()
        || !out.phi_rr.is_finite()
        || !out.phi_rrr.is_finite()
        || !out.q.is_finite()
        || !out.q_r.is_finite()
        || !out.q_rr.is_finite()
        || !out.lap.is_finite()
        || !out.lap_r.is_finite()
        || !out.lap_rr.is_finite()
        || !out.t.is_finite()
        || !out.t_r.is_finite()
        || !out.t_rr.is_finite()
    {
        crate::bail_invalid_basis!(
            "non-finite Duchon radial jets at r={r}, length_scale={length_scale}, p={p_order}, s={s_order}, dim={k_dim}"
        );
    }
    Ok(out)
}

pub(crate) fn duchon_radial_core_psi_triplet(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRadialCore, BasisError> {
    // Duchon spectral derivation
    // Start from the isotropic spectrum
    //   K^(ω; kappa) ∝ 1 / (|ω|^(2p) * (kappa^2 + |ω|^2)^s),
    // with fixed integer orders p,s and continuous scale
    //   psi = log(kappa),   kappa = 1 / length_scale.
    //
    // Rescaling frequency by ω = kappa ξ gives the full spatial kernel scaling law
    //   phi(r; kappa) = kappa^delta H(kappa r),
    //   delta = d - 2p - 2s.
    //
    // Therefore the exact full-kernel psi derivatives are
    //   phi_psi     = delta * phi + r * phi_r
    //   phi_psipsi  = delta^2 * phi + (2 delta + 1) r phi_r + r^2 phi_rr.
    //
    // The operator scalars are
    //   q(r; kappa) = phi_r(r; kappa) / r
    //   ell(r; kappa) = Δphi(r; kappa) = phi_rr + (d-1) q.
    // Both q and ell scale with exponent delta + 2, so
    //   q_psi       = (delta + 2) q + r q_r
    //   q_psipsi    = (delta + 2)^2 q + (2 delta + 5) r q_r + r^2 q_rr
    // and identically for ell.
    //
    // Once {phi, q, ell} and their psi derivatives are known, the collocation
    // operators follow exactly:
    //   D0[k,j]         = phi(r_kj)
    //   D1[(k,a), j]    = q(r_kj) * (x_{k,a} - c_{j,a})
    //   D2[k,j]         = ell(r_kj)
    // and the penalty Hessians come from the Gram identities
    //   S_psi     = D_psi^T D + D^T D_psi
    //   S_psipsi  = D_psipsi^T D + 2 D_psi^T D_psi + D^T D_psipsi.
    //
    // This helper computes exactly that minimal scalar core:
    //   phi, q = phi_r / r, ell = Δphi
    // together with their first and second psi derivatives.
    //
    // Representation note:
    //   When p > 0 the Duchon kernel is only conditionally positive definite, so
    //   the spatial kernel is canonical only up to polynomial additions. The
    //   formulas in this helper are therefore tied to the specific representative
    //   encoded by the partial-fraction construction and the collision rules used
    //   below. The operator penalties, exact psi derivatives, and center-collision
    //   limits all have to use that same representative or the resulting penalty
    //   geometry will drift across code paths.
    let delta = duchon_scaling_exponent(p_order, s_order, k_dim);
    let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, coeffs)?;
    let phi = jets.phi;
    let (phi_psi, phi_psi_psi) =
        scaled_log_kappa_derivatives(phi, jets.phi_r, jets.phi_rr, delta, r);
    if r > 1e-10 {
        assert!(
            ((delta * phi + r * jets.phi_r) - phi_psi).abs() < 1e-7_f64.max(1e-7_f64 * phi.abs())
        );
        return Ok(DuchonRadialCore {
            phi: PsiTriplet {
                value: phi,
                psi: phi_psi,
                psi_psi: phi_psi_psi,
            },
        });
    }

    // Continuous center-collision extension for the scalar operator core:
    //   q(0; kappa) = phi_rr(0; kappa)
    //   L(0; kappa) = d * phi_rr(0; kappa).
    //
    // The value and psi derivatives are extracted from the same Taylor
    // coefficient of the assembled partial-fraction kernel. In even dimensions
    // this preserves the log-Riesz finite-part constants, so the collision
    // derivative is not the naive `(delta + 2) * phi_rr` scaling shortcut.
    Ok(DuchonRadialCore {
        phi: PsiTriplet {
            value: phi,
            psi: phi_psi,
            psi_psi: phi_psi_psi,
        },
    })
}

pub(crate) fn duchonphi_rr_collision_psi_triplet(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<(f64, f64, f64), BasisError> {
    // Center-collision rule
    // For a C^2 radial kernel one has
    //   lim_{r->0} phi_r(r)/r = phi_rr(0),
    //   lim_{r->0} Δphi(r)    = d * phi_rr(0).
    //
    // Assemble phi_rr and its psi derivatives by summing the partial-fraction
    // blocks directly.  Do not replace this with the tempting scaling shortcut
    // `phi_rr_psi = (delta + 2) phi_rr`: in even dimensions the log-Riesz
    // representative carries kappa-dependent finite-part constants at the
    // origin, so the shortcut gives the wrong center-collision derivative even
    // when the classical C^2 limit exists.
    duchon_phi_even_derivative_collision_psi_triplet(
        length_scale,
        p_order,
        s_order,
        k_dim,
        coeffs,
        1,
    )
}

/// Euler-Mascheroni constant γ ≈ 0.5772.
pub(crate) const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Digamma function ψ(n) for positive integer n.
///
/// ψ(1) = −γ, ψ(n+1) = −γ + H_n where H_n = Σ_{j=1}^{n} 1/j.
#[inline(always)]
pub(crate) fn digamma_pos_int(n: usize) -> f64 {
    assert!(n >= 1, "digamma_pos_int requires n >= 1: n={n}");
    let mut h = 0.0_f64;
    for j in 1..n {
        h += 1.0 / j as f64;
    }
    -EULER_MASCHERONI + h
}

/// Extract the coefficient of r^{2j} (pure and log-r parts) from a single
/// Matérn partial-fraction block g_n(r) = c · r^ν · K_{|ν|}(κr), where
/// ν = n − d/2.
///
/// Returns `(pure_coeff, log_coeff)` such that the r^{2j} piece of g_n is
///   pure_coeff · r^{2j}  +  log_coeff · r^{2j} · ln(r).
///
/// For even d (integer ν) the expansion uses the DLMF 10.31.1 series for
/// K_n(z) at the origin, which involves digamma / harmonic-number terms.
///
/// For odd d (half-integer ν) the Bessel function is elementary; the Taylor
/// coefficients come from convolving a finite polynomial in 1/r with e^{−κr},
/// and there is no log-r contribution.
pub(crate) fn duchon_matern_block_taylor_r2j(
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    j: usize,
) -> (f64, f64) {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    // Normalization constant for the Matérn block.
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));

    if k_dim.is_multiple_of(2) {
        // Integer ν.
        let nu_int = n_order as i64 - (k_dim as i64) / 2;
        duchon_matern_block_taylor_r2j_integer_nu(kappa, c, nu_int, j)
    } else {
        // Half-integer ν.
        duchon_matern_block_taylor_r2j_half_integer_nu(kappa, c, nu, j)
    }
}

#[inline(always)]
pub(crate) fn psi_power_triplet(value: f64, exponent: f64) -> (f64, f64, f64) {
    (value, exponent * value, exponent * exponent * value)
}

#[inline(always)]
pub(crate) fn psi_power_log_triplet(
    base: f64,
    exponent: f64,
    log_kappa_half: f64,
) -> (f64, f64, f64) {
    (
        base * log_kappa_half,
        base * (exponent * log_kappa_half + 1.0),
        base * (exponent * exponent * log_kappa_half + 2.0 * exponent),
    )
}

#[inline(always)]
pub(crate) fn add_triplet(dst: &mut (f64, f64, f64), inc: (f64, f64, f64)) {
    dst.0 += inc.0;
    dst.1 += inc.1;
    dst.2 += inc.2;
}

/// Like [`duchon_matern_block_taylor_r2j`], but also returns exact
/// derivatives of the pure/log Taylor coefficients with respect to
/// `psi = log(kappa)`.
pub(crate) fn duchon_matern_block_taylor_r2j_triplet(
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    j: usize,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c_const = 1.0
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));
    let c_exp = k_half - n;

    let mut pure = (0.0, 0.0, 0.0);
    let mut log_part = (0.0, 0.0, 0.0);
    let log_kappa_half = (0.5 * kappa).ln();

    if k_dim.is_multiple_of(2) {
        let nu_int = n_order as i64 - (k_dim as i64) / 2;
        let mu = nu_int.unsigned_abs() as usize;
        let sign_mu = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };

        if nu_int >= 0 {
            let nu_usize = nu_int as usize;

            if j < nu_usize {
                let sign = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
                let power = 2 * j as i32 - nu_usize as i32;
                let coeff = 0.5 * sign * gamma_lanczos((nu_usize - j) as f64)
                    / gamma_lanczos((j + 1) as f64)
                    * 2.0_f64.powi(-power);
                let exponent = c_exp + power as f64;
                let value = c_const * coeff * kappa.powf(exponent);
                add_triplet(&mut pure, psi_power_triplet(value, exponent));
            }

            if j >= nu_usize {
                let k = j - nu_usize;
                let inv_fac = 1.0
                    / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((nu_usize + k + 1) as f64));
                let power = (2 * k + nu_usize) as i32;
                let exponent = c_exp + power as f64;
                let kp_base = c_const * kappa.powf(exponent) * 2.0_f64.powi(-power);

                let log_base = -sign_mu * kp_base * inv_fac;
                add_triplet(&mut log_part, psi_power_triplet(log_base, exponent));
                add_triplet(
                    &mut pure,
                    psi_power_log_triplet(log_base, exponent, log_kappa_half),
                );

                let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(nu_usize + k + 1);
                let digamma_base = sign_mu * 0.5 * kp_base * inv_fac * psi_sum;
                add_triplet(&mut pure, psi_power_triplet(digamma_base, exponent));
            }
        } else {
            let k = j;
            let inv_fac =
                1.0 / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((mu + k + 1) as f64));
            let power = (mu + 2 * k) as i32;
            let exponent = c_exp + power as f64;
            let kp_base = c_const * kappa.powf(exponent) * 2.0_f64.powi(-power);

            let log_base = -sign_mu * kp_base * inv_fac;
            add_triplet(&mut log_part, psi_power_triplet(log_base, exponent));
            add_triplet(
                &mut pure,
                psi_power_log_triplet(log_base, exponent, log_kappa_half),
            );

            let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(mu + k + 1);
            let digamma_base = sign_mu * 0.5 * kp_base * inv_fac * psi_sum;
            add_triplet(&mut pure, psi_power_triplet(digamma_base, exponent));
        }
    } else {
        let nu_abs = nu.abs();
        let l = (2.0 * nu_abs - 1.0).round().max(0.0) as usize;
        let prefactor_const = (std::f64::consts::PI / 2.0).sqrt();
        let prefactor_exp = -0.5;
        let target = 2 * j;

        for i in 0..=l {
            let c_i = gamma_lanczos((l + i + 1) as f64)
                / (gamma_lanczos((i + 1) as f64) * gamma_lanczos((l - i + 1) as f64));
            let p_f64 = nu - 0.5 - i as f64;
            let p_round = p_f64.round() as i64;
            if (p_f64 - p_round as f64).abs() > 1e-12 {
                continue;
            }
            let q_needed = target as i64 - p_round;
            if q_needed < 0 {
                continue;
            }
            let q = q_needed as usize;
            let sign = if q.is_multiple_of(2) { 1.0 } else { -1.0 };
            let exponent = c_exp + prefactor_exp - i as f64 + q as f64;
            let value = c_const * prefactor_const * c_i * 2.0_f64.powi(-(i as i32)) * sign
                / gamma_lanczos((q + 1) as f64)
                * kappa.powf(exponent);
            add_triplet(&mut pure, psi_power_triplet(value, exponent));
        }
    }

    (pure, log_part)
}

/// Taylor r^{2j} coefficients for integer-ν Matérn block.
///
/// Uses the K_μ(z) expansion for integer μ = |ν| ≥ 0 (A&S 9.6.11 / DLMF 10.31.1):
///
///   K_μ(z) = (−1)^{μ+1} I_μ(z) ln(z/2)
///          + ½ Σ_{k=0}^{μ−1} (−1)^k (μ−k−1)!/k! · (z/2)^{2k−μ}   [singular]
///          + (−1)^μ · ½ Σ_{k≥0} (z/2)^{μ+2k}/(k!(μ+k)!)
///                              · [ψ(k+1)+ψ(μ+k+1)]                  [regular]
///
/// Multiplied by r^ν, the r^{2j} coefficient is assembled from the singular
/// and/or regular+log series depending on the sign and magnitude of ν.
pub(crate) fn duchon_matern_block_taylor_r2j_integer_nu(
    kappa: f64,
    c: f64,
    nu_int: i64,
    j: usize,
) -> (f64, f64) {
    let mu = nu_int.unsigned_abs() as usize; // |ν|

    // Helper: compute (κ/2)^p for integer p.
    let kappa_half = 0.5 * kappa;

    if nu_int >= 0 {
        let nu = nu_int as usize;
        // Two potential sources for the r^{2j} coefficient:
        //
        // 1) Singular sum:  contributes when j ≤ ν−1 (the k=j term gives r^{2j}).
        // 2) Regular+log sum: contributes when 2ν+2k = 2j, i.e. k = j−ν ≥ 0.
        let mut pure = 0.0;
        let mut log_part = 0.0;

        // Source 1: singular sum at k = j.
        if j < nu {
            // (1/2) · (−1)^j · (ν−j−1)!/j! · (κ/2)^{2j−ν}
            let sign = if j.is_multiple_of(2) { 1.0 } else { -1.0 };
            let coeff = sign * gamma_lanczos((nu - j) as f64) / gamma_lanczos((j + 1) as f64)
                * kappa_half.powi(2 * j as i32 - nu as i32)
                * 0.5;
            pure += coeff;
        }

        // Source 2: regular+log sum at k = j − ν.
        if j >= nu {
            let k = j - nu;
            let inv_fac =
                1.0 / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((nu + k + 1) as f64));
            let kp = kappa_half.powi(2 * k as i32 + nu as i32);
            let sign_mu = if mu.is_multiple_of(2) { 1.0 } else { -1.0 }; // (−1)^μ

            // Log coefficient: (−1)^{μ+1} · (κ/2)^{ν+2k} / (k!(ν+k)!)
            log_part += -sign_mu * kp * inv_fac;

            // Pure coefficient from the log series (ln(κ/2) piece):
            //   (−1)^{μ+1} · (κ/2)^{ν+2k} / (k!(ν+k)!) · ln(κ/2)
            // Plus the digamma series:
            //   (−1)^μ · ½ · (κ/2)^{ν+2k} / (k!(ν+k)!) · [ψ(k+1)+ψ(ν+k+1)]
            let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(nu + k + 1);
            pure += -sign_mu * kp * inv_fac * kappa_half.ln();
            pure += sign_mu * 0.5 * kp * inv_fac * psi_sum;
        }

        (c * pure, c * log_part)
    } else {
        // ν < 0: mu = |ν| > 0.
        // Singular sum gives powers r^{2ν}, ..., r^{−2} (all negative).
        // Regular+log sum gives r^0, r^2, r^4, ... at k = j.
        let k = j;
        let inv_fac = 1.0 / (gamma_lanczos((k + 1) as f64) * gamma_lanczos((mu + k + 1) as f64));
        let kp = kappa_half.powi(mu as i32 + 2 * k as i32);
        let sign_mu = if mu.is_multiple_of(2) { 1.0 } else { -1.0 };

        // Log coefficient: (−1)^{μ+1} · (κ/2)^{μ+2k} / (k!(μ+k)!)
        let log_part = -sign_mu * kp * inv_fac;

        // Pure coefficient: log-series ln(κ/2) piece + digamma piece.
        let psi_sum = digamma_pos_int(k + 1) + digamma_pos_int(mu + k + 1);
        let pure =
            -sign_mu * kp * inv_fac * kappa_half.ln() + sign_mu * 0.5 * kp * inv_fac * psi_sum;

        (c * pure, c * log_part)
    }
}

/// Taylor r^{2j} coefficients for half-integer-ν Matérn block.
///
/// For half-integer |ν| = l + ½, K_{l+½}(z) is elementary:
///   K_{l+½}(z) = √(π/(2z)) · e^{−z} · Σ_{i=0}^{l} C_i · (2z)^{−i}
/// where C_i = (l+i)! / (i! · (l−i)!).
///
/// The product r^ν · K_{|ν|}(κr) expands as an explicit polynomial in r
/// (including possible negative powers) times e^{−κr}.  The r^{2j} Taylor
/// coefficient is obtained by convolving with the exponential series
/// e^{−κr} = Σ_q (−κ)^q r^q / q!.  There is never a log-r contribution.
pub(crate) fn duchon_matern_block_taylor_r2j_half_integer_nu(
    kappa: f64,
    c: f64,
    nu: f64,
    j: usize,
) -> (f64, f64) {
    let nu_abs = nu.abs();
    let l = (2.0 * nu_abs - 1.0).round().max(0.0) as usize;
    // Compute the polynomial coefficients C_i / (2κ)^i for each r-power.
    //
    // r^ν · K_{l+½}(κr) = √(π/(2κ)) · e^{−κr} · Σ_{i=0}^{l} C_i (2κ)^{−i} r^{ν−½−i}
    //
    // (since K_{l+½}(z) = √(π/(2z)) e^{−z} Σ C_i (2z)^{−i}, multiplying by
    // r^ν gives r^{ν−½} from the √(π/(2κr)) factor, then each (2κr)^{−i}
    // contributes r^{−i}.)
    let prefactor = (std::f64::consts::PI / (2.0 * kappa)).sqrt();

    // Polynomial term i has r-power = ν − 0.5 − i.  We need to convolve
    // each monomial with e^{−κr} = Σ_q (−κ)^q r^q / q! and extract the
    // r^{2j} coefficient.
    //
    // For monomial r^p (p = ν−½−i) times e^{−κr}: the r^{2j} coefficient is
    //   (−κ)^{2j−p} / (2j−p)!   when 2j−p is a non-negative integer.
    let target = 2 * j;
    let mut pure = 0.0;

    for i in 0..=l {
        let c_i = gamma_lanczos((l + i + 1) as f64)
            / (gamma_lanczos((i + 1) as f64) * gamma_lanczos((l - i + 1) as f64));
        let inv_2kappa_i = (2.0 * kappa).powi(-(i as i32));

        // r-power of this polynomial term.
        let p_f64 = nu - 0.5 - i as f64;
        let p_round = p_f64.round() as i64;
        if (p_f64 - p_round as f64).abs() > 1e-12 {
            // Not integer/half-integer aligned — should not happen for half-integer ν.
            continue;
        }
        let q_needed = target as i64 - p_round;
        if q_needed < 0 {
            continue;
        }
        let q = q_needed as usize;
        let exp_coeff = (-kappa).powi(q as i32) / gamma_lanczos((q + 1) as f64);
        pure += c_i * inv_2kappa_i * exp_coeff;
    }

    (c * prefactor * pure, 0.0) // No log contribution for half-integer ν.
}

/// Extract the r^{2j} Taylor coefficient from a polyharmonic block Φ_m(r).
///
/// Non-log case (d odd, or d even with m < d/2): Φ_m = c · r^α with α = 2m − d.
///   Only contributes when α = 2j exactly: pure_coeff = c, log_coeff = 0.
///
/// Log case (d even, m ≥ d/2): Φ_m = c · r^α · ln(r).
///   Only contributes when α = 2j: pure_coeff = 0, log_coeff = c.
pub(crate) fn duchon_polyharmonic_block_taylor_r2j(m: usize, k_dim: usize, j: usize) -> (f64, f64) {
    let k_half = 0.5 * k_dim as f64;
    let alpha = 2 * m as i64 - k_dim as i64;

    if alpha != 2 * j as i64 {
        return (0.0, 0.0);
    }

    // α = 2j: compute the normalization constant.
    if k_dim.is_multiple_of(2) && m >= k_dim / 2 {
        // Log case: Φ_m = c · r^α · ln(r).
        let c = polyharmonic_log_sign(m, k_dim)
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64)
                * gamma_lanczos((m - k_dim / 2 + 1) as f64));
        (0.0, c)
    } else {
        // Non-log case: Φ_m = c · r^α.
        let c = gamma_lanczos(k_half - m as f64)
            / (4.0_f64.powi(m as i32)
                * std::f64::consts::PI.powf(k_half)
                * gamma_lanczos(m as f64));
        (c, 0.0)
    }
}

/// Compute the even-order radial derivative φ^{(2j)}(0) from analytic Taylor
/// coefficients of the partial-fraction blocks.
///
/// For a C^{2j} radial kernel with Taylor expansion φ(r) = Σ_k a_{2k} r^{2k},
/// φ^{(2j)}(0) = (2j)! · a_{2j}.  Each partial-fraction block (polyharmonic
/// and Matérn) has a computable r^{2j} Taylor coefficient (both pure and
/// ln(r) parts).  The ln(r) contributions cancel across blocks whenever the
/// kernel is sufficiently smooth; the pure coefficients sum to give a_{2j}.
///
/// Existence condition (kernel is C^{2j} at the origin):
///   2(p + s) > d + 2j.
///
/// When this condition fails (borderline or insufficient smoothness), the
/// derivative is not a finite collision limit. Callers must reject that model
/// upstream rather than regularize it at an arbitrary floor radius.
pub(crate) fn duchon_phi_even_derivative_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
    j: usize,
) -> Result<f64, BasisError> {
    let smoothness_order = 2 * (p_order + s_order);
    let required = k_dim + 2 * j;

    if smoothness_order <= required {
        // Smallest integer power admitting phi^{(2j)}(0): 2(p+s) > k_dim+2j.
        let min_power = (required / 2 + 1).saturating_sub(p_order);
        crate::bail_invalid_basis!(
            "Duchon collision derivative phi^({}) requires 2*(p+s) > dimension+{}; got 2*(p+s)={}, dimension={}, p={}, s={}. \
             This path needs the {}-order radial-kernel derivative at the origin, which is finite only for a smoother spline: raise power to >= {} (or reduce the joint smooth's dimension).",
            2 * j,
            2 * j,
            smoothness_order,
            k_dim,
            p_order,
            s_order,
            2 * j,
            min_power
        );
    }

    // Analytic path: extract per-block Taylor r^{2j} coefficients and sum.
    let kappa = 1.0 / length_scale.max(1e-300);
    let mut total_pure = KahanSum::default();
    let mut total_log = KahanSum::default();
    let mut total_log_abs_scale = KahanSum::default();

    // Polyharmonic blocks.
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let (pure, log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, j);
        total_pure.add(a_m * pure);
        total_log.add(a_m * log);
        total_log_abs_scale.add((a_m * log).abs());
    }

    // Matérn blocks.
    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let (pure, log) = duchon_matern_block_taylor_r2j(kappa, n, k_dim, j);
        total_pure.add(b_n * pure);
        total_log.add(b_n * log);
        total_log_abs_scale.add((b_n * log).abs());
    }
    let total_pure = total_pure.sum();
    let total_log = total_log.sum();
    let total_log_abs_scale = total_log_abs_scale.sum();

    // The ln(r) coefficients should cancel to zero (guaranteed by the PFD
    // identity when 2(p+s) > d+2j).  Check this as a sanity guard.
    let log_cancel_tol = 1e-10 * total_log_abs_scale.max(total_pure.abs()).max(1e-30);
    if total_log.abs() > log_cancel_tol {
        crate::bail_invalid_basis!(
            "Duchon Taylor a_{} log-coefficient did not cancel: log={total_log:.6e}, pure={total_pure:.6e}; \
             log_abs_scale={total_log_abs_scale:.6e}, tol={log_cancel_tol:.6e}; p={p_order}, s={s_order}, d={k_dim}",
            2 * j
        );
    }

    // φ^{(2j)}(0) = (2j)! · a_{2j}
    let factorial_2j = gamma_lanczos((2 * j + 1) as f64);
    Ok(factorial_2j * total_pure)
}

pub(crate) fn duchon_phi_even_derivative_collision_psi_triplet(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
    j: usize,
) -> Result<(f64, f64, f64), BasisError> {
    let smoothness_order = 2 * (p_order + s_order);
    let required = k_dim + 2 * j;

    if smoothness_order <= required {
        // Smallest integer power admitting the phi^{(2j)} psi triplet: 2(p+s) > k_dim+2j.
        let min_power = (required / 2 + 1).saturating_sub(p_order);
        crate::bail_invalid_basis!(
            "Duchon collision derivative phi^({}) psi triplet requires 2*(p+s) > dimension+{}; got 2*(p+s)={}, dimension={}, p={}, s={}. \
             The exact two-block / transformation-normal path needs analytic length-scale derivatives of the kernel, which are finite only for a smoother spline: raise power to >= {} (or reduce the joint smooth's dimension).",
            2 * j,
            2 * j,
            smoothness_order,
            k_dim,
            p_order,
            s_order,
            min_power
        );
    }

    let kappa = 1.0 / length_scale.max(1e-300);
    let mut value = KahanSum::default();
    let mut psi = KahanSum::default();
    let mut psi_psi = KahanSum::default();
    let mut log_value = KahanSum::default();
    let mut log_psi = KahanSum::default();
    let mut log_psi_psi = KahanSum::default();
    let mut log_abs_scale = KahanSum::default();

    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let alpha_m = duchon_coeff_exponents(p_order, s_order, m);
        let (pure, log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, j);
        value.add(a_m * pure);
        psi.add(alpha_m * a_m * pure);
        psi_psi.add(alpha_m * alpha_m * a_m * pure);
        log_value.add(a_m * log);
        log_psi.add(alpha_m * a_m * log);
        log_psi_psi.add(alpha_m * alpha_m * a_m * log);
        log_abs_scale.add((a_m * log).abs());
        log_abs_scale.add((alpha_m * a_m * log).abs());
        log_abs_scale.add((alpha_m * alpha_m * a_m * log).abs());
    }

    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let beta_n = duchon_coeff_exponents(p_order, s_order, n);
        let (pure, log) = duchon_matern_block_taylor_r2j_triplet(kappa, n, k_dim, j);
        value.add(b_n * pure.0);
        psi.add(beta_n * b_n * pure.0 + b_n * pure.1);
        psi_psi.add(beta_n * beta_n * b_n * pure.0 + 2.0 * beta_n * b_n * pure.1 + b_n * pure.2);
        log_value.add(b_n * log.0);
        log_psi.add(beta_n * b_n * log.0 + b_n * log.1);
        log_psi_psi.add(beta_n * beta_n * b_n * log.0 + 2.0 * beta_n * b_n * log.1 + b_n * log.2);
        let log_v = b_n * log.0;
        let log_p = beta_n * b_n * log.0 + b_n * log.1;
        let log_pp = beta_n * beta_n * b_n * log.0 + 2.0 * beta_n * b_n * log.1 + b_n * log.2;
        log_abs_scale.add(log_v.abs());
        log_abs_scale.add(log_p.abs());
        log_abs_scale.add(log_pp.abs());
    }

    let value = value.sum();
    let psi = psi.sum();
    let psi_psi = psi_psi.sum();
    let log_value = log_value.sum();
    let log_psi = log_psi.sum();
    let log_psi_psi = log_psi_psi.sum();
    let log_abs_scale = log_abs_scale.sum();
    let scale = value.abs().max(psi.abs()).max(psi_psi.abs()).max(1e-30);
    let log_cancel_tol = 1e-10 * log_abs_scale.max(scale);
    if log_value.abs().max(log_psi.abs()).max(log_psi_psi.abs()) > log_cancel_tol {
        crate::bail_invalid_basis!(
            "Duchon Taylor a_{} log-coefficient derivative did not cancel: \
             log=({log_value:.6e}, {log_psi:.6e}, {log_psi_psi:.6e}), \
             value=({value:.6e}, {psi:.6e}, {psi_psi:.6e}), log_abs_scale={log_abs_scale:.6e}, tol={log_cancel_tol:.6e}; \
             p={p_order}, s={s_order}, d={k_dim}",
            2 * j
        );
    }

    let factorial_2j = gamma_lanczos((2 * j + 1) as f64);
    Ok((
        factorial_2j * value,
        factorial_2j * psi,
        factorial_2j * psi_psi,
    ))
}

/// Assemble φ''''(0) from the partial-fraction blocks using analytic Taylor
/// coefficients.
///
/// For a radial kernel with Taylor expansion φ(r) = a₀ + a₂r² + a₄r⁴ + ...,
/// we have φ''''(0) = 24 a₄.  This is used to compute the collision limit
/// t(0) = φ''''(0) / 3, where t = R²φ = (φ'' - q) / r².
///
/// Each partial-fraction block (polyharmonic and Matérn) has a known Taylor
/// expansion around r = 0; the r⁴ coefficient a₄ is extracted from the series
/// and summed.  This avoids the catastrophic cancellation that occurs when
/// evaluating divergent block derivatives at a small floor radius.
pub(crate) fn duchon_phi_rrrr_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    duchon_phi_even_derivative_collision(length_scale, p_order, s_order, k_dim, coeffs, 2)
}

/// Assemble φ⁽⁶⁾(0) from the partial-fraction blocks using analytic Taylor
/// coefficients.
///
/// For a radial kernel with Taylor expansion φ(r) = a₀ + a₂r² + a₄r⁴ + a₆r⁶ + ...,
/// we have φ⁽⁶⁾(0) = 720 a₆. This gives the collision limit
///   t_rr(0) = φ⁽⁶⁾(0) / 15
/// for t = R²φ.
///
/// Like [`duchon_phi_rrrr_collision`], this extracts per-block Taylor
/// coefficients analytically rather than evaluating divergent derivatives at
/// a small floor radius.
pub(crate) fn duchon_phi_rrrrrr_collision(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    duchon_phi_even_derivative_collision(length_scale, p_order, s_order, k_dim, coeffs, 3)
}

pub(crate) fn build_duchon_design_psi_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
    let length_scale = spec.length_scale.ok_or_else(|| {
        BasisError::InvalidInput(
            "exact Duchon log-kappa derivatives require hybrid Duchon with length_scale"
                .to_string(),
        )
    })?;
    // Exact Duchon design derivatives:
    // 1. evaluate phi_psi and phi_psipsi at each data/center distance
    // 2. project the kernel block with the same nullspace constraint used by the basis
    // 3. append polynomial columns; their psi derivatives are zero because p and s are fixed
    // 4. apply any frozen identifiability transform
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.power_as_usize();
    let kappa = 1.0 / length_scale;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
    let mut z_kernel =
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?;
    // #1355: fold the frozen data-metric reparam `Z' = Z·V` so the design
    // ψ-derivatives assemble in the SAME rotated radial basis as the forward
    // design and penalty (bit-consistent nonlinear/hybrid Duchon arm).
    if let Some(v) = spec.radial_reparam.as_ref() {
        if v.nrows() != z_kernel.ncols() {
            crate::bail_dim_basis!(
                "Duchon frozen radial reparam shape {:?} does not match constrained kernel dimension {}",
                v.dim(),
                z_kernel.ncols()
            );
        }
        z_kernel = fast_ab(&z_kernel, v);
    }
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();
    let p_padded = z_kernel.ncols() + poly_cols;
    if let Some(zf) = identifiability_transform
        && p_padded != zf.nrows()
    {
        crate::bail_dim_basis!(
            "Duchon identifiability transform mismatch in design derivatives: local cols={}, transform rows={}",
            p_padded,
            zf.nrows()
        );
    }
    let p_final = identifiability_transform
        .map(|zf| zf.ncols())
        .unwrap_or(p_padded);
    build_scalar_design_psi_derivatives_shared(
        data,
        centers,
        spec.aniso_log_scales.as_deref(),
        p_final,
        Some(z_kernel),
        identifiability_transform.cloned(),
        poly_cols,
        RadialScalarKind::Duchon {
            length_scale,
            p_order,
            s_order,
            dim: data.ncols(),
            coeffs,
        },
        duchon_scaling_exponent(p_order, s_order, data.ncols()),
    )
}

pub fn build_duchon_basis_log_kappa_derivative(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappa_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_duchon_basis_log_kappa_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeResult, BasisError> {
    let mut bundle = build_duchon_basis_log_kappa_derivativeswithworkspace(data, spec, workspace)?;
    bundle.first.implicit_operator = bundle.implicit_operator;
    Ok(bundle.first)
}

pub fn build_duchon_basis_log_kappa_derivatives(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappa_derivativeswithworkspace(data, spec, &mut workspace)
}

pub(crate) fn duchon_operator_penalties_requested(spec: &DuchonOperatorPenaltySpec) -> bool {
    matches!(spec.mass, OperatorPenaltySpec::Active { .. })
        || matches!(spec.tension, OperatorPenaltySpec::Active { .. })
        || matches!(spec.stiffness, OperatorPenaltySpec::Active { .. })
}

pub fn build_duchon_basis_log_kappa_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    if spec.periodic.is_some() {
        return build_periodic_duchon_basis_log_kappa_derivativeswithworkspace(
            data, spec, workspace,
        );
    }
    let (centers, identifiability_transform) =
        prepare_duchon_derivative_contextwithworkspace(data, spec, workspace)?;
    let operator_collocation_points =
        if duchon_operator_penalties_requested(&spec.operator_penalties) {
            let m = (DUCHON_COLLOCATION_OVERSAMPLE * centers.nrows()).min(data.nrows());
            Some(select_thin_plate_knots(data, m)?)
        } else {
            None
        };
    build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace(
        data,
        spec,
        centers.view(),
        identifiability_transform.as_ref(),
        operator_collocation_points
            .as_ref()
            .map(|points| points.view()),
        workspace,
    )
}

pub fn build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    centers: ArrayView2<'_, f64>,
    identifiability_transform: Option<&Array2<f64>>,
    operator_collocation_points: Option<ArrayView2<'_, f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiDerivativeBundle, BasisError> {
    let design_derivatives = build_duchon_design_psi_derivativeswithworkspace(
        data,
        centers,
        spec,
        identifiability_transform,
        workspace,
    )?;
    let (native_sources, native_first, native_second) =
        build_duchon_native_penalty_psi_derivatives(
            centers,
            spec,
            identifiability_transform,
            workspace,
        )?;
    let (operator_sources, operator_first, operator_second) = if duchon_operator_penalties_requested(
        &spec.operator_penalties,
    ) {
        let Some(collocation_points) = operator_collocation_points else {
            crate::bail_invalid_basis!(
                "Duchon log-kappa operator penalty derivatives require realized collocation points"
            );
        };
        build_duchon_operator_penalty_psi_derivatives(
            collocation_points,
            centers,
            spec,
            identifiability_transform,
            workspace,
        )?
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let mut penalties_derivative = Vec::with_capacity(native_first.len() + operator_first.len());
    penalties_derivative.extend(native_first);
    penalties_derivative.extend(operator_first);
    let mut penaltiessecond_derivative =
        Vec::with_capacity(native_second.len() + operator_second.len());
    penaltiessecond_derivative.extend(native_second);
    penaltiessecond_derivative.extend(operator_second);
    let expected_derivative_count = native_sources.len() + operator_sources.len();
    if penalties_derivative.len() != expected_derivative_count {
        crate::bail_invalid_basis!(
            "Duchon penalty derivative count mismatch: assembled {}, expected {} from active penalty sources",
            penalties_derivative.len(),
            expected_derivative_count
        );
    }
    Ok(BasisPsiDerivativeBundle {
        first: BasisPsiDerivativeResult {
            design_derivative: design_derivatives.design_first,
            penalties_derivative,
            implicit_operator: None,
        },
        second: BasisPsiSecondDerivativeResult {
            designsecond_derivative: design_derivatives.design_second_diag,
            penaltiessecond_derivative,
            implicit_operator: None,
        },
        implicit_operator: design_derivatives.implicit_operator,
    })
}

pub fn build_duchon_basis_log_kappasecond_derivative(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basis_log_kappasecond_derivativewithworkspace(data, spec, &mut workspace)
}

pub fn build_duchon_basis_log_kappasecond_derivativewithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    workspace: &mut BasisWorkspace,
) -> Result<BasisPsiSecondDerivativeResult, BasisError> {
    let mut bundle = build_duchon_basis_log_kappa_derivativeswithworkspace(data, spec, workspace)?;
    bundle.second.implicit_operator = bundle.implicit_operator;
    Ok(bundle.second)
}

/// Multiplicative amplification factor that lifts an underflowing Duchon
/// kernel back into a representable range. Probes max|K_CC| (the kernel at
/// every center pair) and returns `1/max` when the kernel collapses to the
/// double-precision noise floor; otherwise returns `1.0`.
///
/// **Why**: in high d with a small length scale the spectral normalization
/// `c = κ^{d/2-n} / ((2π)^{d/2}·2^{n-1}·Γ(n))` of the Matérn block is `~1e-14`,
/// driving every `K(r) = c · r^ν · K_ν(κr)` to `~1e-16`. Downstream
/// `B^T B` is then at `~1e-32` — below `eps²` — and the spectral whitener
/// truncates everything as noise, even though the basis is mathematically
/// well-defined.
///
/// Rescaling the basis by α = 1/max|K_CC| produces the same predictions
/// (β rescales by α, REML's λ adapts). Since the probe is computed from
/// `centers + kernel parameters` which are stored verbatim in
/// `BasisMetadata::Duchon`, prediction recomputes an identical α — so
/// fit-time and predict-time bases share a single coefficient frame.
pub(crate) fn duchon_kernel_amplification(
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    d: usize,
    aniso_log_scales: Option<&[f64]>,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
    pure_poly_coeff: Option<&PolyharmonicBlockCoeff>,
) -> f64 {
    let k = centers.nrows();
    if k == 0 {
        return 1.0;
    }
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let mut max_abs = 0.0_f64;
    for i in 0..k {
        for j in i..k {
            let r = if let Some(scales) = axis_scales.as_deref() {
                aniso_distance_rows_with_scales(centers, i, centers, j, scales)
            } else {
                euclidean_distance_rows(centers, i, centers, j)
            };
            let val = if let Some(ppc) = pure_poly_coeff {
                ppc.eval(r)
            } else {
                match duchon_matern_kernel_general_from_distance(
                    r,
                    length_scale,
                    p_order,
                    s_order,
                    d,
                    coeffs,
                ) {
                    Ok(v) => v,
                    Err(_) => continue,
                }
            };
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }
    }
    // Only amplify when the kernel has underflowed. The 1e-10 threshold is
    // well above any meaningful smoothing-relevant kernel scale yet far from
    // 1.0, so well-conditioned kernels pass through unchanged (α = 1).
    if max_abs > 0.0 && max_abs < 1e-10 {
        1.0 / max_abs
    } else {
        1.0
    }
}

/// Scalar kernel amplification `α` that [`build_duchon_basis`] applies to the
/// pure scale-free polyharmonic Duchon kernel block (`length_scale = None`,
/// `power = 0`, no anisotropy) for the given requested null-space `order`.
///
/// This is the exact factor the forward design multiplies into `K(t,C)` before
/// the null-space projection `Z`, so any derivative path that differentiates
/// that forward design (e.g. the `duchon_basis_with_jet` FFI, which builds its
/// forward via [`build_duchon_basis`] with these same parameters) must scale
/// its raw radial jet by the identical `α`. Returning it from the Rust core —
/// rather than recomputing the amplification probe in a wrapper — keeps the
/// derivative bit-for-bit consistent with the forward and avoids duplicating
/// the spectral-normalization math outside this module.
///
/// The requested `order` is internally degraded via
/// [`duchon_effective_nullspace_order`] exactly as the forward builder does, so
/// the polyharmonic order `p` used by the amplification probe matches.
pub fn duchon_pure_kernel_amplification(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
    power: f64,
) -> f64 {
    let dim = centers.ncols();
    if dim == 0 || centers.nrows() == 0 {
        return 1.0;
    }
    let effective_order = duchon_effective_nullspace_order(centers, order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = power;
    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    )
}

pub(crate) fn build_duchon_basis_designwithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    radial_reparam: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<DuchonBasisDesign, BasisError> {
    let n = data.nrows();
    let d = data.ncols();
    let k = centers.nrows();

    if d == 0 {
        crate::bail_invalid_basis!("Duchon basis requires at least one covariate dimension");
    }
    if k == 0 {
        crate::bail_invalid_basis!("Duchon basis requires at least one center");
    }
    if centers.ncols() != d {
        crate::bail_dim_basis!(
            "Duchon basis dimension mismatch: data has {d} columns, centers have {}",
            centers.ncols()
        );
    }
    if data.iter().any(|v| !v.is_finite()) || centers.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("Duchon basis requires finite data and center values");
    }
    // Auto-degrade the null-space order to Zero when centers are insufficient
    // to span the requested polynomial block; emits a warning inside the helper.
    let nullspace_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order: f64 = power;
    // Gate on the spectral power the kernel actually evaluates: the scale-free
    // native Gram uses the literal fractional `power`, but the hybrid
    // (`length_scale=Some`) partial-fraction kernel reads `s` back through
    // `duchon_power_to_usize` (truncating a fractional `power`). Validating the
    // raw fractional power on the hybrid path would desync the `2(p+s) > d`
    // gate from the realized kernel and let the non-finite-at-origin case
    // through (gh#750).
    let validation_power = if length_scale.is_some() {
        duchon_power_to_usize(s_order) as f64
    } else {
        s_order
    };
    validate_duchon_kernel_orders(length_scale, p_order, validation_power, d)?;

    // Translation-invariant polynomial frame (#1375, mirroring the #1269 tp fix).
    // The Duchon kernel reads only coordinate *differences* `data − centers`, so
    // the `K·Z` block is already invariant to a covariate translation `x → x + b`.
    // The polynomial null-space block `P = {1, x, x², …}` (appended as explicit
    // unpenalized design columns) and the side-condition `P(centers)ᵀα = 0` that
    // defines `Z`, however, are assembled at the *absolute* coordinate. With a
    // large covariate mean the `{1, x}` columns become near-collinear, the design
    // ill-conditions, and REML λ-selection lands in a slightly different basin —
    // moving the fit even though `{1, x − x̄}` spans the same model space. Subtract
    // the CENTER-CLOUD per-axis mean from both `data` and `centers` before every
    // polynomial / side-condition assembly so the polynomial frame is
    // location-standardized. The mean is a fixed property of the frozen
    // (`UserProvided`) centers — recomputed identically at predict — and under
    // `x → x + b` the centers (selected from the data) shift by the same `b`, so
    // the centred coordinate, hence the whole basis, is invariant.
    let center_mean: Vec<f64> = (0..d)
        .map(|c| centers.column(c).sum() / (k.max(1) as f64))
        .collect();
    let mut data_centered = data.to_owned();
    for c in 0..d {
        let mu = center_mean[c];
        data_centered.column_mut(c).mapv_inplace(|v| v - mu);
    }

    let poly_block = polynomial_block_from_order(data_centered.view(), nullspace_order);
    // Z spans null(Q^T), where Q contains polynomial side conditions at centers.
    // Reparameterizing alpha = Z gamma enforces conditional-PD constraints once
    // and yields free-parameter penalty gamma^T (Z^T K_CC Z) gamma.
    // `kernel_constraint_nullspace` centers `centers` by the same center-cloud
    // mean internally (#1375), so the side-condition factorisation matches the
    // centered polynomial design columns above and is translation-stable; this is
    // the SAME `Z` the penalty path assembles, keeping design and penalty
    // consistent.
    let z_raw = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;
    // #1355: fold the frozen data-metric radial reparameterization `V` into the
    // constrained kernel transform (`Z' = Z·V`) so the realized design columns
    // `K·Z·V` rotate into the `G_c`-orthonormal generalized eigenbasis. Applied
    // here identically to the penalty assembly keeps design and penalty
    // bit-consistent at fit, predict, and κ-trial time.
    let z = if let Some(v) = radial_reparam {
        if v.nrows() != z_raw.ncols() {
            crate::bail_dim_basis!(
                "Duchon radial reparam shape {:?} does not match constrained kernel dimension {}",
                v.dim(),
                z_raw.ncols()
            );
        }
        fast_ab(&z_raw, v)
    } else {
        z_raw
    };

    let coeffs = length_scale.map(|ls| {
        duchon_partial_fraction_coeffs(
            p_order,
            duchon_power_to_usize(s_order),
            1.0 / ls.max(1e-300),
        )
    });

    // Practical safe operating range (document Eq. D.2):
    //   κ in [1e-2 / r_max, 1e2 / r_min]
    // where r_min/r_max are pairwise center distance extrema. Under
    // anisotropy the kernel metric is y-space (y_a = exp(η_a) x_a), so
    // the relevant r_min/r_max are y-space pairwise distances, not raw.
    // We keep user-provided κ but emit a warning outside this regime.
    let warn_bounds = match (length_scale, aniso_log_scales) {
        (Some(_), Some(eta)) => {
            let y_centers = points_in_aniso_y_space(centers, eta);
            pairwise_distance_bounds(y_centers.view())
        }
        (Some(_), None) => pairwise_distance_bounds(centers),
        (None, _) => None,
    };
    if let (Some(length_scale), Some((r_min, r_max))) = (length_scale, warn_bounds) {
        let kappa = 1.0 / length_scale.max(1e-300);
        let kappa_lo = 1e-2 / r_max;
        let kappa_hi = 1e2 / r_min;
        if kappa < kappa_lo || kappa > kappa_hi {
            log::debug!(
                "Duchon κ={} is outside recommended range [{}, {}] derived from centers (r_min={}, r_max={}); numerical conditioning may degrade",
                kappa,
                kappa_lo,
                kappa_hi,
                r_min,
                r_max
            );
        }
    }

    let kernel_cols = z.ncols();
    let poly_cols = poly_block.ncols();
    let total_cols = kernel_cols + poly_cols;

    // Pre-compute polyharmonic coefficient for the pure Duchon case (no length_scale).
    // This avoids 2 gamma_lanczos calls per kernel evaluation (n × k total).
    let pure_poly_coeff = if length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            (pure_duchon_block_order(p_order, s_order)) as f64,
            d,
        ))
    } else {
        None
    };

    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        length_scale,
        p_order,
        duchon_power_to_usize(s_order),
        d,
        aniso_log_scales,
        coeffs.as_ref(),
        pure_poly_coeff.as_ref(),
    );
    // Certified radial value profile for the hybrid path (#979): one exact
    // hybrid-Duchon kernel value costs microseconds across its
    // partial-fraction blocks, and this n·k materialization loop runs on
    // every design rebuild of every κ-trial. For large sweeps, profile φ
    // once over the observed radius range (distance-only pre-pass) and
    // answer per-pair queries by Clenshaw; out-of-range radii and
    // uncertified builds fall back to the exact evaluator (the profile's
    // exact fallback IS `duchon_radial_jets`, whose value channel is the
    // same `duchon_matern_kernel_general_from_distance` evaluated below).
    let hybrid_kind = match (length_scale, coeffs.as_ref()) {
        (Some(ls), Some(c)) if pure_poly_coeff.is_none() => Some(RadialScalarKind::Duchon {
            length_scale: ls,
            p_order,
            s_order: duchon_power_to_usize(s_order),
            dim: d,
            coeffs: c.clone(),
        }),
        _ => None,
    };
    let value_profile = hybrid_kind.as_ref().and_then(|kind| {
        if n.saturating_mul(k) < RADIAL_PROFILE_MIN_PAIRS {
            return None;
        }
        let (r_lo, r_hi) = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut lo = f64::INFINITY;
                let mut hi = 0.0_f64;
                for j in 0..k {
                    let r = if let Some(scales) = axis_scales.as_deref() {
                        aniso_distance_rows_with_scales(data, i, centers, j, scales)
                    } else {
                        euclidean_distance_rows(data, i, centers, j)
                    };
                    if r > 0.0 {
                        lo = lo.min(r);
                        hi = hi.max(r);
                    }
                }
                (lo, hi)
            })
            .reduce(
                || (f64::INFINITY, 0.0_f64),
                |a, b| (a.0.min(b.0), a.1.max(b.1)),
            );
        if r_lo.is_finite() && r_hi > r_lo {
            radial_profile::RadialProfile::build(kind, r_lo, r_hi)
        } else {
            None
        }
    });
    let mut basis = Array2::<f64>::zeros((n, total_cols));
    // Process rows in chunks to amortize thread-local allocation across many rows.
    // Use larger chunks (1024) for better cache utilization at large scale.
    let chunk_size = 1024.min(n);
    let basis_result: Result<(), BasisError> = basis
        .axis_chunks_iter_mut(Axis(0), chunk_size)
        .into_par_iter()
        .enumerate()
        .try_for_each(|(ci, mut chunk)| {
            let mut kernel_row = vec![0.0; k];
            let chunk_start = ci * chunk_size;
            for local_i in 0..chunk.nrows() {
                let i = chunk_start + local_i;
                for j in 0..k {
                    let r = if let Some(scales) = axis_scales.as_deref() {
                        aniso_distance_rows_with_scales(data, i, centers, j, scales)
                    } else {
                        euclidean_distance_rows(data, i, centers, j)
                    };
                    let raw = if let Some(ref ppc) = pure_poly_coeff {
                        // Pure Duchon: use precomputed coefficient, skip gamma calls.
                        ppc.eval(r)
                    } else if let (Some(profile), Some(kind)) =
                        (value_profile.as_ref(), hybrid_kind.as_ref())
                    {
                        profile.eval_or_exact(kind, r)?.0
                    } else {
                        duchon_matern_kernel_general_from_distance(
                            r,
                            length_scale,
                            p_order,
                            duchon_power_to_usize(s_order),
                            d,
                            coeffs.as_ref(),
                        )?
                    };
                    kernel_row[j] = raw * kernel_amp;
                }
                // Write basis row = kernel_row^T × Z using scatter-accumulate
                // pattern: for each knot j with nonzero kernel, add its
                // contribution to all columns at once. This is more cache-
                // friendly than the column-by-column gather pattern since
                // Z rows are contiguous in memory.
                let mut row = chunk.row_mut(local_i);
                row.slice_mut(s![..kernel_cols]).fill(0.0);
                for j in 0..k {
                    let kv = kernel_row[j];
                    if kv != 0.0 {
                        let z_row = z.row(j);
                        for col in 0..kernel_cols {
                            row[col] += kv * z_row[col];
                        }
                    }
                }
            }
            Ok(())
        });
    basis_result?;
    if poly_cols > 0 {
        basis.slice_mut(s![.., kernel_cols..]).assign(&poly_block);
    }

    Ok(DuchonBasisDesign { basis })
}

pub(crate) fn build_cyclic_duchon_basis_1dwithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    start: f64,
    end: f64,
) -> Result<BasisBuildResult, BasisError> {
    if data.ncols() != 1 {
        crate::bail_invalid_basis!("cyclic Duchon smooths currently require exactly one covariate");
    }
    if end <= start {
        return Err(BasisError::InvalidRange(start, end));
    }
    let centers = select_centers_by_strategy(data, &spec.center_strategy)?;
    if centers.ncols() != 1 {
        crate::bail_dim_basis!(
            "cyclic Duchon centers must have one column, got {}",
            centers.ncols()
        );
    }
    let k = centers.nrows();
    let s_order_usize = spec.power_as_usize();
    if k <= s_order_usize.max(1) {
        crate::bail_invalid_basis!(
            "cyclic Duchon basis requires more centers ({k}) than power ({})",
            spec.power
        );
    }
    let period = end - start;
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Zero);
    // Hybrid kernel evaluates the truncated integer `s` (`power_as_usize`);
    // scale-free uses the literal fractional power. Gate on the realized value.
    let validation_power = if spec.length_scale.is_some() {
        s_order_usize as f64
    } else {
        spec.power
    };
    validate_duchon_kernel_orders(spec.length_scale, p_order, validation_power, 1)?;
    let coeffs = spec
        .length_scale
        .map(|ls| duchon_partial_fraction_coeffs(p_order, s_order_usize, 1.0 / ls.max(1e-300)));
    let pure_poly_coeff = if spec.length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            pure_duchon_block_order(p_order, spec.power),
            1,
        ))
    } else {
        None
    };
    let kernel_amp = duchon_kernel_amplification(
        centers.view(),
        spec.length_scale,
        p_order,
        duchon_power_to_usize(spec.power),
        1,
        None,
        coeffs.as_ref(),
        pure_poly_coeff.as_ref(),
    );
    let mut basis = Array2::<f64>::zeros((data.nrows(), k + 1));
    for i in 0..data.nrows() {
        let x = wrap_to_period(data[[i, 0]], start, period);
        for j in 0..k {
            let c = wrap_to_period(centers[[j, 0]], start, period);
            let r = cyclic_distance_1d(x, c, period);
            let raw = if let Some(ref ppc) = pure_poly_coeff {
                ppc.eval(r)
            } else {
                duchon_matern_kernel_general_from_distance(
                    r,
                    spec.length_scale,
                    p_order,
                    s_order_usize,
                    1,
                    coeffs.as_ref(),
                )?
            };
            basis[[i, j]] = raw * kernel_amp;
        }
        basis[[i, k]] = 1.0;
    }

    let identifiability_transform = spatial_identifiability_transform_from_design(
        data,
        basis.view(),
        &spec.identifiability,
        "cyclic Duchon",
    )?;
    let (design_matrix, transform_for_penalty) = if let Some(z) = identifiability_transform.as_ref()
    {
        (fast_ab(&basis, z), Some(z))
    } else {
        (basis, None)
    };

    let mut s_kernel = Array2::<f64>::zeros((k + 1, k + 1));
    let s_cyclic = create_cyclic_difference_penalty_matrix(k, s_order_usize.max(1).min(k - 1))?;
    s_kernel.slice_mut(s![..k, ..k]).assign(&s_cyclic);
    let s_final = if let Some(z) = transform_for_penalty {
        fast_ab(&fast_atb(z, &s_kernel), z)
    } else {
        s_kernel
    };
    // Frobenius-normalize the cyclic roughness penalty so its smoothing
    // parameter shares the unit-Frobenius scale of every other basis (cr /
    // duchon / tensor / open-and-cyclic ps, #1365); a raw operator (scale 1.0)
    // would put `λ` on a basis-dependent scale and miscalibrate the outer
    // λ-search. Fit-invariant at the REML optimum (only `λ̂` rescales by `c`).
    let (s_final_norm, s_final_scale) = normalize_penalty(&s_final);
    let candidates = vec![PenaltyCandidate {
        matrix: s_final_norm,
        nullspace_dim_hint: 1,
        source: PenaltySource::Primary,
        normalization_scale: s_final_scale,
        kronecker_factors: None,
        op: None,
    }];
    let (penalties, nullspace_dims, penaltyinfo, null_eigenvectors, ops) =
        filter_active_penalty_candidates_with_ops(candidates)?;
    Ok(BasisBuildResult {
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design_matrix)),
        penalties,
        nullspace_dims,
        penaltyinfo,
        metadata: BasisMetadata::Duchon {
            centers,
            length_scale: spec.length_scale,
            periodic: Some(vec![Some(period)]),
            power: spec.power,
            nullspace_order: DuchonNullspaceOrder::Zero,
            identifiability_transform,
            input_scales: None,
            aniso_log_scales: None,
            operator_collocation_points: None,
            radial_reparam: None,
        },
        kronecker_factored: None,
        ops,
        null_eigenvectors,
        joint_null_rotation: None,
    })
}

/// Generic Duchon builder returning design + penalty list.
pub fn build_duchon_basis(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) -> Result<BasisBuildResult, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_basiswithworkspace(data, spec, &mut workspace)
}

pub fn create_duchon_basis_1d_derivative_dense(
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    periodic: bool,
    period: Option<f64>,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    if order > 2 {
        crate::bail_invalid_basis!(
            "Duchon basis derivative supports orders 0, 1, and 2; got order={order}"
        );
    }
    if t.is_empty() || centers.is_empty() {
        crate::bail_invalid_basis!("Duchon basis derivative requires non-empty t and centers");
    }
    if t.iter().any(|v| !v.is_finite()) || centers.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("Duchon basis derivative requires finite t and center values");
    }
    if !periodic && period.is_some() {
        crate::bail_invalid_basis!(
            "Duchon basis derivative period is only valid when periodic=true"
        );
    }

    let data = t.to_owned().insert_axis(Axis(1));
    let center_matrix = centers.to_owned().insert_axis(Axis(1));
    let mut workspace = BasisWorkspace::default();
    // The user-requested Duchon order ``m`` is encoded in ``nullspace_order``;
    // the PERIODIC kernel is the Bernoulli Green's function of ``(d²/dx²)^m``
    // (PSD on the circle, gam#580) so it needs the original ``m`` even though
    // the periodic *constraint* nullspace is forced to constants only.
    let user_m = duchon_p_from_nullspace_order(nullspace_order);
    let effective_order = if periodic {
        DuchonNullspaceOrder::Zero
    } else {
        duchon_effective_nullspace_order(center_matrix.view(), nullspace_order)
    };
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order = duchon_power_to_usize(power);
    validate_duchon_kernel_orders(None, p_order, s_order as f64, 1)?;

    if periodic {
        // Periodic case: mirror the forward Bernoulli Green's-function design
        // (`build_periodic_duchon_basis_1d`) EXACTLY — same collapsed centers,
        // same domain-wrap period, same constant-only constraint nullspace —
        // so the analytic derivative is the true ∂/∂t of the forward design
        // (gam#580). Using the polyharmonic triangle-wave kernel here (the old
        // path) was inconsistent with the Bernoulli forward and silently wrong.
        let (collapsed_centers, left, resolved_period) =
            prepare_periodic_duchon_centers_1d_with_period(center_matrix, period)?;
        let z = kernel_constraint_nullspace(
            collapsed_centers.view(),
            effective_order,
            &mut workspace.cache,
        )?;
        let kernel_cols = z.ncols();
        let k_centers = collapsed_centers.nrows();
        let centers_col0: Vec<f64> = collapsed_centers.column(0).to_vec();
        let mut raw_kernel = Array2::<f64>::zeros((t.len(), k_centers));
        for i in 0..t.len() {
            let x = wrap_to_period(t[i], left, resolved_period);
            for j in 0..k_centers {
                // Signed offset reduced to [−period/2, period/2]; r = |offset|.
                let mut delta = (x - centers_col0[j]).rem_euclid(resolved_period);
                if delta > 0.5 * resolved_period {
                    delta -= resolved_period;
                }
                let r = delta.abs();
                let sign = if delta > 0.0 {
                    1.0
                } else if delta < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let (phi, phi_r, phi_rr) =
                    periodic_duchon_kernel_bernoulli_triplet(r, user_m, resolved_period)?;
                raw_kernel[[i, j]] = match order {
                    0 => phi,
                    1 => phi_r * sign,
                    2 => phi_rr,
                    other => {
                        crate::bail_invalid_basis!(
                            "Duchon basis derivative supports orders 0, 1, and 2; got order={other}"
                        );
                    }
                };
            }
        }
        // Forward design appends a single constant column; its t-derivative is
        // zero (order ≥ 1) or one (order 0). Match that layout exactly.
        let mut basis = Array2::<f64>::zeros((t.len(), kernel_cols + 1));
        let design_kernel = fast_ab(&raw_kernel, &z);
        basis
            .slice_mut(s![.., 0..kernel_cols])
            .assign(&design_kernel);
        if order == 0 {
            basis.column_mut(kernel_cols).fill(1.0);
        }
        return Ok(basis);
    }

    let z =
        kernel_constraint_nullspace(center_matrix.view(), effective_order, &mut workspace.cache)?;
    let kernel_cols = z.ncols();
    let poly_cols = polynomial_block_from_order(data.view(), effective_order).ncols();

    let pure_coeff =
        PolyharmonicBlockCoeff::new((pure_duchon_block_order(p_order, s_order as f64)) as f64, 1);
    let kernel_amp = duchon_kernel_amplification(
        center_matrix.view(),
        None,
        p_order,
        s_order,
        1,
        None,
        None,
        Some(&pure_coeff),
    );

    let mut raw_kernel = Array2::<f64>::zeros((t.len(), centers.len()));
    for i in 0..t.len() {
        let x = t[i];
        for j in 0..centers.len() {
            let delta = x - centers[j];
            let r = delta.abs();
            let sign = if delta > 0.0 {
                1.0
            } else if delta < 0.0 {
                -1.0
            } else {
                0.0
            };
            let (phi, phi_r, phi_rr) =
                duchon_kernel_radial_triplet(r, None, p_order, s_order as f64, 1, None)?;
            raw_kernel[[i, j]] = match order {
                0 => phi,
                1 => phi_r * sign,
                2 => phi_rr,
                other => {
                    crate::bail_invalid_basis!(
                        "Duchon basis derivative supports orders 0, 1, and 2; got order={other}"
                    );
                }
            } * kernel_amp;
        }
    }

    let mut basis = Array2::<f64>::zeros((t.len(), kernel_cols + poly_cols));
    let design_kernel = fast_ab(&raw_kernel, &z);
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&design_kernel);
    fill_duchon_1d_polynomial_derivative(&mut basis, kernel_cols, t, effective_order, order);
    Ok(basis)
}
