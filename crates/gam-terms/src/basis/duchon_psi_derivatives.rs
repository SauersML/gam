use super::*;

/// Process-wide counter of full `n×k` Duchon kernel-design materializations
/// performed by [`build_duchon_basis_designwithworkspace`]. Each increment is a
/// full kernel evaluation over every (data-row, center) pair — the dominant
/// cold-build cost. It exists so regression tests can pin STRUCTURALLY that the
/// default `duchon(x, z)` cold build materializes the design ONCE, not twice
/// (the #1718 redundant second kernel pass); it never affects the numeric
/// result.
pub(crate) static DUCHON_DESIGN_BUILD_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

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

/// Exact `psi = log(kappa)` value jet of the low-dimensional hybrid Duchon
/// representative that [`duchon_matern_kernel_general_from_distance`] ships.
///
/// The usual scaling identity
///
/// `K_psi = delta K + r K_r`, `delta = d - 2(p+s)`,
///
/// is exact for the cancellation-free integral representation (`2p < d`).
/// On the complementary partial-fraction route, however, the Riesz blocks in
/// even dimension contain `r^(2m-d) log(r)`.  Rescaling such a block adds a
/// polynomial in `r`; the conditionally-positive-definite kernel is equivalent
/// as a function space, but its *shipped coefficient representative* is not
/// strictly homogeneous.  Differentiating the scaling identity there therefore
/// differentiates a different design (gam#979's 2-D identity-chart control).
///
/// This routine differentiates the actual partial-fraction sum instead.  Riesz
/// blocks move only through their coefficient powers.  For a Matérn block
/// `M_n = F^-1[(kappa^2 + |omega|^2)^-n]`, differentiation under the spectrum
/// gives the exact recurrence
///
/// `M_n,psi = -2 n kappa^2 M_(n+1)`.
///
/// The second derivative follows by applying the same identity again.  At the
/// collision we use the same analytic Taylor coefficients as the forward value,
/// so divergent partial-fraction terms are never evaluated separately.
pub(crate) fn duchon_partial_fraction_kernel_psi_triplet(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<(f64, f64, f64), BasisError> {
    assert!(
        !duchon_hybrid_stable_integral_applies(p_order, s_order, k_dim),
        "partial-fraction psi jet called for the stable-integral Duchon route"
    );
    let smoothness_order = 2 * (p_order + s_order);
    // Refuse a length scale that is not finite and positive before either
    // branch, so the collision radius is proportional to the validated scale.
    let kappa = duchon_inverse_length_scale(length_scale, "Duchon partial-fraction ψ-triplet")?;
    let collision_taylor_radius = DUCHON_COLLISION_TAYLOR_REL * length_scale;
    if r <= collision_taylor_radius && smoothness_order > k_dim {
        let r2 = r * r;
        let mut value = 0.0_f64;
        let mut first = 0.0_f64;
        let mut second = 0.0_f64;
        let mut r_power = 1.0_f64;
        for j in 0..=3 {
            if smoothness_order <= k_dim + 2 * j {
                break;
            }
            let (derivative, derivative_first, derivative_second) =
                duchon_phi_even_derivative_collision_psi_triplet(
                    length_scale,
                    p_order,
                    s_order,
                    k_dim,
                    coeffs,
                    j,
                )?;
            let factorial = gamma_lanczos((2 * j + 1) as f64);
            let weight = r_power / factorial;
            value += weight * derivative;
            first += weight * derivative_first;
            second += weight * derivative_second;
            r_power *= r2;
        }
        return Ok((value, first, second));
    }

    let kappa2 = kappa * kappa;
    let mut value = CompensatedSum::default();
    let mut first = CompensatedSum::default();
    let mut second = CompensatedSum::default();
    for (m, &coefficient) in coeffs.a.iter().enumerate().skip(1) {
        if coefficient == 0.0 {
            continue;
        }
        let block = polyharmonic_kernel(r, m as f64, k_dim);
        let exponent = duchon_coeff_exponents(p_order, s_order, m);
        value.add(coefficient * block);
        first.add(exponent * coefficient * block);
        second.add(exponent * exponent * coefficient * block);
    }
    for (n, &coefficient) in coeffs.b.iter().enumerate().skip(1) {
        if coefficient == 0.0 {
            continue;
        }
        let block = duchon_matern_block(r, kappa, n, k_dim)?;
        let next = duchon_matern_block(r, kappa, n + 1, k_dim)?;
        let next_next = duchon_matern_block(r, kappa, n + 2, k_dim)?;
        let block_first = -2.0 * n as f64 * kappa2 * next;
        let block_second = -4.0 * n as f64 * kappa2 * next
            + 4.0 * n as f64 * (n + 1) as f64 * kappa2 * kappa2 * next_next;
        let exponent = duchon_coeff_exponents(p_order, s_order, n);
        value.add(coefficient * block);
        first.add(coefficient * (exponent * block + block_first));
        second.add(
            coefficient
                * (exponent * exponent * block + 2.0 * exponent * block_first + block_second),
        );
    }
    let triplet = (value.value(), first.value(), second.value());
    if !(triplet.0.is_finite() && triplet.1.is_finite() && triplet.2.is_finite()) {
        crate::bail_invalid_basis!(
            "non-finite Duchon partial-fraction psi jet at r={r}, length_scale={length_scale}, p={p_order}, s={s_order}, dim={k_dim}"
        );
    }
    Ok(triplet)
}

/// The outer coordinate a Duchon ψ-derivative differentiates (gam#2735).
///
/// The anisotropic Duchon metric is `u² = Σ_a exp(2 ψ_a) h_a²`, and the ψ
/// coordinates the outer REML solve owns are the **raw** `ψ_a`: each one
/// decodes simultaneously into the global scale `κ = exp(mean ψ)` and the
/// centered contrast `η_a = ψ_a − mean ψ`, so
///
/// ```text
///     ∂ log κ / ∂ψ_a = 1/d           ∂η_b / ∂ψ_a = δ_ab − 1/d
/// ```
///
/// `Global` is the all-ones direction of that frame — moving every `ψ_a` by the
/// same amount leaves every contrast fixed and multiplies `κ`. That is not a
/// convention, it is an identity, and it is what
/// `duchon_axis_log_kappa_derivatives` reproduces by construction:
/// summing its first derivative over `a`, and its second over `(a, b)`, gives
/// `scaled_log_kappa_derivatives` exactly. The isotropic route is therefore a
/// contraction of the anisotropic one rather than a parallel derivation that
/// could drift from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DuchonPsiDirection {
    /// `ψ = log κ`. Every contrast — and therefore every metric weight
    /// `w_a = exp(2η_a)` appearing explicitly in the operator-penalty blocks —
    /// is constant along this direction.
    Global,
    /// The raw per-axis coordinate `ψ_a`.
    Axis(usize),
}

/// Per-axis ψ derivatives of a radial scalar `F(r; κ) = κ^E G(κ r)`.
///
/// `axis_share[a] = s_a / r²` where `s_a = exp(2 η_a) h_a²` is the per-axis
/// weighted squared displacement produced by `aniso_distance_and_components`;
/// the shares are non-negative and sum to one. Writing
///
/// ```text
///     A = r F_r                 B = r² F_rr − r F_r                 c = E/d
/// ```
///
/// the exact chain rule through `(κ, η)` collapses to
///
/// ```text
///     ∂F/∂ψ_a        = c F + A σ_a
///     ∂²F/∂ψ_a∂ψ_b   = B σ_a σ_b + c A (σ_a + σ_b) + 2 A σ_a δ_ab + c² F
/// ```
///
/// `A` and `B` are exactly the two combinations `scaled_log_kappa_derivatives`
/// already forms, so the per-axis jet needs no radial quantity the isotropic
/// jet does not, and — crucially — it is finite at collision: `σ` is bounded by
/// 1 and both `A` and `B` vanish with `r`, so no `1/r` ever appears.
///
/// Contracting over the all-ones direction returns the isotropic jet:
/// `Σ_a first = E F + r F_r` and `Σ_{a,b} second = E² F + (2E+1) r F_r + r² F_rr`.
#[inline(always)]
pub(crate) fn duchon_axis_log_kappa_derivatives(
    value: f64,
    radial_first: f64,
    radialsecond: f64,
    exponent: f64,
    r: f64,
    dim: usize,
    axis_share: f64,
    second_axis_share: f64,
    same_axis: bool,
) -> (f64, f64) {
    let a_term = r * radial_first;
    let b_term = r * r * radialsecond - a_term;
    let c = exponent / dim.max(1) as f64;
    let first = c * value + a_term * axis_share;
    let second = b_term * axis_share * second_axis_share
        + c * a_term * (axis_share + second_axis_share)
        + if same_axis {
            2.0 * a_term * axis_share
        } else {
            0.0
        }
        + c * c * value;
    (first, second)
}

/// First and second ψ derivatives of a radial scalar along `direction`.
///
/// The single dispatch point between the isotropic and per-axis routes: every
/// Duchon penalty assembly consumes this and nothing else, so a route can only
/// differ from another by its `direction`.
#[inline(always)]
pub(crate) fn duchon_direction_derivatives(
    direction: DuchonPsiDirection,
    value: f64,
    radial_first: f64,
    radialsecond: f64,
    exponent: f64,
    r: f64,
    dim: usize,
    axis_shares: &[f64],
) -> (f64, f64) {
    match direction {
        DuchonPsiDirection::Global => {
            scaled_log_kappa_derivatives(value, radial_first, radialsecond, exponent, r)
        }
        DuchonPsiDirection::Axis(a) => {
            let share = axis_shares.get(a).copied().unwrap_or(0.0);
            duchon_axis_log_kappa_derivatives(
                value,
                radial_first,
                radialsecond,
                exponent,
                r,
                dim,
                share,
                share,
                true,
            )
        }
    }
}

/// Normalized per-axis shares `σ_a = s_a / r²` of the anisotropic squared
/// distance, with the symmetric convention `σ_a = 1/d` at collision.
///
/// At `r = 0` both `A = r F_r` and `B = r² F_rr − r F_r` vanish for every
/// radial scalar the Duchon jets produce, so the share is multiplied by zero
/// and the convention only has to keep `Σ_a σ_a = 1` — which is what makes the
/// isotropic contraction identity hold at collision too.
#[inline(always)]
pub(crate) fn duchon_axis_shares(components: &[f64], r: f64) -> Vec<f64> {
    let d = components.len().max(1);
    let r2 = r * r;
    if !(r2 > 0.0) || !r2.is_finite() {
        return vec![1.0 / d as f64; components.len()];
    }
    components.iter().map(|&s| s / r2).collect()
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
    let mut q_sum = CompensatedSum::default();
    let mut t_sum = CompensatedSum::default();
    let mut t_r_sum = CompensatedSum::default();
    let mut t_rr_sum = CompensatedSum::default();

    for (m, coeff) in coeffs.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        let (q, t, t_r, t_rr) = duchon_polyharmonic_operator_block_jets(r_eval, m, k_dim)?;
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
        q: q_sum.value(),
        t: t_sum.value(),
        t_r: t_r_sum.value(),
        t_rr: t_rr_sum.value(),
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
    let kappa = duchon_inverse_length_scale(length_scale, "Duchon radial jets")?;
    // `length_scale` is finite and positive (the owner above refused otherwise),
    // so the collision radii are fractions of the real scale, not of a floored
    // one that would have substituted `1e-8` for a genuinely small length scale.
    let r_floor = DUCHON_DERIVATIVE_R_FLOOR_REL * length_scale;
    let collision_taylor_radius = DUCHON_COLLISION_TAYLOR_REL * length_scale;
    let r_eval = r.max(r_floor);
    let d = k_dim as f64;

    // The value and the operator read the same shape at the same κ, so the
    // certified profile and the κ-fixed prefactor are bound once for both
    // rather than resolved twice at every pair of every ψ sweep.
    let hybrid = duchon_hybrid_evaluator(Some(length_scale), p_order, s_order, k_dim)?;
    // Value path keeps the intrinsic diagonal convention used by the actual basis.
    let phi = match hybrid.as_ref() {
        Some(hybrid) => hybrid.value(r)?,
        None => duchon_matern_kernel_general_from_distance(
            r,
            Some(length_scale),
            p_order,
            s_order,
            k_dim,
            Some(coeffs),
        )?,
    };
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
    let operator_core = match hybrid.as_ref() {
        Some(hybrid) => hybrid.operator_core(r_eval)?,
        None => duchon_regularized_operator_core(r_eval, kappa, k_dim, coeffs)?,
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

/// The scalar core's radial jet, before any ψ direction has been chosen.
///
/// **Duchon spectral derivation.** Start from the isotropic spectrum
/// `K^(ω; κ) ∝ 1 / (|ω|^{2p} (κ² + |ω|²)^s)`, with fixed integer orders `p, s`
/// and continuous scale `ψ = log κ`, `κ = 1/length_scale`. Rescaling frequency
/// by `ω = κ ξ` gives the full spatial kernel scaling law
///
/// ```text
///     φ(r; κ) = κ^δ H(κ r),      δ = d − 2p − 2s
/// ```
///
/// so every radial scalar this file forms is `κ^E G(κ r)` for some exponent
/// `E`: `φ` at `δ`, `q = φ_r/r` and `Δφ` at `δ + 2`, `t = q_r/r` at `δ + 4`.
/// `scaled_log_kappa_derivatives` contracts that along the isotropic
/// direction; `duchon_axis_log_kappa_derivatives` contracts it per axis, and
/// summing the latter reproduces the former exactly. Splitting the value jet
/// from the contraction makes the DIRECTION the only thing that differs
/// between the two routes, so a global and a per-axis derivative can never be
/// taken of two different kernels.
///
/// Once `{φ, q, Δφ}` and their ψ derivatives are known the collocation
/// operators follow exactly — `D0[k,j] = φ(r)`, `D1[(k,a),j] = q(r)·h_a`,
/// `D2[k,j] = Δφ(r)` — and the penalty Hessians come from the Gram identities
/// `S_ψ = D_ψᵀD + DᵀD_ψ` and `S_ψψ = D_ψψᵀD + 2D_ψᵀD_ψ + DᵀD_ψψ`.
///
/// **Representation note.** When `p > 0` the Duchon kernel is only
/// conditionally positive definite, so the spatial kernel is canonical only up
/// to polynomial additions. These formulas are tied to the specific
/// representative encoded by the partial-fraction construction and the
/// collision rules; the operator penalties, exact ψ derivatives, and
/// center-collision limits all have to use that same representative or the
/// resulting penalty geometry drifts across code paths. In particular the
/// `r = 0` limit is NOT the naive `(δ+2)·φ_rr` scaling shortcut — in even
/// dimensions the log-Riesz representative carries κ-dependent finite parts at
/// the origin, which is what [`duchonphi_rr_collision_psi_triplet`] exists for.
pub(crate) struct DuchonRadialCoreValueJet {
    pub(crate) value: f64,
    pub(crate) first: f64,
    pub(crate) second: f64,
    pub(crate) exponent: f64,
}

pub(crate) fn duchon_radial_core_value_jet(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<DuchonRadialCoreValueJet, BasisError> {
    let jets = duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, coeffs)?;
    Ok(DuchonRadialCoreValueJet {
        value: jets.phi,
        first: jets.phi_r,
        second: jets.phi_rr,
        exponent: duchon_scaling_exponent(p_order, s_order, k_dim),
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
    let c = duchon_matern_block_normalization(kappa, n_order, k_dim);
    let nu = n_order as f64 - 0.5 * k_dim as f64;

    if k_dim.is_multiple_of(2) {
        // Integer ν.
        let nu_int = n_order as i64 - (k_dim as i64) / 2;
        duchon_matern_block_taylor_r2j_integer_nu(kappa, c, nu_int, j)
    } else {
        // Half-integer ν.
        duchon_matern_block_taylor_r2j_half_integer_nu(kappa, c, nu, j)
    }
}

/// The normalization constant `c` of one Matérn partial-fraction block
/// `g_n(r) = c · r^ν K_{|ν|}(κ r)`, `ν = n − d/2`.
#[inline(always)]
fn duchon_matern_block_normalization(kappa: f64, n_order: usize, k_dim: usize) -> f64 {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half)
            * 2.0_f64.powf(n - 1.0)
            * factorial_f64(n_order - 1))
}

/// The `r^k` Taylor coefficient (pure and `ln r` parts) of one Matérn
/// partial-fraction block, for ANY integer power `k` — not only the even ones
/// [`duchon_matern_block_taylor_r2j`] exposes.
///
/// A collision derivative reads only even powers, because the odd powers of an
/// isotropic kernel carry no even-order radial derivative at the origin. The
/// null-space-reduced kernel (gam#4558) reads both: in odd `d` the hybrid
/// kernel's NON-analytic sector is exactly its odd powers, and that sector is
/// what survives the constraint projection once the polynomial head is gone.
///
/// In even `d` (integer ν) the expansion is a series in `r²`, so an odd `k` is
/// absent rather than small.
pub(crate) fn duchon_matern_block_taylor_rk(
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    k: usize,
) -> (f64, f64) {
    if k.is_multiple_of(2) {
        return duchon_matern_block_taylor_r2j(kappa, n_order, k_dim, k / 2);
    }
    if k_dim.is_multiple_of(2) {
        return (0.0, 0.0);
    }
    let c = duchon_matern_block_normalization(kappa, n_order, k_dim);
    let nu = n_order as f64 - 0.5 * k_dim as f64;
    duchon_matern_block_taylor_rk_half_integer_nu(kappa, c, nu, k)
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
        / ((2.0 * std::f64::consts::PI).powf(k_half)
            * 2.0_f64.powf(n - 1.0)
            * factorial_f64(n_order - 1));
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
                let coeff = 0.5 * sign * factorial_f64(nu_usize - j - 1)
                    / factorial_f64(j)
                    * 2.0_f64.powi(-power);
                let exponent = c_exp + power as f64;
                let value = c_const * coeff * kappa.powf(exponent);
                add_triplet(&mut pure, psi_power_triplet(value, exponent));
            }

            if j >= nu_usize {
                let k = j - nu_usize;
                let inv_fac = 1.0 / (factorial_f64(k) * factorial_f64(nu_usize + k));
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
            let inv_fac = 1.0 / (factorial_f64(k) * factorial_f64(mu + k));
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
        // |ν| = l + ½ ⇒ l = |ν| − ½. (The earlier `2|ν| − 1` form computed `2l`,
        // not `l`: it is correct only at ν = ½, and for |ν| ≥ 3/2 it selected the
        // K_{2|ν|−½} polynomial instead of K_{|ν|}, collapsing the Taylor
        // coefficients — e.g. the r⁰ diagonal term of the ν = 3/2 block to 0,
        // which broke the d=1 / power≥2 Duchon penalty diagonal — gam#1604.)
        let l = (nu_abs - 0.5).round().max(0.0) as usize;
        let prefactor_const = (std::f64::consts::PI / 2.0).sqrt();
        let prefactor_exp = -0.5;
        let target = 2 * j;

        for i in 0..=l {
            let c_i = gamma_lanczos((l + i + 1) as f64)
                / (gamma_lanczos((i + 1) as f64) * gamma_lanczos((l - i + 1) as f64));
            let p_f64 = nu - 0.5 - i as f64;
            let p_round = p_f64.round() as i64;
            if p_f64 != p_round as f64 {
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
            let coeff = sign * factorial_f64(nu - j - 1) / factorial_f64(j)
                * kappa_half.powi(2 * j as i32 - nu as i32)
                * 0.5;
            pure += coeff;
        }

        // Source 2: regular+log sum at k = j − ν.
        if j >= nu {
            let k = j - nu;
            let inv_fac = 1.0 / (factorial_f64(k) * factorial_f64(nu + k));
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
        let inv_fac = 1.0 / (factorial_f64(k) * factorial_f64(mu + k));
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
    duchon_matern_block_taylor_rk_half_integer_nu(kappa, c, nu, 2 * j)
}

/// [`duchon_matern_block_taylor_r2j_half_integer_nu`] at an arbitrary integer
/// power `k`. For half-integer ν every power of `r` in the expansion is an
/// integer, odd as well as even, so the even-only entry point above is this
/// one at `k = 2j`.
pub(crate) fn duchon_matern_block_taylor_rk_half_integer_nu(
    kappa: f64,
    c: f64,
    nu: f64,
    k: usize,
) -> (f64, f64) {
    let nu_abs = nu.abs();
    // |ν| = l + ½ ⇒ l = |ν| − ½. (The earlier `2|ν| − 1` form computed `2l`,
    // not `l` — see the matching note in `duchon_matern_block_taylor_r2j_triplet`;
    // gam#1604.)
    let l = (nu_abs - 0.5).round().max(0.0) as usize;
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
    // For monomial r^p (p = ν−½−i) times e^{−κr}: the r^k coefficient is
    //   (−κ)^{k−p} / (k−p)!   when k−p is a non-negative integer.
    let target = k;
    let mut pure = 0.0;

    for i in 0..=l {
        let c_i = gamma_lanczos((l + i + 1) as f64)
            / (gamma_lanczos((i + 1) as f64) * gamma_lanczos((l - i + 1) as f64));
        let inv_2kappa_i = (2.0 * kappa).powi(-(i as i32));

        // r-power of this polynomial term.
        let p_f64 = nu - 0.5 - i as f64;
        let p_round = p_f64.round() as i64;
        if p_f64 != p_round as f64 {
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
    duchon_polyharmonic_block_taylor_rk(m, k_dim, 2 * j)
}

/// [`duchon_polyharmonic_block_taylor_r2j`] at an arbitrary integer power `k`.
///
/// A polyharmonic block is the single monomial `c · r^{2m−d}` (times `ln r` in
/// the log case), so it contributes to exactly one power. In odd `d` that
/// power is odd, which is why the even-only entry point above always reports
/// zero there and the null-space-reduced kernel (gam#4558) must ask for the
/// odd powers by name.
pub(crate) fn duchon_polyharmonic_block_taylor_rk(m: usize, k_dim: usize, k: usize) -> (f64, f64) {
    let k_half = 0.5 * k_dim as f64;
    let alpha = 2 * m as i64 - k_dim as i64;

    if alpha != k as i64 {
        return (0.0, 0.0);
    }

    // α = 2j: compute the normalization constant.
    if k_dim.is_multiple_of(2) && m >= k_dim / 2 {
        // Log case: Φ_m = c · r^α · ln(r).
        let c = polyharmonic_log_sign(m, k_dim)
            / (2.0_f64.powi((2 * m - 1) as i32)
                * std::f64::consts::PI.powf(k_half)
                * factorial_f64(m - 1)
                * factorial_f64(m - k_dim / 2));
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

/// Roundings that form one polyharmonic block's `ln r` summand `a_m · c_m` of
/// the `r^{2j}` Taylor coefficient.
///
/// The `ln r` parts cancel across the blocks to a real zero for every `κ` (the
/// partial-fraction identity), so a constant every summand shares, `π` or `κ`
/// as represented, cancels with them; only the operations that form each
/// summand are left in the residue:
/// - `a_m = ±κ^e·C`: one `powf` and one product (the binomial is exact);
/// - `c_m = ±1/(2^{2m−1}·π^{d/2}·(m−1)!·(m−d/2)!)`: the power of two is exact,
///   one `powf`, the two factorials, three products and one quotient;
/// - `a_m·c_m`: one product.
pub(crate) fn polyharmonic_log_summand_roundings(m: usize, k_dim: usize) -> usize {
    2 + 1 + factorial_roundings(m - 1) + factorial_roundings(m - k_dim / 2) + 3 + 1 + 1
}

/// Roundings that form one integer-`ν` Matérn block's `ln r` summand
/// `b_n · c · ℓ` of the `r^{2j}` Taylor coefficient (see
/// [`polyharmonic_log_summand_roundings`] for why shared constants do not count):
/// - `b_n = ±κ^e·C`: one `powf` and one product;
/// - `c = κ^{d/2−n}/((2π)^{d/2}·2^{n−1}·(n−1)!)`: two `powf` (doubling `π` is
///   exact), one for the power of two, the factorial, two products and one
///   quotient;
/// - `ℓ = ∓(κ/2)^e/(k!(|ν|+k)!)`: `powi` by repeated squaring, at most two
///   products per bit of `e`; the two factorials, a product and a reciprocal;
///   one product;
/// - `c·ℓ` and `b_n·(c·ℓ)`: two products.
pub(crate) fn matern_log_summand_roundings(n_order: usize, k_dim: usize, j: usize) -> usize {
    let nu = n_order as i64 - (k_dim as i64) / 2;
    let mu = nu.unsigned_abs() as usize;
    let (k, exponent) = if nu >= 0 {
        let k = j.saturating_sub(mu);
        (k, 2 * k + mu)
    } else {
        (j, mu + 2 * j)
    };
    let powi = 2 * (usize::BITS - exponent.leading_zeros()) as usize;
    2 + (3 + factorial_roundings(n_order - 1) + 3)
        + powi
        + (factorial_roundings(k) + factorial_roundings(mu + k) + 2)
        + 1
        + 2
}

/// Roundings that form one integer-`ν` Matérn block's `ln r` summands of the
/// ψ-triplet collision coefficient ([`duchon_matern_block_taylor_r2j_triplet`]),
/// up to and including their combination into the three sums:
/// - `b_n`: one `powf` and one product;
/// - `c = 1/((2π)^{d/2}·2^{n−1}·(n−1)!)`: two `powf`, the factorial, two
///   products and one quotient;
/// - `c·κ^e·2^{−e'}`: one `powf` and one product (the power of two is exact);
/// - `1/(k!(|ν|+k)!)`: the two factorials, a product and a reciprocal;
/// - the log base `∓(…)·(…)`: one product, and the triplet's `e²·v`: two;
/// - the combination `β²·b_n·ℓ₀ + 2β·b_n·ℓ₁ + b_n·ℓ₂`: three products and two
///   additions.
pub(crate) fn matern_log_triplet_summand_roundings(
    n_order: usize,
    k_dim: usize,
    j: usize,
) -> usize {
    let nu = n_order as i64 - (k_dim as i64) / 2;
    let mu = nu.unsigned_abs() as usize;
    let k = if nu >= 0 { j.saturating_sub(mu) } else { j };
    2 + (3 + factorial_roundings(n_order - 1) + 2)
        + 2
        + (factorial_roundings(k) + factorial_roundings(mu + k) + 2)
        + 1
        + 2
        + 5
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
        // This path needs the (2j)-order radial-kernel derivative at the
        // origin, finite only when 2(p+s) > k_dim + 2j.
        return Err(BasisError::duchon_smoothness_insufficient(
            format!("collision derivative phi^({})", 2 * j),
            2 * j,
            k_dim,
            p_order,
            s_order as f64,
        ));
    }

    // Analytic path: extract per-block Taylor r^{2j} coefficients and sum.
    let kappa = duchon_inverse_length_scale(length_scale, "Duchon even-derivative collision")?;
    let mut total_pure = CompensatedSum::default();
    let mut total_log = CompensatedSum::default();
    let mut total_log_abs_scale = CompensatedSum::default();
    // The band of the `ln r` residue, summand by summand: each summand's own
    // formation roundings plus the compensated sum's `2u`.
    let mut log_cancel_band = 0.0_f64;

    // Polyharmonic blocks.
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let (pure, log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, j);
        total_pure.add(a_m * pure);
        total_log.add(a_m * log);
        total_log_abs_scale.add((a_m * log).abs());
        if log != 0.0 {
            log_cancel_band += gam_linalg::roundoff::compensated_band(
                polyharmonic_log_summand_roundings(m, k_dim),
                (a_m * log).abs(),
            );
        }
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
        if log != 0.0 {
            log_cancel_band += gam_linalg::roundoff::compensated_band(
                matern_log_summand_roundings(n, k_dim, j),
                (b_n * log).abs(),
            );
        }
    }
    let total_pure = total_pure.value();
    let total_log = total_log.value();
    let total_log_abs_scale = total_log_abs_scale.value();

    // The `ln r` coefficients cancel exactly (the PFD identity, whenever
    // 2(p+s) > d+2j), so `total_log` sums to a real zero and carries only the
    // roundings of its summands and of their sum. `CompensatedSum` is
    // Kahan-Babuska-Neumaier, whose forward error is `(2 + k)·u·Σ|terms|`
    // independently of the term count (Higham, *ASNA* 2nd ed., §4.3), where `k`
    // is the roundings that formed a summand: counted per block above, not one
    // product (the Γ, power and quotient roundings of each block's coefficient
    // do not cancel; gam#2735 measured a 1.6-ulp residue against a one-product
    // band). A residue above the band is a failure of the identity rather than
    // rounding, and the band is read off the log terms rather than off the pure
    // part, a different quantity.
    if total_log.abs() > log_cancel_band {
        crate::bail_invalid_basis!(
            "Duchon Taylor a_{} log-coefficient did not cancel: log={total_log:.6e}, pure={total_pure:.6e}; \
             log_abs_scale={total_log_abs_scale:.6e}, band={log_cancel_band:.6e}; p={p_order}, s={s_order}, d={k_dim}",
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
        // The exact two-block / transformation-normal path needs analytic
        // length-scale derivatives of the kernel, finite only when
        // 2(p+s) > k_dim + 2j.
        return Err(BasisError::duchon_smoothness_insufficient(
            format!("collision derivative phi^({}) psi triplet", 2 * j),
            2 * j,
            k_dim,
            p_order,
            s_order as f64,
        ));
    }

    let kappa =
        duchon_inverse_length_scale(length_scale, "Duchon even-derivative collision ψ-triplet")?;
    let mut value = CompensatedSum::default();
    let mut psi = CompensatedSum::default();
    let mut psi_psi = CompensatedSum::default();
    let mut log_value = CompensatedSum::default();
    let mut log_psi = CompensatedSum::default();
    let mut log_psi_psi = CompensatedSum::default();
    let mut log_abs_scale = CompensatedSum::default();
    // The band of the three `ln r` residues, product by product: each product's
    // formation roundings plus the compensated sum's `2u`.
    let mut log_cancel_band = 0.0_f64;

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
        if log != 0.0 {
            // `α²·a_m·ℓ` is the longest: two products past the summand.
            log_cancel_band += gam_linalg::roundoff::compensated_band(
                polyharmonic_log_summand_roundings(m, k_dim) + 2,
                (a_m * log).abs()
                    + (alpha_m * a_m * log).abs()
                    + (alpha_m * alpha_m * a_m * log).abs(),
            );
        }
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
        // Every PRODUCT the three sums accumulate, in absolute value. A Matern
        // block's ψ and ψψ summands are themselves two- and three-term sums, so
        // charging only the composed summand would miss the cancellation inside
        // it and understate the band.
        log_abs_scale.add((b_n * log.0).abs());
        log_abs_scale.add((beta_n * b_n * log.0).abs());
        log_abs_scale.add((b_n * log.1).abs());
        log_abs_scale.add((beta_n * beta_n * b_n * log.0).abs());
        log_abs_scale.add((2.0 * beta_n * b_n * log.1).abs());
        log_abs_scale.add((b_n * log.2).abs());
        if log != (0.0, 0.0, 0.0) {
            log_cancel_band += gam_linalg::roundoff::compensated_band(
                matern_log_triplet_summand_roundings(n, k_dim, j),
                (b_n * log.0).abs()
                    + (beta_n * b_n * log.0).abs()
                    + (b_n * log.1).abs()
                    + (beta_n * beta_n * b_n * log.0).abs()
                    + (2.0 * beta_n * b_n * log.1).abs()
                    + (b_n * log.2).abs(),
            );
        }
    }

    let value = value.value();
    let psi = psi.value();
    let psi_psi = psi_psi.value();
    let log_value = log_value.value();
    let log_psi = log_psi.value();
    let log_psi_psi = log_psi_psi.value();
    let log_abs_scale = log_abs_scale.value();
    // All three `ln r` coefficient sums cancel exactly, so each carries only its
    // Kahan-Babuska-Neumaier forward error `(2 + k)·u·Σ|terms|` (Higham, *ASNA*
    // 2nd ed., §4.3), with `k` the roundings that formed each summand: counted
    // per block above, through the block's own Γ, power and quotient roundings
    // and the triplet's combination (the combination alone, `k = 5`, missed the
    // coefficients' formation). The band is taken over every product the three
    // sums accumulate, which majorizes each sum's own, so one band covers all
    // three.
    if log_value.abs().max(log_psi.abs()).max(log_psi_psi.abs()) > log_cancel_band {
        crate::bail_invalid_basis!(
            "Duchon Taylor a_{} log-coefficient derivative did not cancel: \
             log=({log_value:.6e}, {log_psi:.6e}, {log_psi_psi:.6e}), \
             value=({value:.6e}, {psi:.6e}, {psi_psi:.6e}), log_abs_scale={log_abs_scale:.6e}, band={log_cancel_band:.6e}; \
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

/// Resolve the FROZEN radial chart that every Duchon ψ-derivative is taken in.
///
/// `build_duchon_basis` ADOPTS a data-metric radial reparameterization `V`
/// whenever the constrained kernel block has columns and the spec carries no
/// frozen one (#1355), then freezes it into
/// `BasisMetadata::Duchon::radial_reparam`. The design, the native penalties
/// and the operator penalties are all assembled in `Z·V`, and every
/// ψ-derivative on this path is a FROZEN-chart derivative: `V` is held at the
/// cold build and replayed onto the spec at each trial κ, which is exactly what
/// the κ-optimizer does before asking for one.
///
/// A spec that reaches a derivative builder WITHOUT a frozen `V` therefore does
/// not describe the penalty its own forward build would ship — that build would
/// compute a fresh `V(ψ)` — so differentiating in the raw `Z` chart returns the
/// exact derivative of a DIFFERENT matrix. Nothing downstream can notice: the
/// shapes agree and the numbers are finite. That is how three finite-difference
/// gates came to report the chart mismatch as a 10×–290× error in the analytic
/// derivative itself, when the derivative is exact to 1.1e-7 against a
/// same-chart difference (#2638). Refuse instead of returning it.
///
/// This is the ONE place the three fold sites decide the chart, so a missing
/// `V` cannot be absorbed silently at any of them.
pub(crate) fn duchon_frozen_radial_chart(
    z_kernel: Array2<f64>,
    spec: &DuchonBasisSpec,
    site: &str,
) -> Result<Array2<f64>, BasisError> {
    let Some(v) = spec.radial_reparam.as_ref() else {
        if z_kernel.ncols() == 0 {
            // No constrained radial columns ⇒ the forward build has nothing to
            // rotate and adopts no `V` either, so the raw chart IS its chart.
            return Ok(z_kernel);
        }
        crate::bail_invalid_basis!(
            "Duchon {site} ψ-derivative requires the frozen data-metric radial reparam V, but              the spec carries none while the constrained kernel block has {} columns. The              forward build adopts a fresh V(ψ) for this spec, so a derivative taken in the raw              Z chart is the exact derivative of a different penalty (#2638). Replay              BasisMetadata::Duchon::radial_reparam onto the spec first, as the κ-optimizer does.",
            z_kernel.ncols()
        );
    };
    if v.nrows() != z_kernel.ncols() {
        crate::bail_dim_basis!(
            "Duchon frozen radial reparam shape {:?} does not match constrained kernel dimension {}",
            v.dim(),
            z_kernel.ncols()
        );
    }
    Ok(fast_ab(&z_kernel, v))
}

pub(crate) fn build_duchon_design_psi_derivativeswithworkspace(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    identifiability_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<ScalarDesignPsiDerivatives, BasisError> {
    let length_scale = spec.hybrid_length_scale()?.ok_or_else(|| {
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
    let s_order = spec.hybrid_s_order()?;
    let kappa = 1.0 / length_scale;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
    // #1355/#2638: the design ψ-derivatives assemble in the SAME frozen radial
    // chart `Z·V` as the forward design and penalty.
    let z_kernel = duchon_frozen_radial_chart(
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?,
        spec,
        "design",
    )?;
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
    // gam#979: the forward design ships `α·K` (see `duchon_kernel_chart`), so
    // the design ψ-derivative is formed under that same chart.
    let chart = duchon_kernel_chart(
        centers,
        Some(length_scale),
        p_order,
        s_order,
        data.ncols(),
        spec.aniso_log_scales.as_deref(),
        Some(&coeffs),
        None,
    )
    .design_chart();
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
        chart,
    )
}

pub(crate) fn duchon_operator_penalties_requested(spec: &DuchonOperatorPenaltySpec) -> bool {
    matches!(spec.mass, OperatorPenaltySpec::Active { .. })
        || matches!(spec.tension, OperatorPenaltySpec::Active { .. })
        || matches!(spec.stiffness, OperatorPenaltySpec::Active { .. })
}

/// Per-axis ψ derivatives of a hybrid Duchon basis — the anisotropic sibling of
/// [`build_duchon_basis_log_kappa_derivativeswith_collocationwithworkspace`]
/// (gam#2735).
///
/// The design half is the family-agnostic
/// `build_aniso_design_psi_derivatives_shared`, which already handles
/// `RadialScalarKind::Duchon` including its `δ/d` prefactor share; the penalty
/// half is the `_in_directions` entries, called once with `[Axis(0) … Axis(d−1)]`
/// so the whole per-axis surface costs one pass over the pairs.
///
/// Callers must have cleared `crate::basis::duchon_spec_supports_axis_psi`
/// first: this refuses rather than silently degrading, because a per-axis
/// coordinate whose derivative came from the isotropic route would be a
/// value/gradient desync rather than an approximation.
pub fn build_duchon_basis_log_kappa_aniso_derivativeswith_collocationwithworkspace(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
    centers: ArrayView2<'_, f64>,
    identifiability_transform: Option<&Array2<f64>>,
    operator_collocation_points: Option<ArrayView2<'_, f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<AnisoBasisPsiDerivatives, BasisError> {
    let dim = data.ncols();
    if !crate::basis::duchon_spec_supports_axis_psi(spec, dim) {
        crate::bail_invalid_basis!(
            "Duchon per-axis ψ derivatives requested for a spec whose per-axis surface is not \
             derived (dim={dim}, length_scale={:?}, periodic={}, power={})",
            spec.length_scale,
            spec.periodic.is_some(),
            spec.power
        );
    }
    let length_scale = spec
        .hybrid_length_scale()?
        .expect("capability check requires a hybrid scale");
    let eta = spec
        .aniso_log_scales
        .clone()
        .expect("capability check requires resolved anisotropy");
    let effective_nullspace_order = duchon_effective_nullspace_order(centers, spec.nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_nullspace_order);
    let s_order = spec.hybrid_s_order()?;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let z_kernel = duchon_frozen_radial_chart(
        kernel_constraint_nullspace(centers, effective_nullspace_order, &mut workspace.cache)?,
        spec,
        "aniso design",
    )?;
    let poly_cols = polynomial_block_from_order(data, effective_nullspace_order).ncols();
    let p_padded = z_kernel.ncols() + poly_cols;
    if let Some(zf) = identifiability_transform
        && p_padded != zf.nrows()
    {
        crate::bail_dim_basis!(
            "Duchon identifiability transform mismatch in aniso design derivatives: local cols={}, transform rows={}",
            p_padded,
            zf.nrows()
        );
    }
    let p_final = identifiability_transform
        .map(|zf| zf.ncols())
        .unwrap_or(p_padded);
    // gam#979: same chart as the isotropic sibling — the forward design is
    // `α·K` in the anisotropic metric too.
    let chart = duchon_kernel_chart(
        centers,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        Some(eta.as_slice()),
        Some(&coeffs),
        None,
    )
    .design_chart();
    let mut result = build_aniso_design_psi_derivatives_shared(
        data,
        centers,
        &eta,
        p_final,
        Some(z_kernel),
        identifiability_transform.cloned(),
        poly_cols,
        RadialScalarKind::Duchon {
            length_scale,
            p_order,
            s_order,
            dim,
            coeffs,
        },
        chart,
    )?;

    let directions: Vec<DuchonPsiDirection> = (0..dim).map(DuchonPsiDirection::Axis).collect();
    let native = crate::basis::build_duchon_native_penalty_psi_derivatives_in_directions(
        centers,
        spec,
        identifiability_transform,
        workspace,
        &directions,
    )?;
    let operator = if duchon_operator_penalties_requested(&spec.operator_penalties) {
        let Some(collocation_points) = operator_collocation_points else {
            crate::bail_invalid_basis!(
                "Duchon per-axis operator penalty derivatives require realized collocation points"
            );
        };
        crate::basis::build_duchon_operator_penalty_psi_derivatives_in_directions(
            collocation_points,
            centers,
            spec,
            identifiability_transform,
            workspace,
            &directions,
        )?
    } else {
        vec![(Vec::new(), Vec::new(), Vec::new()); dim]
    };

    // Same order the isotropic bundle ships: native candidates then operator
    // candidates, per axis. A mismatch here would misalign the ψ blocks against
    // the realized penalty list, so it is asserted rather than assumed.
    let mut penalties_first = Vec::with_capacity(dim);
    let mut penalties_second_diag = Vec::with_capacity(dim);
    let expected = native[0].0.len() + operator[0].0.len();
    for axis in 0..dim {
        if native[axis].0.len() != native[0].0.len()
            || operator[axis].0.len() != operator[0].0.len()
        {
            crate::bail_invalid_basis!(
                "Duchon per-axis penalty source counts disagree across axes: axis {axis} has \
                 {}+{} blocks, axis 0 has {}+{}",
                native[axis].0.len(),
                operator[axis].0.len(),
                native[0].0.len(),
                operator[0].0.len()
            );
        }
        let mut first = Vec::with_capacity(expected);
        let mut second = Vec::with_capacity(expected);
        first.extend(native[axis].1.iter().cloned());
        first.extend(operator[axis].1.iter().cloned());
        second.extend(native[axis].2.iter().cloned());
        second.extend(operator[axis].2.iter().cloned());
        if first.len() != expected || second.len() != expected {
            crate::bail_invalid_basis!(
                "Duchon per-axis penalty derivative count mismatch on axis {axis}: assembled \
                 {}/{} against {expected} active sources",
                first.len(),
                second.len()
            );
        }
        penalties_first.push(first);
        penalties_second_diag.push(second);
    }
    result.penalties_first = penalties_first;
    result.penalties_second_diag = penalties_second_diag;
    // Cross-axis PENALTY seconds are not provided: the outer solve consumes the
    // per-axis diagonal seconds plus the operator's exact cross-axis DESIGN
    // seconds, and an absent provider is the shape the anisotropic Matérn's
    // operator-triplet path already ships.
    result.penalties_cross_pairs = Vec::new();
    result.penalties_cross_provider = None;
    Ok(result)
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

/// The amplitude `α` the forward Duchon basis multiplies into every kernel
/// value: [`duchon_kernel_chart`]'s `1/|φ̃(r*)|` at the frozen reference pair.
///
/// **Why**: the kernel's own scale is not representable across the length
/// scales a fit visits. In high `d` at a large spectral power the prefactor is
/// `~1e-14`, so `BᵀB` sits near `1e-32` and the spectral frame truncates the
/// basis as noise. At long length scales the surviving kernel grows like
/// `ℓ^{2(b−p)}`, so the kernel block dwarfs the polynomial columns.
///
/// Rescaling the basis by a positive `α` produces the same predictions (β
/// rescales by `α`, REML's λ adapts). `α` is a pure function of the centers and
/// the kernel parameters stored verbatim in `BasisMetadata::Duchon`, so
/// prediction recomputes an identical `α`, and fit-time and predict-time bases
/// share one coefficient frame.
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
    duchon_kernel_chart(
        centers,
        length_scale,
        p_order,
        s_order,
        d,
        aniso_log_scales,
        coeffs,
        pure_poly_coeff,
    )
    .amplification
}

/// The numerical chart of one realized Duchon kernel block (gam#979, gam#2735).
///
/// [`duchon_kernel_amplification`] is the amplitude `α` the forward basis
/// multiplies into every kernel value: `α = 1/|φ̃(r*(η); κ)|`, where `(i*, j*)`
/// is the frozen reference pair [`duchon_kernel_chart`] selects and `r*(η)` is
/// its distance in the current metric. The reference pair depends on the
/// centers alone, so `α` has no branch and no argmax: it is smooth in ψ. The
/// design the criterion is built on is `α(ψ)·K(ψ)`, and a ψ-derivative of the
/// design that differentiates `K` alone is a derivative of something the fit
/// never evaluates. The derivative builders read the reference pair from here
/// and form the exact ψ-jets of `ln α` from the SAME radial jets they use for
/// every other pair, so the charted derivative is the derivative of the charted
/// kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DuchonKernelChart {
    /// `α = 1/|φ̃(r*)|`, or `1.0` for a block with fewer than two distinct centers.
    pub(crate) amplification: f64,
    /// The frozen `(i*, j*)` reference pair; `None` only for the identity chart.
    pub(crate) reference_pair: Option<(usize, usize)>,
}

impl DuchonKernelChart {
    pub(crate) const IDENTITY: Self = Self {
        amplification: 1.0,
        reference_pair: None,
    };

    pub(crate) fn design_chart(&self) -> crate::basis::DesignKernelChart {
        crate::basis::DesignKernelChart {
            scale: self.amplification,
            reference_pair: self.reference_pair,
        }
    }
}

/// Select the frozen reference pair and form `α = 1/|φ̃(r*(η); κ)|`.
///
/// The reference pair is the farthest pair of the frozen standardized centers in
/// the isotropic metric, the lowest `(i, j)` on ties. It is recomputed at every
/// build from the centers alone and never persisted, so it has one source of
/// truth and no dependence on ψ. The previous chart amplified only below a
/// literal `max|K| = 1e-10`, over an argmax the origin constant pinned to the
/// diagonal. That branch switched the realized columns at a length-scale
/// threshold (#979 CTN, gam#2735); this chart has no threshold to cross.
///
/// **Magnitude bound.** For the stable hybrid orders (`2p < d`, `b > 0`) the
/// profile `G(ρ)` decreases, because `d/dz [z^b K_b(z)] = −z^b K_{b−1}(z) < 0`
/// on every slice of the reference integral. So `|φ̃(r)| = pref·κ^{−2b}·(G(0) −
/// G(κr))` increases with `r`, and every center pair satisfies
/// `|α·K_CC| ≤ |φ̃(r_max(η))| / |φ̃(r*(η))|`, where `r_max(η)` is the farthest
/// center distance in the current metric. That ratio is exactly 1 in the
/// isotropic metric. A data row `x` outside the centers' hull can exceed it by
/// `|φ̃(r_x)| / |φ̃(r*)|`. No such proof is claimed for pure log-case kernels,
/// partial-fraction orders or null-space-reduced kernels; their bound is
/// measured on the ψ ladder instead.
pub(crate) fn duchon_kernel_chart(
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    d: usize,
    aniso_log_scales: Option<&[f64]>,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
    pure_poly_coeff: Option<&PolyharmonicBlockCoeff>,
) -> DuchonKernelChart {
    let k = centers.nrows();
    let mut reference_pair = None;
    let mut farthest = 0.0_f64;
    for i in 0..k {
        for j in (i + 1)..k {
            let r = euclidean_distance_rows(centers, i, centers, j);
            if r > farthest {
                farthest = r;
                reference_pair = Some((i, j));
            }
        }
    }
    let Some((i, j)) = reference_pair else {
        return DuchonKernelChart::IDENTITY;
    };
    let axis_scales = aniso_log_scales.map(aniso_axis_scales);
    let r = match axis_scales.as_deref() {
        Some(scales) => aniso_distance_rows_with_scales(centers, i, centers, j, scales),
        None => farthest,
    };
    let value = if let Some(ppc) = pure_poly_coeff {
        Ok(ppc.eval(r))
    } else {
        match duchon_hybrid_evaluator(length_scale, p_order, s_order, d) {
            Ok(Some(hybrid)) => hybrid.value(r),
            Ok(None) => duchon_matern_kernel_general_from_distance(
                r,
                length_scale,
                p_order,
                s_order,
                d,
                coeffs,
            ),
            Err(error) => Err(error),
        }
    };
    match value {
        Ok(magnitude) if magnitude.is_finite() && magnitude != 0.0 => DuchonKernelChart {
            amplification: 1.0 / magnitude.abs(),
            reference_pair: Some((i, j)),
        },
        // A kernel that cannot be evaluated at the reference pair has no chart; the
        // forward build evaluates the same kernel and owns that refusal.
        _ => DuchonKernelChart::IDENTITY,
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
    spectral_kernel_transform: Option<&Array2<f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<DuchonBasisDesign, BasisError> {
    DUCHON_DESIGN_BUILD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    // Validate the requested power itself: the hybrid kernel refuses a
    // fractional `power` (#3541), the scale-free one evaluates it literally.
    validate_duchon_kernel_orders(length_scale, p_order, s_order, d)?;

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
    if radial_reparam.is_some() && spectral_kernel_transform.is_some() {
        crate::bail_invalid_basis!(
            "Duchon design cannot combine landmark radial reparameterization with a direct \
             spectral kernel transform"
        );
    }
    let z_raw = if let Some(spectral) = spectral_kernel_transform {
        if spectral.nrows() != centers.nrows() {
            crate::bail_dim_basis!(
                "Duchon spectral kernel transform shape {:?} does not match {} centers",
                spectral.dim(),
                centers.nrows()
            );
        }
        spectral.clone()
    } else {
        kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?
    };
    // #1355: the frozen data-metric radial reparameterization `V` rotates the
    // constrained kernel columns into the `G_c`-orthonormal generalized eigenbasis,
    // `K·Z·V`. The kernel pass forms `K·Z`, and `V` rotates the assembled block
    // afterwards (below), because that is the product order the cold build takes
    // when it adopts `V` (`duchon_resolve_radial_chart`: `(K·Z)·V` off its one
    // kernel pass). Folding `V` into the transform first gives `K·(Z·V)`: the same
    // matrix in exact arithmetic, not in floating point. The polynomial-orthogonal
    // combinations in `K·Z` cancel large kernel entries, so a frozen replay drifted
    // from its fit-time design while every replayed matrix was bit-identical.
    if let Some(v) = radial_reparam {
        if v.nrows() != z_raw.ncols() {
            crate::bail_dim_basis!(
                "Duchon radial reparam shape {:?} does not match constrained kernel dimension {}",
                v.dim(),
                z_raw.ncols()
            );
        }
    }
    let z = z_raw;

    let coeffs = length_scale
        .map(|ls| {
            duchon_inverse_length_scale(ls, "Duchon basis design").map(|kappa| {
                duchon_partial_fraction_coeffs(p_order, duchon_power_to_usize(s_order), kappa)
            })
        })
        .transpose()?;

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
    // The hybrid orders are answered from the shape's certified universal
    // profile, bound here for the whole n·k sweep.
    let hybrid_eval = if pure_poly_coeff.is_some() {
        None
    } else {
        duchon_hybrid_evaluator(length_scale, p_order, duchon_power_to_usize(s_order), d)?
    };
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
    // A bound evaluator already answers those orders exactly; the per-build
    // Chebyshev profile below would approximate the same universal G a second
    // time, and the n·k distance pre-pass that sizes it would walk every pair
    // before the loop that walks them again.
    let value_profile = hybrid_kind
        .as_ref()
        .filter(|_| hybrid_eval.is_none())
        .and_then(|kind| {
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
            let rows = chunk.nrows();
            let chunk_start = ci * chunk_size;
            // The chunk's kernel block `K` (rows × k), filled row by row and
            // then multiplied by `Z` in ONE call.
            let mut kernel_block = Array2::<f64>::zeros((rows, k));
            for local_i in 0..rows {
                let i = chunk_start + local_i;
                let mut kernel_row = kernel_block.row_mut(local_i);
                // gam#4588: the hybrid row's representative, origin-reduced or
                // with the origin constant, whichever carries the smaller entries
                // over this row's distances; the constraint annihilates either
                // row constant on its own.
                let hybrid_row_keeps_origin = match hybrid_eval.as_ref() {
                    Some(hybrid) if pure_poly_coeff.is_none() => {
                        let (mut r_min, mut r_max) = (f64::INFINITY, 0.0_f64);
                        for j in 0..k {
                            let r = if let Some(scales) = axis_scales.as_deref() {
                                aniso_distance_rows_with_scales(data, i, centers, j, scales)
                            } else {
                                euclidean_distance_rows(data, i, centers, j)
                            };
                            r_min = r_min.min(r);
                            r_max = r_max.max(r);
                        }
                        hybrid.row_keeps_origin(r_min, r_max)?
                    }
                    _ => false,
                };
                for j in 0..k {
                    let r = if let Some(scales) = axis_scales.as_deref() {
                        aniso_distance_rows_with_scales(data, i, centers, j, scales)
                    } else {
                        euclidean_distance_rows(data, i, centers, j)
                    };
                    let raw = if let Some(ref ppc) = pure_poly_coeff {
                        // Pure Duchon: use precomputed coefficient, skip gamma calls.
                        ppc.eval(r)
                    } else if let Some(hybrid) = hybrid_eval.as_ref() {
                        if hybrid_row_keeps_origin {
                            hybrid.value_with_origin(r)?
                        } else {
                            hybrid.value(r)?
                        }
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
            }
            // The kernel block times the identifiability transform,
            // `basis[chunk] = K · Z`, is a matrix product: one GEMM per chunk
            // instead of a scatter-accumulate over `rows · k` rows of `Z`.
            //
            // Written out by hand this is the dominant cost of a wide design:
            // `n · k · kernel_cols` scalar updates through bounds-checked
            // indexing measured ≈ 1 GFLOP/s across the pool on the 6-D
            // isotropic Duchon fit (n = 50 000, k = 100, 8 threads: 19 % of
            // the whole fit's samples sat in this closure's own frame, and the
            // per-κ-trial rebuild grows with `k²`). faer's kernel does the
            // same arithmetic blocked and vectorised. It runs sequentially
            // here because the chunk loop is already the parallel region — the
            // pool is saturated by chunks, not by one product (#2735).
            let mut product = Array2::<f64>::zeros((rows, kernel_cols));
            gam_linalg::faer_ndarray::with_nested_parallel(|| {
                gam_linalg::faer_ndarray::fast_ab_into(&kernel_block, &z, &mut product)
            });
            chunk.slice_mut(s![.., ..kernel_cols]).assign(&product);
            Ok(())
        });
    basis_result?;
    if poly_cols > 0 {
        basis.slice_mut(s![.., kernel_cols..]).assign(&poly_block);
    }
    // Rotate the assembled `K·Z` block by the frozen `V` exactly as
    // `duchon_resolve_radial_chart` rotates its raw basis, so a replay reproduces the
    // cold build's design bit for bit.
    if let Some(v) = radial_reparam {
        let rotated_kernel = fast_ab(&basis.slice(s![.., 0..kernel_cols]), v);
        let mut rotated = Array2::<f64>::zeros((n, rotated_kernel.ncols() + poly_cols));
        rotated
            .slice_mut(s![.., 0..rotated_kernel.ncols()])
            .assign(&rotated_kernel);
        if poly_cols > 0 {
            rotated
                .slice_mut(s![.., rotated_kernel.ncols()..])
                .assign(&basis.slice(s![.., kernel_cols..]));
        }
        basis = rotated;
    }

    Ok(DuchonBasisDesign { basis })
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
    create_duchon_basis_1d_derivative_dense_with_radial_reparam(
        t,
        centers,
        power,
        nullspace_order,
        periodic,
        period,
        None,
        order,
    )
}

/// Evaluate a 1-D Duchon design derivative in an already-frozen radial chart.
/// Position-batched consumers compute the data-metric chart once from the
/// complete ragged batch and reuse it for every segment; without this argument
/// each segment differentiates a different coefficient basis.
pub fn create_duchon_basis_1d_derivative_dense_with_radial_reparam(
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    periodic: bool,
    period: Option<f64>,
    radial_reparam: Option<ArrayView2<'_, f64>>,
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
    if periodic && radial_reparam.is_some() {
        crate::bail_invalid_basis!(
            "periodic 1-D Duchon derivatives do not admit an open-domain radial reparameterization"
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

    if periodic {
        // The periodic kernel is the Bernoulli Green's function of order
        // `user_m` and does not read `s`; validate exactly as the forward
        // periodic builder does.
        validate_duchon_kernel_orders(None, p_order, duchon_power_to_usize(power) as f64, 1)?;
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
    if let Some(radial_reparam) = radial_reparam {
        if radial_reparam.nrows() != z.ncols() {
            crate::bail_dim_basis!(
                "Duchon frozen radial reparam shape {:?} does not match constrained kernel dimension {}",
                radial_reparam.dim(),
                z.ncols()
            );
        }
    }
    let kernel_cols = radial_reparam.map_or(z.ncols(), |v| v.ncols());
    let poly_cols = polynomial_block_from_order(data.view(), effective_order).ncols();
    // The forward design assembles its polynomial null-space block in the
    // center-cloud-centred frame `t − t̄_c` (#1375), so the polynomial columns
    // here — and their t-derivatives — are the monomials of that same centred
    // coordinate. The raw `t` monomials span the same space but are a different
    // basis: coefficients fitted against the forward design would be read back
    // against the wrong columns.
    let center_mean = centers.sum() / centers.len() as f64;
    let t_centered = t.mapv(|v| v - center_mean);

    // The scale-free kernel evaluates the literal spectral power, exactly as
    // the forward 1-D design (`build_duchon_basis`) does. Truncating it to an
    // integer here differentiated a different kernel than the one the fit
    // realized whenever `power` was fractional (#3541).
    validate_duchon_kernel_orders(None, p_order, power, 1)?;
    let pure_coeff = PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, power), 1);
    let kernel_amp = duchon_kernel_amplification(
        center_matrix.view(),
        None,
        p_order,
        duchon_power_to_usize(power),
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
                duchon_kernel_radial_triplet(r, None, p_order, power, 1, None)?;
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
    // `(K·Z)·V`, the product order the forward design takes (#1355): `K·(Z·V)`
    // is the same matrix only in exact arithmetic.
    let constrained_kernel = fast_ab(&raw_kernel, &z);
    let design_kernel = match radial_reparam {
        Some(v) => fast_ab(&constrained_kernel, &v.to_owned()),
        None => constrained_kernel,
    };
    basis
        .slice_mut(s![.., 0..kernel_cols])
        .assign(&design_kernel);
    fill_duchon_1d_polynomial_derivative(
        &mut basis,
        kernel_cols,
        t_centered.view(),
        effective_order,
        order,
    );
    Ok(basis)
}

#[cfg(test)]
mod taylor_degree_tests {
    use super::*;

    /// gam#1604 — the half-integer-ν Matérn block Taylor coefficients. For
    /// |ν| = l + ½ the block has the elementary closed form
    /// `c · r^ν K_ν(κr) = c · √(π/2κ) · e^{−κr} · P(κ,r)` with P a finite
    /// Laurent polynomial, so the exact `r^{2j}` coefficients are clean rationals
    /// (no log term). At κ = 1, d = 1:
    ///   • n = 2 (ν = 3/2): block = ¼ (r + 1) e^{−r}        → [0.25, −0.125, −0.03125]
    ///   • n = 3 (ν = 5/2): block = 1/16 (r² + 3r + 3) e^{−r} → [0.1875, −0.03125, 0.0078125]
    /// The earlier `l = round(2|ν| − 1)` miscount used the K_{5/2} / K_{9/2}
    /// polynomials for these (degree 2|ν|−½, not |ν|), collapsing the j = 0 term
    /// to exactly 0. These references would all fail under that bug.
    #[test]
    fn half_integer_matern_taylor_coeffs_1604() {
        let want_nu_3_2 = [0.25_f64, -0.125, -0.03125];
        let want_nu_5_2 = [0.1875_f64, -0.03125, 0.0078125];
        for (j, &want) in want_nu_3_2.iter().enumerate() {
            let (pure, log) = duchon_matern_block_taylor_r2j(1.0, 2, 1, j);
            assert!(log == 0.0, "no log term for half-integer ν (j={j}): {log}");
            assert!(
                (pure - want).abs() < 1e-13,
                "ν=3/2 r^{{{}}} coeff: got {pure:.15}, want {want}",
                2 * j
            );
        }
        for (j, &want) in want_nu_5_2.iter().enumerate() {
            let (pure, log) = duchon_matern_block_taylor_r2j(1.0, 3, 1, j);
            assert!(log == 0.0, "no log term for half-integer ν (j={j}): {log}");
            assert!(
                (pure - want).abs() < 1e-13,
                "ν=5/2 r^{{{}}} coeff: got {pure:.15}, want {want}",
                2 * j
            );
        }
    }

    /// gam#1604 — the j = 0 Taylor coefficient must equal the r → 0⁺ limit of the
    /// block computed independently via the real Bessel-K value path
    /// (`r^ν K_ν(κr) → 2^{ν−1} Γ(ν) κ^{−ν}` for ν > 0). Sweeps half-integer ν up
    /// to 7/2 and several κ; the regressed code returned 0 for ν ≥ 3/2.
    #[test]
    fn half_integer_matern_taylor_j0_matches_value_limit_1604() {
        let d = 1usize;
        for n in 1..=4usize {
            let nu = n as f64 - 0.5 * d as f64; // ν = n − ½ ∈ {0.5, 1.5, 2.5, 3.5}
            for &kappa in &[0.3_f64, 1.0, 2.0, 7.5] {
                let (pure, _log) = duchon_matern_block_taylor_r2j(kappa, n, d, 0);
                // Independent r→0⁺ limit through the value path.
                let want = duchon_matern_block(0.0, kappa, n, d).expect("r→0 limit");
                let rel = (pure - want).abs() / want.abs().max(1e-300);
                assert!(
                    rel < 1e-12,
                    "ν={nu} κ={kappa}: Taylor j=0 {pure:.15e} vs value limit {want:.15e} (rel {rel:.2e})"
                );
            }
        }
    }
}

#[cfg(test)]
mod end_to_end_1604_tests {
    use super::*;
    use gam_linalg::faer_ndarray::FaerEigh;

    /// #3541 — the 1-D derivative builder must differentiate the kernel the
    /// forward design realized. For a fractional scale-free power it used to
    /// truncate `s` to an integer, so its order-0 output was a different basis
    /// than `build_duchon_basis`. The integer power is the control: it pins the
    /// frame (standardized centers, frozen radial chart) this comparison needs.
    #[test]
    fn d1_pure_fractional_power_derivative_matches_forward_design_3541() {
        let n = 30usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let u = i as f64 / (n as f64 - 1.0);
            data[[i, 0]] = -1.0 + 2.0 * u + 0.15 * (5.0 * u).sin();
        }
        for &power in &[0.0f64, 0.25] {
            let spec = DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 9 },
                periodic: None,
                length_scale: None,
                power,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::None,
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                boundary: OneDimensionalBoundary::Open,
                radial_reparam: None,
            };
            let built = build_duchon_basis(data.view(), &spec)
                .unwrap_or_else(|e| panic!("power={power}: forward build failed: {e}"));
            let BasisMetadata::Duchon {
                centers,
                input_scale,
                radial_reparam,
                ..
            } = &built.metadata
            else {
                panic!("power={power}: expected Duchon metadata");
            };
            let forward = built
                .design
                .try_to_dense_by_chunks("d1_pure_fractional_power_3541")
                .expect("dense forward design");
            let t = data.column(0).mapv(|x| x / input_scale.get());
            let replay = create_duchon_basis_1d_derivative_dense_with_radial_reparam(
                t.view(),
                centers.column(0),
                power,
                DuchonNullspaceOrder::Linear,
                false,
                None,
                radial_reparam.as_ref().map(|v| v.view()),
                0,
            )
            .unwrap_or_else(|e| panic!("power={power}: derivative builder failed: {e}"));
            assert_eq!(replay.dim(), forward.dim(), "power={power}: layout mismatch");
            let scale = forward.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let worst = forward
                .iter()
                .zip(replay.iter())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            // Both sides evaluate the same closed form on the same inputs, so
            // they agree to a few ulps of the design's magnitude.
            assert!(
                worst <= 1e-10 * scale.max(1.0),
                "power={power}: order-0 derivative basis differs from the forward design \
                 by {worst:.3e} (design scale {scale:.3e})"
            );
        }
    }

    /// gam#1604 — end-to-end: a 1-D hybrid Duchon smooth with power ≥ 2 must
    /// build successfully through the public `build_duchon_basis` path and emit
    /// numerically-PSD penalties. Before the half-integer-ν Taylor-degree fix the
    /// corrupted collision diagonal made the constrained native penalty
    /// indefinite, so the build's PSD guard rejected it outright — the issue's
    /// "any d=1 Duchon smooth with power ≥ 2 currently cannot be fitted".
    #[test]
    fn d1_hybrid_duchon_power_ge_2_builds_psd() {
        // A clustered + spread 1-D sample so center spacing is non-trivial.
        let n = 40usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            data[[i, 0]] = -1.0 + 2.0 * (i as f64) / (n as f64 - 1.0);
        }
        for &power in &[2.0f64, 3.0] {
            let spec = DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
                periodic: None,
                length_scale: Some(crate::basis::MaternLengthScale::fixed(0.5)),
                power,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::None,
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                boundary: OneDimensionalBoundary::Open,
                radial_reparam: None,
            };
            let result = build_duchon_basis(data.view(), &spec).unwrap_or_else(|e| {
                panic!("d=1 hybrid Duchon power={power} build rejected (gam#1604): {e}")
            });
            assert!(
                !result.active_penalties.is_empty(),
                "d=1 hybrid Duchon power={power} produced no penalty"
            );
            for (k, penalty) in result.active_penalties.iter().enumerate() {
                let sym = symmetrize_penalty(&penalty.matrix);
                let (evals, _) =
                    FaerEigh::eigh(&sym, faer::Side::Lower).expect("symmetric eigendecomposition");
                let lam_min = evals.iter().copied().fold(f64::INFINITY, f64::min);
                let lam_max = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let tol = 1e-9 * lam_max.abs().max(1.0);
                assert!(
                    lam_min >= -tol,
                    "d=1 hybrid Duchon power={power} penalty[{k}] not PSD: λ_min={lam_min:.6e} (tol={tol:.3e})"
                );
            }
        }
    }
}
