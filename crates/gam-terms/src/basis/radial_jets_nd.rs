use super::*;

/// Shared N-D Duchon radial-jet matrices: the per-`(row, center)` scalar
/// radial derivatives `φ'(r)`, `φ''(r)`, `φ'''(r)` of the Duchon kernel.
///
/// This is the single place that performs the expensive, error-prone work
/// behind every N-D Duchon derivative consumer: distance evaluation, the
/// effective-scale default, partial-fraction-coefficient derivation, and the
/// per-pair [`duchon_radial_jets`] call. The first/second/third radial
/// derivative helpers, and the analytic-penalty Cartesian-derivative tensors,
/// are all thin adapters over the matrices produced here.
///
/// Only the radial orders `1..=max_order` are materialized; higher matrices
/// are left empty. `max_order` must be in `1..=3` for the supported latent /
/// isometry paths.
pub(crate) struct DuchonRadialJetsNd {
    /// `(n_rows, n_centers)` matrix of `φ'(r_{nk})`; always populated.
    pub(crate) phi_r: Array2<f64>,
    /// `(n_rows, n_centers)` matrix of `φ''(r_{nk})`; populated iff `max_order ≥ 2`.
    pub(crate) phi_rr: Array2<f64>,
    /// `(n_rows, n_centers)` matrix of `φ'''(r_{nk})`; populated iff `max_order ≥ 3`.
    pub(crate) phi_rrr: Array2<f64>,
}

/// Effective length scale used by [`duchon_radial_jets`]'s near-origin guards
/// (`r_floor`, collision-Taylor radius) when the caller selects the scale-free
/// pure-Duchon spectrum via `length_scale = None`. This only sets the
/// numerical guards near `r = 0` and does not change the analytic kernel; we
/// pick the typical inter-center distance (or `1.0` as a last resort).
pub(crate) fn duchon_effective_length_scale(
    length_scale: Option<f64>,
    centers: ArrayView2<'_, f64>,
) -> f64 {
    length_scale.unwrap_or_else(|| {
        let n_centers = centers.nrows();
        let dim = centers.ncols();
        let mut acc = 0.0_f64;
        let mut cnt = 0usize;
        for i in 0..n_centers.min(8) {
            for j in (i + 1)..n_centers.min(8) {
                let mut r2 = 0.0_f64;
                for a in 0..dim {
                    let dv = centers[[i, a]] - centers[[j, a]];
                    r2 += dv * dv;
                }
                acc += r2.sqrt();
                cnt += 1;
            }
        }
        if cnt == 0 || acc <= 0.0 {
            1.0
        } else {
            acc / cnt as f64
        }
    })
}

/// Evaluate the shared N-D Duchon radial jets up to `max_order` (`1..=3`).
///
/// `caller` is used only to give callers a precise validation message.
pub(crate) fn duchon_radial_jets_nd(
    max_order: usize,
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
    caller: &str,
) -> Result<DuchonRadialJetsNd, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!("{caller}: centers must have at least one column");
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "{caller}: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    assert!(
        (1..=3).contains(&max_order),
        "duchon_radial_jets_nd supports radial orders 1..=3; got {max_order}"
    );
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    // Resolve the same hybrid `(p, s, κ)` triple the forward `build_duchon_basis`
    // uses, so the radial jets differentiate the *exact* forward Green's
    // function `φ_{p,s,κ}` rather than a hard-coded `s = 0` surrogate
    // (issue #440). `power` is the forward spectral order `s = spec.power`.
    //
    //   * Hybrid (`length_scale = Some`): spectrum `‖w‖^{2p}(κ²+‖w‖²)^s`, built
    //     from the integer partial-fraction blocks at the real κ = 1/length_scale.
    //   * Pure scale-free (`length_scale = None`): the forward kernel collapses
    //     to the polyharmonic of total order `p + s` (`pure_duchon_block_order`,
    //     `duchon_matern_kernel_general_from_distance`'s `None` branch). Folding
    //     `s` into the polyharmonic order (`s_jets = 0`, `a_{p+s} = 1`) makes the
    //     operator core κ-independent and reproduces that exact kernel and all
    //     of its radial derivatives.
    let (jet_p_order, s_order) = match length_scale {
        Some(_) => (p_order, power),
        None => (p_order + power, 0usize),
    };
    let kappa = length_scale.map(|l| 1.0 / l.max(1e-300)).unwrap_or(0.0);
    let coeffs = duchon_partial_fraction_coeffs(jet_p_order, s_order, kappa);
    let effective_length_scale = duchon_effective_length_scale(length_scale, centers);

    let mut phi_r = Array2::<f64>::zeros((n_rows, n_centers));
    let mut phi_rr = if max_order >= 2 {
        Array2::<f64>::zeros((n_rows, n_centers))
    } else {
        Array2::<f64>::zeros((0, 0))
    };
    let mut phi_rrr = if max_order >= 3 {
        Array2::<f64>::zeros((n_rows, n_centers))
    } else {
        Array2::<f64>::zeros((0, 0))
    };
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let dv = t[[n, a]] - centers[[k, a]];
                r2 += dv * dv;
            }
            let r = r2.sqrt();
            let jets = duchon_radial_jets(
                r,
                effective_length_scale,
                jet_p_order,
                s_order,
                dim,
                &coeffs,
            )?;
            phi_r[[n, k]] = jets.phi_r;
            if max_order >= 2 {
                phi_rr[[n, k]] = jets.phi_rr;
            }
            if max_order >= 3 {
                phi_rrr[[n, k]] = jets.phi_rrr;
            }
        }
    }
    Ok(DuchonRadialJetsNd {
        phi_r,
        phi_rr,
        phi_rrr,
    })
}

/// Map shared N-D Duchon radial jets into the Cartesian input-location
/// derivative tensor of `order` (2 or 3), contracted against per-center
/// decoder coefficients.
///
/// `coeffs` is the `(n_centers, p_out)` matrix of radial-basis coefficients
/// `c_{k,i}`. The result is a flat `(n_rows, p_out · dⁿ)` matrix whose entry
/// for row `n`, output `i`, and Cartesian multi-index `m` lives at column
/// `i · dⁿ + m`, where `m` enumerates the axes in row-major order
/// (`(a·d + c)` for `order = 2`, `((a·d + c)·d + e)` for `order = 3`).
///
/// The radial→Cartesian maps are:
///
/// ```text
/// order 2: ∂²Φ/∂t_a∂t_c = q δ_ac + (φ'' − q) u_a u_c,   q = φ'/r
/// order 3: ∂³Φ/∂t_a∂t_c∂t_e = a u_a u_c u_e
///                            + b (δ_ac u_e + δ_ae u_c + δ_ce u_a)
///          b = (φ'' − q)/r,   a = φ''' − 3b
/// ```
///
/// with `u = (t_n − c_k)/r`. At `r = 0` the order-2 collision limit is the
/// isotropic `φ''(0) δ_ac`; the order-3 tensor vanishes there.
pub(crate) fn radial_basis_cartesian_derivative(
    order: usize,
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    coeffs: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
) -> Result<Array2<f64>, BasisError> {
    assert!(
        order == 2 || order == 3,
        "radial_basis_cartesian_derivative supports Cartesian orders 2 and 3; got {order}"
    );
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let d = centers.ncols();
    let p_out = coeffs.ncols();
    assert_eq!(
        coeffs.nrows(),
        n_centers,
        "radial_basis_cartesian_derivative: coeffs has {} rows but centers have {n_centers}",
        coeffs.nrows()
    );
    let jets = duchon_radial_jets_nd(
        order,
        t,
        centers,
        length_scale,
        nullspace_order,
        power,
        "radial_basis_cartesian_derivative",
    )?;
    let d_pow = d.pow(order as u32);
    let mut out = Array2::<f64>::zeros((n_rows, p_out * d_pow));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..d {
                let delta = t[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            match order {
                2 => {
                    for a in 0..d {
                        for c in 0..d {
                            let basis_hess = if r == 0.0 {
                                if a == c { jets.phi_rr[[n, k]] } else { 0.0 }
                            } else {
                                let inv_r = 1.0 / r;
                                let u_a = (t[[n, a]] - centers[[k, a]]) * inv_r;
                                let u_c = (t[[n, c]] - centers[[k, c]]) * inv_r;
                                let q = jets.phi_r[[n, k]] * inv_r;
                                let eye = if a == c { 1.0 } else { 0.0 };
                                q * eye + (jets.phi_rr[[n, k]] - q) * u_a * u_c
                            };
                            if basis_hess == 0.0 {
                                continue;
                            }
                            let m = a * d + c;
                            for i in 0..p_out {
                                out[[n, i * d_pow + m]] += coeffs[[k, i]] * basis_hess;
                            }
                        }
                    }
                }
                _ => {
                    if r == 0.0 {
                        continue;
                    }
                    let inv_r = 1.0 / r;
                    let q = jets.phi_r[[n, k]] * inv_r;
                    let b_coef = (jets.phi_rr[[n, k]] - q) * inv_r;
                    let a_coef = jets.phi_rrr[[n, k]] - 3.0 * b_coef;
                    for a in 0..d {
                        let u_a = (t[[n, a]] - centers[[k, a]]) * inv_r;
                        for c in 0..d {
                            let u_c = (t[[n, c]] - centers[[k, c]]) * inv_r;
                            for e in 0..d {
                                let u_e = (t[[n, e]] - centers[[k, e]]) * inv_r;
                                let eye_ac = if a == c { 1.0 } else { 0.0 };
                                let eye_ae = if a == e { 1.0 } else { 0.0 };
                                let eye_ce = if c == e { 1.0 } else { 0.0 };
                                let basis_third = a_coef * u_a * u_c * u_e
                                    + b_coef * (eye_ac * u_e + eye_ae * u_c + eye_ce * u_a);
                                if basis_third == 0.0 {
                                    continue;
                                }
                                let m = (a * d + c) * d + e;
                                for i in 0..p_out {
                                    out[[n, i * d_pow + m]] += coeffs[[k, i]] * basis_third;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// N-D Duchon radial first-derivative `φ'(r)` evaluated for every
/// `(row, center)` pair.
///
/// Returns an `(n_rows, n_centers)` matrix whose `(n, k)` entry is the
/// scalar radial derivative `φ'(r_{nk})` of the Duchon kernel,
/// where `r_{nk} = ‖t_n − c_k‖_2`.
///
/// This is the load-bearing primitive for differentiating a Duchon design
/// against its *first* kernel argument (i.e. per-row latent coordinates
/// `t_n`); the full per-row gradient is reconstructed at the call site as
/// `∂Φ_{n,k}/∂t_n = φ'(r_{n,k}) · (t_n − c_k) / r_{n,k}` (see
/// [`crate::latent::LatentCoordValues::design_gradient_wrt_t`]).
///
/// `length_scale = None` selects the scale-free pure-Duchon spectrum
/// (matches `gam_pyffi::position_basis_derivative` for the 1-D case).
///
/// Thin adapter over [`duchon_radial_jets_nd`].
pub fn duchon_radial_first_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
) -> Result<Array2<f64>, BasisError> {
    Ok(duchon_radial_jets_nd(
        1,
        t,
        centers,
        length_scale,
        nullspace_order,
        power,
        "duchon_radial_first_derivative_nd",
    )?
    .phi_r)
}

/// N-D Duchon radial second derivative `φ''(r)` evaluated for every
/// `(row, center)` pair.
///
/// Returns an `(n_rows, n_centers)` matrix whose `(n, k)` entry is the scalar
/// radial second derivative `φ''(r_{nk})` of the Duchon kernel, where
/// `r_{nk} = ‖t_n − c_k‖_2`.
///
/// This is the companion primitive to
/// [`duchon_radial_first_derivative_nd`]. Together the two scalars reconstruct
/// the full input-location Hessian:
///
/// ```text
/// ∂²φ/∂t_a∂t_b = (φ'(r)/r) δ_ab
///              + (φ''(r) − φ'(r)/r) (t_a − c_a)(t_b − c_b) / r².
/// ```
///
/// At `r = 0`, consumers should use the isotropic collision limit
/// `φ''(0) δ_ab`; `duchon_radial_jets` supplies that finite scalar whenever
/// the selected Duchon order is smooth enough for the supported latent path.
///
/// Thin adapter over [`duchon_radial_jets_nd`].
pub fn duchon_radial_second_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
) -> Result<Array2<f64>, BasisError> {
    Ok(duchon_radial_jets_nd(
        2,
        t,
        centers,
        length_scale,
        nullspace_order,
        power,
        "duchon_radial_second_derivative_nd",
    )?
    .phi_rr)
}

/// N-D Duchon radial third derivative `φ'''(r)` evaluated for every
/// `(row, center)` pair.
///
/// Returns an `(n_rows, n_centers)` matrix whose `(n, k)` entry is the scalar
/// third radial derivative `φ'''(r_{nk})` from the same
/// [`duchon_radial_jets`] path used by the first/second derivative helpers.
///
/// Thin adapter over [`duchon_radial_jets_nd`].
pub fn duchon_radial_third_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    nullspace_order: DuchonNullspaceOrder,
    power: usize,
) -> Result<Array2<f64>, BasisError> {
    Ok(duchon_radial_jets_nd(
        3,
        t,
        centers,
        length_scale,
        nullspace_order,
        power,
        "duchon_radial_third_derivative_nd",
    )?
    .phi_rrr)
}

pub(crate) fn fill_duchon_1d_polynomial_derivative(
    basis: &mut Array2<f64>,
    start_col: usize,
    t: ArrayView1<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
    derivative_order: usize,
) {
    let exponents: Vec<usize> = match nullspace_order {
        DuchonNullspaceOrder::Zero => vec![0],
        DuchonNullspaceOrder::Linear => vec![0, 1],
        DuchonNullspaceOrder::Degree(degree) => monomial_exponents(1, degree)
            .into_iter()
            .map(|exponent| exponent[0])
            .collect(),
    };
    for (offset, exponent) in exponents.into_iter().enumerate() {
        if exponent < derivative_order {
            continue;
        }
        let coefficient =
            (0..derivative_order).fold(1.0, |acc, step| acc * (exponent - step) as f64);
        let remaining = exponent - derivative_order;
        for row in 0..t.len() {
            basis[[row, start_col + offset]] = if remaining == 0 {
                coefficient
            } else {
                coefficient * t[row].powi(remaining as i32)
            };
        }
    }
}

/// N-D Duchon polynomial-nullspace first derivative `∂P/∂t` per row.
///
/// Generalises [`fill_duchon_1d_polynomial_derivative`] to arbitrary spatial
/// dimension `d` and arbitrary nullspace degree. For the monomial
/// `m_α(t) = ∏_a t_a^{α_a}`, the partial derivative w.r.t. `t_axis` is
///
/// ```text
///     ∂ m_α / ∂ t_axis = α_axis · t_axis^{α_axis − 1} · ∏_{a ≠ axis} t_a^{α_a}
/// ```
///
/// (and is zero if `α_axis == 0`).
///
/// Returned tensor shape: `(n_rows, n_poly_cols, d)`, ordered with the same
/// `monomial_exponents(d, max_total_degree)` enumeration that
/// `monomial_basis_block` uses to build the polynomial-tail columns of the
/// Duchon design, so the column index `k` aligns directly with the design.
pub fn duchon_polynomial_first_derivative_nd(
    t: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Array3<f64> {
    let n_rows = t.nrows();
    let dim = t.ncols();
    let max_degree = match nullspace_order {
        DuchonNullspaceOrder::Zero => 0usize,
        DuchonNullspaceOrder::Linear => 1usize,
        DuchonNullspaceOrder::Degree(k) => k,
    };
    let exponents = monomial_exponents(dim, max_degree);
    let n_poly = exponents.len();
    let mut out = Array3::<f64>::zeros((n_rows, n_poly, dim));
    if dim == 0 || n_poly == 0 {
        return out;
    }
    for (col, alpha) in exponents.iter().enumerate() {
        for axis in 0..dim {
            let a_axis = alpha[axis];
            if a_axis == 0 {
                continue;
            }
            for row in 0..n_rows {
                let mut value = a_axis as f64;
                for a in 0..dim {
                    let exp_a = if a == axis { a_axis - 1 } else { alpha[a] };
                    if exp_a != 0 {
                        value *= t[[row, a]].powi(exp_a as i32);
                    }
                }
                out[[row, col, axis]] = value;
            }
        }
    }
    out
}

/// Per-`(row, center)` input-location jet of a radial kernel from its scalar
/// first derivative `φ'(r)`.
///
/// `phi_r[n, k] = φ'(r_{nk})`. The full gradient w.r.t. the latent input is
///
/// ```text
/// ∂Φ_{n,k}/∂t_{n,a} = φ'(r_{nk}) · (t_{n,a} − c_{k,a}) / r_{nk}.
/// ```
///
/// At a collision (`r ≤ 1e-12`) the gradient is the zero vector — the radial
/// kernel has a stationary point at the center, so every axis derivative
/// vanishes there. Output shape `(n_rows, n_centers, dim)`.
pub fn radial_input_location_jet_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    phi_r: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = t.ncols();
    if phi_r.dim() != (n_rows, n_centers) {
        crate::bail_dim_basis!(
            "radial_input_location_jet_nd: phi_r shape {:?} != ({n_rows}, {n_centers})",
            phi_r.dim()
        );
    }
    if centers.ncols() != dim {
        crate::bail_dim_basis!(
            "radial_input_location_jet_nd: t has {dim} cols but centers have {}",
            centers.ncols()
        );
    }
    let mut out = Array3::<f64>::zeros((n_rows, n_centers, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let delta = t[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            if r <= 1.0e-12 {
                continue;
            }
            let scale = phi_r[[n, k]] / r;
            for a in 0..dim {
                out[[n, k, a]] = scale * (t[[n, a]] - centers[[k, a]]);
            }
        }
    }
    Ok(out)
}

/// Forward design and input-location first jet of the scale-free Duchon atom
/// used by the SAE-manifold path, recomputed self-consistently at arbitrary
/// latent coordinates `t`.
///
/// The column layout matches [`build_duchon_basis`] under the SAE atom's spec
/// (`length_scale = None`, `power = 0`, no identifiability transform): the
/// kernel block `Φ_radial(t) · Z` (where `Z = null(P_centersᵀ)` is the
/// polynomial-constraint null space) followed by the polynomial block
/// `P(t)`. Both blocks of `Φ` **and** the matching blocks of the jet carry
/// the identical scalar kernel amplification `α` that
/// [`build_duchon_basis`] applies, so the returned `(Φ, ∂Φ/∂t)` pair is a
/// true jet — i.e. the kernel block of the jet is exactly the `t`-derivative
/// of the kernel block of `Φ`, with no stray `α` mismatch (the precise
/// failure mode of issue #247: a forward design and derivative jet built from
/// inconsistent scalings/column counts).
///
/// `t` is `(n_rows, dim)`, `centers` is `(n_centers, dim)`. Returns
/// `(Φ, jet)` with `Φ` shape `(n_rows, n_kernel + n_poly)` and `jet` shape
/// `(n_rows, n_kernel + n_poly, dim)`.
pub fn duchon_sae_atom_basis_with_jet(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Result<(Array2<f64>, Array3<f64>), BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_sae_atom_basis_with_jet: centers must have at least one column"
        );
    }
    if t.ncols() != dim {
        crate::bail_dim_basis!(
            "duchon_sae_atom_basis_with_jet: t has {} cols but centers have {dim}",
            t.ncols()
        );
    }
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = 0.0;

    // Polynomial-constraint null space `Z` (same construction as the design).
    let poly_block_centers = polynomial_block_from_order(centers, effective_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    // Scalar kernel amplification `α`, identical to the value the design path
    // applies (pure scale-free polyharmonic block).
    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    );

    // Forward radial kernel block `(Φ_radial · α) · Z`.
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let mut radial_value = Array2::<f64>::zeros((n_rows, n_centers));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let r = euclidean_distance_rows(t, n, centers, k);
            radial_value[[n, k]] = pure_poly_coeff.eval(r) * kernel_amp;
        }
    }
    let kernel_design = fast_ab(&radial_value, &z);

    // Polynomial forward block `P(t)`.
    let poly_design = polynomial_block_from_order(t, effective_order);
    let n_poly = poly_design.ncols();

    let mut phi = Array2::<f64>::zeros((n_rows, n_kernel + n_poly));
    phi.slice_mut(s![.., ..n_kernel]).assign(&kernel_design);
    if n_poly > 0 {
        phi.slice_mut(s![.., n_kernel..]).assign(&poly_design);
    }

    // Input-location first jet, scaled by the *same* `α` on the kernel block.
    let radial_first = duchon_radial_first_derivative_nd(t, centers, None, effective_order, 0)?;
    let radial_jet = radial_input_location_jet_nd(t, centers, radial_first.view())?;
    let poly_jet = duchon_polynomial_first_derivative_nd(t, effective_order);
    if poly_jet.shape()[1] != n_poly {
        crate::bail_dim_basis!(
            "duchon_sae_atom_basis_with_jet: polynomial jet has {} columns but design has {n_poly}",
            poly_jet.shape()[1]
        );
    }
    let mut jet = Array3::<f64>::zeros((n_rows, n_kernel + n_poly, dim));
    for axis in 0..dim {
        let projected = radial_jet.index_axis(Axis(2), axis).dot(&z);
        let mut block = jet.slice_mut(s![.., ..n_kernel, axis]);
        block.assign(&projected);
        block *= kernel_amp;
    }
    jet.slice_mut(s![.., n_kernel.., ..]).assign(&poly_jet);

    Ok((phi, jet))
}

/// Reproducing-norm smoothness penalty for the scale-free Duchon SAE atom,
/// consistent column-for-column with [`duchon_sae_atom_basis_with_jet`].
///
/// The atom basis is `[ (Φ_radial·α)·Z | P ]` (the `Z`-constrained, `α`-amplified
/// kernel block followed by the polynomial nullspace `P`). The native
/// reproducing (bending-energy) norm of a function in this basis is
/// `ω = α²·Zᵀ K_CC Z` on the constrained kernel coordinates, with the polynomial
/// nullspace unpenalized — so the `(n_kernel + n_poly)²` penalty is
/// `blockdiag(α²·Zᵀ K_CC Z, 0)`.
///
/// Unlike the design-path `build_duchon_basis`, this does NOT run the TPRS
/// generalized-eigen reparameterization / near-null mode dropping (#1347): the
/// SAE atom keeps ALL `m` evaluator columns (the evaluator re-evaluates Φ at
/// updated latent coords every inner step, so its width is fixed at `m`), and
/// the SAE-specific arc-length reweighting in
/// `SaeManifoldAtom::refresh_intrinsic_smooth_penalty` plays the metric role
/// TPRS plays on the design path. With coincident/duplicate seed centers (the
/// over-complete large-K regime) `Zᵀ K_CC Z` is merely rank-deficient — its
/// degenerate directions get ~zero penalty, which the inner solve's per-row
/// Tikhonov ridge conditions — rather than changing the basis width and
/// desyncing `phi` (width `m`) from the penalty (the #1221/#1026 32K bug).
pub fn duchon_sae_atom_penalty(
    centers: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Result<Array2<f64>, BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_sae_atom_penalty: centers must have at least one column"
        );
    }
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = 0.0;
    let poly_block_centers = polynomial_block_from_order(centers, effective_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();
    let n_poly = poly_block_centers.ncols();
    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    );
    // Center-to-center reproducing kernel Gram `K_CC[i,j] = φ(|c_i − c_j|)`.
    let n_centers = centers.nrows();
    let mut k_cc = Array2::<f64>::zeros((n_centers, n_centers));
    for i in 0..n_centers {
        for j in 0..n_centers {
            let r = euclidean_distance_rows(centers, i, centers, j);
            k_cc[[i, j]] = pure_poly_coeff.eval(r);
        }
    }
    // ω_kernel = α² · Zᵀ K_CC Z (n_kernel × n_kernel), symmetrized for safety.
    let kz = fast_ab(&k_cc, &z);
    let z_t = z.t().to_owned();
    let ztkz = fast_ab(&z_t, &kz);
    let amp2 = kernel_amp * kernel_amp;
    let m = n_kernel + n_poly;
    let mut penalty = Array2::<f64>::zeros((m, m));
    for a in 0..n_kernel {
        for b in 0..n_kernel {
            penalty[[a, b]] = 0.5 * amp2 * (ztkz[[a, b]] + ztkz[[b, a]]);
        }
    }
    Ok(penalty)
}

/// Second input-location jet of the scale-free Duchon SAE atom (the analytic
/// Hessian `∂²Φ / ∂t_a ∂t_c`), consistent with
/// [`duchon_sae_atom_basis_with_jet`] column-for-column and `α`-for-`α`.
///
/// The kernel block uses the standard radial Hessian decomposition
///
/// ```text
/// ∂²φ/∂t_a∂t_c = (φ'(r)/r) δ_ac + (φ''(r) − φ'(r)/r) (t−c)_a (t−c)_c / r²,
/// ```
///
/// projected through `Z` and scaled by `α`; the polynomial block carries the
/// monomial Hessian. Output shape `(n_rows, n_kernel + n_poly, dim, dim)`.
pub fn duchon_sae_atom_second_jet(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Result<Array4<f64>, BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_sae_atom_second_jet: centers must have at least one column"
        );
    }
    if t.ncols() != dim {
        crate::bail_dim_basis!(
            "duchon_sae_atom_second_jet: t has {} cols but centers have {dim}",
            t.ncols()
        );
    }
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = 0.0;

    let poly_block_centers = polynomial_block_from_order(centers, effective_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    );

    let n_rows = t.nrows();

    let poly_block_t_cols = polynomial_block_from_order(t, effective_order).ncols();

    // Kernel-block Cartesian Hessian `∂²Φ/∂t_a∂t_c`, projected through `Z` and
    // scaled by `α`, via the shared radial→Cartesian engine. Folding `α` into
    // the per-center coefficient matrix `Z` (so `coeffs = α·Z`, shape
    // `(n_centers, n_kernel)`) makes the shared helper emit the already-
    // amplified, already-projected kernel Hessian directly: its flat
    // `(n_rows, n_kernel·d²)` output places output `i`, multi-index `(a,c)` at
    // column `i·d² + (a·d + c)`, including the `φ''(0) δ_ac` collision limit.
    let coeffs = &z * kernel_amp;
    let flat =
        radial_basis_cartesian_derivative(2, t, centers, coeffs.view(), None, effective_order, 0)?;

    let mut out = Array4::<f64>::zeros((n_rows, n_kernel + poly_block_t_cols, dim, dim));
    for n in 0..n_rows {
        for i in 0..n_kernel {
            for a in 0..dim {
                for c in 0..dim {
                    out[[n, i, a, c]] = flat[[n, i * dim * dim + a * dim + c]];
                }
            }
        }
    }

    // Polynomial Hessian block.
    let exponents = monomial_exponents(
        dim,
        match effective_order {
            DuchonNullspaceOrder::Zero => 0,
            DuchonNullspaceOrder::Linear => 1,
            DuchonNullspaceOrder::Degree(k) => k,
        },
    );
    if exponents.len() != poly_block_t_cols {
        crate::bail_dim_basis!(
            "duchon_sae_atom_second_jet: monomial count {} != polynomial block columns {poly_block_t_cols}",
            exponents.len()
        );
    }
    for (col, alpha) in exponents.iter().enumerate() {
        for a in 0..dim {
            for c in 0..dim {
                // ∂²(Π t_i^{α_i}) / ∂t_a ∂t_c.
                for row in 0..n_rows {
                    let coeff_a = alpha[a];
                    if coeff_a == 0 {
                        continue;
                    }
                    let coeff_c = if a == c {
                        alpha[c].saturating_sub(1)
                    } else {
                        alpha[c]
                    };
                    if a != c && coeff_c == 0 {
                        continue;
                    }
                    let lead = (coeff_a as f64) * (coeff_c as f64);
                    if lead == 0.0 {
                        continue;
                    }
                    let mut value = lead;
                    for axis in 0..dim {
                        let mut exp = alpha[axis];
                        if axis == a {
                            exp = exp.saturating_sub(1);
                        }
                        if axis == c {
                            exp = exp.saturating_sub(1);
                        }
                        if exp != 0 {
                            value *= t[[row, axis]].powi(exp as i32);
                        }
                    }
                    out[[row, n_kernel + col, a, c]] = value;
                }
            }
        }
    }

    Ok(out)
}

/// Third input-location jet of the scale-free Duchon SAE atom (the analytic
/// `∂³Φ / ∂t_a ∂t_c ∂t_e`), consistent with
/// [`duchon_sae_atom_basis_with_jet`] and [`duchon_sae_atom_second_jet`]
/// column-for-column and `α`-for-`α`.
///
/// The kernel block uses the standard radial third-derivative decomposition
///
/// ```text
/// ∂³φ/∂t_a∂t_c∂t_e = a_coef·u_a u_c u_e
///                  + b_coef·(δ_ac u_e + δ_ae u_c + δ_ce u_a),
/// a_coef = φ'''(r) − 3·b_coef,   b_coef = (φ''(r) − φ'(r)/r)/r,   u = (t−c)/r,
/// ```
///
/// projected through `Z` and scaled by `α` (emitted directly by the shared
/// [`radial_basis_cartesian_derivative`] engine at order 3); the polynomial
/// block carries the monomial third derivative. At a coincident point `r = 0`
/// the kernel third jet vanishes (odd-order radial derivative of an even kernel
/// in the collision limit), which the engine encodes by skipping `r == 0`.
/// Output shape `(n_rows, n_kernel + n_poly, dim, dim, dim)`. Symmetric in its
/// three trailing axes by construction.
pub fn duchon_sae_atom_third_jet(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
) -> Result<Array5<f64>, BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "duchon_sae_atom_third_jet: centers must have at least one column"
        );
    }
    if t.ncols() != dim {
        crate::bail_dim_basis!(
            "duchon_sae_atom_third_jet: t has {} cols but centers have {dim}",
            t.ncols()
        );
    }
    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let p_order = duchon_p_from_nullspace_order(effective_order);
    let s_order: f64 = 0.0;

    let poly_block_centers = polynomial_block_from_order(centers, effective_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    let pure_poly_coeff =
        PolyharmonicBlockCoeff::new(pure_duchon_block_order(p_order, s_order), dim);
    let kernel_amp = duchon_kernel_amplification(
        centers,
        None,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        None,
        None,
        Some(&pure_poly_coeff),
    );

    let n_rows = t.nrows();
    let poly_block_t_cols = polynomial_block_from_order(t, effective_order).ncols();

    // Kernel-block Cartesian third jet `∂³Φ/∂t_a∂t_c∂t_e`, projected through `Z`
    // and scaled by `α`, via the shared radial→Cartesian engine. Folding `α`
    // into the per-center coefficient matrix (`coeffs = α·Z`, shape
    // `(n_centers, n_kernel)`) makes the helper emit the already-amplified,
    // already-projected kernel third jet directly: its flat
    // `(n_rows, n_kernel·d³)` output places output `i`, multi-index `(a,c,e)` at
    // column `i·d³ + ((a·d + c)·d + e)`.
    let coeffs = &z * kernel_amp;
    let flat =
        radial_basis_cartesian_derivative(3, t, centers, coeffs.view(), None, effective_order, 0)?;

    let d_pow = dim * dim * dim;
    let mut out = Array5::<f64>::zeros((n_rows, n_kernel + poly_block_t_cols, dim, dim, dim));
    for n in 0..n_rows {
        for i in 0..n_kernel {
            for a in 0..dim {
                for c in 0..dim {
                    for e in 0..dim {
                        out[[n, i, a, c, e]] = flat[[n, i * d_pow + ((a * dim) + c) * dim + e]];
                    }
                }
            }
        }
    }

    // Polynomial third-derivative block: `∂³(Π t_i^{α_i}) / ∂t_a ∂t_c ∂t_e`.
    // Differentiating axis `j` a total of `k_j` times (its multiplicity in
    // `{a, c, e}`) contracts that factor to `falling(α_j, k_j)·t_j^{α_j − k_j}`,
    // with `falling(α, k) = α(α−1)…(α−k+1)`; the term vanishes when `α_j < k_j`.
    let exponents = monomial_exponents(
        dim,
        match effective_order {
            DuchonNullspaceOrder::Zero => 0,
            DuchonNullspaceOrder::Linear => 1,
            DuchonNullspaceOrder::Degree(k) => k,
        },
    );
    if exponents.len() != poly_block_t_cols {
        crate::bail_dim_basis!(
            "duchon_sae_atom_third_jet: monomial count {} != polynomial block columns {poly_block_t_cols}",
            exponents.len()
        );
    }
    let falling = |alpha: usize, k: usize| -> f64 {
        let mut acc = 1.0_f64;
        for j in 0..k {
            acc *= (alpha as f64) - (j as f64);
        }
        acc
    };
    for (col, alpha) in exponents.iter().enumerate() {
        for a in 0..dim {
            if alpha[a] == 0 {
                continue;
            }
            for c in 0..dim {
                for e in 0..dim {
                    // Per-axis differentiation order in this (a, c, e) cell.
                    let mut order = vec![0usize; dim];
                    order[a] += 1;
                    order[c] += 1;
                    order[e] += 1;
                    if (0..dim).any(|axis| order[axis] > alpha[axis]) {
                        continue;
                    }
                    let mut lead = 1.0_f64;
                    for axis in 0..dim {
                        lead *= falling(alpha[axis], order[axis]);
                    }
                    if lead == 0.0 {
                        continue;
                    }
                    for row in 0..n_rows {
                        let mut value = lead;
                        for axis in 0..dim {
                            let exp = alpha[axis] - order[axis];
                            if exp != 0 {
                                value *= t[[row, axis]].powi(exp as i32);
                            }
                        }
                        out[[row, n_kernel + col, a, c, e]] = value;
                    }
                }
            }
        }
    }

    Ok(out)
}

/// Forward design, input-location first jet, and input-location second jet
/// (Hessian) of the **general** Duchon basis — the same matrix
/// [`build_duchon_basis`] / [`build_duchon_basis_mixed_periodicity_auto`]
/// produce for the resolved spec, differentiated **exactly**.
///
/// This is the analytic-derivative companion to the basis-only Python FFI
/// `duchon_basis`. The forward Duchon design is **not** the raw centerwise
/// radial kernel `K(x, C)`; it is
///
/// ```text
/// X(x) = [ α · K(x, C) · Z ,  P(x) ],
/// ```
///
/// where `Z = null(P(C)ᵀ)` is the polynomial-constraint null space, `P(x)`
/// is the monomial nullspace block, and `α` is the kernel amplification the
/// design path applies. The jets returned here are the exact `x`-derivatives
/// of that built matrix, column-for-column:
///
/// ```text
/// J(x)   = [ α · ∂ₓK(x,C) · Z ,  ∂ₓP(x) ],
/// H(x)   = [ α · ∂²ₓK(x,C) · Z , ∂²ₓP(x) ].
/// ```
///
/// Both the non-periodic radial path (pure polyharmonic and hybrid Matérn
/// length-scale, `s_order = power`) and the mixed-periodicity chord-embedding
/// path are handled, matching whichever forward builder the spec selects.
///
/// `t` is `(n_rows, dim)`, `centers` is `(n_centers, dim)`. The returned
/// triple is `(Φ, J, H)` with `Φ` shape `(n_rows, n_kernel + n_poly)`,
/// `J` shape `(n_rows, n_kernel + n_poly, dim)`, `H` shape
/// `(n_rows, n_kernel + n_poly, dim, dim)`.
pub fn build_duchon_basis_design_and_jets(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    periodic_per_axis: &[bool],
    periods: &[f64],
) -> Result<(Array2<f64>, Array3<f64>, Array4<f64>), BasisError> {
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "build_duchon_basis_design_and_jets: centers must have at least one column"
        );
    }
    if t.ncols() != dim {
        crate::bail_dim_basis!(
            "build_duchon_basis_design_and_jets: t has {} cols but centers have {dim}",
            t.ncols()
        );
    }
    if periodic_per_axis.len() != dim {
        crate::bail_invalid_basis!(
            "build_duchon_basis_design_and_jets: periodic_per_axis must have length {dim}, got {}",
            periodic_per_axis.len()
        );
    }
    if periods.len() != dim {
        crate::bail_invalid_basis!(
            "build_duchon_basis_design_and_jets: periods must have length {dim}, got {}",
            periods.len()
        );
    }
    let any_periodic = periodic_per_axis.iter().any(|&p| p);

    let n_rows = t.nrows();
    let n_centers = centers.nrows();

    // ----------------------------------------------------------------- spec
    // The mixed-periodicity forward forces the constraint nullspace to
    // {constants} (the only polynomial periodic on every periodic axis),
    // gates the spectrum to the pure polyharmonic case (length_scale = None,
    // power = 0), and applies NO kernel amplification. The non-periodic
    // forward keeps the (auto-degraded) requested nullspace, the hybrid
    // length-scale / power, and the amplification α. Mirror both exactly.
    if any_periodic {
        if length_scale.is_some() {
            crate::bail_invalid_basis!(
                "mixed-periodicity Duchon basis currently only supports the pure polyharmonic spectrum (length_scale=None)"
            );
        }
        if power != 0.0 {
            crate::bail_invalid_basis!(
                "mixed-periodicity Duchon basis currently requires power = 0 (pure polyharmonic); got power={power}"
            );
        }
        for (j, (&per, &period)) in periodic_per_axis.iter().zip(periods.iter()).enumerate() {
            if per && !(period.is_finite() && period > 0.0) {
                crate::bail_invalid_basis!(
                    "axis {j} is periodic but period={period} is not finite & positive"
                );
            }
        }
    }

    // gam#1422 / gam#1423 — the mixed-periodicity forward now builds the design
    // and penalty from the ADDITIVE per-axis PSD reproducing kernel
    // (`mixed_periodicity_additive_kernel`) with a non-periodic-only polynomial
    // null space. Mirror that EXACTLY here so prediction / position-API jets
    // stay consistent with the realized design (and remain seam-wrapping).
    if any_periodic {
        return mixed_periodicity_additive_design_and_jets(
            t,
            centers,
            nullspace_order,
            periodic_per_axis,
            periods,
        );
    }

    let effective_order = duchon_effective_nullspace_order(centers, nullspace_order);
    let kernel_nullspace_order = if any_periodic {
        DuchonNullspaceOrder::Zero
    } else {
        effective_order
    };
    let p_order = duchon_p_from_nullspace_order(kernel_nullspace_order);
    // For the periodic chord path the kernel smoothness order tracks the
    // user's requested m (carried in `nullspace_order`, not the forced-to-
    // constant kernel-nullspace order), matching the mixed-periodicity forward.
    let kernel_m = if any_periodic {
        duchon_p_from_nullspace_order(nullspace_order)
    } else {
        p_order
    };
    let s_order_int = duchon_power_to_usize(power);
    let s_order_f = power;

    // Polynomial-constraint null space `Z` (same construction as every design
    // path: null of the polynomial side-condition block at the centers).
    let poly_block_centers = polynomial_block_from_order(centers, kernel_nullspace_order);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    // Hybrid partial-fraction coefficients (None ⇒ pure polyharmonic).
    let coeffs = length_scale
        .map(|ls| duchon_partial_fraction_coeffs(p_order, s_order_int, 1.0 / ls.max(1e-300)));
    let pure_poly_coeff = if length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            pure_duchon_block_order(kernel_m, s_order_f),
            dim,
        ))
    } else {
        None
    };

    // Kernel amplification α — identical to the value the design path applies.
    // The mixed-periodicity forward applies no amplification (α = 1).
    let kernel_amp = if any_periodic {
        1.0
    } else {
        duchon_kernel_amplification(
            centers,
            length_scale,
            p_order,
            s_order_int,
            dim,
            None,
            coeffs.as_ref(),
            pure_poly_coeff.as_ref(),
        )
    };

    // --------------------------------------------------- radial value block
    // Per (row, center) metric distance `r` plus the scalar radial value
    // φ(r), first derivative φ'(r), and second derivative φ''(r) of the SAME
    // kernel the forward design evaluates. `metric_axis[(n,k,a)]` is the
    // per-axis chord-embedding partial dδ_a/dx_a and `metric_axis2` its second
    // derivative (zero on linear / non-periodic axes); the displacement
    // `delta_a = δ_a(x_a − c_a)` is the chord (or plain difference) used to
    // contract the radial scalars into the input-location jet.
    let mut radial_value = Array2::<f64>::zeros((n_rows, n_centers));
    let mut radial_first = Array2::<f64>::zeros((n_rows, n_centers));
    let mut radial_second = Array2::<f64>::zeros((n_rows, n_centers));
    // delta[(n,k,a)] = embedded displacement along axis a (chord or plain).
    let mut delta = Array3::<f64>::zeros((n_rows, n_centers, dim));
    // d1[(n,k,a)] = ∂δ_a/∂x_a, d2 = ∂²δ_a/∂x_a²  (chord-embedding metric jets).
    let mut metric_d1 = Array3::<f64>::zeros((n_rows, n_centers, dim));
    let mut metric_d2 = Array3::<f64>::zeros((n_rows, n_centers, dim));

    // Pure-polyharmonic radial scalars φ, φ', φ'' all come from the SAME
    // analytic jet of the polyharmonic block `c · r^(2m_pure − d)` (or its log
    // variant) that defines the forward value, via `polyharmonic_block_jet4`.
    // This is the *exact* derivative of `ppc.eval(r)` — `polyharmonic_block_jet4`
    // and `PolyharmonicBlockCoeff::new` share the identical coefficient `c`,
    // power, and log-case branch — so the returned φ' and φ'' differentiate the
    // forward kernel value column-for-column with no Matérn-regularized
    // surrogate. (The earlier `duchon_radial_jets` path injected a fabricated
    // length scale + partial-fraction coeffs, producing φ', φ'' of a *hybrid*
    // kernel that is NOT the derivative of the pure polyharmonic `ppc.eval`.)
    // The kernel smoothness order `m_pure` is the one that built `pure_poly_coeff`.
    let m_pure = pure_duchon_block_order(kernel_m, s_order_f);

    let pi = std::f64::consts::PI;
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let raw = t[[n, a]] - centers[[k, a]];
                let (d_a, d1_a, d2_a) = if periodic_per_axis[a] {
                    // Chord embedding on the circle of circumference P_a:
                    //   δ_a   = (P/π) sin(π·raw/P)
                    //   δ_a'  = cos(π·raw/P)
                    //   δ_a'' = −(π/P) sin(π·raw/P)
                    let p = periods[a];
                    let theta = pi * raw / p;
                    ((p / pi) * theta.sin(), theta.cos(), -(pi / p) * theta.sin())
                } else {
                    (raw, 1.0, 0.0)
                };
                delta[[n, k, a]] = d_a;
                metric_d1[[n, k, a]] = d1_a;
                metric_d2[[n, k, a]] = d2_a;
                r2 += d_a * d_a;
            }
            let r = r2.sqrt();
            let (phi, phi_r, phi_rr) = if pure_poly_coeff.is_some() {
                // Exact analytic (value, φ', φ'') of the pure polyharmonic block,
                // i.e. the true derivatives of the forward `ppc.eval(r)`.
                polyharmonic_kernel_triplet(r, m_pure, dim)?
            } else {
                let jets = duchon_radial_jets(
                    r,
                    length_scale.expect("hybrid Duchon requires length_scale"),
                    p_order,
                    s_order_int,
                    dim,
                    coeffs.as_ref().expect("hybrid Duchon requires coeffs"),
                )?;
                (jets.phi, jets.phi_r, jets.phi_rr)
            };
            radial_value[[n, k]] = phi * kernel_amp;
            radial_first[[n, k]] = phi_r;
            radial_second[[n, k]] = phi_rr;
        }
    }

    // ---------------------------------------------------- forward kernel block
    let kernel_design = fast_ab(&radial_value, &z);

    // Polynomial forward block P(x).  Periodic ⇒ constant-only column.
    let poly_design = polynomial_block_from_order(t, kernel_nullspace_order);
    let n_poly = poly_design.ncols();

    let mut phi_design = Array2::<f64>::zeros((n_rows, n_kernel + n_poly));
    phi_design
        .slice_mut(s![.., ..n_kernel])
        .assign(&kernel_design);
    if n_poly > 0 {
        phi_design
            .slice_mut(s![.., n_kernel..])
            .assign(&poly_design);
    }

    // -------------------------------------------------- radial input-location
    // jets, contracted through the chord-embedding metric.  For each (n, k):
    //   ∂φ/∂x_a   = φ'(r) · (δ_a · δ_a') / r
    //   ∂²φ/∂x_a∂x_c = (φ''(r) − φ'(r)/r)/r² · (δ_a δ_a')(δ_c δ_c')
    //                + [a == c] · (φ'(r)/r) · ( (δ_a')² + δ_a · δ_a'' )
    // The first form collapses to the standard radial jet when every axis is
    // non-periodic (δ_a = x_a − c_a, δ_a' = 1, δ_a'' = 0).
    let mut radial_jet = Array3::<f64>::zeros((n_rows, n_centers, dim));
    let mut radial_hess = Array4::<f64>::zeros((n_rows, n_centers, dim, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let da = delta[[n, k, a]];
                r2 += da * da;
            }
            let r = r2.sqrt();
            let phi_r = radial_first[[n, k]];
            let phi_rr = radial_second[[n, k]];
            if r <= 1.0e-12 {
                // Collision limit: the radial gradient vanishes and the Hessian
                // is the isotropic φ''(0) δ_ac scaled by (δ_a')² (= 1 at the
                // center for both linear and chord embeddings).
                for a in 0..dim {
                    let d1a = metric_d1[[n, k, a]];
                    radial_hess[[n, k, a, a]] = phi_rr * kernel_amp * d1a * d1a;
                }
                continue;
            }
            let q = phi_r / r;
            let s_scalar = (phi_rr - q) / r2;
            // g_a = δ_a · δ_a' is ∂r/∂x_a · r (the metric-contracted gradient
            // of ½r²).
            for a in 0..dim {
                let ga = delta[[n, k, a]] * metric_d1[[n, k, a]];
                radial_jet[[n, k, a]] = q * ga * kernel_amp;
            }
            for a in 0..dim {
                let ga = delta[[n, k, a]] * metric_d1[[n, k, a]];
                for c in 0..dim {
                    let gc = delta[[n, k, c]] * metric_d1[[n, k, c]];
                    let mut value = s_scalar * ga * gc;
                    if a == c {
                        let d1a = metric_d1[[n, k, a]];
                        let curvature = d1a * d1a + delta[[n, k, a]] * metric_d2[[n, k, a]];
                        value += q * curvature;
                    }
                    radial_hess[[n, k, a, c]] = value * kernel_amp;
                }
            }
        }
    }

    // ---------------------------------------------------- assemble jet (J)
    let poly_jet = duchon_polynomial_first_derivative_nd(t, kernel_nullspace_order);
    if poly_jet.shape()[1] != n_poly {
        crate::bail_dim_basis!(
            "build_duchon_basis_design_and_jets: polynomial jet has {} columns but design has {n_poly}",
            poly_jet.shape()[1]
        );
    }
    let mut jet = Array3::<f64>::zeros((n_rows, n_kernel + n_poly, dim));
    for axis in 0..dim {
        let projected = radial_jet.index_axis(Axis(2), axis).dot(&z);
        jet.slice_mut(s![.., ..n_kernel, axis]).assign(&projected);
    }
    if n_poly > 0 {
        jet.slice_mut(s![.., n_kernel.., ..]).assign(&poly_jet);
    }

    // ---------------------------------------------------- assemble Hessian (H)
    let mut hess = Array4::<f64>::zeros((n_rows, n_kernel + n_poly, dim, dim));
    for a in 0..dim {
        for c in 0..dim {
            let slab = radial_hess.slice(s![.., .., a, c]);
            let projected = slab.dot(&z);
            hess.slice_mut(s![.., ..n_kernel, a, c]).assign(&projected);
        }
    }
    if n_poly > 0 {
        // Polynomial Hessian block: ∂²(Π t_i^{α_i}) / ∂t_a ∂t_c.
        let max_degree = match kernel_nullspace_order {
            DuchonNullspaceOrder::Zero => 0,
            DuchonNullspaceOrder::Linear => 1,
            DuchonNullspaceOrder::Degree(deg) => deg,
        };
        let exponents = monomial_exponents(dim, max_degree);
        if exponents.len() != n_poly {
            crate::bail_dim_basis!(
                "build_duchon_basis_design_and_jets: monomial count {} != polynomial block columns {n_poly}",
                exponents.len()
            );
        }
        for (col, alpha) in exponents.iter().enumerate() {
            for a in 0..dim {
                let coeff_a = alpha[a];
                if coeff_a == 0 {
                    continue;
                }
                for c in 0..dim {
                    let coeff_c = if a == c {
                        alpha[c].saturating_sub(1)
                    } else {
                        alpha[c]
                    };
                    if a != c && coeff_c == 0 {
                        continue;
                    }
                    let lead = (coeff_a as f64) * (coeff_c as f64);
                    if lead == 0.0 {
                        continue;
                    }
                    for row in 0..n_rows {
                        let mut value = lead;
                        for axis in 0..dim {
                            let mut exp = alpha[axis];
                            if axis == a {
                                exp = exp.saturating_sub(1);
                            }
                            if axis == c {
                                exp = exp.saturating_sub(1);
                            }
                            if exp != 0 {
                                value *= t[[row, axis]].powi(exp as i32);
                            }
                        }
                        hess[[row, n_kernel + col, a, c]] = value;
                    }
                }
            }
        }
    }

    Ok((phi_design, jet, hess))
}

/// Design + input-location jets for the ADDITIVE mixed-periodicity Duchon
/// kernel (gam#1422 / gam#1423). The kernel is the sum of per-axis PSD
/// reproducing kernels (periodic Bernoulli on periodic axes, 1-D Sobolev on
/// non-periodic axes); it is separable, so its gradient is per-axis and its
/// Hessian is diagonal. The null space is the polynomials of total degree
/// ``< m`` in the NON-periodic coordinates only. This mirrors
/// `build_duchon_basis_mixed_periodicity` exactly so prediction matches the
/// realized design and the periodic axes wrap cleanly at the seam.
fn mixed_periodicity_additive_design_and_jets(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    nullspace_order: DuchonNullspaceOrder,
    periodic_per_axis: &[bool],
    periods: &[f64],
) -> Result<(Array2<f64>, Array3<f64>, Array4<f64>), BasisError> {
    let dim = centers.ncols();
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let m = duchon_p_from_nullspace_order(nullspace_order);
    let axis_bounds = crate::basis::mixed_periodicity_axis_bounds(centers, periodic_per_axis);

    // Non-periodic-only polynomial null space `Z = null(Pᵀ)`.
    let poly_block_centers =
        crate::basis::mixed_periodicity_nullspace_poly_block(centers, m, periodic_per_axis);
    let z = kernel_constraint_nullspace_from_matrix(poly_block_centers.view())?;
    let n_kernel = z.ncols();

    // Non-periodic axis order + monomial exponents over the non-periodic axes
    // (used for the explicit polynomial columns + their jet/Hessian).
    let nonperiodic_axes: Vec<usize> = (0..dim).filter(|&j| !periodic_per_axis[j]).collect();
    let max_degree = m.saturating_sub(1);
    let exps = monomial_exponents(nonperiodic_axes.len(), max_degree);
    let n_poly = exps.len();

    // --- value / first / second (diagonal) of the additive kernel ---
    let mut value = Array2::<f64>::zeros((n_rows, n_centers));
    let mut grad = Array3::<f64>::zeros((n_rows, n_centers, dim));
    let mut hdiag = Array3::<f64>::zeros((n_rows, n_centers, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let (v, g, h) = crate::basis::mixed_periodicity_additive_kernel_jet(
                t.row(n),
                centers.row(k),
                m,
                periodic_per_axis,
                periods,
                &axis_bounds,
            )?;
            value[[n, k]] = v;
            for a in 0..dim {
                grad[[n, k, a]] = g[a];
                hdiag[[n, k, a]] = h[a];
            }
        }
    }

    // --- design = [value·Z, P(t)] ---
    let kernel_design = fast_ab(&value, &z);
    let poly_design = crate::basis::mixed_periodicity_nullspace_poly_block(t, m, periodic_per_axis);
    if poly_design.ncols() != n_poly {
        crate::bail_dim_basis!(
            "mixed-periodicity additive jets: polynomial block has {} columns but expected {n_poly}",
            poly_design.ncols()
        );
    }
    let mut phi_design = Array2::<f64>::zeros((n_rows, n_kernel + n_poly));
    phi_design
        .slice_mut(s![.., ..n_kernel])
        .assign(&kernel_design);
    if n_poly > 0 {
        phi_design
            .slice_mut(s![.., n_kernel..])
            .assign(&poly_design);
    }

    // --- jet (J): kernel block projects grad by Z; polynomial block is the
    // monomial gradient in the non-periodic coordinates. ---
    let mut jet = Array3::<f64>::zeros((n_rows, n_kernel + n_poly, dim));
    for axis in 0..dim {
        let projected = grad.index_axis(Axis(2), axis).dot(&z);
        jet.slice_mut(s![.., ..n_kernel, axis]).assign(&projected);
    }
    if n_poly > 0 {
        for (col, alpha) in exps.iter().enumerate() {
            for (local_axis, &deriv_axis) in nonperiodic_axes.iter().enumerate() {
                let power = alpha[local_axis];
                if power == 0 {
                    continue;
                }
                for row in 0..n_rows {
                    let mut val = power as f64;
                    for (la, &ax) in nonperiodic_axes.iter().enumerate() {
                        let mut e = alpha[la];
                        if la == local_axis {
                            e = e.saturating_sub(1);
                        }
                        if e != 0 {
                            val *= t[[row, ax]].powi(e as i32);
                        }
                    }
                    jet[[row, n_kernel + col, deriv_axis]] = val;
                }
            }
        }
    }

    // --- Hessian (H): the kernel is separable so its Hessian is DIAGONAL in
    // the axes (∂²K/∂x_a∂x_c = δ_{ac} R_a''). Project the diagonal by Z. The
    // polynomial Hessian is the monomial second derivative in non-periodic
    // coordinates. ---
    let mut hess = Array4::<f64>::zeros((n_rows, n_kernel + n_poly, dim, dim));
    for a in 0..dim {
        let slab = hdiag.index_axis(Axis(2), a);
        let projected = slab.dot(&z);
        hess.slice_mut(s![.., ..n_kernel, a, a]).assign(&projected);
    }
    if n_poly > 0 {
        for (col, alpha) in exps.iter().enumerate() {
            for (la_a, &axis_a) in nonperiodic_axes.iter().enumerate() {
                let coeff_a = alpha[la_a];
                if coeff_a == 0 {
                    continue;
                }
                for (la_c, &axis_c) in nonperiodic_axes.iter().enumerate() {
                    let coeff_c = if la_a == la_c {
                        alpha[la_c].saturating_sub(1)
                    } else {
                        alpha[la_c]
                    };
                    if la_a != la_c && coeff_c == 0 {
                        continue;
                    }
                    let lead = (coeff_a as f64) * (coeff_c as f64);
                    if lead == 0.0 {
                        continue;
                    }
                    for row in 0..n_rows {
                        let mut val = lead;
                        for (la, &ax) in nonperiodic_axes.iter().enumerate() {
                            let mut e = alpha[la];
                            if la == la_a {
                                e = e.saturating_sub(1);
                            }
                            if la == la_c {
                                e = e.saturating_sub(1);
                            }
                            if e != 0 {
                                val *= t[[row, ax]].powi(e as i32);
                            }
                        }
                        hess[[row, n_kernel + col, axis_a, axis_c]] = val;
                    }
                }
            }
        }
    }

    Ok((phi_design, jet, hess))
}

/// N-D Matérn radial first-derivative `φ'(r)` evaluated for every
/// `(row, center)` pair.
///
/// Returns an `(n_rows, n_centers)` matrix whose `(n, k)` entry is the
/// scalar radial derivative `φ'(r_{nk})` of the Matérn kernel,
/// where `r_{nk} = ‖t_n − c_k‖_2`.
///
/// This is the Matérn analogue of [`duchon_radial_first_derivative_nd`].
/// The full per-row gradient is reconstructed at the call site as
/// `∂Φ_{n,k}/∂t_n = φ'(r_{n,k}) · (t_n − c_k) / r_{n,k}` (chain rule of the
/// radial kernel w.r.t. its first argument), reusing
/// [`crate::latent::LatentCoordValues::design_gradient_wrt_t`].
///
/// All radial derivatives are obtained in closed form from the half-integer
/// Matérn polynomial-times-exponential representation; the underlying scalar
/// arithmetic is [`matern_kernel_radial_tripletwith_safe_ratio`].
pub fn matern_radial_first_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
) -> Result<Array2<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "matern_radial_first_derivative_nd: centers must have at least one column".into(),
        );
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "matern_radial_first_derivative_nd: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_centers));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let dv = t[[n, a]] - centers[[k, a]];
                r2 += dv * dv;
            }
            let r = r2.sqrt();
            let (_phi, phi_r, _phi_rr, _ratio) =
                matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?;
            out[[n, k]] = phi_r;
        }
    }
    Ok(out)
}

/// N-D Matérn radial second derivative `φ''(r)` evaluated for every
/// `(row, center)` pair.
///
/// Companion to [`matern_radial_first_derivative_nd`]. Together they give
/// the full input-location Hessian:
///
/// ```text
/// ∂²φ/∂t_i∂t_j = (φ'(r)/r) (δ_ij − u_i u_j) + φ''(r) u_i u_j,
/// ```
/// where `u_a = (t_a − c_a) / r`. At `r = 0`, the limit reduces to the
/// isotropic `φ''(0) δ_ij`.
pub fn matern_radial_second_derivative_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
) -> Result<Array2<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "matern_radial_second_derivative_nd: centers must have at least one column".into(),
        );
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "matern_radial_second_derivative_nd: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    let mut out = Array2::<f64>::zeros((n_rows, n_centers));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let dv = t[[n, a]] - centers[[k, a]];
                r2 += dv * dv;
            }
            let r = r2.sqrt();
            let (_phi, _phi_r, phi_rr, _ratio) =
                matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?;
            out[[n, k]] = phi_rr;
        }
    }
    Ok(out)
}

/// Resolve the per-axis metric weights `w_a = exp(2·ψ_a)` that the Matérn
/// forward design (`build_matern_basis` → `StreamingMaternEvaluator` /
/// `ChunkedKernelDesignOperator`) applies, **bit-for-bit**.
///
/// The forward path centres the supplied anisotropy log-scales through
/// [`centered_aniso_contrasts`] (subtract the mean, zero tiny residuals) and
/// then squares `exp(ψ_a)`. Replicating that exact transform here is what lets
/// the input-location jet/Hessian differentiate the *same* function the forward
/// evaluates under anisotropy. Like the forward design, this is a pure function
/// of the supplied `η`: an explicit all-zero vector yields the isotropic
/// all-ones metric, matching the closed-form isotropic Matérn (#437, #1042).
///
/// `None` (or a 1-D problem, where the centred contrast is a no-op) yields the
/// isotropic all-ones metric.
pub(crate) fn matern_metric_weights(dim: usize, aniso: Option<&[f64]>) -> Vec<f64> {
    match centered_aniso_contrasts(aniso) {
        Some(psi) => psi.iter().map(|&v| (2.0 * v).exp()).collect(),
        None => vec![1.0; dim],
    }
}

/// N-D Matérn input-location **jet** `∂Φ/∂t` under the anisotropic metric.
///
/// Returns an `(n_rows, n_centers, dim)` tensor whose `(n, k, a)` entry is the
/// exact partial derivative of the (un-projected) kernel value
/// `Φ_{n,k} = φ(r_A)` w.r.t. the input coordinate `t_{n,a}`, where the
/// anisotropic radius is `r_A = √(Σ_b w_b (t_b − c_b)²)` with the forward
/// metric weights `w_b` from [`matern_metric_weights`]:
///
/// ```text
/// ∂Φ_{n,k}/∂t_{n,a} = φ'(r_A) · w_a (t_{n,a} − c_{k,a}) / r_A.
/// ```
///
/// At `r_A = 0` the kernel is at a smooth maximum and the jet is exactly `0`.
/// This is the metric-aware analogue of pairing
/// [`matern_radial_first_derivative_nd`] with `(t − c)/r`; combining the two
/// isotropic helpers ignores `w_a` and therefore differentiates a *different*
/// function whenever anisotropy is active (issue #437).
pub fn matern_input_location_jet_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array3<f64>, BasisError> {
    let n_rows = t.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "matern_input_location_jet_nd: centers must have at least one column".into(),
        );
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "matern_input_location_jet_nd: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    if let Some(eta) = aniso_log_scales
        && eta.len() != dim
    {
        crate::bail_invalid_basis!(
            "matern_input_location_jet_nd: aniso_log_scales len {} != dim {}",
            eta.len(),
            dim
        );
    }
    // Match the forward Matérn basis builder's realized column geometry: an
    // over-specified center set is RRQR-reduced before the design matrix is
    // formed. Input-location jets must differentiate those same realized basis
    // columns; evaluating all pre-reduction centers desynchronizes derivative
    // column indices from finite differences of the forward design.
    let reduced_centers =
        matern_rank_reduce_centers(t, &centers.to_owned(), length_scale, nu, aniso_log_scales)?;
    let centers = reduced_centers.view();
    let n_centers = centers.nrows();
    let weights = matern_metric_weights(dim, aniso_log_scales);
    let mut out = Array3::<f64>::zeros((n_rows, n_centers, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let h = t[[n, a]] - centers[[k, a]];
                r2 += weights[a] * h * h;
            }
            let r = r2.sqrt();
            if r <= 0.0 {
                continue;
            }
            let (_phi, phi_r, _phi_rr, _ratio) =
                matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?;
            let scale = phi_r / r;
            for a in 0..dim {
                let h = t[[n, a]] - centers[[k, a]];
                out[[n, k, a]] = scale * weights[a] * h;
            }
        }
    }
    Ok(out)
}

/// N-D Matérn input-location **Hessian** `∂²Φ/∂t∂tᵀ` under the anisotropic
/// metric.
///
/// Returns an `(n_rows, n_centers, dim, dim)` tensor whose `(n, k, a, c)`
/// entry is the exact second partial of `Φ_{n,k} = φ(r_A)` (same `r_A` and
/// forward metric weights `w` as [`matern_input_location_jet_nd`]):
///
/// ```text
/// H_{ac} = φ''(r_A) · (w_a h_a / r_A)(w_c h_c / r_A)
///        + (φ'(r_A)/r_A) · (w_a δ_{ac} − w_a h_a w_c h_c / r_A²),
/// ```
/// with `h = t − c`. At `r_A = 0` the smooth limit collapses to the diagonal
/// `(φ'/r)|_0 · w_a δ_{ac}` (the regularized ratio from
/// [`matern_kernel_radial_tripletwith_safe_ratio`], which equals `φ''(0)` for
/// ν ≥ 3/2 and carries the genuine ν = 1/2 singularity floor).
pub fn matern_input_location_hessian_nd(
    t: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
    nu: MaternNu,
    aniso_log_scales: Option<&[f64]>,
) -> Result<Array4<f64>, BasisError> {
    let n_rows = t.nrows();
    let n_centers = centers.nrows();
    let dim = centers.ncols();
    if dim == 0 {
        crate::bail_invalid_basis!(
            "matern_input_location_hessian_nd: centers must have at least one column".into(),
        );
    }
    if t.ncols() != dim {
        crate::bail_invalid_basis!(
            "matern_input_location_hessian_nd: t has {} cols but centers have {}",
            t.ncols(),
            dim
        );
    }
    if let Some(eta) = aniso_log_scales
        && eta.len() != dim
    {
        crate::bail_invalid_basis!(
            "matern_input_location_hessian_nd: aniso_log_scales len {} != dim {}",
            eta.len(),
            dim
        );
    }
    let weights = matern_metric_weights(dim, aniso_log_scales);
    let mut out = Array4::<f64>::zeros((n_rows, n_centers, dim, dim));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..dim {
                let h = t[[n, a]] - centers[[k, a]];
                r2 += weights[a] * h * h;
            }
            let r = r2.sqrt();
            let (_phi, phi_r, phi_rr, phi_r_over_r) =
                matern_kernel_radial_tripletwith_safe_ratio(r, length_scale, nu)?;
            if r <= 0.0 {
                // Smooth collision limit: gradient component `w_a h_a / r` → 0,
                // so only the diagonal `(φ'/r)·w_a` term survives.
                for a in 0..dim {
                    out[[n, k, a, a]] = phi_r_over_r * weights[a];
                }
                continue;
            }
            let q = phi_r / r; // = φ'(r)/r at this r > 0.
            let inv_r2 = 1.0 / r2;
            for a in 0..dim {
                let ga = weights[a] * (t[[n, a]] - centers[[k, a]]); // W h, component a
                for c in 0..dim {
                    let gc = weights[c] * (t[[n, c]] - centers[[k, c]]);
                    // φ''(r)·(W h/r)_a (W h/r)_c
                    let mut value = phi_rr * (ga / r) * (gc / r);
                    // (φ'/r)·(−w_a h_a w_c h_c / r²)
                    value -= q * ga * gc * inv_r2;
                    if a == c {
                        // (φ'/r)·w_a δ_ac
                        value += q * weights[a];
                    }
                    out[[n, k, a, c]] = value;
                }
            }
        }
    }
    Ok(out)
}

/// N-D Sobolev-sphere first-derivative jet `∂Φ/∂t` per row, on the unit
/// sphere `S^{dim−1}`.
///
/// `points` is `(n_rows, dim)` ambient unit vectors `t_n ∈ ℝ^dim`,
/// `centers` is `(n_centers, dim)` ambient unit vectors `c_k`. The kernel
/// is `K(cos γ)` with `cos γ = t · c`, and the chain rule gives
///
/// ```text
///     ∂Φ_{n,k} / ∂t_n = K'(cos γ_{n,k}) · c_k,
/// ```
///
/// where `K'` is `dK/d(cos γ)` from
/// [`wahba_sphere_kernel_sobolev_derivative_dcos`].
///
/// When `project_to_tangent` is `true`, each per-row gradient is projected
/// through [`crate::latent::LatentManifold::Sphere`] onto
/// `T_{t_n} S^{dim-1}` as `g − (g · t_n) t_n`, which is the correct
/// Riemannian input-location derivative for embedded-sphere latent updates.
/// Passing `false` returns the un-projected ambient jet.
///
/// Returned tensor shape: `(n_rows, n_centers, dim)`.
pub fn sphere_first_derivative_nd(
    points: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    project_to_tangent: bool,
) -> Result<Array3<f64>, BasisError> {
    let n_rows = points.nrows();
    let n_centers = centers.nrows();
    let dim = points.ncols();
    if !(1..=4).contains(&penalty_order) {
        crate::bail_invalid_basis!(
            "sphere_first_derivative_nd: penalty_order must be in 1..=4; got {penalty_order}"
        );
    }
    if dim == 0 {
        crate::bail_invalid_basis!(
            "sphere_first_derivative_nd: points must have at least one column".into(),
        );
    }
    if centers.ncols() != dim {
        crate::bail_invalid_basis!(
            "sphere_first_derivative_nd: points have dim {} but centers have dim {}",
            dim,
            centers.ncols()
        );
    }
    let tangent_projector =
        project_to_tangent.then_some(crate::latent::LatentManifold::Sphere { dim });
    let mut out = Array3::<f64>::zeros((n_rows, n_centers, dim));
    let mut ambient = Array1::<f64>::zeros(dim);
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut cos_g = 0.0_f64;
            for a in 0..dim {
                cos_g += points[[n, a]] * centers[[k, a]];
            }
            let dk = wahba_sphere_kernel_sobolev_derivative_dcos(cos_g, penalty_order);
            for a in 0..dim {
                ambient[a] = dk * centers[[k, a]];
            }
            if let Some(manifold) = &tangent_projector {
                let tangent = manifold.project_to_tangent(points.row(n), ambient.view());
                for a in 0..dim {
                    out[[n, k, a]] = tangent[a];
                }
            } else {
                for a in 0..dim {
                    out[[n, k, a]] = ambient[a];
                }
            }
        }
    }
    Ok(out)
}

/// Raw (pre-identifiability) Wahba sphere DESIGN jet `∂Φ_raw/∂(lat, lon)`.
///
/// `data` is `(N, 2)` lat/lon, `centers` is `(K, 2)` lat/lon, both in the same
/// angular convention selected by `radians`. Returns `(N, K, 2)` where the
/// last axis is `(∂col/∂lat, ∂col/∂lon)` in the SAME angular units as the
/// input — i.e. the radian-space derivative scaled by `deg = radians ? 1 :
/// π/180`.
///
/// With `cos γ = sinφ sinφc + cosφ cosφc cos(ψ − ψc)` (φ, ψ in radians):
///   ∂cosγ/∂φ = cosφ sinφc − sinφ cosφc cos(ψ − ψc),
///   ∂cosγ/∂ψ = −cosφ cosφc sin(ψ − ψc),
/// and ∂Φ/∂φ = K'(cosγ)·∂cosγ/∂φ, ∂Φ/∂ψ = K'(cosγ)·∂cosγ/∂ψ. The raw-radian
/// derivatives are multiplied by `deg` to express them per raw input unit.
pub(crate) fn spherical_wahba_kernel_jet_with_kind(
    data: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    penalty_order: usize,
    radians: bool,
    kernel: SphereWahbaKernel,
) -> Result<Array3<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical spline jet data", radians)?;
    validate_lat_lon_matrix(centers, "spherical spline jet centers", radians)?;
    if !(1..=4).contains(&penalty_order) {
        crate::bail_invalid_basis!(
            "spherical spline jet penalty_order must be one of 1, 2, 3, 4; got {penalty_order}"
        );
    }
    let n = data.nrows();
    let k = centers.nrows();
    let deg = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    let mut sin_lat_c = Vec::<f64>::with_capacity(k);
    let mut cos_lat_c = Vec::<f64>::with_capacity(k);
    let mut sin_lon_c = Vec::<f64>::with_capacity(k);
    let mut cos_lon_c = Vec::<f64>::with_capacity(k);
    for c in centers.outer_iter() {
        let (s_lat, c_lat) = (c[0] * deg).sin_cos();
        let (s_lon, c_lon) = (c[1] * deg).sin_cos();
        sin_lat_c.push(s_lat);
        cos_lat_c.push(c_lat);
        sin_lon_c.push(s_lon);
        cos_lon_c.push(c_lon);
    }
    let mut out = Array3::<f64>::zeros((n, k, 2));
    let err_flag = std::sync::atomic::AtomicBool::new(false);
    out.axis_chunks_iter_mut(ndarray::Axis(0), 256)
        .into_par_iter()
        .enumerate()
        .for_each(|(chunk_idx, mut block)| {
            let row_offset = chunk_idx * 256;
            for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                let i = row_offset + local_i;
                let (sin_lat, cos_lat) = (data[(i, 0)] * deg).sin_cos();
                let (sin_lon, cos_lon) = (data[(i, 1)] * deg).sin_cos();
                for j in 0..k {
                    // cos(ψ − ψc) and sin(ψ − ψc) via angle-subtraction.
                    let dlon_cos = cos_lon * cos_lon_c[j] + sin_lon * sin_lon_c[j];
                    let dlon_sin = sin_lon * cos_lon_c[j] - cos_lon * sin_lon_c[j];
                    let cos_gamma = sin_lat * sin_lat_c[j] + cos_lat * cos_lat_c[j] * dlon_cos;
                    let dk =
                        wahba_sphere_kernel_derivative_dcos_kind(cos_gamma, penalty_order, kernel);
                    // ∂cosγ/∂φ and ∂cosγ/∂ψ (radian space).
                    let dcos_dphi = cos_lat * sin_lat_c[j] - sin_lat * cos_lat_c[j] * dlon_cos;
                    let dcos_dpsi = -cos_lat * cos_lat_c[j] * dlon_sin;
                    let dphi = dk * dcos_dphi * deg;
                    let dpsi = dk * dcos_dpsi * deg;
                    if !dphi.is_finite() || !dpsi.is_finite() {
                        err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                    out_row[[j, 0]] = dphi;
                    out_row[[j, 1]] = dpsi;
                }
            }
        });
    if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
        crate::bail_invalid_basis!("spherical spline kernel jet produced a non-finite value");
    }
    Ok(out)
}

/// Apply the `(K × K')` identifiability transform `z` to a raw Wahba jet
/// `(N, K, 2)`, producing the realized-design jet `(N, K', 2)` whose column
/// `c` aligns with column `c` of `raw_design.dot(z)`. The transform is linear
/// in the kernel columns, so `∂(Φ_raw z)/∂t = (∂Φ_raw/∂t) z` axis-by-axis.
pub(crate) fn apply_identifiability_to_jet(raw_jet: &Array3<f64>, z: &Array2<f64>) -> Array3<f64> {
    let n = raw_jet.shape()[0];
    let k = raw_jet.shape()[1];
    let kp = z.ncols();
    assert_eq!(
        z.nrows(),
        k,
        "apply_identifiability_to_jet: identifiability transform rows ({}) must match raw jet basis dim ({})",
        z.nrows(),
        k
    );
    let mut out = Array3::<f64>::zeros((n, kp, 2));
    for axis in 0..2 {
        // raw_axis: (N, K); out_axis = raw_axis · z → (N, K').
        let raw_axis: ndarray::ArrayView2<'_, f64> = raw_jet.index_axis(ndarray::Axis(2), axis);
        let projected = raw_axis.dot(z);
        out.slice_mut(ndarray::s![.., .., axis]).assign(&projected);
    }
    out
}

/// Real-spherical-harmonic DESIGN jet `∂Φ/∂(lat, lon)`, shape `(N, p, 2)` with
/// `p = L(L+2)` and column order matching [`fill_real_spherical_harmonics_row`].
///
/// With `x = sinφ`, column `= N_{lm}·T_m(ψ)·P_{lm}(x)` where `T_m` is
/// `sin(mψ)`, `1`, or `cos(mψ)`:
///   ∂col/∂φ = N_{lm}·T_m(ψ)·P'_{lm}(x)·cosφ   (dx/dφ = cosφ),
///   ∂col/∂ψ = N_{lm}·T'_m(ψ)·P_{lm}(x)         (T' = m cos(mψ), 0, −m sin(mψ)).
/// `P'_{lm}(x)` from `(1 − x²) P'_{lm}(x) = −l x P_{lm}(x) + (l+m) P_{l−1,m}(x)`,
/// with the forward's latitude clamp and `somx2` floor reused for the poles.
/// The radian-space derivatives are scaled by `deg` to per-raw-unit values.
pub(crate) fn spherical_harmonic_jet(
    data: ArrayView2<'_, f64>,
    max_degree: usize,
    radians: bool,
) -> Result<Array3<f64>, BasisError> {
    validate_lat_lon_matrix(data, "spherical-harmonic jet", radians)?;
    if max_degree < 1 {
        crate::bail_invalid_basis!("spherical-harmonic jet max_degree must be >= 1");
    }
    if max_degree > 32 {
        crate::bail_invalid_basis!(
            "spherical-harmonic jet max_degree {max_degree} too large; cap is 32"
        );
    }
    let n = data.nrows();
    let p = max_degree * (max_degree + 2);
    let deg = if radians {
        1.0
    } else {
        std::f64::consts::PI / 180.0
    };
    let norms = precompute_harmonic_norms(max_degree);
    let l_cap = max_degree + 1;
    let mut out = Array3::<f64>::zeros((n, p, 2));
    let idx = |l: usize, m: usize| l * l_cap + m;
    {
        let mut row_blocks = out
            .axis_chunks_iter_mut(ndarray::Axis(0), 1024)
            .collect::<Vec<_>>();
        let chunk_size = 1024usize;
        row_blocks
            .par_iter_mut()
            .enumerate()
            .for_each(|(chunk_idx, block)| {
                let mut p_buf = vec![0.0_f64; l_cap * l_cap];
                let row_offset = chunk_idx * chunk_size;
                for (local_i, mut out_row) in block.outer_iter_mut().enumerate() {
                    let i = row_offset + local_i;
                    let lat_raw = data[(i, 0)] * deg;
                    let lat =
                        lat_raw.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
                    let lon = data[(i, 1)] * deg;
                    let cos_lat = lat.cos();
                    let x = lat.sin();
                    let somx2 = (1.0 - x * x).max(0.0).sqrt();
                    let one_minus_x2 = (1.0 - x * x).max(f64::EPSILON);
                    // Associated Legendre P_{l,m}(x) — identical recurrence to
                    // `fill_real_spherical_harmonics_row`.
                    for slot in p_buf.iter_mut() {
                        *slot = 0.0;
                    }
                    p_buf[idx(0, 0)] = 1.0;
                    for m in 1..=max_degree {
                        p_buf[idx(m, m)] = -((2 * m - 1) as f64) * somx2 * p_buf[idx(m - 1, m - 1)];
                    }
                    for m in 0..max_degree {
                        p_buf[idx(m + 1, m)] = ((2 * m + 1) as f64) * x * p_buf[idx(m, m)];
                    }
                    for m in 0..=max_degree {
                        for l in (m + 2)..=max_degree {
                            p_buf[idx(l, m)] = (((2 * l - 1) as f64) * x * p_buf[idx(l - 1, m)]
                                - ((l + m - 1) as f64) * p_buf[idx(l - 2, m)])
                                / ((l - m) as f64);
                        }
                    }
                    // P'_{l,m}(x) via (1 − x²) P'_{l,m} = −l x P_{l,m} + (l+m) P_{l−1,m}.
                    let dp = |l: usize, m: usize| -> f64 {
                        let p_lm1 = if l >= 1 { p_buf[idx(l - 1, m)] } else { 0.0 };
                        (-(l as f64) * x * p_buf[idx(l, m)] + ((l + m) as f64) * p_lm1)
                            / one_minus_x2
                    };
                    // sin(mψ), cos(mψ) via Chebyshev recurrence (mirror forward).
                    let (sin1, cos1) = lon.sin_cos();
                    let mut sin_buf = [0.0_f64; 33];
                    let mut cos_buf = [0.0_f64; 33];
                    sin_buf[0] = 0.0;
                    cos_buf[0] = 1.0;
                    if max_degree >= 1 {
                        sin_buf[1] = sin1;
                        cos_buf[1] = cos1;
                    }
                    let two_cos1 = 2.0 * cos1;
                    for m in 2..=max_degree {
                        sin_buf[m] = two_cos1 * sin_buf[m - 1] - sin_buf[m - 2];
                        cos_buf[m] = two_cos1 * cos_buf[m - 1] - cos_buf[m - 2];
                    }
                    let mut col = 0usize;
                    for l in 1..=max_degree {
                        // sin(mψ) columns for m = l, l-1, ..., 1.
                        for m_pos in (1..=l).rev() {
                            let nlm = norms[idx(l, m_pos)];
                            let mf = m_pos as f64;
                            // ∂/∂φ = N·sin(mψ)·P'·cosφ ; ∂/∂ψ = N·m cos(mψ)·P.
                            out_row[[col, 0]] = nlm * sin_buf[m_pos] * dp(l, m_pos) * cos_lat * deg;
                            out_row[[col, 1]] =
                                nlm * mf * cos_buf[m_pos] * p_buf[idx(l, m_pos)] * deg;
                            col += 1;
                        }
                        // m = 0: no trig factor → ∂/∂ψ = 0.
                        let nl0 = norms[idx(l, 0)];
                        out_row[[col, 0]] = nl0 * dp(l, 0) * cos_lat * deg;
                        out_row[[col, 1]] = 0.0;
                        col += 1;
                        // cos(mψ) columns for m = 1, ..., l.
                        for m in 1..=l {
                            let nlm = norms[idx(l, m)];
                            let mf = m as f64;
                            // ∂/∂φ = N·cos(mψ)·P'·cosφ ; ∂/∂ψ = −N·m sin(mψ)·P.
                            out_row[[col, 0]] = nlm * cos_buf[m] * dp(l, m) * cos_lat * deg;
                            out_row[[col, 1]] = -nlm * mf * sin_buf[m] * p_buf[idx(l, m)] * deg;
                            col += 1;
                        }
                    }
                }
            });
    }
    if out.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_basis!("spherical-harmonic jet produced a non-finite value");
    }
    Ok(out)
}

/// Realized-design DESIGN jet `∂Φ/∂(lat, lon)` for the spherical-spline basis,
/// matching the column layout of [`build_spherical_spline_basis`] with the
/// given `spec`. Returns `(N, K, 2)` where `K` equals the forward design's
/// column count and the last axis is `(∂col/∂lat, ∂col/∂lon)` in the same
/// angular units as the raw input.
///
/// - **Harmonic** (`spec.method == Harmonic`): `K = L(L+2)`, no transform.
/// - **Wahba** (Sobolev/Pseudo/truncated): centers are resolved exactly as the
///   forward does, the raw `(N, K_c, 2)` kernel jet is built, then contracted
///   with the same identity-or-frozen transform `z` so the result aligns
///   column-for-column with `raw_design · z`.
pub fn spherical_spline_design_jet(
    data: ArrayView2<'_, f64>,
    spec: &SphericalSplineBasisSpec,
) -> Result<Array3<f64>, BasisError> {
    if matches!(spec.method, SphereMethod::Harmonic) {
        let l_max = spec
            .max_degree
            .unwrap_or_else(|| default_spherical_harmonic_degree(data.nrows()));
        if !(1..=4).contains(&spec.penalty_order) {
            crate::bail_invalid_basis!(
                "spherical-harmonic jet penalty_order must be one of 1, 2, 3, 4; got {}",
                spec.penalty_order
            );
        }
        return spherical_harmonic_jet(data, l_max, spec.radians);
    }
    // The Pseudo Wahba kernel forward build routes through the harmonic basis
    // (see `build_spherical_spline_basis`), so its design jet must mirror that
    // routing — degree-mapped harmonic columns — rather than the raw Wahba
    // kernel jet, or the analytic jet would have a different column count than
    // the forward design it is meant to differentiate.
    if matches!(spec.wahba_kernel, SphereWahbaKernel::Pseudo) {
        let l_max = spec
            .max_degree
            .unwrap_or_else(|| harmonic_degree_for_wahba_basis_width(spec, data.nrows()));
        return spherical_harmonic_jet(data, l_max, spec.radians);
    }
    validate_lat_lon_matrix(data, "spherical spline jet", spec.radians)?;
    let centers = match realized_center_strategy(&spec.center_strategy) {
        CenterStrategy::FarthestPoint { num_centers } => {
            select_spherical_farthest_point_centers(data, *num_centers, spec.radians)?
        }
        _ => select_centers_by_strategy(data, &spec.center_strategy)?,
    };
    validate_lat_lon_matrix(centers.view(), "spherical spline jet centers", spec.radians)?;
    if centers.nrows() < 2 {
        return Err(BasisError::InsufficientColumnsForConstraint {
            found: centers.nrows(),
        });
    }
    // The Wahba (Sobolev) forward design is the low-degree *decomposed* design,
    // not the raw kernel design: `[ raw_kernel·kernel_basis −
    // low·kernel_low_projection | low ]` (see `build_wahba_decomposed_design`).
    // Build the matching decomposed jet so it aligns column-for-column with the
    // forward design (width = decomposed width, not centers.nrows()).
    let center_kernel = spherical_wahba_kernel_matrix_with_kind(
        centers.view(),
        centers.view(),
        spec.penalty_order,
        spec.radians,
        spec.wahba_kernel,
    )?;
    let decomposition =
        wahba_low_degree_decomposition(centers.view(), spec.radians, center_kernel.view())?;
    let raw_kernel_jet = spherical_wahba_kernel_jet_with_kind(
        data,
        centers.view(),
        spec.penalty_order,
        spec.radians,
        spec.wahba_kernel,
    )?;
    let low_jet = if decomposition.low_degree_centers.is_some() {
        Some(spherical_harmonic_jet(
            data,
            SPHERE_UNPENALIZED_LOW_DEGREE,
            spec.radians,
        )?)
    } else {
        None
    };
    let decomposed_jet =
        build_wahba_decomposed_jet(&raw_kernel_jet, low_jet.as_ref(), &decomposition);
    let raw_width = decomposed_jet.shape()[1];
    let z = match &spec.identifiability {
        SphericalSplineIdentifiability::FrozenTransform { transform } => {
            if transform.nrows() != raw_width {
                crate::bail_dim_basis!(
                    "frozen spherical identifiability transform mismatch: {} raw basis columns but transform has {} rows",
                    raw_width,
                    transform.nrows()
                );
            }
            transform.clone()
        }
        SphericalSplineIdentifiability::CenterSumToZero => Array2::<f64>::eye(raw_width),
    };
    Ok(apply_identifiability_to_jet(&decomposed_jet, &z))
}
