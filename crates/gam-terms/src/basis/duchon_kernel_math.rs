use crate::smooth::center_aniso_log_scales;
use super::*;

pub fn build_duchon_collocation_operator_matrices(
    centers: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    max_operator_derivative_order: usize,
) -> Result<CollocationOperatorMatrices, BasisError> {
    let mut workspace = BasisWorkspace::default();
    build_duchon_collocation_operator_matriceswithworkspace(
        centers,
        centers,
        collocationweights,
        length_scale,
        power,
        nullspace_order,
        aniso_log_scales,
        identifiability_transform,
        max_operator_derivative_order,
        None,
        &mut workspace,
    )
}

pub fn build_duchon_operator_penalty_matrices(
    centers: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
) -> Result<DuchonOperatorPenaltyMatrices, BasisError> {
    let ops = build_duchon_collocation_operator_matrices(
        centers,
        collocationweights,
        length_scale,
        power,
        nullspace_order,
        aniso_log_scales,
        identifiability_transform,
        2,
    )?;
    let (mass, _) = normalize_penalty(&symmetrize(&fast_ata(&ops.d0)));
    let (tension, _) = normalize_penalty(&symmetrize(&fast_ata(&ops.d1)));
    let (stiffness, _) = normalize_penalty(&symmetrize(&fast_ata(&ops.d2)));
    Ok(DuchonOperatorPenaltyMatrices {
        mass,
        tension,
        stiffness,
    })
}

pub fn build_thin_plate_penalty_matrix(
    centers: ArrayView2<'_, f64>,
    length_scale: f64,
) -> Result<ThinPlatePenaltyMatrix, BasisError> {
    let mut workspace = BasisWorkspace::default();
    let kernel_transform = thin_plate_kernel_constraint_nullspace(centers, &mut workspace.cache)?;
    let (penalty, _) =
        build_thin_plate_penalty_matrices(centers, length_scale, &kernel_transform, false)?;
    let (penalty, _) = normalize_penalty(&penalty);
    Ok(ThinPlatePenaltyMatrix { penalty })
}

pub fn build_duchon_collocation_operator_matriceswithworkspace(
    centers: ArrayView2<'_, f64>,
    collocation_points: ArrayView2<'_, f64>,
    collocationweights: Option<ArrayView1<'_, f64>>,
    length_scale: Option<f64>,
    power: f64,
    nullspace_order: DuchonNullspaceOrder,
    aniso_log_scales: Option<&[f64]>,
    identifiability_transform: Option<ArrayView2<'_, f64>>,
    max_operator_derivative_order: usize,
    radial_reparam: Option<ArrayView2<'_, f64>>,
    workspace: &mut BasisWorkspace,
) -> Result<CollocationOperatorMatrices, BasisError> {
    // The operator design rows are the COLLOCATION points (a density-blind,
    // space-filling sample of the data support); the columns are the `k` basis
    // CENTERS. Decoupling them is what makes the operator penalty a faithful
    // quadrature of `∫‖Dᵠf‖²` (collocating at the `k` centers themselves — the
    // old `collocation_points == centers` special case — under-samples a
    // `k`-bump basis and is what made these penalties explode).
    let nullspace_order = duchon_effective_nullspace_order(centers, nullspace_order);
    // Auto-raise the null-space order (p) so the pointwise kernel and every
    // active derivative-collocation operator clear their well-posedness margin
    // `2(p + s) > d + max_op` BEFORE the guard in
    // `validate_duchon_collocation_orders` can fire. Mirrors the auto-degrade
    // above; only `p` is lifted, so the spectral power and CPD condition are
    // untouched. See `duchon_order_for_operator_margin`.
    let nullspace_order = duchon_order_for_operator_margin(
        centers.ncols(),
        power,
        nullspace_order,
        max_operator_derivative_order,
    );
    let p_order = duchon_p_from_nullspace_order(nullspace_order);
    let s_order: f64 = power;
    let p_colloc = collocation_points.nrows();
    let n_basis = centers.nrows();
    let dim = centers.ncols();
    if collocation_points.ncols() != dim {
        crate::bail_dim_basis!(
            "collocation points dim {} != centers dim {dim}",
            collocation_points.ncols()
        );
    }
    validate_duchon_collocation_orders(
        length_scale,
        p_order,
        s_order,
        dim,
        max_operator_derivative_order,
    )?;
    if let Some(eta) = aniso_log_scales
        && eta.len() != dim
    {
        crate::bail_dim_basis!(
            "Duchon anisotropy dimension mismatch: got {}, expected {dim}",
            eta.len()
        );
    }
    // Partial-fraction expansion only runs in the hybrid Matérn branch
    // (`length_scale = Some`). The scale-free path (`length_scale = None`)
    // skips it entirely and is fractional-clean down to the Riesz kernel.
    let coeffs = length_scale
        .map(|scale| {
            let s_int = duchon_power_to_usize(s_order);
            duchon_inverse_length_scale(scale, "Duchon collocation operator")
                .map(|kappa| duchon_partial_fraction_coeffs(p_order, s_int, kappa))
        })
        .transpose()?;
    let metric_weights: Option<Vec<f64>> = aniso_log_scales.map(centered_aniso_metric_weights);
    // gam#979: the shipped kernel block is `α·K` (`duchon_kernel_chart`), so
    // the operator quadratures of that basis carry the same amplitude. Without
    // it the collocation Grams of a high-dimensional hybrid kernel sit ~30
    // decades below one — the #2627 "max|λ| ≈ 2e-17" design — where every
    // absolute floor downstream (the basis-level penalty normalization and its
    // ψ-derivative, the PSD projection, the factorized-operator rescale) acts on
    // rounding noise instead of on the penalty. This is the SAME `α` the design
    // and the `Primary` penalty already carry, computed from the same call.
    let pure_poly_coeff = if length_scale.is_none() {
        Some(PolyharmonicBlockCoeff::new(
            pure_duchon_block_order(p_order, s_order),
            dim,
        ))
    } else {
        None
    };
    let kernel_amplification = duchon_kernel_amplification(
        centers,
        length_scale,
        p_order,
        duchon_power_to_usize(s_order),
        dim,
        aniso_log_scales,
        coeffs.as_ref(),
        pure_poly_coeff.as_ref(),
    );
    let row_scales = if let Some(w) = collocationweights {
        if w.len() != p_colloc {
            crate::bail_dim_basis!(
                "collocation weight length mismatch: got {}, expected {p_colloc}",
                w.len()
            );
        }
        let mut out = Vec::with_capacity(p_colloc);
        for &wk in w {
            if !wk.is_finite() || wk < 0.0 {
                crate::bail_invalid_basis!(
                    "collocation weights must be finite and non-negative; got {wk}"
                );
            }
            out.push(wk.sqrt());
        }
        out
    } else {
        vec![1.0; p_colloc]
    };
    let mut z = kernel_constraint_nullspace(centers, nullspace_order, &mut workspace.cache)?;
    // #1355 cliff reparam consistency: when the design's constrained kernel
    // columns are rotated into the data-metric generalized eigenbasis
    // (`K·Z·V`), the operator collocation designs D0/D1/D2 must live in the SAME
    // `Z·V` frame or their emitted penalties would penalize the wrong
    // coefficients (a design↔penalty basis desync). Fold the frozen `V` into `Z`
    // here so every operator block is assembled directly in the fit-time
    // `K·Z·V` basis — exactly as the native `Primary` penalty already is. Guard
    // on the column count: `V` was solved against the design's constrained
    // kernel dimension (`Z.ncols()` at the design's null-space order); if the
    // operator margin auto-raised the order the dims differ and `V` does not
    // apply, so the block is left in the raw `Z` frame (the pre-reparam
    // behavior, no regression for those configs).
    if let Some(v) = radial_reparam {
        if v.nrows() == z.ncols() {
            z = fast_ab(&z, &v);
        }
    }
    // D0/D1/D2 rows = collocation points (`p_colloc`), columns = basis centers
    // (`n_basis`). Gradients/Hessians are taken w.r.t. the EVALUATION point
    // (the collocation row), so `delta = collocation - center`. No symmetry: the
    // two point sets differ in general.
    // Skip the costly higher-derivative designs the caller doesn't need: mass
    // (D0) + tension (D1) build with `max_op = 1`, so the `O(d²)`-row Hessian
    // (D2) is never allocated or filled — decisive in high `d`.
    let build_d1 = max_operator_derivative_order >= 1;
    let build_d2 = max_operator_derivative_order >= 2;
    let mut d0_raw = Array2::<f64>::zeros((p_colloc, n_basis));
    let mut d1_raw = Array2::<f64>::zeros((if build_d1 { p_colloc * dim } else { 0 }, n_basis));
    let mut d2_raw =
        Array2::<f64>::zeros((if build_d2 { p_colloc * dim * dim } else { 0 }, n_basis));
    const R_EPS: f64 = 1e-10;
    // The anisotropic metric is fixed for the whole block: its per-axis scales
    // are formed once, and the center rows once, instead of `d` exponentials
    // and two allocations per `(collocation, center)` pair.
    let aniso_scales: Option<Vec<f64>> = aniso_log_scales.map(aniso_axis_scales);
    let center_rows: Vec<Vec<f64>> = (0..n_basis)
        .map(|j| (0..dim).map(|a| centers[[j, a]]).collect())
        .collect();
    let mut row_i: Vec<f64> = vec![0.0; dim];
    for i in 0..p_colloc {
        let scale_i = row_scales[i];
        for (a, slot) in row_i.iter_mut().enumerate() {
            *slot = collocation_points[[i, a]];
        }
        for j in 0..n_basis {
            let r = if let Some(scales) = aniso_scales.as_deref() {
                aniso_distance_with_scales(&row_i, &center_rows[j], scales)
            } else {
                stable_euclidean_norm(
                    (0..dim).map(|axis| collocation_points[[i, axis]] - centers[[j, axis]]),
                )
            };
            // Floor coincident collocation/center pairs off the kernel's origin
            // singularity: a farthest-point sample can land exactly on a center.
            // The gradient/Hessian limits at r→0 are the zeros the `r > R_EPS`
            // guards below already produce, so flooring only avoids the log-case
            // `r²·log r` second-derivative blow-up at exact r=0.
            let r_exact = r;
            let r = r.max(R_EPS);
            let (phi, q, t) = if let (Some(length_scale), Some(coeffs)) =
                (length_scale, coeffs.as_ref())
            {
                // The hybrid jets floor their own derivative core
                // (`DUCHON_DERIVATIVE_R_FLOOR_REL`) and take the VALUE at the
                // exact distance, where `r = 0` is a closed form. Handing them
                // the `R_EPS` floor instead evaluated the stable integral at
                // `r = 1e-10`, which for a large Bessel order (3-D, order 0,
                // power 9: `b = p + s − d/2 = 8.5`) is not the collision value:
                // the mass Gram of a collocation sample that lands on centers
                // was 2.4% off its own definition, and its ψ-derivative 3.7×
                // (gam#979 operator-penalty gate, `opers_3d_order0_power9`).
                let jets = duchon_radial_jets(
                    r_exact,
                    length_scale,
                    p_order,
                    s_order as usize,
                    dim,
                    coeffs,
                )?;
                (jets.phi, jets.q, jets.t)
            } else {
                let (phi, phi_r, phi_rr) = duchon_kernel_radial_triplet(
                    r,
                    length_scale,
                    p_order,
                    s_order,
                    dim,
                    coeffs.as_ref(),
                )?;
                let q = if r > R_EPS { phi_r / r } else { phi_rr };
                let t = if r > R_EPS {
                    (phi_rr - q) / (r * r)
                } else {
                    0.0
                };
                (phi, q, t)
            };
            if !phi.is_finite() || !q.is_finite() || !t.is_finite() {
                crate::bail_invalid_basis!(
                    "non-finite Duchon collocation operator derivative at (colloc {i}, center {j}), r={r}"
                );
            }
            // The chart amplitude multiplies every radial scalar of the block
            // (the kernel value, `q = φ'/r`, `t`), exactly as it multiplies the
            // design's kernel column.
            let (phi, q, t) = (
                phi * kernel_amplification,
                q * kernel_amplification,
                t * kernel_amplification,
            );
            d0_raw[[i, j]] = scale_i * phi;
            if build_d2 {
                for axis_a in 0..dim {
                    let h_a = collocation_points[[i, axis_a]] - centers[[j, axis_a]];
                    let w_a = metric_weights
                        .as_ref()
                        .map(|weights| weights[axis_a])
                        .unwrap_or(1.0);
                    for axis_b in 0..dim {
                        let h_b = collocation_points[[i, axis_b]] - centers[[j, axis_b]];
                        let w_b = metric_weights
                            .as_ref()
                            .map(|weights| weights[axis_b])
                            .unwrap_or(1.0);
                        let diagonal = if axis_a == axis_b { q * w_a } else { 0.0 };
                        let mixed = if r > R_EPS {
                            t * w_a * h_a * w_b * h_b
                        } else {
                            0.0
                        };
                        let value = diagonal + mixed;
                        let row_i = (i * dim + axis_a) * dim + axis_b;
                        d2_raw[[row_i, j]] = scale_i * value;
                    }
                }
            }
            if build_d1 && r > R_EPS {
                for axis in 0..dim {
                    let delta = collocation_points[[i, axis]] - centers[[j, axis]];
                    let axis_scale = metric_weights
                        .as_ref()
                        .map(|weights| weights[axis])
                        .unwrap_or(1.0);
                    d1_raw[[i * dim + axis, j]] = scale_i * q * axis_scale * delta;
                }
            }
        }
    }
    let d0_kernel = fast_ab(&d0_raw, &z);
    let poly = polynomial_block_from_order(centers, nullspace_order);
    let poly_collocation = polynomial_block_from_order(collocation_points, nullspace_order);
    let poly_d1 = if build_d1 {
        polynomial_derivative_block(collocation_points, nullspace_order, 1)
    } else {
        Array2::<f64>::zeros((0, poly.ncols()))
    };
    let poly_d2 = if build_d2 {
        polynomial_derivative_block(collocation_points, nullspace_order, 2)
    } else {
        Array2::<f64>::zeros((0, poly.ncols()))
    };
    let kernel_cols = d0_kernel.ncols();
    let poly_cols = poly.ncols();
    let total_cols = kernel_cols + poly_cols;
    // The operator matrices act on the SAME coefficient basis as the emitted
    // design: constrained radial columns followed by explicit polynomial
    // null-space columns.  The lower-order Hilbert-scale penalties are function
    // penalties, not just radial-kernel penalties, so the polynomial block must
    // be evaluated/differentiated at the collocation sites too: D0 sees the
    // polynomial value, D1 its gradient, and D2 its Hessian. Orders the caller
    // skipped stay empty (0 rows).
    let mut d0 = Array2::<f64>::zeros((p_colloc, total_cols));
    d0.slice_mut(s![.., 0..kernel_cols]).assign(&d0_kernel);
    d0.slice_mut(s![.., kernel_cols..total_cols])
        .assign(&poly_collocation);
    let mut d1 = Array2::<f64>::zeros((if build_d1 { p_colloc * dim } else { 0 }, total_cols));
    if build_d1 {
        d1.slice_mut(s![.., 0..kernel_cols])
            .assign(&fast_ab(&d1_raw, &z));
        d1.slice_mut(s![.., kernel_cols..total_cols])
            .assign(&poly_d1);
    }
    let mut d2 =
        Array2::<f64>::zeros((if build_d2 { p_colloc * dim * dim } else { 0 }, total_cols));
    if build_d2 {
        d2.slice_mut(s![.., 0..kernel_cols])
            .assign(&fast_ab(&d2_raw, &z));
        d2.slice_mut(s![.., kernel_cols..total_cols])
            .assign(&poly_d2);
    }
    if let Some(z) = identifiability_transform {
        let z = z.to_owned();
        d0 = fast_ab(&d0, &z);
        d1 = fast_ab(&d1, &z);
        d2 = fast_ab(&d2, &z);
    }
    Ok(CollocationOperatorMatrices {
        d0,
        d1,
        d2,
        collocation_points: collocation_points.to_owned(),
        kernel_nullspace_transform: Some(z),
        polynomial_block_cols: poly_cols,
        kernel_amplification,
    })
}

pub(crate) fn polynomial_derivative_block(
    points: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
    derivative_order: usize,
) -> Array2<f64> {
    let n = points.nrows();
    let d = points.ncols();
    let degree = match order {
        DuchonNullspaceOrder::Zero => 0,
        DuchonNullspaceOrder::Linear => 1,
        DuchonNullspaceOrder::Degree(degree) => degree,
    };
    let exponents = monomial_exponents(d, degree);
    match derivative_order {
        1 => {
            let mut block = Array2::<f64>::zeros((n * d, exponents.len()));
            for row in 0..n {
                for axis in 0..d {
                    let out_row = row * d + axis;
                    for (col, exps) in exponents.iter().enumerate() {
                        block[[out_row, col]] = monomial_derivative_value(points, row, exps, axis);
                    }
                }
            }
            block
        }
        2 => {
            let mut block = Array2::<f64>::zeros((n * d * d, exponents.len()));
            for row in 0..n {
                for axis_a in 0..d {
                    for axis_b in 0..d {
                        let out_row = (row * d + axis_a) * d + axis_b;
                        for (col, exps) in exponents.iter().enumerate() {
                            block[[out_row, col]] =
                                monomial_second_derivative_value(points, row, exps, axis_a, axis_b);
                        }
                    }
                }
            }
            block
        }
        _ => Array2::<f64>::zeros((0, exponents.len())),
    }
}

fn monomial_derivative_value(
    points: ArrayView2<'_, f64>,
    row: usize,
    exponents: &[usize],
    axis: usize,
) -> f64 {
    let exponent = exponents[axis];
    if exponent == 0 {
        return 0.0;
    }
    let mut value = exponent as f64;
    for a in 0..points.ncols() {
        let power = exponents[a] - usize::from(a == axis);
        if power != 0 {
            value *= points[[row, a]].powi(power as i32);
        }
    }
    value
}

fn monomial_second_derivative_value(
    points: ArrayView2<'_, f64>,
    row: usize,
    exponents: &[usize],
    axis_a: usize,
    axis_b: usize,
) -> f64 {
    let coeff = if axis_a == axis_b {
        let exponent = exponents[axis_a];
        if exponent < 2 {
            return 0.0;
        }
        (exponent * (exponent - 1)) as f64
    } else {
        let exponent_a = exponents[axis_a];
        let exponent_b = exponents[axis_b];
        if exponent_a == 0 || exponent_b == 0 {
            return 0.0;
        }
        (exponent_a * exponent_b) as f64
    };
    let mut value = coeff;
    for axis in 0..points.ncols() {
        let consumed = usize::from(axis == axis_a) + usize::from(axis == axis_b);
        let power = exponents[axis] - consumed;
        if power != 0 {
            value *= points[[row, axis]].powi(power as i32);
        }
    }
    value
}

/// Chebyshev coefficients for `√x·e^x·K₀(x)` on `x ≥ 2`, in ASCENDING powers of
/// `y = 2/x ∈ (0, 1]`, and likewise for `K₁` below.
///
/// The scaled function is analytic on the closed interval (its only branch
/// point, `x = 0`, sits at `y = ∞`), so a Chebyshev projection converges
/// geometrically: 12 terms reach `2.5e−11`, 16 reach `1.1e−13`, 20 reach
/// `8.3e−16`, and the 24 kept here reach `8.4e−18` — comfortably under `f64`,
/// so the shipped error is the Horner evaluation's own `2.8e−16` and not the
/// truncation. Every coefficient is `O(1)` (largest `1.33`, smallest `5.2e−4`),
/// so the monomial-basis Horner is well conditioned despite the degree.
///
/// Regenerate with, at 40 digits:
///
/// ```text
/// f = lambda y: sqrt(2/y) * e**(2/y) * besselk(nu, 2/y)   # f(0) := sqrt(pi/2)
/// chebyfit(f, [0, 1], 24, error=False)                    # descending; reverse
/// ```
///
/// These replace the Abramowitz & Stegun 9.8.6 / 9.8.8 seven-term polynomials,
/// whose stated accuracy is `|ε| < 2e−7` and which measured `1.6e−7` relative
/// here — against a small-`x` branch that is already accurate to `1e−15`, so
/// the pair also had a `2.9e−9` STEP at their `x = 2` crossover. That step was
/// a jump discontinuity in the Matérn/Duchon radial kernel, and therefore in
/// every length-scale derivative taken through it.
const SCALED_BESSEL_K0_CHEBYSHEV: [f64; 24] = [
    1.2533141373155003,
    -0.07833213358220663,
    0.02203091256764185,
    -0.011474433446088833,
    0.008785105521551262,
    -0.008894725254872692,
    0.011207719358405647,
    -0.01687069932280815,
    0.0292835658778568,
    -0.05619383717311583,
    0.11288530340577708,
    -0.22351971988707764,
    0.4132521698842443,
    -0.6841363697170683,
    0.9837481719941347,
    -1.2010795750851755,
    1.2217566634595598,
    -1.0164713197726316,
    0.6772223712126879,
    -0.3515643114800813,
    0.13672561929586632,
    -0.03741713905236645,
    0.00641841764170551,
    -0.0005187103462358833,
];

/// Chebyshev coefficients for `√x·e^x·K₁(x)` on `x ≥ 2`; see
/// [`SCALED_BESSEL_K0_CHEBYSHEV`] for the derivation and the regeneration
/// recipe (same call with `nu = 1`).
const SCALED_BESSEL_K1_CHEBYSHEV: [f64; 24] = [
    1.2533141373155003,
    0.23499640074664313,
    -0.03671818761410955,
    0.01606420688270234,
    -0.011295137230303733,
    0.010871358854537931,
    -0.01324584075315714,
    0.019469483748136365,
    -0.03321119404885756,
    0.06293088720301371,
    -0.12530769564341018,
    0.24664898892370152,
    -0.4542359642973804,
    0.7500446332432126,
    -1.0766417854892274,
    1.3129047408768328,
    -1.3343491820484357,
    1.1094354121788925,
    -0.7388012080869869,
    0.38338722849862483,
    -0.14905729821792707,
    0.0407820842387012,
    -0.006994249846147938,
    0.000565154088568589,
];

/// `e^{−x}/√x · Σ_k c_k y^k` with `y = 2/x`, the common envelope of both
/// large-argument branches. Kept in one place so `K₀` and `K₁` cannot drift
/// apart in how they form it.
#[inline(always)]
fn scaled_bessel_k_large(x: f64, coefficients: &[f64; 24]) -> f64 {
    let y = 2.0 / x;
    let series = coefficients.iter().rev().fold(0.0, |acc, &c| acc * y + c);
    (-x).exp() / x.sqrt() * series
}

// The modified Bessel functions of the second kind have a pole at the origin:
// `K_ν(x) → +∞` as `x → 0⁺`. These primitives return that pole honestly (the
// small-`x` series produces `+∞` at `x = 0`) instead of evaluating at a floored
// `max(x, 1e-300)`, which returned a large finite value that is the kernel of
// no argument the caller passed. Every production caller evaluates at `r = 0`
// through its own exact-limit branch (`origin_limit`, the `r <= 0` returns in
// the Matérn block and the stable integrals, the positive collision radius of
// the radial jets), so a `+∞` reaching a Gram assembly is a leaked collision
// that the finiteness validators refuse — not a value to be hidden (#2469).
#[inline(always)]
pub(crate) fn bessel_k0_stable(x: f64) -> f64 {
    if x <= 2.0 {
        return bessel_k0_small_series(x);
    }
    scaled_bessel_k_large(x, &SCALED_BESSEL_K0_CHEBYSHEV)
}

#[inline(always)]
pub(crate) fn bessel_k1_stable(x: f64) -> f64 {
    if x <= 2.0 {
        return bessel_k1_small_series(x);
    }
    scaled_bessel_k_large(x, &SCALED_BESSEL_K1_CHEBYSHEV)
}

#[inline(always)]
pub(crate) fn bessel_k0_k1_small_series(x: f64) -> (f64, f64) {
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let y = 0.25 * x * x;
    let log_half_plus_gamma = 0.5 * y.ln() + EULER_GAMMA;
    let mut i0 = 1.0;
    let mut i1 = 0.5 * x;
    let mut harmonic = 0.0;
    let mut y_power_over_fact_sq = 1.0;
    let mut k0_series = 0.0;
    let mut k0_series_y_derivative_times_y = 0.0;
    for k in 1..=256 {
        let kf = k as f64;
        harmonic += 1.0 / kf;
        y_power_over_fact_sq *= y / (kf * kf);
        let k0_term = harmonic * y_power_over_fact_sq;
        k0_series += k0_term;
        k0_series_y_derivative_times_y += kf * k0_term;
        i0 += y_power_over_fact_sq;
        i1 += 0.5 * x * y_power_over_fact_sq / (kf + 1.0);
        if k0_term.abs() <= f64::EPSILON * i0.abs().max(k0_series.abs()).max(1.0) {
            break;
        }
    }

    let k0 = -log_half_plus_gamma * i0 + k0_series;
    let k1 = i0 / x + log_half_plus_gamma * i1 - (2.0 / x) * k0_series_y_derivative_times_y;
    (k0, k1)
}

#[inline(always)]
pub(crate) fn bessel_k0_small_series(x: f64) -> f64 {
    bessel_k0_k1_small_series(x).0
}

#[inline(always)]
pub(crate) fn bessel_k1_small_series(x: f64) -> f64 {
    bessel_k0_k1_small_series(x).1
}

pub(crate) const DUCHON_DERIVATIVE_R_FLOOR_REL: f64 = 1e-5;

pub(crate) const DUCHON_COLLISION_TAYLOR_REL: f64 = 1e-4;

/// Minimum `(row, center)` pair count before a radial design sweep builds a
/// certified [`radial_profile::RadialProfile`] instead of evaluating every
/// pair exactly. The profile build costs a few hundred exact jet
/// evaluations, so it only pays for itself when the sweep reuses it well
/// beyond that; below the threshold the exact path keeps small fits
/// bit-identical to the pre-profile behavior.
pub(crate) const RADIAL_PROFILE_MIN_PAIRS: usize = 16_384;

/// The one m→order mapping: `m` is the spline ORDER knob (mgcv's `m`), and it
/// selects the polynomial null space the smoother leaves unpenalized
/// (1 → mean only, 2 → mean + linear, k → total degree ≤ k−1). It is NOT the
/// spectral power. Inverse of `duchon_p_from_nullspace_order`.
#[inline(always)]
pub fn duchon_nullspace_order_from_m(m: usize) -> DuchonNullspaceOrder {
    match m {
        1 => DuchonNullspaceOrder::Zero,
        2 => DuchonNullspaceOrder::Linear,
        other => DuchonNullspaceOrder::Degree(other - 1),
    }
}

#[inline(always)]
pub(crate) fn duchon_p_from_nullspace_order(order: DuchonNullspaceOrder) -> usize {
    match order {
        // Duchon null spaces contain all polynomials of degree < m.
        // The public `order` knob chooses that polynomial degree cutoff:
        //   order=0 -> constants only  -> m=1
        //   order=1 -> constants+linear -> m=2
        DuchonNullspaceOrder::Zero => 1,
        DuchonNullspaceOrder::Linear => 2,
        DuchonNullspaceOrder::Degree(degree) => degree + 1,
    }
}

/// Whether a Duchon spec's **per-axis** ψ derivative surface is complete, so
/// its `aniso_log_scales` may be enrolled as outer REML coordinates (gam#2735).
///
/// This is a capability question, not a policy one, and it is asked of the spec
/// alone so the answer cannot depend on which call site is asking. It returns
/// `false` — leaving the term on its single isotropic ψ axis, exactly as
/// before — for every configuration whose per-axis derivative is not derived:
///
/// * a **scale-free** (`length_scale = None`) Duchon: every ψ-derivative
///   builder in the family refuses it, isotropic included;
/// * a **periodic** Duchon: the periodic path is a different builder with its
///   own chart, and no per-axis route through it exists;
/// * a term with no per-axis contrasts to learn (`d ≤ 1`, or η absent / the
///   wrong length);
/// * a spec whose ACTIVE operator penalty routes through the **closed-form
///   Lebesgue block**, which replaces the collocation Gram on the value side
///   and whose ψ-derivative is derived only for the isotropic direction.
///   Enrolling one of those would ship a block whose value and gradient came
///   from two different constructions.
///
/// The closed-form check sweeps every null-space order the realized build could
/// degrade to (`duchon_effective_nullspace_order` only ever reduces), because
/// the predicate is asked here — before centers exist — and a spec that becomes
/// unsupported only after degradation must not be enrolled.
pub fn duchon_spec_supports_axis_psi(spec: &DuchonBasisSpec, dim: usize) -> bool {
    if dim <= 1 || spec.length_scale.is_none() || spec.periodic.is_some() {
        return false;
    }
    match spec.aniso_log_scales.as_deref() {
        Some(eta) if eta.len() == dim => {}
        _ => return false,
    }
    let s_order = spec.power_as_usize() as f64;
    if s_order != spec.power {
        // The partial-fraction jets require an integer spectral power; the
        // fractional path never reaches the ψ-derivative surface at all.
        return false;
    }
    let requested_p = duchon_p_from_nullspace_order(spec.nullspace_order);
    let tension_requested = matches!(
        spec.operator_penalties.tension,
        OperatorPenaltySpec::Active { .. }
    );
    let stiffness_requested = matches!(
        spec.operator_penalties.stiffness,
        OperatorPenaltySpec::Active { .. }
    );
    for p_order in 1..=requested_p.max(1) {
        let two_pps = 2.0 * (p_order as f64 + spec.power);
        // Mirror the builder's auto-disable: a penalty the kernel is too rough
        // to admit is never assembled, so it cannot reach the closed form.
        let tension_active = tension_requested && two_pps > dim as f64 + 1.0;
        let stiffness_active = stiffness_requested && two_pps > dim as f64 + 2.0;
        if tension_active
            && crate::basis::duchon_closed_form_operator_penalty_converges(
                1, p_order, s_order, dim,
            )
        {
            return false;
        }
        if stiffness_active
            && crate::basis::duchon_closed_form_operator_penalty_converges(
                2, p_order, s_order, dim,
            )
        {
            return false;
        }
    }
    true
}

/// Returns the effective Duchon null-space order, auto-degrading when the
/// requested order leaves no radial kernel degrees of freedom.
///
/// The constrained kernel block has `centers.nrows() - rank(P)` columns, where
/// `P` is the polynomial null-space block. A valid polynomial block with
/// exactly as many centers as columns is still useless for smoothing: every
/// center is consumed by the side constraints and the design collapses to the
/// polynomial tail. Degrade to the highest lower null-space order with at
/// least one constrained kernel column.
pub fn duchon_effective_nullspace_order(
    centers: ArrayView2<'_, f64>,
    order: DuchonNullspaceOrder,
) -> DuchonNullspaceOrder {
    if order == DuchonNullspaceOrder::Zero {
        return order;
    }
    let mut effective = order;
    while effective != DuchonNullspaceOrder::Zero
        && centers.nrows() <= polynomial_block_from_order(centers, effective).ncols()
    {
        effective = duchon_previous_nullspace_order(effective);
    }
    if effective != order {
        // Dedup: warn only once per (rows, cols, requested_order) per process.
        // BFGS × P-IRLS × derivative callsites hit this path many times.
        static SEEN: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashSet<(usize, usize, DuchonNullspaceOrder)>>,
        > = std::sync::OnceLock::new();
        let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        let key = (centers.nrows(), centers.ncols(), order);
        let fresh = seen.lock().map(|mut s| s.insert(key)).unwrap_or(true);
        if fresh {
            let requested_cols = polynomial_block_from_order(centers, order).ncols();
            let effective_cols = polynomial_block_from_order(centers, effective).ncols();
            log::warn!(
                "Duchon nullspace order={:?} in dim={} with {} centers leaves no radial kernel columns (polynomial_cols={}); degrading to {:?} (polynomial_cols={})",
                order,
                centers.ncols(),
                centers.nrows(),
                requested_cols,
                effective,
                effective_cols
            );
        }
    }
    effective
}

/// Auto-*raise* the Duchon null-space order so the polyharmonic kernel — and
/// any active derivative-collocation operators — clear their pointwise
/// well-posedness margin `2(p + s) > dimension + max_operator_derivative_order`
/// *before* the hard guard in [`validate_duchon_collocation_orders`] can fire.
///
/// This is the escalating twin of [`duchon_effective_nullspace_order`], which
/// auto-*degrades* the order when too few centers remain to leave any radial
/// kernel columns. Here the failure mode is the opposite end: a low
/// order/power pair in dimension `dim` (e.g. `d=2`, `Linear` ⇒ `p=2`, `s=0`
/// with stiffness/D2 active) leaves the kernel value — or its `k`-th
/// derivative collocation — divergent at the origin, so `2(p + s) ≤ d + k`
/// trips the guard mid-fit. Lifting `p` (the null-space order) by the smallest
/// amount that restores the strict margin makes the guard unreachable for
/// otherwise valid-intent configs.
///
/// Only `p` is lifted; the spectral power `s` and the CPD condition `2s < d`
/// (which involves `s` and `d` alone) are untouched, so raising `p` can never
/// invalidate a config that the requested power already satisfied.
///
/// `max_operator_derivative_order` is the max derivative order among the
/// *active* operators (0 = mass/pointwise, 1 = tension/D1, 2 = stiffness/D2),
/// exactly the value threaded into [`validate_duchon_collocation_orders`].
pub(crate) fn duchon_order_for_operator_margin(
    dim: usize,
    power: f64,
    order: DuchonNullspaceOrder,
    max_operator_derivative_order: usize,
) -> DuchonNullspaceOrder {
    let margin = dim as f64 + max_operator_derivative_order as f64;
    let mut effective = order;
    // 2(p + s) > margin  ⇔  p > margin/2 − s. Each escalation lifts `p` by 1, so
    // `2(p + s)` grows by 2 per step and the loop is bounded by ⌈margin/2⌉.
    while 2.0 * (duchon_p_from_nullspace_order(effective) as f64 + power) <= margin {
        effective = duchon_next_nullspace_order(effective);
    }
    if effective != order {
        // Dedup: warn only once per (dim, power, requested_order, max_op) per
        // process — the escalation is hit from many rebuild callsites.
        static SEEN: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashSet<(usize, u64, DuchonNullspaceOrder, usize)>>,
        > = std::sync::OnceLock::new();
        let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
        let key = (dim, power.to_bits(), order, max_operator_derivative_order);
        let fresh = seen.lock().map(|mut s| s.insert(key)).unwrap_or(true);
        if fresh {
            log::warn!(
                "Duchon nullspace order={:?} with power={} in dim={} leaves 2*(p+s)={} \
                 below the pointwise/collocation margin dimension+{}={} required by the \
                 active operators; auto-raising to {:?} so the kernel stays well-posed",
                order,
                power,
                dim,
                2.0 * (duchon_p_from_nullspace_order(order) as f64 + power),
                max_operator_derivative_order,
                margin,
                effective,
            );
        }
    }
    effective
}

#[inline(always)]
pub(crate) fn gamma_lanczos(x: f64) -> f64 {
    // Numerical Recipes / Lanczos approximation with reflection formula.
    const G: f64 = 7.0;
    const P: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pix = std::f64::consts::PI * x;
        return std::f64::consts::PI / (pix.sin() * gamma_lanczos(1.0 - x));
    }
    let z = x - 1.0;
    let mut a = P[0];
    for (i, coeff) in P.iter().enumerate().skip(1) {
        a += coeff / (z + i as f64);
    }
    let t = z + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * a
}

#[inline(always)]
pub(crate) fn bessel_k_integer_order(n: usize, z: f64) -> f64 {
    if n == 0 {
        return bessel_k0_stable(z);
    }
    if n == 1 {
        return bessel_k1_stable(z);
    }
    let mut km1 = bessel_k0_stable(z);
    let mut k = bessel_k1_stable(z);
    for m in 1..n {
        let kp1 = km1 + 2.0 * (m as f64) * k / z;
        km1 = k;
        k = kp1;
    }
    k
}

#[inline(always)]
pub(crate) fn bessel_k_half_integer_order(l: usize, z: f64) -> f64 {
    // Exact closed-form seeds and the stable upward recurrence
    //   K_{1/2}(z) = sqrt(π/(2z))·e^{−z},
    //   K_{3/2}(z) = K_{1/2}(z)·(1 + 1/z),
    //   K_{ν+1}(z) = K_{ν−1}(z) + (2ν/z)·K_ν(z)   (ν = 1/2 + m, m ≥ 1).
    // Equivalent to the closed-form polynomial sum, but uses EXACT integer
    // coefficients via the recurrence instead of approximate Lanczos-gamma
    // values for `c_j = (l+j)!/(j!(l−j)!)`. The Lanczos approximation is
    // accurate to ~1 ULP at integer arguments; that error gets amplified
    // through catastrophic cancellation in derivative lattices of the
    // r^μ·K_μ(κr) family. Matching the [`BesselKLadder`] arithmetic byte-
    // for-byte also ensures the ladder/per-call paths agree exactly.
    let k_half = (std::f64::consts::PI / (2.0 * z)).sqrt() * (-z).exp();
    if l == 0 {
        return k_half;
    }
    let mut km1 = k_half;
    let mut k = k_half * (1.0 + 1.0 / z);
    for m in 1..l {
        let nu = m as f64 + 0.5;
        let kp1 = km1 + 2.0 * nu * k / z;
        km1 = k;
        k = kp1;
    }
    k
}

#[inline(always)]
pub(crate) fn bessel_k_real_half_integer_or_integer(
    nu_abs: f64,
    z: f64,
) -> Result<f64, BasisError> {
    let two_nu = (2.0 * nu_abs).round();
    if (two_nu - 2.0 * nu_abs).abs() > 1e-12 {
        crate::bail_invalid_basis!(
            "unsupported Bessel-K order ν={nu_abs}; only integer/half-integer orders are supported"
        );
    }
    let two_nu_i = two_nu as i64;
    if two_nu_i % 2 == 0 {
        let n = (two_nu_i / 2).max(0) as usize;
        Ok(bessel_k_integer_order(n, z))
    } else {
        let l = ((two_nu_i - 1) / 2).max(0) as usize;
        Ok(bessel_k_half_integer_order(l, z))
    }
}

/// Exact `i32` representation of a floating-point exponent, when one exists.
///
/// Duchon block orders are allowed to be fractional in the public pure-kernel
/// API, but the auto-topology/operator path carries integer `m` and therefore
/// integer exponents `2m-d-j`. Keeping the distinction explicit lets that hot
/// path use LLVM's integral-power primitive while fractional callers retain
/// the existing `powf` domain and rounding semantics.
#[inline(always)]
fn exact_i32_exponent(exponent: f64) -> Option<i32> {
    if !exponent.is_finite() {
        return None;
    }
    let integral = exponent as i32;
    (integral as f64 == exponent).then_some(integral)
}

/// Power for a positive base and an integer/half-integer exponent represented
/// as `2 * exponent`. This uses one `sqrt` only for the half-integer case and
/// otherwise routes directly to `powi`; it is never used for a non-positive
/// base, so it does not alter `powf`'s signed-zero/negative-base domain rules.
#[inline(always)]
fn positive_base_half_integer_power(base: f64, twice_exponent: i32) -> f64 {
    if twice_exponent % 2 == 0 {
        return base.powi(twice_exponent / 2);
    }
    let integer_part = twice_exponent / 2;
    if twice_exponent > 0 {
        base.powi(integer_part) * base.sqrt()
    } else {
        base.powi(integer_part) / base.sqrt()
    }
}

#[inline(always)]
fn positive_base_power_integral_or_half(base: f64, exponent: f64) -> f64 {
    exact_i32_exponent(2.0 * exponent).map_or_else(
        || base.powf(exponent),
        |twice| positive_base_half_integer_power(base, twice),
    )
}

/// Precomputed coefficient for `polyharmonic_kernel` that depends only on
/// `m` and `k_dim`, not on `r`.  Avoids repeated gamma_lanczos calls in the
/// hot kernel evaluation loop (called n × k times per basis build).
#[derive(Clone, Copy)]
pub(crate) struct PolyharmonicBlockCoeff {
    pub(crate) c: f64,
    pub(crate) power: f64,
    power_i32: Option<i32>,
    pub(crate) is_log_case: bool,
}

impl PolyharmonicBlockCoeff {
    pub(crate) fn new(m: f64, k_dim: usize) -> Self {
        assert!(
            m.is_finite() && m > 0.0,
            "PolyharmonicBlockCoeff::new: m must be finite and > 0, got {m}"
        );
        let k_half = 0.5 * k_dim as f64;
        let power = 2.0 * m - k_dim as f64;
        // Log case: k_dim is even and `2m − k_dim` is a non-negative even
        // integer (within ε). For fractional `m` this never fires; for
        // integer `m` it matches the original integer modulo check exactly.
        const LOG_EPS: f64 = 1e-12;
        let two_m = 2.0 * m;
        let is_log_case = k_dim.is_multiple_of(2) && {
            let n_f = (power / 2.0).round();
            n_f >= 0.0 && (n_f * 2.0 - power).abs() < LOG_EPS
        };
        if is_log_case {
            let m_int = m.round() as i64;
            let m_minus_half_d_plus_one = (m - k_half + 1.0).round() as i64;
            let c = polyharmonic_log_sign(m_int as usize, k_dim)
                / (2.0_f64.powi((two_m.round() as i32) - 1)
                    * positive_base_power_integral_or_half(std::f64::consts::PI, k_half)
                    * gamma_lanczos(m)
                    * gamma_lanczos(m_minus_half_d_plus_one as f64));
            Self {
                c,
                power,
                power_i32: exact_i32_exponent(power),
                is_log_case: true,
            }
        } else {
            let c = gamma_lanczos(k_half - m)
                / (positive_base_power_integral_or_half(4.0, m)
                    * positive_base_power_integral_or_half(std::f64::consts::PI, k_half)
                    * gamma_lanczos(m));
            Self {
                c,
                power,
                power_i32: exact_i32_exponent(power),
                is_log_case: false,
            }
        }
    }

    #[inline(always)]
    pub(crate) fn eval(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return self.origin_limit();
        }
        let radial_power = self
            .power_i32
            .map_or_else(|| r.powf(self.power), |power| r.powi(power));
        if self.is_log_case {
            self.c * radial_power * r.ln()
        } else {
            self.c * radial_power
        }
    }

    #[inline(always)]
    pub(crate) fn origin_limit(&self) -> f64 {
        if self.is_log_case {
            log_power_origin_limit(self.c, self.power, 1.0, 0.0)
        } else {
            log_power_origin_limit(self.c, self.power, 0.0, 1.0)
        }
    }
}

pub(crate) fn polyharmonic_kernel(r: f64, m: f64, k_dim: usize) -> f64 {
    PolyharmonicBlockCoeff::new(m, k_dim).eval(r)
}

#[inline(always)]
pub(crate) fn signed_infinity(sign: f64) -> f64 {
    if sign.is_sign_negative() {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    }
}

#[inline(always)]
pub(crate) fn log_power_origin_limit(
    coeff: f64,
    exponent: f64,
    log_coeff: f64,
    pure_coeff: f64,
) -> f64 {
    if log_coeff == 0.0 && pure_coeff == 0.0 {
        return 0.0;
    }
    if exponent > 0.0 {
        return 0.0;
    }
    if exponent == 0.0 {
        if log_coeff != 0.0 {
            signed_infinity(-coeff * log_coeff)
        } else {
            coeff * pure_coeff
        }
    } else if log_coeff != 0.0 {
        signed_infinity(-coeff * log_coeff)
    } else {
        signed_infinity(coeff * pure_coeff)
    }
}

#[inline(always)]
pub(crate) fn polyharmonic_log_sign(m: usize, k_dim: usize) -> f64 {
    assert!(
        k_dim.is_multiple_of(2),
        "polyharmonic_log_sign requires even kernel dimension: k_dim={k_dim}, m={m}"
    );
    (-1.0_f64).powi(m as i32 - (k_dim as i32 / 2) + 1)
}

#[inline(always)]
pub(crate) fn duchon_matern_block(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
) -> Result<f64, BasisError> {
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let nu_abs = nu.abs();
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));
    if r <= 0.0 {
        if nu > 0.0 {
            // r^ν K_ν(κr) → 2^(ν−1) Γ(ν) κ^(−ν) as r→0+.
            return Ok(c * 2.0_f64.powf(nu - 1.0) * gamma_lanczos(nu) * kappa.powf(-nu));
        }
        // ν ≤ 0: c·r^ν·K_|ν|(κr) is divergent at r=0 (logarithmically for ν=0,
        // power-law for ν<0). The hybrid-kernel diagonal must be evaluated via
        // duchon_hybrid_kernel_collision_value, which sums the divergent
        // Matérn and polyharmonic blocks so the singularities cancel exactly
        // (guaranteed by the PFD identity when 2(p+s) > d).
        crate::bail_invalid_basis!(
            "Duchon Matérn block at r=0 with ν={nu} ≤ 0 is divergent; \
             evaluate the hybrid kernel diagonal via the collision routine"
        );
    }
    let z = kappa * r;
    let k_nu = bessel_k_real_half_integer_or_integer(nu_abs, z)?;
    Ok(c * r.powf(nu) * k_nu)
}

#[inline(always)]
pub(crate) fn polyharmonic_kernel_triplet(
    r: f64,
    m: f64,
    k_dim: usize,
) -> Result<(f64, f64, f64), BasisError> {
    let (value, first, second, _, _) = polyharmonic_block_jet4(r, m, k_dim)?;
    Ok((value, first, second))
}

#[inline(always)]
pub(crate) fn falling_factorial(alpha: f64, order: usize) -> f64 {
    (0..order).fold(1.0, |acc, idx| acc * (alpha - idx as f64))
}

#[inline(always)]
pub(crate) fn falling_factorial_derivative(alpha: f64, order: usize) -> f64 {
    if order == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for omit in 0..order {
        let mut term = 1.0;
        for idx in 0..order {
            if idx != omit {
                term *= alpha - idx as f64;
            }
        }
        total += term;
    }
    total
}

/// Unified radial jet for one polyharmonic partial-fraction block.
///
/// Returns (φ, φ', φ'', φ''', φ'''') from a single consistent evaluation,
/// sharing normalization constant, r_safe, and log_r. This eliminates the
/// possibility of numerical drift between the triplet and higher-order
/// derivative paths.
pub(crate) fn polyharmonic_block_jet4(
    r: f64,
    m: f64,
    k_dim: usize,
) -> Result<(f64, f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!("polyharmonic distance must be finite and non-negative");
    }
    assert!(
        m.is_finite() && m > 0.0,
        "polyharmonic_block_jet4: m must be finite and > 0, got {m}"
    );

    let k_half = 0.5 * k_dim as f64;
    let alpha = 2.0 * m - k_dim as f64;
    let alpha_i32 = exact_i32_exponent(alpha);
    // Log case: k_dim even and `2m − k_dim` is a non-negative even integer
    // (within ε). For fractional `m` this never fires.
    const LOG_EPS: f64 = 1e-12;
    let is_log_case = k_dim.is_multiple_of(2) && {
        let n_f = (alpha / 2.0).round();
        n_f >= 0.0 && (n_f * 2.0 - alpha).abs() < LOG_EPS
    };
    if is_log_case {
        let m_int = m.round() as usize;
        let c = polyharmonic_log_sign(m_int, k_dim)
            / (2.0_f64.powi((2 * m_int - 1) as i32)
                * positive_base_power_integral_or_half(std::f64::consts::PI, k_half)
                * gamma_lanczos(m)
                * gamma_lanczos((m_int - k_dim / 2 + 1) as f64));
        let mut out = [0.0; 5];
        let log_r = (r > 0.0).then(|| r.ln());
        for d in 0..5 {
            let e = alpha - d as f64;
            let ff = falling_factorial(alpha, d);
            let ff_d = falling_factorial_derivative(alpha, d);
            out[d] = if r <= 0.0 {
                log_power_origin_limit(c, e, ff, ff_d)
            } else {
                let radial_power =
                    alpha_i32.map_or_else(|| r.powf(e), |integral| r.powi(integral - d as i32));
                c * radial_power * (ff * log_r.expect("positive radius has a logarithm") + ff_d)
            };
        }
        return Ok((out[0], out[1], out[2], out[3], out[4]));
    }

    let c = gamma_lanczos(k_half - m)
        / (positive_base_power_integral_or_half(4.0, m)
            * positive_base_power_integral_or_half(std::f64::consts::PI, k_half)
            * gamma_lanczos(m));
    let mut out = [0.0; 5];
    for d in 0..5 {
        let e = alpha - d as f64;
        let ff = falling_factorial(alpha, d);
        out[d] = if r <= 0.0 {
            log_power_origin_limit(c, e, 0.0, ff)
        } else {
            let radial_power =
                alpha_i32.map_or_else(|| r.powf(e), |integral| r.powi(integral - d as i32));
            c * ff * radial_power
        };
    }
    Ok((out[0], out[1], out[2], out[3], out[4]))
}

#[inline(always)]
pub(crate) fn log_power_family_derivative(
    exponent: i32,
    log_coeff: f64,
    pure_coeff: f64,
) -> (i32, f64, f64) {
    let exponent_f64 = exponent as f64;
    (
        exponent - 1,
        exponent_f64 * log_coeff,
        exponent_f64 * pure_coeff + log_coeff,
    )
}

#[inline(always)]
pub(crate) fn log_power_family_value(
    r: f64,
    coeff: f64,
    exponent: i32,
    log_coeff: f64,
    pure_coeff: f64,
) -> f64 {
    if r <= 0.0 {
        log_power_origin_limit(coeff, exponent as f64, log_coeff, pure_coeff)
    } else {
        coeff * r.powi(exponent) * (log_coeff * r.ln() + pure_coeff)
    }
}

#[inline(always)]
pub(crate) fn duchon_polyharmonic_operator_block_jets(
    r: f64,
    m: usize,
    k_dim: usize,
) -> Result<(f64, f64, f64, f64), BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!("polyharmonic distance must be finite and non-negative");
    }
    assert!(
        m > 0,
        "duchon_polyharmonic_operator_block_jets: m must be > 0, got {m}"
    );

    let Ok(m_i32) = i32::try_from(m) else {
        crate::bail_invalid_basis!("polyharmonic order {m} exceeds the supported i32 range");
    };
    let Ok(k_dim_i32) = i32::try_from(k_dim) else {
        crate::bail_invalid_basis!("Duchon dimension {k_dim} exceeds the supported i32 range");
    };
    let Some(alpha) = m_i32
        .checked_mul(2)
        .and_then(|twice_m| twice_m.checked_sub(k_dim_i32))
    else {
        crate::bail_invalid_basis!("Duchon exponent 2*{m}-{k_dim} exceeds the supported i32 range");
    };
    let m_f64 = m as f64;
    let k_half = 0.5 * k_dim as f64;
    let is_log_case = k_dim.is_multiple_of(2) && alpha >= 0;
    let (c, phi_log_coeff, phi_pure_coeff) = if is_log_case {
        (
            polyharmonic_log_sign(m, k_dim)
                / (2.0_f64.powi(2 * m_i32 - 1)
                    * positive_base_half_integer_power(std::f64::consts::PI, k_dim_i32)
                    * gamma_lanczos(m_f64)
                    * gamma_lanczos((m - k_dim / 2 + 1) as f64)),
            1.0,
            0.0,
        )
    } else {
        (
            gamma_lanczos(k_half - m_f64)
                / (4.0_f64.powi(m_i32)
                    * positive_base_half_integer_power(std::f64::consts::PI, k_dim_i32)
                    * gamma_lanczos(m_f64)),
            0.0,
            1.0,
        )
    };

    let (phi_r_exp, phi_r_log, phi_r_pure) =
        log_power_family_derivative(alpha, phi_log_coeff, phi_pure_coeff);
    let q_exp = phi_r_exp - 1;
    let q = log_power_family_value(r, c, q_exp, phi_r_log, phi_r_pure);

    let (q_r_exp_raw, q_r_log, q_r_pure) =
        log_power_family_derivative(q_exp, phi_r_log, phi_r_pure);
    let t_exp = q_r_exp_raw - 1;
    let t = log_power_family_value(r, c, t_exp, q_r_log, q_r_pure);

    let (t_r_exp, t_r_log, t_r_pure) = log_power_family_derivative(t_exp, q_r_log, q_r_pure);
    let t_r = log_power_family_value(r, c, t_r_exp, t_r_log, t_r_pure);

    let (t_rr_exp, t_rr_log, t_rr_pure) = log_power_family_derivative(t_r_exp, t_r_log, t_r_pure);
    let t_rr = log_power_family_value(r, c, t_rr_exp, t_rr_log, t_rr_pure);

    Ok((q, t, t_r, t_rr))
}

/// Shared Bessel-K ladder for one evaluation point `z = κ·r`.
///
/// Every Matérn partial-fraction block and every term of its radial
/// derivative lattice consumes `K_ν(z)` at orders from ONE parity class
/// (integer when the covariate dimension is even, half-integer when odd),
/// differing by integers — and all at the SAME `z`. The previous code
/// restarted the `K₀/K₁` (or closed-form half-integer) seed evaluation and
/// the upward recurrence inside every per-term Bessel call: hundreds of
/// redundant seed+recurrence runs per `(row, center)` pair, which the #979
/// CTN stage-1 stack profile showed to be the dominant cost of every Duchon
/// κ-trial at scale. One ladder per point replaces all of them: two seed
/// evaluations plus the standard upward recurrence
/// `K_{ν+1}(z) = K_{ν−1}(z) + (2ν/z)·K_ν(z)`, which is the numerically
/// STABLE direction for `K` (it grows with ν). For integer orders this is
/// arithmetic-identical to the old per-call `bessel_k_integer_order`, which
/// ran the same seeds and recurrence internally; for half-integer orders the
/// recurrence is exact and replaces the per-order closed-form sum.
pub(crate) struct BesselKLadder {
    /// `values[i] = K_{base + i}(z)` with `base ∈ {0, ½}`.
    pub(crate) values: SmallVec<[f64; 16]>,
    pub(crate) half_integer: bool,
}

impl BesselKLadder {
    pub(crate) fn build(z: f64, half_integer: bool, max_order_steps: usize) -> Self {
        let mut values: SmallVec<[f64; 16]> = SmallVec::with_capacity(max_order_steps + 2);
        if half_integer {
            // K_{1/2}(z) = √(π/(2z))·e^{−z};  K_{3/2}(z) = K_{1/2}(z)·(1 + 1/z).
            let k_half = (std::f64::consts::PI / (2.0 * z)).sqrt() * (-z).exp();
            values.push(k_half);
            values.push(k_half * (1.0 + 1.0 / z));
        } else {
            values.push(bessel_k0_stable(z));
            values.push(bessel_k1_stable(z));
        }
        let base = if half_integer { 0.5 } else { 0.0 };
        for i in 1..max_order_steps {
            let nu = base + i as f64;
            let next = values[i - 1] + 2.0 * nu * values[i] / z;
            values.push(next);
        }
        Self {
            values,
            half_integer,
        }
    }

    /// `K_{|order|}(z)` from the ladder (`K_{−ν} = K_ν`).
    #[inline]
    pub(crate) fn k_abs(&self, order_abs: f64) -> f64 {
        let base = if self.half_integer { 0.5 } else { 0.0 };
        let idx = (order_abs - base).round() as usize;
        self.values[idx]
    }
}

/// Radial-derivative jets of the Matérn family `coeff·r^μ·K_μ(κr)` up to
/// order `max_j ≤ 4`, evaluated against a shared [`BesselKLadder`].
///
/// Exact recurrence derived from `d/dr[r^ν K_ν(κr)]` and the Bessel identity
/// `dK_ν/dz = −K_{ν−1}(z) − (ν/z)K_ν(z)`:
///
///   g⁽⁰⁾ = c · r^ν · K_ν(z)
///   g⁽¹⁾ = −c · κ · r^ν · K_{ν−1}(z)
///   g⁽²⁾ = c·κ² r^ν K_{ν−2} − c·κ r^{ν−1} K_{ν−1}, ...
///
/// Same derivative lattice as the per-order reference implementation
/// `duchon_matern_family_radial_derivative_reference` (kept in the test
/// module as the equivalence oracle)
/// (term-for-term, in the same order), but: (a) the lattice is expanded
/// incrementally once instead of rebuilt from scratch per derivative order,
/// (b) terms live in a fixed-capacity stack buffer instead of per-call heap
/// `Vec`s (≤ 2^max_j ≤ 16 terms), and (c) every Bessel factor is an indexed
/// ladder read instead of a fresh seed+recurrence evaluation. Only orders
/// `0..=max_j` are computed — the q-family consumes order 0 only and the
/// t-family orders ≤ 2, where the old path always expanded to order 4 and
/// discarded the tail.
pub(crate) fn duchon_matern_family_jets_with_ladder(
    r: f64,
    kappa: f64,
    coeff: f64,
    mu: f64,
    max_j: usize,
    ladder: &BesselKLadder,
    out: &mut [f64],
) -> Result<(), BasisError> {
    if max_j > 4 || out.len() <= max_j {
        crate::bail_invalid_basis!(
            "Duchon Matérn-family ladder jets support derivative orders 0..=4 with an output slot per order"
        );
    }
    if r <= 0.0 {
        out[..=max_j].fill(0.0);
        if mu > 0.0 {
            out[0] = coeff * 2.0_f64.powf(mu - 1.0) * gamma_lanczos(mu) * kappa.powf(-mu);
        }
        return Ok(());
    }
    let mut terms: SmallVec<[DuchonMaternDerivativeTerm; 16]> =
        smallvec![DuchonMaternDerivativeTerm {
            coeff,
            kappa_power: 0,
            r_power: mu,
            bessel_order: mu,
        }];
    for (j, slot) in out.iter_mut().enumerate().take(max_j + 1) {
        if j > 0 {
            let mut next: SmallVec<[DuchonMaternDerivativeTerm; 16]> =
                SmallVec::with_capacity(terms.len() * 2);
            for term in &terms {
                let stay_coeff = term.coeff * (term.r_power - term.bessel_order);
                if stay_coeff != 0.0 {
                    next.push(DuchonMaternDerivativeTerm {
                        coeff: stay_coeff,
                        kappa_power: term.kappa_power,
                        r_power: term.r_power - 1.0,
                        bessel_order: term.bessel_order,
                    });
                }
                next.push(DuchonMaternDerivativeTerm {
                    coeff: -term.coeff,
                    kappa_power: term.kappa_power + 1,
                    r_power: term.r_power,
                    bessel_order: term.bessel_order - 1.0,
                });
            }
            terms = next;
        }
        let mut value = KahanSum::default();
        for term in &terms {
            if term.coeff == 0.0 {
                continue;
            }
            value.add(
                term.coeff
                    * kappa.powi(term.kappa_power as i32)
                    * r.powf(term.r_power)
                    * ladder.k_abs(term.bessel_order.abs()),
            );
        }
        *slot = value.sum();
    }
    Ok(())
}

/// Maximum ladder steps (`K_base ..= K_{base+steps}`) needed by the q/t
/// operator families of Matérn block `n` in dimension `k_dim`: the q-family
/// reads `K_{|ν−1|}` and the t-family `K_{|ν−2−j|}` for `j ≤ 2`, ν = n − d/2.
pub(crate) fn duchon_matern_block_max_ladder_steps(n_order: usize, k_dim: usize) -> usize {
    let nu = n_order as f64 - 0.5 * k_dim as f64;
    let candidates = [
        (nu - 1.0).abs(),
        (nu - 2.0).abs(),
        (nu - 3.0).abs(),
        (nu - 4.0).abs(),
    ];
    let max_abs = candidates.iter().copied().fold(0.0_f64, f64::max);
    max_abs.floor() as usize + 1
}

pub(crate) fn duchon_matern_operator_block_jets_with_ladder(
    r: f64,
    kappa: f64,
    n_order: usize,
    k_dim: usize,
    ladder: &BesselKLadder,
) -> Result<(f64, f64, f64, f64), BasisError> {
    if r <= 0.0 {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    let n = n_order as f64;
    let k_half = 0.5 * k_dim as f64;
    let nu = n - k_half;
    let c = kappa.powf(k_half - n)
        / ((2.0 * std::f64::consts::PI).powf(k_half) * 2.0_f64.powf(n - 1.0) * gamma_lanczos(n));

    let mut q_out = [0.0_f64; 1];
    duchon_matern_family_jets_with_ladder(r, kappa, -c * kappa, nu - 1.0, 0, ladder, &mut q_out)?;
    let mut t_out = [0.0_f64; 3];
    duchon_matern_family_jets_with_ladder(
        r,
        kappa,
        c * kappa * kappa,
        nu - 2.0,
        2,
        ladder,
        &mut t_out,
    )?;
    Ok((q_out[0], t_out[0], t_out[1], t_out[2]))
}

#[inline(always)]
pub(crate) fn pure_duchon_block_order(p_order: usize, s_order: f64) -> f64 {
    p_order as f64 + s_order
}

pub(crate) fn validate_duchon_kernel_orders(
    length_scale: Option<f64>,
    p_order: usize,
    s_order: f64,
    k_dim: usize,
) -> Result<(), BasisError> {
    if k_dim == 0 {
        crate::bail_invalid_basis!("Duchon basis requires at least one covariate dimension");
    }
    if let Some(scale) = length_scale
        && (!scale.is_finite() || scale <= 0.0)
    {
        crate::bail_invalid_basis!("Duchon hybrid length_scale must be finite and positive");
    }
    // Two independent well-posedness conditions on (p, s, d) for pure Duchon.
    //
    // (1) CPD-vs-nullspace adequacy — gated below on `length_scale.is_none()`.
    //     The pure-polyharmonic kernel of effective order m = p+s in R^d is
    //     phi(r) = r^{2m-d}, or r^{2m-d}·log r when 2m-d is a non-negative
    //     even integer (the "log case", reached precisely when d is even
    //     and m >= d/2). Wendland's Theorem 8.17 / 8.18 give its
    //     conditional-positive-definiteness order:
    //
    //         d odd,  exponent half-integer:  ceil((2m-d)/2) = m - (d-1)/2
    //         d even, log case:               (2m-d)/2 + 1   = m - d/2 + 1
    //
    //     Duchon interpolation with polynomial nullspace P_p (polynomials
    //     of degree < p) is uniquely solvable iff the kernel's CPD order
    //     does not exceed p. Substituting m = p + s:
    //
    //         d odd:  s <= (d-1)/2     <=>  2s <= d - 1
    //         d even: s <= d/2 - 1     <=>  2s <= d - 2
    //
    //     Both branches collapse to `2s < d` once we use that s and d are
    //     integers and 2s is therefore even (so `2s = d - 1` is impossible
    //     for even d, and `2s <= d - 2` is just `2s < d`).
    //
    //     Counter-example admitted if this guard is dropped: d=2, p=1, s=1
    //     passes the spectral check (2(1+1)=4 > 2) and builds the TPS
    //     kernel c·r²·log r against a constants-only nullspace P_1; the
    //     interpolation form is not PD on lambda perp P_1 and the fitted
    //     penalty is meaningless.
    //
    //     The hybrid (Matérn-blended) Duchon kernel sidesteps this entirely:
    //     the Matérn remainder is strictly positive definite (CPD order 0),
    //     so any P_p suffices — hence the `length_scale.is_none()` gate.
    //
    // (2) Spectral kernel-existence — universal, gated below on the sum.
    //     The pointwise kernel comes from the inverse Fourier of
    //     1/|xi|^{2(p+s)}, which is a finite distribution at the origin
    //     iff `2(p+s) > d`. Below that threshold the radial kernel value
    //     diverges and there is nothing to evaluate.
    if !s_order.is_finite() || s_order < 0.0 {
        crate::bail_invalid_basis!("Duchon spectral power must be finite and ≥ 0; got s={s_order}");
    }
    if length_scale.is_none() && 2.0 * s_order >= k_dim as f64 {
        // The `2s >= d` boundary is INDEPENDENT of the nullspace degree p (it
        // cancels in the CPD-order-vs-p derivation above, #2278): a former
        // `p_order < 2` conjunct here wrongly let `p >= 2` configs (e.g. d=2,
        // Linear nullspace p=2, explicit power s=1) bypass the check and build a
        // penalty from a kernel that is not CPD on the nullspace complement.
        crate::bail_invalid_basis!(
            "pure Duchon requires spectral power < dimension/2 (2s < d), independent of nullspace degree; got power={s_order}, dimension={k_dim}"
        );
    }
    let spectral_order = 2.0 * (p_order as f64 + s_order);
    if spectral_order <= k_dim as f64 {
        return Err(BasisError::duchon_smoothness_insufficient(
            "pointwise kernel values",
            0,
            k_dim,
            p_order,
            s_order,
        ));
    }
    Ok(())
}

pub(crate) fn validate_duchon_collocation_orders(
    length_scale: Option<f64>,
    p_order: usize,
    s_order: f64,
    k_dim: usize,
    max_operator_derivative_order: usize,
) -> Result<(), BasisError> {
    // Kernel-level conditions (existence + CPD/nullspace adequacy) come first;
    // the operator-level conditions below build on a pointwise-valid kernel.
    validate_duchon_kernel_orders(length_scale, p_order, s_order, k_dim)?;
    // The spectral_order > k_dim + k checks below are C^k-at-origin
    // conditions: for the polyharmonic kernel r^{2(p+s)-d} (or the log
    // variant) to admit k-th radial derivatives in the distributional sense
    // — and therefore for k-th-order derivative *collocation* of the
    // kernel against centers to produce a finite operator — we need its
    // exponent to clear the next k orders of differentiation at the
    // origin. Equivalently: 2(p+s) - d > k.
    //
    // Note these are independent of the CPD/nullspace check. The penalty
    // matrices ultimately built from these collocation operators are of
    // the form S = D_k^T D_k and are PSD by construction; the discipline
    // here is purely about *existence* of D_k itself.
    let spectral_order = 2.0 * (p_order as f64 + s_order);
    if max_operator_derivative_order >= 1 && spectral_order <= k_dim as f64 + 1.0 {
        return Err(BasisError::duchon_smoothness_insufficient(
            "D1 collocation",
            1,
            k_dim,
            p_order,
            s_order,
        ));
    }
    if max_operator_derivative_order >= 2 && spectral_order <= k_dim as f64 + 2.0 {
        return Err(BasisError::duchon_smoothness_insufficient(
            "D2 collocation",
            2,
            k_dim,
            p_order,
            s_order,
        ));
    }
    Ok(())
}

/// The inverse length scale `κ = 1/ℓ` of a Matérn / hybrid-Duchon kernel.
///
/// One owner for a conversion every kernel site used to spell as
/// `1.0 / length_scale.max(1e-300)`. That floor was not a bound of the
/// arithmetic: a length scale of `0` is not a kernel with `κ = 1e300`, it is a
/// construction defect — the spectral density `1/(ρ^{2p}(κ²+ρ²)^s)` has no
/// hybrid limit there (the scale-free family is `length_scale = None`, a
/// different spectrum) — and the floor kept `κ·r` finite out to `r ≈ 1.8e8`
/// while evaluating a kernel no caller asked for. A length scale that is not
/// finite and positive, or whose inverse is not representable, is refused with
/// the calling context named (#2469).
#[inline]
pub(crate) fn duchon_inverse_length_scale(
    length_scale: f64,
    context: &str,
) -> Result<f64, BasisError> {
    let kappa = 1.0 / length_scale;
    if !(length_scale.is_finite() && length_scale > 0.0 && kappa.is_finite()) {
        crate::bail_invalid_basis!(
            "{context}: kernel length_scale must be finite and positive with a finite \
             inverse; got {length_scale}"
        );
    }
    Ok(kappa)
}

#[derive(Debug, Clone)]
pub struct DuchonPartialFractionCoeffs {
    pub(crate) a: Vec<f64>,
    pub(crate) b: Vec<f64>,
}

#[inline(always)]
pub(crate) fn duchon_partial_fraction_coeffs(
    p_order: usize,
    s_order: usize,
    kappa: f64,
) -> DuchonPartialFractionCoeffs {
    // 1/(ρ^{2p}(κ²+ρ²)^s) = Σ a_m/ρ^{2m} + Σ b_n/(κ²+ρ²)^n
    let mut a = vec![0.0_f64; p_order + 1]; // 1-based m
    let mut b = vec![0.0_f64; s_order + 1]; // 1-based n
    if s_order == 0 {
        if p_order > 0 {
            // Pure intrinsic polyharmonic case: no Matérn tail remains, so the
            // spectrum is exactly 1 / ρ^(2p).
            a[p_order] = 1.0;
        }
        return DuchonPartialFractionCoeffs { a, b };
    }
    for m in 1..=p_order {
        let sign = if (p_order - m).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let expo = -2.0 * (s_order + p_order - m) as f64;
        let comb = binomial_f64(s_order + p_order - m - 1, p_order - m);
        a[m] = sign * kappa.powf(expo) * comb;
    }
    for n in 1..=s_order {
        let sign = if p_order.is_multiple_of(2) { 1.0 } else { -1.0 };
        let expo = -2.0 * (p_order + s_order - n) as f64;
        let comb = if p_order == 0 && n == s_order {
            // p=0 reduces to the pure Matérn block 1/(κ²+ρ²)^s.
            1.0
        } else {
            let top = p_order + s_order - n - 1;
            binomial_f64(top, s_order - n)
        };
        b[n] = sign * kappa.powf(expo) * comb;
    }
    DuchonPartialFractionCoeffs { a, b }
}

/// Evaluate the hybrid Duchon–Matérn kernel
/// `φ(r) = F^{-1}[ ρ^{-2p} (κ²+ρ²)^{-s} ](r)` via a single, cancellation-free
/// 1-D integral (gam#1424).
///
/// The partial-fraction expansion `Σ a_m/ρ^{2m} + Σ b_n/(κ²+ρ²)^n` evaluates
/// the radial kernel as an alternating sum of individually huge polyharmonic
/// (`r^{2m-d}`) and Matérn blocks whose leading singular parts cancel. For
/// high `d` (e.g. d=16, p=2, s=7) the largest block is ~1e3 while the true
/// value is ~1e-13, so f64 loses *every* significant digit and the resulting
/// Gram matrix is no longer PSD (λ_min ≈ −0.26 after normalization).
///
/// Using the Schwinger / Feynman parametrization of both rational factors and
/// performing the Gaussian (radial inverse-FT) integral analytically reduces
/// the kernel to
///
/// ```text
///   φ(r) = (4π)^{-d/2} / (Γ(p)Γ(s))
///          · ∫₀¹ (1-w)^{p-1} w^{s-1} · 2(B/A)^{b/2} K_b(2√(AB)) dw,
///   with  b = p + s − d/2,  A = w κ²,  B = r²/4.
/// ```
///
/// With `ρ = κ r` every `κ` leaves the integrand: `φ(r; κ) = pref · κ^{-2b}
/// · G(ρ)` for the universal profile `G` of `(p, s, d)`, which
/// [`duchon_radial_profile`] represents once per process, certified against
/// its own adaptive reference integral (a fixed 64-node Gauss–Legendre rule
/// on `w` was measured 1 % off for half-integer `b` at every distance and
/// 100 % off for `κ r ≳ 10³`; see that module). The `r = 0` diagonal is the
/// closed form `φ(0) = (4π)^{-d/2} Γ(b)/(Γ(p)Γ(s)) κ^{-2b} B(s−b, p)`.
///
/// Requires `b = p + s − d/2 > 0` (kernel existence, `2(p+s) > d`) and
/// `s − b = d/2 − p > 0` (integrable `w → 0` endpoint), i.e. `2p < d`. Callers
/// must check [`duchon_hybrid_stable_integral_applies`] before invoking.
pub(crate) fn duchon_hybrid_kernel_stable_integral(
    r: f64,
    kappa: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
) -> Result<f64, BasisError> {
    DuchonHybridEvaluator::new(kappa, p_order, s_order, k_dim)?.value(r)
}


/// The hybrid Duchon–Matérn kernel of one shape at one length scale, bound
/// once for a whole sweep.
///
/// `φ(r) = pref · κ^{-2b} · G(κ r)` and `φ^{(m)}(r) = pref · κ^{m-2b} ·
/// G^{(m)}(κ r)` split into a factor fixed by `(p, s, d, κ)` and the shape's
/// certified profile. A design build walks its `n · k` pairs at one
/// `(shape, κ)`, and so does every ψ sweep, so re-deriving that factor per
/// pair — two `Γ` evaluations, two `powf`s, and a lookup of the profile —
/// costs more than the Chebyshev evaluation it sets up: the lookup alone
/// measured 37 % of a 6-D `n = 50 000`, `k = 100` fit (gam#2735). Binding it
/// once leaves the per-pair work at the profile evaluation itself.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DuchonHybridEvaluator {
    profile: &'static DuchonRadialProfile,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    kappa: f64,
    /// `pref · κ^{-2b}`.
    scale: f64,
    /// `pref · κ^{-2b} · G(0)`, the closed-form `r = 0` diagonal, or `None`
    /// for a kernel singular at the origin (`2(p+s) ≤ d`).
    origin: Option<f64>,
}

impl DuchonHybridEvaluator {
    /// Bind the certified profile of `(p, s, d)` and the `κ`-fixed factor.
    ///
    /// Callers must check [`duchon_hybrid_stable_integral_applies`] first;
    /// [`duchon_hybrid_evaluator`] does that and is the usual entry point.
    pub(crate) fn new(
        kappa: f64,
        p_order: usize,
        s_order: usize,
        k_dim: usize,
    ) -> Result<Self, BasisError> {
        assert!(
            duchon_hybrid_stable_integral_applies(p_order, s_order, k_dim),
            "DuchonHybridEvaluator precondition violated: 2(p+s) > d and 2p < d required (p={p_order}, s={s_order}, d={k_dim})"
        );
        let profile = duchon_radial_profile(p_order, s_order, k_dim)?;
        let scale = profile.kappa_scale(kappa);
        let origin = profile.origin_value().ok().map(|g0| scale * g0);
        Ok(Self {
            profile,
            p_order,
            s_order,
            k_dim,
            kappa,
            scale,
            origin,
        })
    }

    /// `φ(r)`, including the closed-form `r = 0` diagonal.
    pub(crate) fn value(&self, r: f64) -> Result<f64, BasisError> {
        if !r.is_finite() || r < 0.0 {
            crate::bail_invalid_basis!("Duchon kernel distance must be finite and non-negative");
        }
        let value = if r == 0.0 {
            match self.origin {
                Some(origin) => origin,
                // The profile owns the refusal for a kernel singular at `r = 0`.
                None => self.scale * self.profile.origin_value()?,
            }
        } else {
            self.scale * self.profile.value(self.kappa * r)
        };
        if !value.is_finite() {
            crate::bail_invalid_basis!(
                "non-finite Duchon hybrid value (stable form) at r={r}, p={}, s={}, d={}",
                self.p_order,
                self.s_order,
                self.k_dim
            );
        }
        Ok(value)
    }

    /// Radial operator scalars `(q, t, t_r, t_rr)` of the hybrid Duchon–Matérn
    /// kernel via the same cancellation-free single integral as
    /// [`Self::value`], differentiated under the integral
    /// sign (gam#1424 / gam#1453).
    ///
    /// The partial-fraction operator core (`duchon_regularized_operator_core`)
    /// assembles `q, t` as a sign-alternating sum of polyharmonic and Matérn
    /// *operator* blocks. In high dimensions (e.g. d=16, p=1, s=9) each block is
    /// ~1e3 while the true operator scalar is ~1e-13, so f64 loses every
    /// significant digit — Kahan summation fixes accumulation, not the
    /// cancellation between huge opposing terms, leaving `q, t` with ~1e-2 relative
    /// noise. That floor sits above the Chebyshev profile certificate, so the
    /// production profile cannot certify (gam#1453).
    ///
    /// This reads the radial derivatives `φ′…φ⁗` off the
    /// universal profile of `(p, s, d)`: `φ^{(m)}(r) = pref · κ^{m−2b} ·
    /// G^{(m)}(κ r)`, where [`duchon_radial_profile`] carries `G` and its first
    /// four derivatives, each the smooth per-`w` integrand differentiated under
    /// the integral sign (a single well-conditioned `z^a K_ν(z)` term list per
    /// slice, no cross-block cancellation) and certified against the adaptive
    /// reference. The operator scalars are then assembled from the standard
    /// radial relations
    /// `q = φ′/r`, `t = q′/r`, `t_r = (q″−t)/r`, `t_rr = q‴/r − 2q″/r² + 2q′/r³`.
    ///
    /// Requires `r > 0`; the shape precondition is the constructor's.
    pub(crate) fn operator_core(
        &self,
        r: f64,
    ) -> Result<DuchonRegularizedOperatorCore, BasisError> {
        assert!(
            r > 0.0 && r.is_finite(),
            "DuchonHybridEvaluator::operator_core requires r > 0, got r={r}"
        );
        let kappa = self.kappa;
        let g = self.profile.derivatives(kappa * r);
        // φ^{(m)}(r) = pref · κ^{m−2b} · G^{(m)}(κ r).
        let phi1 = self.scale * kappa * g[1];
        let phi2 = self.scale * kappa * kappa * g[2];
        let phi3 = self.scale * kappa * kappa * kappa * g[3];
        let phi4 = self.scale * kappa * kappa * kappa * kappa * g[4];
        if !(phi1.is_finite() && phi2.is_finite() && phi3.is_finite() && phi4.is_finite()) {
            crate::bail_invalid_basis!(
                "non-finite Duchon hybrid operator (stable form) at r={r}, p={}, s={}, d={}",
                self.p_order,
                self.s_order,
                self.k_dim
            );
        }
        // Assemble the operator scalars from the radial derivatives. For r > 0
        // these divisions are removable-singularity quotients of moderate
        // quantities (no cancellation between blocks remains).
        let inv_r = 1.0 / r;
        let q = phi1 * inv_r;
        // q′ = φ″/r − φ′/r²; q″ = φ‴/r − 2φ″/r² + 2φ′/r³;
        // q‴ = φ⁗/r − 3φ‴/r² + 6φ″/r³ − 6φ′/r⁴.
        let q_r = phi2 * inv_r - phi1 * inv_r * inv_r;
        let q_rr = phi3 * inv_r - 2.0 * phi2 * inv_r * inv_r + 2.0 * phi1 * inv_r * inv_r * inv_r;
        let q_rrr = phi4 * inv_r - 3.0 * phi3 * inv_r * inv_r + 6.0 * phi2 * inv_r * inv_r * inv_r
            - 6.0 * phi1 * inv_r * inv_r * inv_r * inv_r;
        let t = q_r * inv_r;
        let t_r = q_rr * inv_r - q_r * inv_r * inv_r;
        let t_rr = q_rrr * inv_r - 2.0 * q_rr * inv_r * inv_r + 2.0 * q_r * inv_r * inv_r * inv_r;
        Ok(DuchonRegularizedOperatorCore { q, t, t_r, t_rr })
    }
}

/// The bound hybrid evaluator for these orders, or `None` when the orders or
/// the missing length scale put the evaluation on the partial-fraction path.
///
/// One owner for the question every caller of the hybrid kernel asks before
/// its own loop: is this shape on the cancellation-free single-integral path,
/// and at which `κ`?
pub(crate) fn duchon_hybrid_evaluator(
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
) -> Result<Option<DuchonHybridEvaluator>, BasisError> {
    let Some(length_scale) = length_scale else {
        return Ok(None);
    };
    if !duchon_hybrid_stable_integral_applies(p_order, s_order, k_dim) {
        return Ok(None);
    }
    let kappa = duchon_inverse_length_scale(length_scale, "Duchon hybrid kernel")?;
    Ok(Some(DuchonHybridEvaluator::new(
        kappa, p_order, s_order, k_dim,
    )?))
}

/// Whether the cancellation-free [`duchon_hybrid_kernel_stable_integral`] is
/// applicable for these orders: a genuine Matérn blend (`s ≥ 1`) whose
/// single-integral reduction has an integrable `w → 0` endpoint (`2p < d`).
///
/// The complementary cases — `s = 0` (pure polyharmonic, already evaluated
/// directly with no cancellation), `p = 0` (a bare Matérn block: the Schwinger
/// parametrization behind the single integral carries `1/Γ(p)`, so the
/// reduction does not exist there and the previous evaluator returned exactly
/// `0` for it) and `2p ≥ d` (only reachable at low `d`, where the
/// partial-fraction sum has no meaningful cancellation) — retain the existing
/// partial-fraction path.
#[inline]
pub(crate) fn duchon_hybrid_stable_integral_applies(
    p_order: usize,
    s_order: usize,
    k_dim: usize,
) -> bool {
    p_order >= 1 && s_order >= 1 && 2 * p_order < k_dim
}

pub(crate) fn duchon_matern_kernel_general_from_distance(
    r: f64,
    length_scale: Option<f64>,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: Option<&DuchonPartialFractionCoeffs>,
) -> Result<f64, BasisError> {
    if !r.is_finite() || r < 0.0 {
        crate::bail_invalid_basis!("Duchon kernel distance must be finite and non-negative");
    }
    let Some(length_scale) = length_scale else {
        return Ok(polyharmonic_kernel(
            r,
            pure_duchon_block_order(p_order, s_order as f64),
            k_dim,
        ));
    };
    let kappa = duchon_inverse_length_scale(length_scale, "Duchon hybrid kernel")?;

    // gam#1424: for genuine high-dimensional Matérn blends the partial-fraction
    // sum below cancels catastrophically (the largest block dwarfs the true
    // ~1e-13 kernel value, destroying every significant digit and the PSD
    // property of the Gram matrix). Evaluate those orders with the
    // cancellation-free single-integral form instead — it also handles the
    // `r = 0` diagonal in closed form, so it short-circuits before the
    // near-collision Taylor branch.
    if duchon_hybrid_stable_integral_applies(p_order, s_order, k_dim) {
        return duchon_hybrid_kernel_stable_integral(r, kappa, p_order, s_order, k_dim);
    }

    let coeffs_local;
    let coeffs_ref = if let Some(c) = coeffs {
        c
    } else {
        coeffs_local = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
        &coeffs_local
    };
    let collision_taylor_radius = DUCHON_COLLISION_TAYLOR_REL * length_scale.max(1e-8);
    // The near-collision Taylor expansion uses phi(0) plus even-order
    // derivative collision limits. Those limits only exist when the kernel
    // is finite at the origin, i.e. when 2(p+s) > d. Below that threshold
    // the partial-fraction blocks individually diverge at r=0 but their
    // sum is still a well-defined function for any r > 0 (each Bessel-K
    // and r^{2m-d}-type block is finite away from origin). Fall through
    // to the direct sum in that regime; r=0 itself remains an error.
    let kernel_finite_at_origin = 2 * (p_order + s_order) > k_dim;
    if r <= collision_taylor_radius && kernel_finite_at_origin {
        return duchon_hybrid_kernel_near_collision_value(
            r,
            length_scale,
            p_order,
            s_order,
            k_dim,
            coeffs_ref,
        );
    }
    let mut val = KahanSum::default();
    for (m, coeff) in coeffs_ref.a.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val.add(coeff * polyharmonic_kernel(r, (m) as f64, k_dim));
    }
    for (n, coeff) in coeffs_ref.b.iter().enumerate().skip(1) {
        if *coeff == 0.0 {
            continue;
        }
        val.add(coeff * duchon_matern_block(r, kappa, n, k_dim)?);
    }
    Ok(val.sum())
}

pub(crate) fn duchon_hybrid_kernel_collision_value(
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    let spectral_order = 2 * (p_order + s_order);
    if spectral_order <= k_dim {
        return Err(BasisError::duchon_smoothness_insufficient(
            "hybrid diagonal",
            0,
            k_dim,
            p_order,
            s_order as f64,
        ));
    }

    let kappa = duchon_inverse_length_scale(length_scale, "Duchon hybrid collision value")?;
    let mut pure = KahanSum::default();
    let mut log_part = KahanSum::default();
    for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
        if a_m == 0.0 {
            continue;
        }
        let (block_pure, block_log) = duchon_polyharmonic_block_taylor_r2j(m, k_dim, 0);
        pure.add(a_m * block_pure);
        log_part.add(a_m * block_log);
    }
    for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
        if b_n == 0.0 {
            continue;
        }
        let (block_pure, block_log) = duchon_matern_block_taylor_r2j(kappa, n, k_dim, 0);
        pure.add(b_n * block_pure);
        log_part.add(b_n * block_log);
    }
    let value = pure.sum();
    let log_value = log_part.sum();
    if log_value.abs() > 1e-8 * value.abs().max(1e-30) {
        crate::bail_invalid_basis!(
            "Duchon hybrid diagonal log terms did not cancel: log={log_value:.6e}, value={value:.6e}; p={p_order}, s={s_order}, d={k_dim}"
        );
    }
    if !value.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Duchon hybrid diagonal value for p={p_order}, s={s_order}, d={k_dim}"
        );
    }
    Ok(value)
}

pub(crate) fn duchon_hybrid_kernel_near_collision_value(
    r: f64,
    length_scale: f64,
    p_order: usize,
    s_order: usize,
    k_dim: usize,
    coeffs: &DuchonPartialFractionCoeffs,
) -> Result<f64, BasisError> {
    let mut value =
        duchon_hybrid_kernel_collision_value(length_scale, p_order, s_order, k_dim, coeffs)?;
    if r == 0.0 {
        return Ok(value);
    }

    // Radial Taylor expansion about the center collision:
    //
    //   phi(r) = phi(0)
    //          + phi''(0) r^2 / 2
    //          + phi''''(0) r^4 / 24
    //          + phi''''''(0) r^6 / 720 + ...
    //
    // Odd terms vanish for an isotropic radial kernel. A finite 2q-th
    // derivative at zero requires spectral smoothness 2(p+s) > d + 2q.
    // Terms whose collision derivative does not exist are omitted; this is
    // still strictly better than evaluating the raw partial-fraction sum at a
    // tiny nonzero radius, where large singular components cancel only after
    // losing many digits.
    let smoothness_order = 2 * (p_order + s_order);
    let r2 = r * r;
    if smoothness_order > k_dim + 2 {
        let (phi_rr, _, _) =
            duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, coeffs)?;
        value += 0.5 * phi_rr * r2;
    }
    if smoothness_order > k_dim + 4 {
        let phi_rrrr = duchon_phi_rrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)?;
        value += (1.0 / 24.0) * phi_rrrr * r2 * r2;
    }
    if smoothness_order > k_dim + 6 {
        let phi_rrrrrr =
            duchon_phi_rrrrrr_collision(length_scale, p_order, s_order, k_dim, coeffs)?;
        value += (1.0 / 720.0) * phi_rrrrrr * r2 * r2 * r2;
    }
    if !value.is_finite() {
        crate::bail_invalid_basis!(
            "non-finite Duchon hybrid near-collision value at r={r}, p={p_order}, s={s_order}, d={k_dim}"
        );
    }
    Ok(value)
}

#[inline(always)]
pub(crate) fn stable_euclidean_norm<I>(components: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut scale = 0.0_f64;
    let mut sumsq = 1.0_f64;
    let mut has_nonzero = false;
    for component in components {
        let abs = component.abs();
        if abs == 0.0 {
            continue;
        }
        if !abs.is_finite() {
            return f64::INFINITY;
        }
        if !has_nonzero {
            scale = abs;
            has_nonzero = true;
            continue;
        }
        if scale < abs {
            let ratio = scale / abs;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = abs;
        } else {
            let ratio = abs / scale;
            sumsq += ratio * ratio;
        }
    }
    if has_nonzero {
        scale * sumsq.sqrt()
    } else {
        0.0
    }
}

#[inline]
pub(crate) fn centered_aniso_log_scale_mean(eta: &[f64]) -> f64 {
    if eta.len() <= 1 {
        0.0
    } else {
        eta.iter().sum::<f64>() / eta.len() as f64
    }
}

#[inline]
pub(crate) fn centered_aniso_log_scale(value: f64, mean: f64) -> f64 {
    // This bound exists solely to keep the downstream `.exp()` (axis scale and
    // metric weight) finite. `f64::clamp` leaves NaN as NaN, so a non-finite
    // contrast (e.g. an `inf − inf` from a degenerate anisotropy `eta`) would
    // slip through and poison the Gram matrix. Map any non-finite difference to
    // the saturating bound explicitly; finite inputs take the identical clamp.
    let centered = value - mean;
    if centered.is_finite() {
        centered.clamp(-50.0, 50.0)
    } else if centered > 0.0 {
        50.0
    } else {
        -50.0
    }
}

#[inline]
pub(crate) fn aniso_axis_scale(value: f64, mean: f64) -> f64 {
    centered_aniso_log_scale(value, mean).exp()
}

#[inline]
pub(crate) fn aniso_metric_weight(value: f64, mean: f64) -> f64 {
    (2.0 * centered_aniso_log_scale(value, mean)).exp()
}

pub(crate) fn centered_aniso_metric_weights(eta: &[f64]) -> Vec<f64> {
    let mean = centered_aniso_log_scale_mean(eta);
    eta.iter()
        .map(|&value| aniso_metric_weight(value, mean))
        .collect()
}

/// Compute anisotropic squared distance components and total distance.
///
/// This is the core of **geometric anisotropy**: a linear warp Λ = diag(κ_a)
/// turns ellipsoidal correlation contours into isotropic ones. Writing h = x − c,
/// z = Λh, the anisotropic distance is r = |z| = |Λh|.
///
/// We decompose Λ = κ · A where det(A) = 1, parameterized as
///   ψ_a = ψ̄ + η_a,   Σ η_a = 0
/// where ψ̄ is the global scale (existing scalar κ) and η_a are d−1 anisotropy
/// contrasts. This separates scale from shape and preserves the Duchon scaling
/// law φ(r;κ) = κ^δ H(κr) for the global part.
///
/// Given per-axis log-scales `eta`, the identifiable centered contrasts are
/// ψ_a = eta_a - mean(eta). The metric uses those contrasts so Σ_a ψ_a = 0
/// even when a caller passes an uncentered vector:
///
///   r = √( Σ_a exp(2·ψ_a) · (x_a - c_a)² )
///
/// Returns `(r, s_vec)` where `s_vec[a] = exp(2·ψ_a) · h_a²` is the
/// per-axis weighted squared displacement. These components are needed for
/// per-axis derivatives: `∂φ/∂ψ_a = q · s_a`.
///
/// The derivative chain through r gives:
///   ∇_ψ r      = s / r
///   ∇²_ψ r     = (2/r) Diag(s) − (1/r³) ss'
/// which is diagonal + rank-1, so Hessian-vector products are O(d).
#[inline]
pub(crate) fn aniso_distance_and_components(
    data_row: &[f64],
    center: &[f64],
    eta: &[f64],
) -> (f64, Vec<f64>) {
    assert_eq!(data_row.len(), center.len());
    assert_eq!(data_row.len(), eta.len());
    let d = data_row.len();
    let eta_mean = centered_aniso_log_scale_mean(eta);
    let mut s_vec = Vec::with_capacity(d);
    let mut scaled_components = Vec::with_capacity(d);
    for a in 0..d {
        let h_a = data_row[a] - center[a];
        // Clamp exp(2ψ) to avoid overflow/underflow: ψ in [-50, 50].
        let scale_a = aniso_axis_scale(eta[a], eta_mean);
        let scaled_h_a = scale_a * h_a;
        let s_a = scaled_h_a * scaled_h_a;
        scaled_components.push(scaled_h_a);
        s_vec.push(s_a);
    }
    (stable_euclidean_norm(scaled_components), s_vec)
}

/// Compute anisotropic distance without returning per-axis components.
///
/// This is the lightweight version of [`aniso_distance_and_components`] for
/// call sites that only need the scalar distance `r`.
#[inline]
pub(crate) fn aniso_distance(data_row: &[f64], center: &[f64], eta: &[f64]) -> f64 {
    assert_eq!(data_row.len(), center.len());
    assert_eq!(data_row.len(), eta.len());
    let eta_mean = centered_aniso_log_scale_mean(eta);
    stable_euclidean_norm(
        (0..data_row.len()).map(|a| aniso_axis_scale(eta[a], eta_mean) * (data_row[a] - center[a])),
    )
}

/// [`aniso_distance`] with the per-axis scales precomputed once by
/// [`aniso_axis_scales`]. The per-call form recomputes the log-scale mean and
/// `d` exponentials for every `(row, center)` pair — `1.5e8` exponentials per
/// design pass at `n = 50,000`, `k = 500`, `d = 6`, measured as 2.6–3.8 % of a
/// Duchon fit (`expf32x`) before the hot loops moved here.
#[inline]
pub(crate) fn aniso_distance_with_scales(data_row: &[f64], center: &[f64], scales: &[f64]) -> f64 {
    assert_eq!(data_row.len(), center.len());
    assert_eq!(data_row.len(), scales.len());
    stable_euclidean_norm((0..data_row.len()).map(|a| scales[a] * (data_row[a] - center[a])))
}

/// [`aniso_distance_and_components`] with precomputed scales, writing the
/// per-axis squared scaled components into `s_vec` instead of allocating them.
#[inline]
pub(crate) fn aniso_distance_and_components_with_scales(
    data_row: &[f64],
    center: &[f64],
    scales: &[f64],
    s_vec: &mut [f64],
) -> f64 {
    assert_eq!(data_row.len(), center.len());
    assert_eq!(data_row.len(), scales.len());
    assert_eq!(data_row.len(), s_vec.len());
    for a in 0..data_row.len() {
        let scaled_h_a = scales[a] * (data_row[a] - center[a]);
        s_vec[a] = scaled_h_a * scaled_h_a;
    }
    stable_euclidean_norm((0..data_row.len()).map(|a| scales[a] * (data_row[a] - center[a])))
}

#[inline(always)]
pub(crate) fn euclidean_distance_rows(
    lhs: ArrayView2<'_, f64>,
    lhs_row: usize,
    rhs: ArrayView2<'_, f64>,
    rhs_row: usize,
) -> f64 {
    assert_eq!(lhs.ncols(), rhs.ncols());
    stable_euclidean_norm((0..lhs.ncols()).map(|axis| lhs[[lhs_row, axis]] - rhs[[rhs_row, axis]]))
}

#[inline(always)]
pub(crate) fn aniso_axis_scales(eta: &[f64]) -> Vec<f64> {
    let eta_mean = centered_aniso_log_scale_mean(eta);
    eta.iter()
        .map(|&value| aniso_axis_scale(value, eta_mean))
        .collect()
}

#[inline(always)]
pub(crate) fn aniso_distance_rows_with_scales(
    lhs: ArrayView2<'_, f64>,
    lhs_row: usize,
    rhs: ArrayView2<'_, f64>,
    rhs_row: usize,
    axis_scales: &[f64],
) -> f64 {
    assert_eq!(lhs.ncols(), rhs.ncols());
    assert_eq!(lhs.ncols(), axis_scales.len());
    stable_euclidean_norm(
        (0..lhs.ncols())
            .map(|axis| axis_scales[axis] * (lhs[[lhs_row, axis]] - rhs[[rhs_row, axis]])),
    )
}

pub(crate) fn fill_symmetric_from_row_kernel<F>(
    matrix: &mut Array2<f64>,
    kernel: F,
) -> Result<(), BasisError>
where
    F: Fn(usize, usize) -> Result<f64, BasisError> + Sync,
{
    assert_eq!(matrix.nrows(), matrix.ncols());
    let k = matrix.nrows();
    // The kernels passed here are pure functions of the (symmetric) pairwise
    // center distance, so `kernel(i, j) == kernel(j, i)`. Evaluate only the
    // upper triangle (including the diagonal) in parallel — each row task
    // touches only its own `j >= i` cells, so the borrows stay disjoint — then
    // mirror into the lower triangle. This halves the (sqrt + special-function)
    // kernel evaluations relative to filling every cell independently, with no
    // change to the resulting matrix (still exactly symmetric).
    matrix
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .try_for_each(|(i, mut row)| {
            for j in i..k {
                row[j] = kernel(i, j)?;
            }
            Ok::<(), BasisError>(())
        })?;
    for i in 1..k {
        for j in 0..i {
            matrix[[i, j]] = matrix[[j, i]];
        }
    }
    Ok(())
}

/// Return y-space points `y_{i,a} = exp(ψ_a) x_{i,a}` with
/// `ψ_a = η_a - mean(η)` so Euclidean pairwise
/// distances in y equal anisotropic kernel distances in x:
///   |y_i - y_j|² = Σ_a exp(2 ψ_a) (x_{i,a} - x_{j,a})² = aniso_distance²(x_i, x_j, η).
/// Use this before `pairwise_distance_bounds` whenever κ conditioning
/// bounds must match the kernel's actual metric (anisotropic case). For
/// isotropic terms, pass `None` and keep using the raw centers.
pub(crate) fn points_in_aniso_y_space(points: ArrayView2<'_, f64>, eta: &[f64]) -> Array2<f64> {
    assert_eq!(points.ncols(), eta.len());
    let mut y = points.to_owned();
    let eta_mean = centered_aniso_log_scale_mean(eta);
    let weights: Vec<f64> = eta.iter().map(|&e| aniso_axis_scale(e, eta_mean)).collect();
    for a in 0..eta.len() {
        let w_a = weights[a];
        y.column_mut(a).mapv_inplace(|v| v * w_a);
    }
    y
}

/// Compute per-axis standard deviations of knot center coordinates.
///
/// Returns σ_a for each axis column of `centers`. Axes with zero variance
/// (constant column) get σ_a = 1.0. All values are clamped to [1e-6, 1e6].
pub fn knot_cloud_axis_scales(centers: ArrayView2<'_, f64>) -> Vec<f64> {
    let k = centers.nrows();
    let d = centers.ncols();
    if k < 2 || d == 0 {
        return vec![1.0; d];
    }
    let n = k as f64;
    let mut scales = Vec::with_capacity(d);
    for a in 0..d {
        let col = centers.column(a);
        let mean = col.sum() / n;
        let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let sigma = var.sqrt();
        // If variance is zero (constant column), use 1.0 (no scaling).
        let sigma = if sigma < 1e-12 { 1.0 } else { sigma };
        scales.push(sigma.clamp(1e-6, 1e6));
    }
    scales
}

/// Compute initial anisotropy contrasts η_a from knot center geometry.
///
/// Returns η_a = −ln(σ_a) + (1/d) Σ_b ln(σ_b), which satisfies Ση_a = 0
/// by construction. Axes with more spread get negative η_a (smaller κ_a,
/// longer correlation range), axes with less spread get positive η_a.
///
/// If d ≤ 1, returns an empty vector (anisotropy is meaningless for 1-D).
pub fn initial_aniso_contrasts(centers: ArrayView2<'_, f64>) -> Vec<f64> {
    let d = centers.ncols();
    if d <= 1 {
        return Vec::new();
    }
    let scales = knot_cloud_axis_scales(centers);
    let mean_neg_log: f64 = scales.iter().map(|&s| -s.ln()).sum::<f64>() / d as f64;
    // η_a = −ln(σ_a) + (1/d) Σ_b ln(σ_b)
    //     = −ln(σ_a) − mean(−ln(σ_b))
    //     = neg_log_scales[a] − mean(neg_log_scales)
    scales
        .iter()
        .map(|&scale| -scale.ln() - mean_neg_log)
        .collect()
}

/// Pure forward transform of the supplied anisotropy log-scales: subtract the
/// mean (so Σ η = 0) and zero tiny residuals. `None` (or a 1-D problem, where
/// centering is a no-op) means *no* anisotropy.
///
/// This is a **continuous function of η with no hidden data dependence**: an
/// explicit all-zero vector centers to all-zero, i.e. the isotropic metric
/// (weights `exp(2·0) = 1`, Euclidean radius). It is therefore identical, as a
/// design, to the `None` path through `η = 0`, and is continuous across it —
/// `[1e-9, -1e-9]` and `[0, 0]` map to neighboring designs, not a jump.
///
/// The Matérn input-location jet/Hessian (`matern_metric_weights`, the public
/// `matern_input_location_first_jet`/`_hessian` FFI) and the `UserProvided`-center
/// forward design both apply *this* transform, so the jet differentiates exactly
/// the function the public design evaluates (#437), and an explicit isotropic
/// request reduces to the closed-form isotropic Matérn kernel rather than a
/// data-driven anisotropic one (#1042).
///
/// Auto-initialization of `η` from knot-cloud geometry is a *separate* concern
/// handled by [`auto_seed_aniso_contrasts`]; it is reserved for callers that
/// opt into data-derived geometry (the κ-optimizer's data-driven center
/// strategies and the pure-Duchon `scale_dims` path), selected by
/// [`resolve_matern_forward_aniso`].
pub(crate) fn centered_aniso_contrasts(aniso: Option<&[f64]>) -> Option<Vec<f64>> {
    match aniso {
        Some(v) if v.len() > 1 => Some(center_aniso_log_scales(v)),
        Some(v) => Some(v.to_vec()),
        None => None,
    }
}

/// Auto-seed anisotropy contrasts from knot-cloud geometry for callers that use
/// an all-zero vector as the "initialize me" sentinel.
///
/// Used by (a) the pure-Duchon `scale_dims` path, where `η` is a FIXED,
/// geometry-derived basis parameter that is never enrolled as a REML hyper-axis
/// (see `spatial_term_supports_hyper_optimization`): "standardize the geometry,
/// then learn the smoothness"; and (b) the Matérn forward design when the term
/// uses a **data-driven** center strategy, i.e. the κ-optimizer's seeding
/// sentinel (the optimizer's analytic ψ-gradient is computed against the same
/// auto-seeded design, so the pair stays consistent). A non-zero (or absent)
/// vector is honored verbatim (centered, exactly like [`centered_aniso_contrasts`]);
/// only an *exactly* all-zero vector is replaced by `initial_aniso_contrasts(centers)`.
///
/// A `UserProvided`-center Matérn term does NOT use this — its geometry is fully
/// caller-specified, so an explicit all-zero η must be honored literally; folding
/// the geometry seed into that path made the public design discontinuous at
/// `η = 0` and hijacked explicit isotropic requests (#1042).
pub(crate) fn auto_seed_aniso_contrasts(
    centers: ArrayView2<'_, f64>,
    aniso: Option<&[f64]>,
) -> Option<Vec<f64>> {
    let eta = match aniso {
        Some(v) if v.len() > 1 => v,
        Some(v) => return Some(v.to_vec()),
        None => return None,
    };
    let all_zero = eta.iter().all(|&e| e == 0.0);
    if !all_zero {
        return Some(center_aniso_log_scales(eta));
    }
    let contrasts = initial_aniso_contrasts(centers);
    if contrasts.is_empty() {
        Some(center_aniso_log_scales(eta))
    } else {
        Some(center_aniso_log_scales(&contrasts))
    }
}

/// How the Matérn forward design build interprets an *exactly all-zero*
/// `aniso_log_scales` vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnisoSeedMode {
    /// All-zero `η` is the κ-optimizer / `scale_dims` seeding sentinel: replace
    /// it with geometry-derived contrasts from the knot cloud
    /// (`auto_seed_aniso_contrasts`). This is the default for every internal
    /// build entry; the optimizer's analytic ψ-gradient is computed against the
    /// same auto-seeded design, so value/gradient stay consistent. Note that by
    /// the time the κ-optimizer rebuilds a frozen design the center strategy has
    /// usually been resolved to `UserProvided`, so center provenance cannot be
    /// used to distinguish this from a genuine literal request — the mode must
    /// be carried explicitly.
    AutoSeedFromGeometry,
    /// All-zero `η` is an explicit isotropic request and is honored literally
    /// (`centered_aniso_contrasts`): the design reduces to the closed-form
    /// isotropic Matérn and varies continuously through `η = 0`. The public
    /// `matern_basis` FFI (and its input-location jet/Hessian) selects this so a
    /// caller's explicit isotropic request is not hijacked into a data-driven
    /// anisotropic kernel (#1042).
    Literal,
}

/// Resolve the anisotropy contrasts the Matérn forward design build applies,
/// dispatching on the explicit [`AnisoSeedMode`].
pub(crate) fn resolve_matern_forward_aniso(
    mode: AnisoSeedMode,
    centers: ArrayView2<'_, f64>,
    aniso: Option<&[f64]>,
) -> Option<Vec<f64>> {
    match mode {
        AnisoSeedMode::Literal => centered_aniso_contrasts(aniso),
        AnisoSeedMode::AutoSeedFromGeometry => auto_seed_aniso_contrasts(centers, aniso),
    }
}

pub(crate) fn pairwise_distance_bounds(points: ArrayView2<'_, f64>) -> Option<(f64, f64)> {
    let n = points.nrows();
    let d = points.ncols();
    if n < 2 || d == 0 {
        return None;
    }
    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = stable_euclidean_norm((0..d).map(|c| points[[i, c]] - points[[j, c]]));
            if r.is_finite() && r > 0.0 {
                r_min = r_min.min(r);
                r_max = r_max.max(r);
            }
        }
    }
    if r_min.is_finite() && r_max.is_finite() && r_min > 0.0 && r_max > 0.0 {
        Some((r_min, r_max))
    } else {
        None
    }
}

/// Capped-sample pairwise distance bounds for large point clouds.
///
/// Returns `(r_min_hat, r_max_hat)` such that:
/// - `r_max_hat <= true r_max`  (pairwise max over a sub-sample is monotone
///    in the sample, so the sampled max underestimates the true max).
/// - `r_min_hat >= true r_min`  (pairwise min over a sub-sample can only
///    exclude some pairs, so the sampled min overestimates the true min).
///
/// Both approximations are conservative for κ-bound derivation:
///   kappa_lo = 1e-2 / r_max_hat  >=  1e-2 / true r_max  (wider window, low κ)
///   kappa_hi = 1e2  / r_min_hat  <=  1e2  / true r_min  (tighter window, high κ)
/// so no feasible κ that the exact bound would include is excluded by the
/// approximation — it can only slightly shrink the high-κ tail, which is
/// exactly the regime (κ → ∞ ⇒ degenerate kernel) that we want the outer
/// optimizer to avoid anyway.
///
/// Sampling picks `K = 1024` indices spaced evenly across the FULL index range
/// `[0, n-1]` (endpoints included): `idx(s) = round(s·(n-1)/(K-1))`. For a cap
/// of `K = 1024` and n up to ~10⁹ this yields O(K²·d) work per call — a few
/// hundred μs. For n ≤ K the exact pairwise is used.
///
/// #1033: spanning the full range (rather than the contiguous prefix `0,
/// stride, …, (K-1)·stride` that an integer `stride = n/K` produces — which
/// floors to `1` for every n in `(K, 2K]` and so visits only indices `0..K`,
/// dropping the entire tail of the cloud) is what makes the diameter estimate
/// `r_max_hat` n-STABLE. The κ/ψ window is derived once from `r_max_hat`
/// (`psi_lo = ln(diameter_fraction / r_max_hat)`); a prefix-sampled `r_max_hat`
/// shrinks with the prefix's spatial extent as n crosses `K`, which silently
/// makes the outer optimizer's box — and therefore its whole trajectory —
/// n-dependent. Even spacing keeps `r_max_hat` ≈ the true cloud diameter at
/// every n, so the sufficient-statistic outer loop touches the same ψ window
/// regardless of n.
pub(crate) fn pairwise_distance_bounds_sampled(points: ArrayView2<'_, f64>) -> Option<(f64, f64)> {
    const K_CAP: usize = 1024;
    let n = points.nrows();
    let d = points.ncols();
    if n < 2 || d == 0 {
        return None;
    }
    if n <= K_CAP {
        return pairwise_distance_bounds(points);
    }
    // Evenly spaced indices spanning `[0, n-1]` inclusive (n > K_CAP ⇒ k = K_CAP
    // ≥ 2, so the denominator is positive and `idx(0)=0`, `idx(k-1)=n-1`). The
    // spacing `(n-1)/(k-1) > 1`, so distinct `s` map to distinct rows; any rare
    // rounding collision is harmless (the `r > 0.0` guard drops a zero pair).
    let k = K_CAP;
    let denom = (k - 1) as f64;
    let span = (n - 1) as f64;
    let sample_index = |s: usize| -> usize { ((s as f64) * span / denom).round() as usize };
    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;
    for i_idx in 0..k {
        let i = sample_index(i_idx);
        for j_idx in (i_idx + 1)..k {
            let j = sample_index(j_idx);
            let r = stable_euclidean_norm((0..d).map(|c| points[[i, c]] - points[[j, c]]));
            if r.is_finite() && r > 0.0 {
                r_min = r_min.min(r);
                r_max = r_max.max(r);
            }
        }
    }
    if r_min.is_finite() && r_max.is_finite() && r_min > 0.0 && r_max > 0.0 {
        Some((r_min, r_max))
    } else {
        None
    }
}

#[cfg(test)]
mod bessel_k_accuracy_tests {
    use super::*;

    /// `K₀` and `K₁` against a 50-digit `mpmath.besselk`, across both branches.
    ///
    /// The small-argument branch (`x ≤ 2`) was always accurate — it sums a
    /// convergent series to an `f64::EPSILON` break. The large-argument branch
    /// was the A&S 9.8.6 / 9.8.8 minimax polynomials at `1.6e−7`, so the pair
    /// disagreed by `2.9e−9` across their own crossover: a jump discontinuity
    /// in a radial kernel that `duchon_matern_block_jet4` differentiates with
    /// respect to the length scale.
    #[test]
    fn bessel_k_matches_independent_high_precision_reference() {
        // (x, K₀(x), K₁(x))
        const BESSEL_K_REFERENCE: [[f64; 3]; 20] = [
            [1e-08, 18.536612259610777, 99999999.9999999],
            [0.0001, 9.326271913450276, 9999.999508686404],
            [0.01, 4.721244730161095, 99.97389411829624],
            [0.1, 2.4270690247020164, 9.853844780870606],
            [0.5, 0.9244190712276659, 1.656441120003301],
            [1.0, 0.42102443824070834, 0.6019072301972346],
            [1.5, 0.21380556264752573, 0.2773878004568438],
            [1.99, 0.1153017675517768, 0.14171756162240132],
            [2.0, 0.11389387274953344, 0.13986588181652243],
            [2.01, 0.11250436099872804, 0.1380408773192077],
            [2.5, 0.06234755320036619, 0.07389081634774707],
            [3.0, 0.03473950438627925, 0.040156431128194184],
            [5.0, 0.0036910983340425942, 0.004044613445452165],
            [8.0, 0.0001464707052228154, 0.00015536921180500115],
            [12.0, 2.2008253973114916e-06, 2.290757464767188e-06],
            [20.0, 5.741237815336525e-10, 5.883057969557038e-10],
            [50.0, 3.4101677497894956e-23, 3.4441022267175555e-23],
            [150.0, 7.336371406107646e-67, 7.36078548876807e-67],
            [400.0, 1.199780043200976e-175, 1.2012788332610325e-175],
            [700.0, 4.669776431685377e-306, 4.6731107967079664e-306],
        ];

        // The ascending series carries `−[ln(x/2)+γ]·I₀(x)` against a positive
        // sum, and both grow like `e^x` while `K` decays like `e^{−x}`, so it
        // loses `~e^{2x}` — about 1.7 digits at the `x = 2` crossover and less
        // below. The Chebyshev branch is limited only by its own Horner.
        const TOLERANCE: f64 = 8e-15;
        for [x, want_k0, want_k1] in BESSEL_K_REFERENCE {
            for (order, got, want) in [
                (0, bessel_k0_stable(x), want_k0),
                (1, bessel_k1_stable(x), want_k1),
            ] {
                let error = (got - want).abs() / want.abs();
                assert!(
                    error < TOLERANCE,
                    "K{order}({x}): got {got:.17e}, want {want:.17e} (rel {error:.3e})"
                );
            }
        }
    }

    /// The two branches must not be distinguishable at their crossover. Before
    /// the Chebyshev fit they stepped by `2.9e−9` (`K₀`) and `2.4e−9` (`K₁`).
    #[test]
    fn bessel_k_branch_crossover_has_no_step() {
        // 1e-13 is ~230 ulp of 2.0, so the two sides land in different branches
        // while the true functions have barely moved.
        let delta = 1.0e-13;
        // `dK₀/dx = −K₁` and `dK₁/dx = −K₀ − K₁/x`, so `K₀(2) + K₁(2)` bounds
        // both slopes at the crossover.
        let slope = bessel_k0_stable(2.0) + bessel_k1_stable(2.0);
        for (order, f) in [
            (0, bessel_k0_stable as fn(f64) -> f64),
            (1, bessel_k1_stable),
        ] {
            let below = f(2.0 - delta);
            let above = f(2.0 + delta);
            let budget = 2.0 * delta * slope + 8.0 * f64::EPSILON * below.abs();
            assert!(
                (above - below).abs() < budget,
                "K{order} steps at the x=2 crossover: {below:.17e} -> {above:.17e} \
                 (change {:.3e} > budget {budget:.3e})",
                (above - below).abs()
            );
        }
    }

    /// The crate carries a SECOND `K` — `closed_form_penalty::bessel_k`, a
    /// Temme-series/Steed-continued-fraction evaluator for arbitrary real order.
    /// It is full precision and always was, so before the Chebyshev fit the two
    /// implementations of `K₀`/`K₁` in this crate disagreed by up to `1.6e−7`
    /// depending on which one a caller happened to reach. They must not.
    #[test]
    fn the_two_bessel_k_implementations_in_this_crate_agree() {
        use crate::basis::closed_form_penalty::bessel_k;
        for x in [
            0.01_f64, 0.1, 0.5, 1.0, 1.99, 2.0, 2.01, 2.5, 4.0, 7.0, 15.0, 40.0, 120.0,
        ] {
            for (order, fast) in [(0.0_f64, bessel_k0_stable(x)), (1.0, bessel_k1_stable(x))] {
                let general = bessel_k(order, x);
                let error = (fast - general).abs() / general.abs();
                assert!(
                    error < 1e-13,
                    "K{order}({x}): fast path {fast:.17e} vs Temme/Steed {general:.17e} \
                     (rel {error:.3e})"
                );
            }
        }
    }

    /// `K` obeys `K_{ν−1}(x) − K_{ν+1}(x) = −(2ν/x)·K_ν(x)` and the Wronskian
    /// `I₀(x)K₁(x) + I₁(x)K₀(x) = 1/x`. The Wronskian ties `K` to an evaluator
    /// it shares no code with (`gam_math`'s modified Bessel `I`), so it is a
    /// genuine cross-check rather than a restatement of either one.
    #[test]
    fn bessel_k_satisfies_the_wronskian_against_bessel_i() {
        for x in [
            0.05_f64, 0.5, 1.0, 1.99, 2.0, 2.01, 3.0, 6.0, 12.0, 30.0, 80.0,
        ] {
            let (centered_log_i0, ratio, _) = gam_math::special::bessel_i0_centered_terms(x);
            // I₀ = exp(centered + x); I₁ = ratio·I₀. Both are formed here rather
            // than cancelled against K, so the identity is checked on the values
            // themselves.
            let i0 = (centered_log_i0 + x).exp();
            let i1 = ratio * i0;
            let wronskian = i0 * bessel_k1_stable(x) + i1 * bessel_k0_stable(x);
            let want = 1.0 / x;
            let error = (wronskian - want).abs() / want;
            // `I₀` reaches 1e34 by x = 80 while `K₀` is 1e-36, so the product is
            // formed from numbers whose exponents differ by 70; the tolerance
            // tracks that reconstruction, not the evaluators.
            assert!(
                error < 1e-13,
                "Wronskian at x={x}: got {wronskian:.17e}, want {want:.17e} (rel {error:.3e})"
            );
        }
    }
}

#[cfg(test)]
mod duchon_hybrid_psd_tests {
    use super::*;
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;

    fn assert_pow_parity(label: &str, got: f64, reference: f64) {
        if got.to_bits() == reference.to_bits() || (got.is_nan() && reference.is_nan()) {
            return;
        }
        if got.is_infinite() || reference.is_infinite() {
            assert_eq!(got, reference, "{label}: infinity/sign mismatch");
            return;
        }
        let scale = got.abs().max(reference.abs()).max(f64::MIN_POSITIVE);
        let relative = (got - reference).abs() / scale;
        assert!(
            relative <= 2.0e-12 || (got - reference).abs() <= 1.0e-300,
            "{label}: got {got:.17e}, powf reference {reference:.17e}, relative error {relative:.3e}"
        );
    }

    fn powf_polyharmonic_constants(m: f64, d: usize) -> (f64, f64, bool) {
        let half_d = 0.5 * d as f64;
        let alpha = 2.0 * m - d as f64;
        let log_case = d.is_multiple_of(2) && alpha >= 0.0 && (alpha % 2.0).abs() < 1.0e-12;
        let c = if log_case {
            let m_int = m.round() as usize;
            polyharmonic_log_sign(m_int, d)
                / (2.0_f64.powi((2 * m_int - 1) as i32)
                    * std::f64::consts::PI.powf(half_d)
                    * gamma_lanczos(m)
                    * gamma_lanczos((m_int - d / 2 + 1) as f64))
        } else {
            gamma_lanczos(half_d - m)
                / (4.0_f64.powf(m) * std::f64::consts::PI.powf(half_d) * gamma_lanczos(m))
        };
        (c, alpha, log_case)
    }

    fn powf_family_value(r: f64, c: f64, exponent: f64, log: f64, pure: f64) -> f64 {
        if r <= 0.0 {
            log_power_origin_limit(c, exponent, log, pure)
        } else {
            c * r.powf(exponent) * (log * r.ln() + pure)
        }
    }

    fn differentiate_powf_family(exponent: &mut f64, log: &mut f64, pure: &mut f64) {
        let old_exponent = *exponent;
        *exponent -= 1.0;
        *pure = old_exponent * *pure + *log;
        *log *= old_exponent;
    }

    fn powf_operator_reference(r: f64, m: usize, d: usize) -> [f64; 4] {
        let (c, alpha, log_case) = powf_polyharmonic_constants(m as f64, d);
        let (mut exponent, mut log, mut pure) = if log_case {
            (alpha, 1.0, 0.0)
        } else {
            (alpha, 0.0, 1.0)
        };
        differentiate_powf_family(&mut exponent, &mut log, &mut pure);
        exponent -= 1.0; // q = phi'/r
        let q = powf_family_value(r, c, exponent, log, pure);
        differentiate_powf_family(&mut exponent, &mut log, &mut pure);
        exponent -= 1.0; // t = q'/r
        let t = powf_family_value(r, c, exponent, log, pure);
        differentiate_powf_family(&mut exponent, &mut log, &mut pure);
        let t_r = powf_family_value(r, c, exponent, log, pure);
        differentiate_powf_family(&mut exponent, &mut log, &mut pure);
        let t_rr = powf_family_value(r, c, exponent, log, pure);
        [q, t, t_r, t_rr]
    }

    #[test]
    fn pure_polyharmonic_integer_powers_match_powf_at_zero_tiny_and_large_radius() {
        let radii = [0.0_f64, 1.0e-40, 1.0e-12, 0.2, 1.0, 12.0, 1.0e40];
        for &(m, d) in &[(2usize, 1usize), (3, 2), (4, 5), (5, 6), (7, 9)] {
            let block = PolyharmonicBlockCoeff::new(m as f64, d);
            assert!(block.power_i32.is_some());
            let (reference_c, alpha, log_case) = powf_polyharmonic_constants(m as f64, d);
            assert_pow_parity("coefficient", block.c, reference_c);
            for &r in &radii {
                let reference_value = if r <= 0.0 {
                    block.origin_limit()
                } else if log_case {
                    reference_c * r.powf(alpha) * r.ln()
                } else {
                    reference_c * r.powf(alpha)
                };
                assert_pow_parity("block value", block.eval(r), reference_value);

                let got = polyharmonic_block_jet4(r, m as f64, d)
                    .expect("the polyharmonic block jet is defined at this fixture radius");
                let got = [got.0, got.1, got.2, got.3, got.4];
                for derivative in 0..5 {
                    let exponent = alpha - derivative as f64;
                    let falling = falling_factorial(alpha, derivative);
                    let reference = if log_case {
                        powf_family_value(
                            r,
                            reference_c,
                            exponent,
                            falling,
                            falling_factorial_derivative(alpha, derivative),
                        )
                    } else {
                        powf_family_value(r, reference_c, exponent, 0.0, falling)
                    };
                    assert_pow_parity("jet channel", got[derivative], reference);
                }

                let got = duchon_polyharmonic_operator_block_jets(r, m, d)
                    .expect("the operator block jets are defined at this fixture radius");
                let got = [got.0, got.1, got.2, got.3];
                let reference = powf_operator_reference(r, m, d);
                for channel in 0..4 {
                    assert_pow_parity("operator channel", got[channel], reference[channel]);
                }
            }
        }
    }

    #[test]
    fn fractional_polyharmonic_power_retains_powf_path() {
        let (m, d) = (2.125_f64, 3usize);
        let block = PolyharmonicBlockCoeff::new(m, d);
        assert_eq!(block.power, 1.25);
        assert!(block.power_i32.is_none());
        let (c, alpha, log_case) = powf_polyharmonic_constants(m, d);
        assert!(!log_case);
        for &r in &[0.0_f64, 1.0e-40, 0.25, 3.0, 1.0e40] {
            let reference = if r <= 0.0 {
                log_power_origin_limit(c, alpha, 0.0, 1.0)
            } else {
                c * r.powf(alpha)
            };
            assert_pow_parity("fractional block", block.eval(r), reference);
        }
    }

    #[test]
    fn pure_polyharmonic_powi_microbenchmark() {
        const N: usize = 20_000;
        let (m, d, r) = (7usize, 9usize, 0.731_f64);
        let start = std::time::Instant::now();
        let powf_sum = (0..N).fold(0.0, |sum, i| {
            let radius = std::hint::black_box(r + (i % 17) as f64 * 1.0e-6);
            sum + powf_operator_reference(radius, m, d)[0]
        });
        let powf_time = start.elapsed();
        let start = std::time::Instant::now();
        let powi_sum = (0..N).fold(0.0, |sum, i| {
            let radius = std::hint::black_box(r + (i % 17) as f64 * 1.0e-6);
            sum + duchon_polyharmonic_operator_block_jets(radius, m, d)
                .expect("the operator block jets are defined at this benchmark radius")
                .0
        });
        let powi_time = start.elapsed();
        assert_pow_parity("benchmark accumulator", powi_sum, powf_sum);
        eprintln!(
            "pure operator {N} calls: powf={powf_time:?}, powi={powi_time:?}, speedup={:.2}x",
            powf_time.as_secs_f64() / powi_time.as_secs_f64().max(f64::MIN_POSITIVE)
        );
    }

    /// #2278: the pure-Duchon CPD-adequacy boundary `2s >= d` is INDEPENDENT of
    /// the nullspace degree `p` (it cancels in the derivation above), so it must
    /// reject for all `p` — not only `p < 2`. Regression for the former spurious
    /// `p_order < 2` conjunct.
    #[test]
    fn pure_duchon_cpd_guard_is_nullspace_degree_independent_issue_2278() {
        // d = 2, Linear nullspace (p = 2), explicit integer power s = 1:
        // 2s = 2 >= d = 2 is ill-posed and previously slipped through.
        let err = validate_duchon_kernel_orders(None, 2, 1.0, 2)
            .expect_err("pure Duchon d=2, p=2, s=1 (2s>=d) must be rejected as ill-posed");
        let BasisError::InvalidInput(msg) = err else {
            panic!("expected an InvalidInput well-posedness error, got {err}");
        };
        assert!(
            msg.contains("dimension/2") || msg.contains("2s < d"),
            "message must name the CPD/well-posedness cause: {msg}"
        );
        // The p < 2 sibling that was already rejected still is (no regression).
        assert!(validate_duchon_kernel_orders(None, 1, 1.0, 2).is_err());
        // Control: a well-posed pure config (default fractional power
        // s = (d-1)/2 = 0.5 at p = 2, giving 2s = 1 < d = 2) must still build —
        // the guard must not over-reject.
        assert!(validate_duchon_kernel_orders(None, 2, 0.5, 2).is_ok());
        // Control: the hybrid (Matérn-blended) path is exempt (CPD order 0), so
        // even 2s >= d builds — the `length_scale.is_none()` gate is preserved.
        assert!(validate_duchon_kernel_orders(Some(1.0), 2, 1.0, 2).is_ok());
    }

    /// #1033: the capped-sample diameter estimate must be n-STABLE on a fixed
    /// point cloud. A uniform grid on `[-3, 3]` has a fixed true diameter (6.0)
    /// and minimum spacing that shrinks like `6/(n-1)` regardless of how finely
    /// it is sampled. The κ/ψ window is derived ONCE from `r_max_hat`, so if the
    /// sampler underestimates the diameter as n crosses the `K_CAP = 1024`
    /// threshold (the old `stride = n/K` prefix bug visited only indices
    /// `0..K`, i.e. the LEFT HALF of the domain for n in `(1024, 2048]`,
    /// halving `r_max_hat`), the outer optimizer's box — and hence its whole
    /// trajectory — becomes n-dependent, which is exactly the invariant #1033
    /// forbids. This pins `r_max_hat ≈ 6.0` across the threshold.
    #[test]
    fn sampled_diameter_is_n_stable_across_cap_threshold() {
        let grid = |n: usize| -> Array2<f64> {
            let mut x = Array2::<f64>::zeros((n, 1));
            for i in 0..n {
                x[[i, 0]] = (i as f64) / (n as f64 - 1.0) * 6.0 - 3.0;
            }
            x
        };
        // Below the cap (exact), straddling it, and well above it.
        let exact_diam = 6.0_f64;
        let mut last_rmax: Option<f64> = None;
        for &n in &[1000usize, 1025, 1500, 2000, 4000, 50_000] {
            let x = grid(n);
            let (r_min, r_max) =
                pairwise_distance_bounds_sampled(x.view()).expect("bounds for dense grid");
            // The sampled diameter must stay within 1% of the true 6.0 at every
            // n — NOT collapse to ~3.0 as the prefix bug did for n in (1024,2048].
            assert!(
                (r_max - exact_diam).abs() <= 0.01 * exact_diam,
                "sampled r_max at n={n} = {r_max:.6} drifted from the true diameter \
                 {exact_diam:.6}: the diameter estimate is n-dependent (#1033)"
            );
            // Cross-n stability: consecutive n's must agree on r_max to <2%.
            if let Some(prev) = last_rmax {
                let rel = (r_max - prev).abs() / exact_diam;
                assert!(
                    rel <= 0.02,
                    "sampled r_max jumped {rel:.4} (rel) between n steps near n={n}: \
                     {prev:.6} -> {r_max:.6}; outer-loop box is not n-stable (#1033)"
                );
            }
            last_rmax = Some(r_max);
            // r_min is positive and finite (the floor used for the high-κ ceiling).
            assert!(
                r_min.is_finite() && r_min > 0.0,
                "r_min must be positive at n={n}"
            );
        }
    }

    /// The sampler's chosen indices must span the FULL range `[0, n-1]`
    /// (endpoints included) — the property that makes the diameter estimate
    /// stable. Reconstruct the index set the implementation uses and assert it
    /// reaches both ends with no contiguous-prefix clustering.
    #[test]
    fn sampled_indices_span_full_range() {
        const K_CAP: usize = 1024;
        let n = 2000usize; // stride = n/K_CAP would floor to 1 → prefix bug regime
        let k = K_CAP;
        let denom = (k - 1) as f64;
        let span = (n - 1) as f64;
        let idx = |s: usize| -> usize { ((s as f64) * span / denom).round() as usize };
        assert_eq!(idx(0), 0, "first sample must be index 0");
        assert_eq!(idx(k - 1), n - 1, "last sample must be the final index n-1");
        // The largest gap between consecutive samples must be ≈ (n-1)/(k-1),
        // i.e. roughly 2 here — NOT a single dense prefix followed by a void.
        let mut max_gap = 0usize;
        for s in 1..k {
            max_gap = max_gap.max(idx(s) - idx(s - 1));
        }
        assert!(
            max_gap <= 2,
            "evenly-spaced samples should step by ~{:.2}; saw a gap of {max_gap} \
             (prefix clustering would leave one huge gap)",
            span / denom
        );
    }

    /// Deterministic, well-separated centers on `[-1, 1]^d` (a Halton-style
    /// low-discrepancy lattice over the radical-inverse base sequence). Mirrors
    /// the `4*d` random centers the Python fixture
    /// (`tests/test_python_api.py`'s high-dimensional hybrid Duchon penalty PSD
    /// check) draws, but without an RNG so the regression is byte-stable.
    fn fixture_centers(d: usize, n: usize) -> Array2<f64> {
        const BASES: [u64; 24] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
            89,
        ];
        let mut centers = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            for axis in 0..d {
                let base = BASES[axis % BASES.len()];
                // Van der Corput radical inverse of (i + 1) in `base`, mapped to
                // [-1, 1]. Different axes use different primes, so the cloud is
                // affinely full-rank and spans the linear null space.
                let mut f = 1.0_f64;
                let mut idx = (i + 1) as u64;
                let mut value = 0.0_f64;
                while idx > 0 {
                    f /= base as f64;
                    value += f * (idx % base) as f64;
                    idx /= base;
                }
                centers[[i, axis]] = 2.0 * value - 1.0;
            }
        }
        centers
    }

    /// Smallest symmetric eigenvalue of `matrix` (the matrix is symmetrized
    /// first; the constrained Duchon penalty is symmetric by construction).
    fn lambda_min(matrix: &Array2<f64>) -> f64 {
        let sym = symmetrize_penalty(matrix);
        let (evals, _) = FaerEigh::eigh(&sym, Side::Lower).expect("symmetric eigendecomposition");
        evals.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// gam#1424: the (d=16, m=2, s=7) hybrid Duchon–Matérn fixture used to lose
    /// positive definiteness through catastrophic cancellation in the
    /// partial-fraction kernel expansion — the constrained, post-normalization
    /// penalty had λ_min ≈ −0.26442 even though the kernel's spectral density
    /// `ρ^{-2p}(κ²+ρ²)^{-s}` is nonnegative (so the true penalty is PSD). The
    /// kernel now routes through the cancellation-free single-integral form, so
    /// the spectrum is numerically PSD. This mirrors the production penalty path
    /// `duchon_constrained_bending_penalty` → `normalize_penalty`.
    #[test]
    fn high_dim_hybrid_penalty_is_numerically_psd_1424() {
        let d = 16usize;
        // m=2 ⇒ Linear null space. The cubic default spectral power is the
        // fractional (d-1)/2 = 7.5; the production hybrid config resolves it to
        // the integer spectral order the closed-form kernel consumes, s = 7
        // (`duchon_constrained_bending_penalty` itself takes the integer view via
        // `duchon_power_to_usize`, and the reroute predicate needs s ≥ 1). This is
        // the (d=16, m=2, s=7) fixture from the issue and the Python
        // `duchon_function_norm_penalty` PSD test.
        let (nullspace_order, default_power) = duchon_cubic_default(d);
        assert!(matches!(nullspace_order, DuchonNullspaceOrder::Linear));
        assert!(
            (default_power - 7.5).abs() < 1e-12,
            "cubic-default power for d=16 is 7.5"
        );
        let power = 7.0_f64;
        assert_eq!(duchon_power_to_usize(power), 7);
        // The reroute must engage for this fixture (s = 7 ≥ 1, 2p = 4 < d = 16).
        assert!(duchon_hybrid_stable_integral_applies(
            duchon_p_from_nullspace_order(nullspace_order),
            duchon_power_to_usize(power),
            d,
        ));
        let length_scale = Some(1.0_f64);
        let centers = fixture_centers(d, 4 * d);

        let mut cache = BasisCacheContext::default();
        let z = kernel_constraint_nullspace(centers.view(), nullspace_order, &mut cache)
            .expect("constraint null space");

        let omega = duchon_constrained_bending_penalty(
            centers.view(),
            length_scale,
            power,
            nullspace_order,
            None,
            &z,
        )
        .expect("constrained bending penalty assembles for the hybrid fixture");
        let (penalty, _scale) = normalize_penalty(&omega);

        let lam_min = lambda_min(&penalty);
        assert!(
            lam_min >= -1e-10,
            "gam#1424: (d=16, m=2, s=7) hybrid penalty is not numerically PSD: \
             λ_min={lam_min:.6e} (was ≈ −0.26442 with the cancellation-prone \
             partial-fraction kernel)"
        );
    }

    /// gam#1604: independent closed form for the hybrid-kernel origin value
    /// `φ(0) = F⁻¹[ρ^{-2p}(κ²+ρ²)^{-s}](0)` in `d` dimensions, derived by
    /// Schwinger-parametrizing both rational factors and evaluating the radial
    /// inverse-FT integral at `r = 0`:
    ///
    ///   φ(0) = (4π)^{-d/2} / Γ(s) · Γ(b) · κ^{-2b} · Γ(d/2 − p) / Γ(d/2),
    ///   b = p + s − d/2.
    ///
    /// Finite whenever `2(p+s) > d` and (for the Γ(d/2 − p) factor to avoid a
    /// pole) `d` is odd or `2p < d`. This reuses none of the Taylor-coefficient
    /// machinery under test, so it is a true oracle for the collision diagonal.
    fn phi0_closed_form(p: usize, s: usize, d: usize, kappa: f64) -> f64 {
        let half_d = 0.5 * d as f64;
        let b = p as f64 + s as f64 - half_d;
        (4.0 * std::f64::consts::PI).powf(-half_d) / gamma_lanczos(s as f64)
            * gamma_lanczos(b)
            * kappa.powf(-2.0 * b)
            * gamma_lanczos(half_d - p as f64)
            / gamma_lanczos(half_d)
    }

    /// gam#1604 — the collision (r = 0) diagonal of the hybrid Duchon–Matérn
    /// kernel must equal the independent closed form above. The half-integer-ν
    /// Taylor coefficients that feed `duchon_hybrid_kernel_collision_value`
    /// previously miscounted the K_{l+½} polynomial degree (`l = 2|ν| − 1`
    /// instead of `|ν| − ½`), zeroing the r⁰ term of every |ν| ≥ 3/2 block and
    /// silently dropping their contribution to φ(0).
    #[test]
    fn hybrid_collision_diagonal_matches_closed_form_1604() {
        // Odd dimensions exercise the half-integer-ν path. For each, sweep p, s
        // (with 2(p+s) > d) and κ. d = 1, n ≥ 2 ⇒ ν ≥ 3/2 is the regressed case.
        for &d in &[1usize, 3, 5] {
            for &p in &[1usize, 2, 3] {
                for &s in &[1usize, 2, 3, 4] {
                    if 2 * (p + s) <= d {
                        continue;
                    }
                    for &kappa in &[0.5f64, 1.0, 2.5] {
                        let coeffs = duchon_partial_fraction_coeffs(p, s, kappa);
                        let got =
                            duchon_hybrid_kernel_collision_value(1.0 / kappa, p, s, d, &coeffs)
                                .expect("collision diagonal");
                        let want = phi0_closed_form(p, s, d, kappa);
                        let rel = (got - want).abs() / want.abs().max(1e-300);
                        assert!(
                            rel < 1e-10,
                            "φ(0) mismatch d={d} p={p} s={s} κ={kappa}: got {got:.12e}, want {want:.12e} (rel {rel:.2e})"
                        );
                    }
                }
            }
        }
    }

    /// gam#1604 — the near-collision Taylor branch must be continuous with the
    /// direct partial-fraction sum: assembling φ(r) from φ(0), φ″(0), φ⁗(0),
    /// φ⁽⁶⁾(0) (all built from the same half-integer-ν Taylor coefficients) must
    /// match the cancellation-free direct block sum at a small radius where both
    /// are individually accurate. This exercises the j ≥ 1 coefficients (the
    /// diagonal test only pins j = 0).
    #[test]
    fn hybrid_near_collision_continuous_with_direct_1604() {
        for &d in &[1usize, 3] {
            for &p in &[1usize, 2] {
                for &s in &[2usize, 3] {
                    if 2 * (p + s) <= d + 6 {
                        // Need φ⁽⁶⁾(0) to exist for the full 6th-order Taylor.
                        continue;
                    }
                    for &kappa in &[0.5f64, 1.0, 2.0] {
                        let length_scale = 1.0 / kappa;
                        let coeffs = duchon_partial_fraction_coeffs(p, s, kappa);
                        // r small enough that the truncated 6th-order Taylor is
                        // accurate to ~r⁸, yet large enough that the direct block
                        // sum has not lost precision (d = 1/3, moderate κ).
                        let r = 0.02 * length_scale;
                        let taylor = duchon_hybrid_kernel_near_collision_value(
                            r,
                            length_scale,
                            p,
                            s,
                            d,
                            &coeffs,
                        )
                        .expect("near-collision value");
                        // Direct partial-fraction sum (real Bessel-K, no Taylor).
                        let mut direct = 0.0f64;
                        for (m, &a_m) in coeffs.a.iter().enumerate().skip(1) {
                            if a_m != 0.0 {
                                direct += a_m * polyharmonic_kernel(r, m as f64, d);
                            }
                        }
                        for (n, &b_n) in coeffs.b.iter().enumerate().skip(1) {
                            if b_n != 0.0 {
                                direct += b_n
                                    * duchon_matern_block(r, kappa, n, d).expect("matern block");
                            }
                        }
                        let rel = (taylor - direct).abs() / direct.abs().max(1e-300);
                        assert!(
                            rel < 1e-9,
                            "near-collision vs direct mismatch d={d} p={p} s={s} κ={kappa} r={r}: \
                             taylor {taylor:.12e}, direct {direct:.12e} (rel {rel:.2e})"
                        );
                    }
                }
            }
        }
    }

    /// gam#1604 — the production constrained Duchon penalty `Ω_c = α²·ZᵀK_CC Z`
    /// for a `d = 1` hybrid smooth with power ≥ 2 must be numerically PSD across
    /// realistic length scales. Before the Taylor-degree fix the corrupted
    /// diagonal made `Ω_c ≈ Ω_true − δ·I` (δ = the dropped diagonal mass),
    /// giving λ_min ≈ −δ < 0 at *every* length scale — the issue's report.
    #[test]
    fn d1_hybrid_penalty_is_psd_1604() {
        let d = 1usize;
        let nullspace_order = DuchonNullspaceOrder::Linear; // p = 2
        let centers = fixture_centers(d, 12);
        let mut cache = BasisCacheContext::default();
        let z = kernel_constraint_nullspace(centers.view(), nullspace_order, &mut cache)
            .expect("constraint null space");
        for &power in &[2.0f64, 3.0] {
            for &length_scale in &[0.5f64, 1.0, 10.0, 100.0] {
                let omega = duchon_constrained_bending_penalty(
                    centers.view(),
                    Some(length_scale),
                    power,
                    nullspace_order,
                    None,
                    &z,
                )
                .unwrap_or_else(|e| {
                    panic!("d=1 p=2 s={power} ls={length_scale} penalty rejected: {e}")
                });
                let (penalty, _scale) = normalize_penalty(&omega);
                let lam_min = lambda_min(&penalty);
                assert!(
                    lam_min >= -1e-9,
                    "d=1 p=2 s={power} ls={length_scale}: λ_min={lam_min:.6e} (not PSD)"
                );
            }
        }
    }

    /// No-regression guard: a well-conditioned low-dimensional fixture must keep
    /// the exact kernel VALUES the partial-fraction path produced before the
    /// gam#1424 fix. For d=2 the stable-integral reroute does not apply
    /// (`2p=4 ≥ d=2`), so `duchon_matern_kernel_general_from_distance` still runs
    /// the original sum verbatim; pinning it against an independent direct
    /// evaluation of the same partial-fraction blocks proves the production
    /// routing is unchanged for low `d`.
    #[test]
    fn low_dim_hybrid_kernel_values_unchanged_1424() {
        let d = 2usize;
        let p_order = 2usize; // Linear null space (m=2)
        let s_order = 2usize;
        let kappa = 1.0_f64;
        let length_scale = Some(1.0_f64);
        // The d=2 case is NOT rerouted to the stable integral.
        assert!(!duchon_hybrid_stable_integral_applies(p_order, s_order, d));
        let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);

        for &r in &[0.25_f64, 0.75, 1.5] {
            // Independent reference: the raw partial-fraction sum
            // Σ a_m·r^{2m-d}(·log) + Σ b_n·matern_block, identical in form to the
            // production direct-sum branch but assembled here from scratch.
            let mut reference = 0.0_f64;
            for (m, &coeff) in coeffs.a.iter().enumerate().skip(1) {
                if coeff != 0.0 {
                    reference += coeff * polyharmonic_kernel(r, m as f64, d);
                }
            }
            for (n, &coeff) in coeffs.b.iter().enumerate().skip(1) {
                if coeff != 0.0 {
                    reference += coeff * duchon_matern_block(r, kappa, n, d).expect("matern block");
                }
            }

            let got = duchon_matern_kernel_general_from_distance(
                r,
                length_scale,
                p_order,
                s_order,
                d,
                Some(&coeffs),
            )
            .expect("low-d hybrid kernel value");
            assert!(
                (got - reference).abs() <= 1e-10,
                "low-d hybrid kernel value regressed at r={r}: got {got:.15e}, reference {reference:.15e}"
            );
        }
    }

    /// #1817: a low-order/low-power Duchon config with the stiffness (D2)
    /// operator active — d=2, `Linear` null space (p=2), power s=0 — has
    /// `2(p+s)=4`, which clears the pointwise margin (>d=2) and D1 (>d+1=3) but
    /// NOT D2 (>d+2=4), so the collocation guard used to fire mid-fit. The order
    /// must now auto-raise so the mass+tension+stiffness penalty matrices build
    /// cleanly, and the effective order must satisfy the strict D2 margin.
    #[test]
    fn operator_penalties_auto_raise_order_issue_1817() {
        // 4×3 grid on [0,1]² — 12 centers, comfortably above the 6 polynomial
        // columns of the auto-raised Degree(2) null space, so the auto-DEGRADE
        // path does not interfere with the auto-RAISE under test.
        let mut centers = Array2::<f64>::zeros((12, 2));
        let mut row = 0;
        for i in 0..4 {
            for j in 0..3 {
                centers[[row, 0]] = i as f64 / 3.0;
                centers[[row, 1]] = j as f64 / 2.0;
                row += 1;
            }
        }

        let dim = 2usize;
        let power = 0.0_f64;
        let requested = DuchonNullspaceOrder::Linear;

        // The unraised config is exactly the one that trips the D2 guard.
        let requested_p = duchon_p_from_nullspace_order(requested);
        assert!(
            2.0 * (requested_p as f64 + power) <= dim as f64 + 2.0,
            "precondition: requested (p,s) must be on the failing side of the D2 margin"
        );

        // Auto-raise (max_op = 2 ⇒ stiffness/D2 active) must clear the strict D2
        // margin 2(p+s) > d+2.
        let effective = duchon_order_for_operator_margin(dim, power, requested, 2);
        let effective_p = duchon_p_from_nullspace_order(effective);
        assert!(
            2.0 * (effective_p as f64 + power) > dim as f64 + 2.0,
            "auto-raised order must satisfy 2(p+s) > d+2: got 2*({}+{})={} vs d+2={}",
            effective_p,
            power,
            2.0 * (effective_p as f64 + power),
            dim as f64 + 2.0
        );

        // End-to-end: the mass+tension+stiffness penalty matrices (max_op=2)
        // must now build without an InvalidInput from the pointwise/collocation
        // guard, because the order was raised before the guard could fire.
        let penalties = build_duchon_operator_penalty_matrices(
            centers.view(),
            None,
            None, // pure (scale-free) Duchon — the guarded branch
            power,
            requested,
            None,
            None,
        )
        .expect("Duchon mass+tension+stiffness penalties must build after auto-raise (#1817)");
        for m in [&penalties.mass, &penalties.tension, &penalties.stiffness] {
            assert!(
                m.iter().all(|v| v.is_finite()),
                "auto-raised operator penalty matrices must be finite"
            );
        }
    }
}

#[cfg(test)]
mod inverse_length_scale_tests {
    use super::*;

    /// The owner refuses every length scale the old `1.0 / ls.max(1e-300)`
    /// floors silently mapped to a huge finite κ, and returns the exact
    /// inverse for a legitimate one (#2469).
    #[test]
    fn inverse_length_scale_refuses_what_the_floor_used_to_swallow() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e-310] {
            let refused = duchon_inverse_length_scale(bad, "test").is_err();
            assert!(refused, "length_scale {bad:e} must be refused, not floored");
        }
        let kappa = duchon_inverse_length_scale(0.25, "test").expect("positive length scale");
        assert_eq!(kappa.to_bits(), 4.0_f64.to_bits());
        let err = duchon_inverse_length_scale(0.0, "Duchon N-D radial jets")
            .err()
            .expect("refusal");
        let text = err.to_string();
        assert!(
            text.contains("Duchon N-D radial jets") && text.contains("got 0"),
            "refusal names its context and the value: {text}"
        );
    }
}
