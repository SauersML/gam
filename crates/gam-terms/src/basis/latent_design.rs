//! Latent-coordinate designs and their input-location jets: the forward designs the latent REML outer problem
//! (`gam_models::latent_outer`) differentiates, moved from gam-pyffi (#2899 F6).

use crate::basis::{bspline_tensor_first_derivative, build_duchon_basis, build_duchon_basis_mixed_periodicity_auto, build_matern_basis, build_periodic_bspline_basis_1d, build_spherical_spline_basis, CenterStrategy, create_duchon_basis_1d_derivative_dense, duchon_effective_nullspace_order, duchon_kernel_constraint_nullspace, duchon_nullspace_order_from_m, duchon_polynomial_first_derivative_nd, duchon_pure_kernel_amplification, duchon_radial_first_derivative_nd, DuchonBasisSpec, evaluate_bspline_basis_scalar, matern_radial_first_derivative_nd, MaternBasisSpec, MaternIdentifiability, MaternLengthScale, MaternNu, OneDimensionalBoundary, periodic_bspline_first_derivative_nd, PeriodicBSplineBasisSpec, resolve_duchon_orders, SpatialIdentifiability, sphere_first_derivative_nd, SphereMethod, SphereWahbaKernel, SphericalSplineBasisSpec, SphericalSplineIdentifiability, SplineScratch};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, s};

pub fn latent_basis_kind(value: &str) -> Result<&'static str, String> {
    match value.to_ascii_lowercase().replace(['_', '-'], "").as_str() {
        "duchon" | "duchonspline" => Ok("duchon"),
        // Dispatch hooks for the in-flight non-Duchon derivative helpers.
        // The call sites below are intentionally shaped around
        // `InputLocationDerivative::{Radial, Jet}` so Matérn can plug into
        // the radial path and sphere / tensor / periodic bases can plug into
        // the pre-computed-jet path without changing the contraction code.
        "matern" | "maternradial" => Ok("matern"),
        "sphere" | "spherical" => Ok("sphere"),
        "bsplinetensor" | "tensorbspline" => Ok("bspline_tensor"),
        "periodicbspline" | "periodicspline" => Ok("periodic_bspline"),
        other => Err(format!("unsupported latent basis_kind {other:?}")),
    }
}

pub fn radial_input_location_jet(
    t_mat: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    phi_r: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, String> {
    if phi_r.dim() != (t_mat.nrows(), centers.nrows()) {
        return Err(format!(
            "radial derivative shape {:?} does not match t/centers ({}, {})",
            phi_r.dim(),
            t_mat.nrows(),
            centers.nrows()
        ));
    }
    if t_mat.ncols() != centers.ncols() {
        return Err(format!(
            "radial derivative dimension mismatch: t has {} cols, centers has {}",
            t_mat.ncols(),
            centers.ncols()
        ));
    }
    let mut out = Array3::<f64>::zeros((t_mat.nrows(), centers.nrows(), t_mat.ncols()));
    for n in 0..t_mat.nrows() {
        for k in 0..centers.nrows() {
            let mut r2 = 0.0;
            for a in 0..t_mat.ncols() {
                let delta = t_mat[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            if r <= 1.0e-12 {
                continue;
            }
            let scale = phi_r[[n, k]] / r;
            for a in 0..t_mat.ncols() {
                out[[n, k, a]] = scale * (t_mat[[n, a]] - centers[[k, a]]);
            }
        }
    }
    Ok(out)
}

pub fn latent_input_location_jet(
    basis_kind: &str,
    t_mat: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    m: usize,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
) -> Result<Array3<f64>, String> {
    match latent_basis_kind(basis_kind)? {
        "duchon" => {
            // Mirror the column layout used by `build_latent_duchon_design` /
            // `build_duchon_basis`: the forward design is the radial block
            // projected through the kernel-constraint nullspace Z
            // (p_constrained = n_centers − n_poly cols) concatenated with the
            // polynomial nullspace block (n_poly cols). The derivative jet
            // must use the same effective nullspace order and the same Z
            // projection so its column count matches the design exactly.
            // Resolve the SAME admissible (nullspace_order, power) pair the
            // forward `build_latent_duchon_design` resolves for this ambient
            // dimension (issue #875): the pure polyharmonic kernel exists only
            // when 2(p + s) > d, so `resolve_duchon_orders` may lift the
            // spectral power s (and null-space order) above the m-derived
            // request. The jet must differentiate the *resolved* forward kernel
            // — same power, same null-space — or its column count and scaling
            // would diverge from the design.
            let dim_ambient = centers.ncols();
            let (resolved_nullspace, resolved_power) =
                resolve_duchon_orders(dim_ambient, duchon_nullspace_order_from_m(m), 0, None);
            let effective_nullspace = duchon_effective_nullspace_order(centers, resolved_nullspace);
            // The canonical construction (shared with the forward design via
            // `build_duchon_basis`) mean-centers the centers before the RRQR
            // (#1375), so the jet's `Z` is bit-identical to the design's `Z`
            // instead of a pivot-drifted basis of the same null space.
            let radial_transform = duchon_kernel_constraint_nullspace(centers, effective_nullspace)
                .map_err(|err| err.to_string())?;
            // The radial derivative differentiates the exact forward Green's
            // function: same scale-free pure Duchon spectrum (`length_scale =
            // None`, the resolved spectral power `s`), not a hard-coded
            // surrogate (issue #440).
            let phi_r = duchon_radial_first_derivative_nd(
                t_mat,
                centers,
                None,
                effective_nullspace,
                resolved_power,
            )
            .map_err(|err| err.to_string())?;
            let radial_jet = radial_input_location_jet(t_mat, centers, phi_r.view())?;
            let poly_jet = duchon_polynomial_first_derivative_nd(t_mat, effective_nullspace);

            let n_rows = radial_jet.shape()[0];
            let dim = radial_jet.shape()[2];
            let n_kernel = radial_transform.ncols();
            let n_poly = poly_jet.shape()[1];
            if poly_jet.shape()[0] != n_rows || poly_jet.shape()[2] != dim {
                return Err(format!(
                    "Duchon polynomial derivative shape mismatch: radial jet is \
                     {}x{}x{}, polynomial jet is {}x{}x{}",
                    n_rows,
                    radial_jet.shape()[1],
                    dim,
                    poly_jet.shape()[0],
                    n_poly,
                    poly_jet.shape()[2],
                ));
            }
            // Scalar kernel amplification `α` the forward
            // `build_latent_duchon_design` (→ `build_duchon_basis` with the
            // resolved `power`, `length_scale = None`) applies to the kernel
            // block `α·K(t,C)·Z`. The input-location derivative is
            // `α·K'(t,C)·Z`, so the raw radial jet must carry the same `α`
            // computed against the same resolved spectral power; the appended
            // polynomial columns are un-amplified, matching the forward.
            let kernel_amp = duchon_pure_kernel_amplification(
                centers,
                resolved_nullspace,
                resolved_power as f64,
            );
            let mut jet = Array3::<f64>::zeros((n_rows, n_kernel + n_poly, dim));
            for axis in 0..dim {
                let projected = radial_jet.index_axis(Axis(2), axis).dot(&radial_transform);
                let mut block = jet.slice_mut(s![.., ..n_kernel, axis]);
                block.assign(&projected);
                block *= kernel_amp;
            }
            jet.slice_mut(s![.., n_kernel.., ..]).assign(&poly_jet);
            Ok(jet)
        }
        "matern" => {
            // Fixes audit-revised claim that non-Duchon latent input-location
            // derivatives must use the closed-form helper instead of stubbing.
            let phi_r =
                matern_radial_first_derivative_nd(t_mat, centers, 1.0, MaternNu::ThreeHalves)
                    .map_err(|err| err.to_string())?;
            radial_input_location_jet(t_mat, centers, phi_r.view())
        }
        "sphere" => {
            // Fixes audit-revised claim that sphere latent derivatives are
            // analytic jets, not unsupported hooks.
            let jet = sphere_first_derivative_nd(t_mat, centers, m, true)
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        "bspline_tensor" => {
            let knots = tensor_knots_concat.ok_or_else(|| {
                "tensor B-spline latent derivative requires knots_concat".to_string()
            })?;
            let offsets = tensor_knot_offsets.ok_or_else(|| {
                "tensor B-spline latent derivative requires knot_offsets".to_string()
            })?;
            let degrees = tensor_degrees
                .ok_or_else(|| "tensor B-spline latent derivative requires degrees".to_string())?;
            let per_axis = split_tensor_knots_owned(knots, offsets, t_mat.ncols())?;
            let per_axis_views = per_axis
                .iter()
                .map(|axis_knots| axis_knots.view())
                .collect::<Vec<_>>();
            let jet = bspline_tensor_first_derivative(t_mat, &per_axis_views, degrees)
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        "periodic_bspline" => {
            // Fixes audit-revised claim that periodic latent derivatives are
            // analytic jets. The latent pyffi path carries only centers today,
            // so infer the period from the first center column.
            if centers.ncols() != 1 || centers.nrows() == 0 {
                return Err(
                    "periodic B-spline latent derivative requires one-column centers".to_string(),
                );
            }
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &value in centers.column(0).iter() {
                lo = lo.min(value);
                hi = hi.max(value);
            }
            if !(lo.is_finite() && hi.is_finite() && hi > lo) {
                return Err("periodic B-spline centers must define a finite range".to_string());
            }
            let jet = periodic_bspline_first_derivative_nd(t_mat, (lo, hi), m, centers.nrows())
                .map_err(|err| err.to_string())?;
            Ok(jet)
        }
        other => Err(format!(
            "latent_basis_kind returned an unknown normalized kind: {other}"
        )),
    }
}

pub fn periodic_bspline_basis_dense_via_spec(
    t: ArrayView1<'_, f64>,
    domain: (f64, f64),
    degree: usize,
    num_basis: usize,
) -> Result<Array2<f64>, String> {
    let (left, right) = domain;
    let period = right - left;
    if !(period.is_finite() && period > 0.0) {
        return Err(format!(
            "periodic B-spline domain must be a finite ordered interval; got ({left}, {right})"
        ));
    }
    // The FFI returns only the dense value basis, but the shared periodic spec
    // validator requires a realizable derivative order. Use curvature when
    // the polynomial degree supports it and slope roughness otherwise.
    let penalty_order = degree.min(2);
    let spec = PeriodicBSplineBasisSpec::new(degree, num_basis, period, left, penalty_order);
    build_periodic_bspline_basis_1d(t, &spec)
        .map_err(|err| format!("failed to evaluate periodic B-spline basis: {err}"))
}

pub fn build_latent_duchon_design(
    t_flat: ArrayView1<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
    periodic: Option<&[Option<f64>]>,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if t_flat.len() != n_obs * latent_dim {
        return Err(format!(
            "latent t length {} != n_obs * latent_dim = {}",
            t_flat.len(),
            n_obs * latent_dim
        ));
    }
    if centers.ncols() != latent_dim {
        return Err(format!(
            "centers must have {latent_dim} columns to match latent_dim; got {}",
            centers.ncols()
        ));
    }
    if m == 0 {
        return Err("LatentCoord Duchon m must be at least 1".into());
    }
    // Materialize t as a (n_obs, latent_dim) matrix.
    let mut t_mat = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            t_mat[[n, a]] = t_flat[n * latent_dim + a];
        }
    }
    let center_matrix = centers.to_owned();
    // Resolve a fully admissible (nullspace_order, power) for THIS ambient
    // latent dimension. The pure scale-free polyharmonic kernel exists only
    // when 2(p + s) > d; with the requested null space alone (s = 0) this
    // fails whenever 2p <= d — e.g. m = 2 (p = 2) at latent_dim >= 4, which is
    // exactly issue #875. `resolve_duchon_orders` lifts the spectral power s
    // (and, if pure-mode CPD requires it, the null-space order) until the
    // kernel is well-posed for any d, including the even-d `r^{2m-d} log r`
    // log case. The latent forward design assembles no operator penalties
    // (`operator_penalties: Default::default()`), so `max_op = 0`: only the
    // kernel-existence / CPD guards apply, matching every other Duchon entry
    // point which routes through this same resolver.
    let (resolved_nullspace, resolved_power) =
        resolve_duchon_orders(latent_dim, duchon_nullspace_order_from_m(m), 0, None);
    // When the optimizer retracts the latent coordinates on a PERIODIC manifold
    // (circle / torus), the decoder MUST be a function on that manifold:
    // Φ(θ) = Φ(θ + period) per circular axis, with the kernel distance measured
    // across the seam. We mirror the POSITION periodic-Duchon path exactly —
    // route through `build_duchon_basis_mixed_periodicity_auto`, which sends the
    // 1-D circle to the Bernoulli Green's-function builder (the true PSD circle
    // kernel, gam#580) and a multi-axis torus to the chord-distance polyharmonic
    // builder. `periodic` carries a per-axis optional period (radians, the chart
    // wrap = TAU for circle/torus); a `None` axis is a Euclidean (open) axis.
    // When `periodic` is `None`/all-open the basis stays byte-identical to the
    // open Euclidean construction (euclidean / sphere / matern latent fits).
    let periodic_flags: Option<Vec<bool>> = periodic.and_then(|axes| {
        if axes.len() == latent_dim && axes.iter().any(|p| p.is_some()) {
            Some(axes.iter().map(|p| p.is_some()).collect())
        } else {
            None
        }
    });
    // The caller's penalty and coefficient adjoints use a fixed coefficient
    // frame. Re-estimating a data-metric radial chart here would silently move
    // that frame with the whole latent batch, changing both the represented
    // prior and the derivative (issue #2833). Freeze the canonical constrained
    // kernel frame explicitly; its row-local jets then differentiate exactly
    // the design used by Gaussian, GLM, and latent optimization entrypoints.
    // Periodic builders already have a fixed frame and do not use this chart.
    let radial_reparam = if periodic_flags.is_none() {
        let effective_nullspace =
            duchon_effective_nullspace_order(centers, resolved_nullspace);
        let constraint = duchon_kernel_constraint_nullspace(centers, effective_nullspace)
            .map_err(|err| err.to_string())?;
        Some(Array2::eye(constraint.ncols()))
    } else {
        None
    };
    let spec = DuchonBasisSpec {
        radial_reparam,
        center_strategy: CenterStrategy::UserProvided(center_matrix.clone()),
        length_scale: None,
        power: resolved_power as f64,
        nullspace_order: resolved_nullspace,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    let built = if let Some(flags) = periodic_flags {
        // `periodic` is Some with the same arity (checked above). Each periodic
        // axis carries an explicit chart period (TAU); non-periodic axes get a
        // placeholder period (unused by the builder for `!periodic` axes).
        let axes = periodic.expect("periodic_flags is only Some when periodic is Some");
        let periods: Vec<f64> = axes.iter().map(|p| p.unwrap_or(1.0)).collect();
        build_duchon_basis_mixed_periodicity_auto(t_mat.view(), &spec, &flags, Some(&periods))
            .map_err(|err| {
                format!("failed to evaluate periodic N-D Duchon basis for LatentCoord: {err}")
            })?
    } else {
        build_duchon_basis(t_mat.view(), &spec)
            .map_err(|err| format!("failed to evaluate N-D Duchon basis for LatentCoord: {err}"))?
    };
    let design = built
        .design
        .try_to_dense_by_chunks("latent_duchon_design")
        .map_err(|err| format!("failed to evaluate N-D Duchon basis for LatentCoord: {err}"))?;
    Ok((design, t_mat))
}

/// Input-location jet `∂Φ/∂t` of the PERIODIC latent Duchon design, matching the
/// per-manifold forward `build_latent_duchon_design` builds: the 1-D circle
/// routes through the Bernoulli Green's-function design (gam#580) and the
/// multi-axis torus through the chord-distance polyharmonic design. Returns
/// `Ok(None)` when no axis is periodic (the caller then uses the open Euclidean
/// jet, which is correct for euclidean / sphere / matern latents).
///
/// The two branches differentiate the SAME kernel, with the SAME resolved orders
/// and the SAME constraint nullspace `Z`, as the forward — so the returned jet is
/// the exact derivative of the forward design column-for-column. Building the
/// open Euclidean jet here instead (the issue #876 bug) gave a wrong gradient and
/// a column-count mismatch that nulled the outer gradient and collapsed the
/// latent.
pub fn build_latent_duchon_periodic_jet(
    t_mat: ArrayView2<'_, f64>,
    centers: ArrayView2<'_, f64>,
    m: usize,
    periodic: Option<&[Option<f64>]>,
) -> Result<Option<Array3<f64>>, String> {
    let latent_dim = t_mat.ncols();
    // Mirror `build_latent_duchon_design`'s gate: a per-axis period descriptor of
    // the right arity with at least one periodic axis.
    let axes = match periodic {
        Some(axes) if axes.len() == latent_dim && axes.iter().any(|p| p.is_some()) => axes,
        _ => return Ok(None),
    };
    // Same resolved (nullspace_order, power) the forward design uses for this
    // ambient latent dimension, so the kernel smoothness order and the Bernoulli
    // order (`user_m = duchon_p_from_nullspace_order(resolved_nullspace)`) match.
    let (resolved_nullspace, resolved_power) =
        resolve_duchon_orders(latent_dim, duchon_nullspace_order_from_m(m), 0, None);

    if latent_dim == 1 {
        // 1-D circle: the forward routes to `build_periodic_duchon_basis_1d`
        // (Bernoulli kernel). `create_duchon_basis_1d_derivative_dense` with
        // `periodic = true, order = 1` differentiates that exact forward — same
        // collapsed centers, same domain wrap, same constant-only constraint
        // nullspace — and returns the dense `(n, kernel_cols + 1)` first
        // derivative `∂Φ/∂t` (the trailing constant column's derivative is 0).
        let period = axes.first().copied().flatten().ok_or_else(|| {
            "periodic one-dimensional latent basis requires a period for its axis".to_string()
        })?;
        let dphi_dt = create_duchon_basis_1d_derivative_dense(
            t_mat.column(0),
            centers.column(0),
            resolved_power as f64,
            resolved_nullspace,
            true,
            Some(period),
            1,
        )
        .map_err(|err| format!("failed to evaluate periodic latent Duchon jet: {err}"))?;
        let n_rows = dphi_dt.nrows();
        let n_cols = dphi_dt.ncols();
        let mut jet = Array3::<f64>::zeros((n_rows, n_cols, 1));
        jet.slice_mut(s![.., .., 0]).assign(&dphi_dt);
        return Ok(Some(jet));
    }

    // Multi-axis torus: the forward routes to `build_duchon_basis_mixed_periodicity`
    // (chord-distance polyharmonic, pure spectrum, constant-only nullspace). The
    // `build_duchon_basis_design_and_jets` builder reproduces that SAME design and
    // returns its exact chord-embedding jet, so we take its `J` block. The mixed
    // periodicity path requires the pure polyharmonic spectrum (`power = 0`); the
    // resolver returns `power = 0` for the periodic latent configurations, but
    // assert it so a future order change fails loudly rather than silently
    // diverging from the forward.
    if resolved_power != 0 {
        return Err(format!(
            "periodic torus latent Duchon requires pure polyharmonic spectrum (power = 0); \
             resolver returned power = {resolved_power}"
        ));
    }
    let periodic_flags: Vec<bool> = axes.iter().map(|p| p.is_some()).collect();
    let periods: Vec<f64> = axes.iter().map(|p| p.unwrap_or(1.0)).collect();
    let (_phi, jet, _hess) = crate::basis::build_duchon_basis_design_and_jets(
        t_mat,
        centers,
        None,
        0.0,
        resolved_nullspace,
        &periodic_flags,
        &periods,
    )
    .map_err(|err| format!("failed to evaluate periodic torus latent Duchon jet: {err}"))?;
    Ok(Some(jet))
}

pub fn t_matrix_from_flat(
    t_flat: ArrayView1<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
) -> Result<Array2<f64>, String> {
    if t_flat.len() != n_obs * latent_dim {
        return Err(format!(
            "latent t length {} != n_obs * latent_dim = {}",
            t_flat.len(),
            n_obs * latent_dim
        ));
    }
    let mut t_mat = Array2::<f64>::zeros((n_obs, latent_dim));
    for n in 0..n_obs {
        for a in 0..latent_dim {
            t_mat[[n, a]] = t_flat[n * latent_dim + a];
        }
    }
    Ok(t_mat)
}

pub fn split_tensor_knots_owned(
    knots_concat: ArrayView1<'_, f64>,
    knot_offsets: &[usize],
    n_axes: usize,
) -> Result<Vec<Array1<f64>>, String> {
    if knot_offsets.len() != n_axes + 1 {
        return Err(format!(
            "tensor B-spline knot_offsets must have length n_axes + 1 = {}, got {}",
            n_axes + 1,
            knot_offsets.len()
        ));
    }
    let mut per_axis = Vec::with_capacity(n_axes);
    for axis in 0..n_axes {
        let lo = knot_offsets[axis];
        let hi = knot_offsets[axis + 1];
        if lo > hi || hi > knots_concat.len() {
            return Err(format!(
                "tensor B-spline knot_offsets axis {axis} out of range \
                 (lo={lo}, hi={hi}, total={})",
                knots_concat.len()
            ));
        }
        per_axis.push(knots_concat.slice(s![lo..hi]).to_owned());
    }
    Ok(per_axis)
}

pub fn build_latent_tensor_bspline_design(
    t_flat: ArrayView1<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    knots_concat: ArrayView1<'_, f64>,
    knot_offsets: &[usize],
    degrees: &[usize],
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if degrees.len() != latent_dim {
        return Err(format!(
            "tensor B-spline degrees length {} must equal latent_dim {}",
            degrees.len(),
            latent_dim
        ));
    }
    let t_mat = t_matrix_from_flat(t_flat, n_obs, latent_dim)?;
    let knots_per_axis = split_tensor_knots_owned(knots_concat, knot_offsets, latent_dim)?;
    let knot_views = knots_per_axis
        .iter()
        .map(|knots| knots.view())
        .collect::<Vec<_>>();
    let mut k_per_axis = Vec::<usize>::with_capacity(latent_dim);
    let mut total_cols = 1usize;
    for axis in 0..latent_dim {
        let k = knot_views[axis]
            .len()
            .checked_sub(degrees[axis] + 1)
            .ok_or_else(|| {
                format!(
                    "tensor B-spline axis {axis} knot vector too short for degree {}",
                    degrees[axis]
                )
            })?;
        k_per_axis.push(k);
        total_cols = total_cols
            .checked_mul(k)
            .ok_or_else(|| "tensor B-spline basis size overflow".to_string())?;
    }

    let mut design = Array2::<f64>::zeros((n_obs, total_cols));
    let mut values_per_axis: Vec<Vec<f64>> = k_per_axis.iter().map(|&k| vec![0.0; k]).collect();
    let mut scratch: Vec<SplineScratch> = (0..latent_dim)
        .map(|axis| SplineScratch::new(degrees[axis]))
        .collect();
    let mut idx = vec![0usize; latent_dim];
    for n in 0..n_obs {
        for axis in 0..latent_dim {
            evaluate_bspline_basis_scalar(
                t_mat[[n, axis]],
                knot_views[axis],
                degrees[axis],
                &mut values_per_axis[axis],
                &mut scratch[axis],
            )
            .map_err(|err| {
                format!("failed to evaluate tensor B-spline latent axis {axis}: {err}")
            })?;
        }
        for col in 0..total_cols {
            let mut rem = col;
            for axis in (0..latent_dim).rev() {
                idx[axis] = rem % k_per_axis[axis];
                rem /= k_per_axis[axis];
            }
            let mut prod = 1.0_f64;
            for axis in 0..latent_dim {
                prod *= values_per_axis[axis][idx[axis]];
            }
            design[[n, col]] = prod;
        }
    }
    Ok((design, t_mat))
}

pub fn latent_periodic_range_from_centers(centers: ArrayView2<'_, f64>) -> Result<(f64, f64), String> {
    if centers.ncols() != 1 || centers.nrows() == 0 {
        return Err("periodic B-spline latent design requires one-column centers".to_string());
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &value in centers.column(0).iter() {
        lo = lo.min(value);
        hi = hi.max(value);
    }
    if !(lo.is_finite() && hi.is_finite() && hi > lo) {
        return Err("periodic B-spline centers must define a finite range".to_string());
    }
    Ok((lo, hi))
}

pub fn project_latent_jet_columns(
    raw_jet: &Array3<f64>,
    transform: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, String> {
    let n_rows = raw_jet.shape()[0];
    let raw_cols = raw_jet.shape()[1];
    let latent_dim = raw_jet.shape()[2];
    if transform.nrows() != raw_cols {
        return Err(format!(
            "latent jet transform row mismatch: jet has {raw_cols} columns, transform has {} rows",
            transform.nrows()
        ));
    }
    let mut out = Array3::<f64>::zeros((n_rows, transform.ncols(), latent_dim));
    for n in 0..n_rows {
        for j in 0..transform.ncols() {
            for k in 0..raw_cols {
                let z = transform[[k, j]];
                if z == 0.0 {
                    continue;
                }
                for a in 0..latent_dim {
                    out[[n, j, a]] += raw_jet[[n, k, a]] * z;
                }
            }
        }
    }
    Ok(out)
}

pub fn build_latent_forward_design(
    basis_kind: &str,
    t_flat: ArrayView1<'_, f64>,
    n_obs: usize,
    latent_dim: usize,
    centers: ArrayView2<'_, f64>,
    m: usize,
    tensor_knots_concat: Option<ArrayView1<'_, f64>>,
    tensor_knot_offsets: Option<&[usize]>,
    tensor_degrees: Option<&[usize]>,
    periodic: Option<&[Option<f64>]>,
) -> Result<(Array2<f64>, Array2<f64>, Array3<f64>), String> {
    let basis_kind = latent_basis_kind(basis_kind)?;
    let (design, t_mat) = match basis_kind {
        "duchon" => {
            let (design, t_mat) =
                build_latent_duchon_design(t_flat, n_obs, latent_dim, centers, m, periodic)?;
            // On a PERIODIC latent manifold (circle / torus) the forward design is
            // the periodic Duchon basis (1-D Bernoulli Green's function or the
            // multi-axis chord-distance polyharmonic) — a DIFFERENT kernel and
            // column layout than the open Euclidean Duchon. Its input-location
            // jet must differentiate that SAME periodic forward, not the open
            // Euclidean basis the generic `latent_input_location_jet` builds.
            // Routing the periodic forward through the open jet produced both a
            // wrong gradient direction AND a column-count mismatch (the open jet
            // carries `d+1` polynomial columns vs. the periodic design's single
            // constant column), which made `value_and_grad` fail the
            // design/jet shape check, return `(+∞, None)`, and hand the outer
            // trust region a zero gradient — so the circle/torus optimizer read
            // "stationary" at the start and collapsed every row to one latent
            // coordinate (issue #876). Build the matching periodic jet here and
            // return early, mirroring the per-manifold forward choice exactly.
            if let Some(jet) = build_latent_duchon_periodic_jet(t_mat.view(), centers, m, periodic)?
            {
                if jet.shape()[1] != design.ncols() {
                    return Err(format!(
                        "periodic latent Duchon design/jet column mismatch: design has {}, jet has {}",
                        design.ncols(),
                        jet.shape()[1]
                    ));
                }
                return Ok((design, t_mat, jet));
            }
            (design, t_mat)
        }
        "matern" => {
            if centers.ncols() != latent_dim {
                return Err(format!(
                    "Matérn latent centers must have {latent_dim} columns; got {}",
                    centers.ncols()
                ));
            }
            let t_mat = t_matrix_from_flat(t_flat, n_obs, latent_dim)?;
            let spec = MaternBasisSpec {
                center_strategy: CenterStrategy::UserProvided(centers.to_owned()),
                length_scale: MaternLengthScale::fixed(1.0),
                nu: MaternNu::ThreeHalves,
                include_intercept: false,
                double_penalty: false,
                identifiability: MaternIdentifiability::None,
                aniso_log_scales: None,
                periodic: None,
            };
            let built = build_matern_basis(t_mat.view(), &spec)
                .map_err(|err| format!("failed to evaluate Matérn latent basis: {err}"))?;
            let design = built
                .design
                .try_to_dense_by_chunks("latent_matern_design")
                .map_err(|err| format!("failed to evaluate Matérn latent basis: {err}"))?;
            (design, t_mat)
        }
        "sphere" => {
            if centers.ncols() != latent_dim {
                return Err(format!(
                    "sphere latent centers must have {latent_dim} columns; got {}",
                    centers.ncols()
                ));
            }
            let t_mat = t_matrix_from_flat(t_flat, n_obs, latent_dim)?;
            let spec = SphericalSplineBasisSpec {
                center_strategy: CenterStrategy::UserProvided(centers.to_owned()),
                penalty_order: m,
                double_penalty: false,
                radians: true,
                method: SphereMethod::Wahba,
                max_degree: None,
                wahba_kernel: SphereWahbaKernel::Sobolev,
                identifiability: SphericalSplineIdentifiability::CenterSumToZero,
            };
            let built = build_spherical_spline_basis(t_mat.view(), &spec)
                .map_err(|err| format!("failed to evaluate sphere latent basis: {err}"))?;
            let constraint_transform = match &built.metadata {
                crate::basis::BasisMetadata::Sphere {
                    constraint_transform,
                    ..
                } => constraint_transform.clone(),
                _ => None,
            };
            let design = built
                .design
                .try_to_dense_by_chunks("latent_sphere_design")
                .map_err(|err| format!("failed to evaluate sphere latent basis: {err}"))?;
            let raw_jet = latent_input_location_jet(
                basis_kind,
                t_mat.view(),
                centers,
                m,
                tensor_knots_concat,
                tensor_knot_offsets,
                tensor_degrees,
            )?;
            let jet = match constraint_transform {
                Some(z) => project_latent_jet_columns(&raw_jet, z.view())?,
                _ => raw_jet,
            };
            if jet.shape()[1] != design.ncols() {
                return Err(format!(
                    "sphere latent design/jet column mismatch: design has {}, jet has {}",
                    design.ncols(),
                    jet.shape()[1]
                ));
            }
            return Ok((design, t_mat, jet));
        }
        "bspline_tensor" => {
            let knots = tensor_knots_concat
                .as_ref()
                .ok_or_else(|| "tensor B-spline latent design requires knots_concat".to_string())?
                .clone();
            let offsets = tensor_knot_offsets
                .ok_or_else(|| "tensor B-spline latent design requires knot_offsets".to_string())?;
            let degrees = tensor_degrees
                .ok_or_else(|| "tensor B-spline latent design requires degrees".to_string())?;
            build_latent_tensor_bspline_design(t_flat, n_obs, latent_dim, knots, offsets, degrees)?
        }
        "periodic_bspline" => {
            if latent_dim != 1 {
                return Err(format!(
                    "periodic B-spline latent design requires latent_dim 1; got {latent_dim}"
                ));
            }
            let t_mat = t_matrix_from_flat(t_flat, n_obs, latent_dim)?;
            let range = latent_periodic_range_from_centers(centers)?;
            let design =
                periodic_bspline_basis_dense_via_spec(t_mat.column(0), range, m, centers.nrows())?;
            (design, t_mat)
        }
        other => {
            return Err(format!(
                "gaussian_reml_fit_latent does not support latent basis_kind {other:?}"
            ));
        }
    };
    let jet = latent_input_location_jet(
        basis_kind,
        t_mat.view(),
        centers,
        m,
        tensor_knots_concat,
        tensor_knot_offsets,
        tensor_degrees,
    )?;
    if jet.shape()[1] != design.ncols() {
        return Err(format!(
            "latent design/jet column mismatch for {basis_kind:?}: design has {}, jet has {}",
            design.ncols(),
            jet.shape()[1]
        ));
    }
    Ok((design, t_mat, jet))
}
