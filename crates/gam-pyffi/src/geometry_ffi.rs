
#[pyfunction]
fn poincare_distance<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<f64> {
    let a_owned = a.as_array().to_owned();
    let b_owned = b.as_array().to_owned();
    detach_geometry_result(py, "poincare_distance", move || {
        poincare_distance_impl(a_owned.view(), b_owned.view(), curvature)
    })
}

#[pyfunction]
fn poincare_project_into_ball<'py>(
    py: Python<'py>,
    point: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = point.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_project_into_ball", move || {
        poincare_project_into_ball_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_log_origin<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = y.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_log_origin", move || {
        poincare_log_origin_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_exp_origin<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = v.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_exp_origin", move || {
        poincare_exp_origin_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_conformal_factor<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<f64> {
    let p_owned = p.as_array().to_owned();
    detach_geometry_result(py, "poincare_conformal_factor", move || {
        poincare_conformal_factor_impl(p_owned.view(), curvature)
    })
}

#[pyfunction]
fn poincare_exp_map<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    v: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_owned = p.as_array().to_owned();
    let v_owned = v.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_exp_map", move || {
        poincare_exp_map_impl(p_owned.view(), v_owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_log_map<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let p_owned = p.as_array().to_owned();
    let q_owned = q.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_log_map", move || {
        poincare_log_map_impl(p_owned.view(), q_owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_to_lorentz<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = y.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_to_lorentz", move || {
        poincare_to_lorentz_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_from_lorentz<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = x.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_from_lorentz", move || {
        poincare_from_lorentz_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_lorentz_log_origin<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = x.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_lorentz_log_origin", move || {
        poincare_lorentz_log_origin_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn poincare_lorentz_exp_origin<'py>(
    py: Python<'py>,
    v_spatial: PyReadonlyArray1<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = v_spatial.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_lorentz_exp_origin", move || {
        poincare_lorentz_exp_origin_impl(owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// Forward pass for the Poincaré tangent-space-at-origin decoder.
///
/// Returns `(x_hat, atoms_projected, v, tangents, proj_scale)` so the Python
/// autograd.Function can stash them and use them on the backward pass without
/// recomputing. `proj_scale` (shape `(F,)`) is the per-atom radial
/// ball-projection factor needed by the backward to chain gradients through
/// the projection back to the raw atom storage.
#[pyfunction]
fn poincare_tangent_decode_forward<'py>(
    py: Python<'py>,
    atoms: PyReadonlyArray2<'py, f64>,
    gates: PyReadonlyArray2<'py, f64>,
    curvature: f64,
) -> PyResult<(
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray2<f64>>,
    Py<PyArray1<f64>>,
)> {
    let atoms_owned = atoms.as_array().to_owned();
    let gates_owned = gates.as_array().to_owned();
    let (x_hat, cache) =
        detach_geometry_result(py, "poincare_tangent_decode_forward", move || {
            poincare_tangent_decode_forward_impl(atoms_owned.view(), gates_owned.view(), curvature)
        })?;
    Ok((
        x_hat.into_pyarray(py).unbind(),
        cache.atoms_projected.into_pyarray(py).unbind(),
        cache.v.into_pyarray(py).unbind(),
        cache.tangents.into_pyarray(py).unbind(),
        cache.proj_scale.into_pyarray(py).unbind(),
    ))
}

/// Backward pass that consumes the cached state returned by the forward.
///
/// Returns `(grad_gates, grad_atoms)`.
#[pyfunction]
fn poincare_tangent_decode_backward<'py>(
    py: Python<'py>,
    atoms_projected: PyReadonlyArray2<'py, f64>,
    gates: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    tangents: PyReadonlyArray2<'py, f64>,
    proj_scale: PyReadonlyArray1<'py, f64>,
    grad_x_hat: PyReadonlyArray2<'py, f64>,
    curvature: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let atoms_p = atoms_projected.as_array().to_owned();
    let gates_owned = gates.as_array().to_owned();
    let v_owned = v.as_array().to_owned();
    let tangents_owned = tangents.as_array().to_owned();
    let proj_scale_owned = proj_scale.as_array().to_owned();
    let grad_owned = grad_x_hat.as_array().to_owned();
    let (gg, ga) = detach_geometry_result(py, "poincare_tangent_decode_backward", move || {
        let cache = gam::geometry::poincare::TangentDecodeCache {
            atoms_projected: atoms_p,
            gates: gates_owned,
            v: v_owned,
            tangents: tangents_owned,
            proj_scale: proj_scale_owned,
            curvature,
        };
        poincare_tangent_decode_backward_impl(&cache, grad_owned.view())
    })?;
    Ok((gg.into_pyarray(py).unbind(), ga.into_pyarray(py).unbind()))
}

#[pyfunction]
fn poincare_lorentz_decode_forward<'py>(
    py: Python<'py>,
    atoms: PyReadonlyArray2<'py, f64>,
    gates: PyReadonlyArray2<'py, f64>,
    curvature: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let atoms_owned = atoms.as_array().to_owned();
    let gates_owned = gates.as_array().to_owned();
    let out = detach_geometry_result(py, "poincare_lorentz_decode_forward", move || {
        poincare_lorentz_decode_forward_impl(atoms_owned.view(), gates_owned.view(), curvature)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// Backward pass for the Lorentz-path decoder. Consumes the cache returned
/// by `poincare_tangent_decode_forward` (the Poincaré and Lorentz tangent
/// decoders are algebraically the same function of the inputs, so the
/// Poincaré cache is the right state to differentiate from). Returns
/// `(grad_gates, grad_atoms)`.
#[pyfunction]
fn poincare_lorentz_decode_backward<'py>(
    py: Python<'py>,
    atoms_projected: PyReadonlyArray2<'py, f64>,
    gates: PyReadonlyArray2<'py, f64>,
    v: PyReadonlyArray2<'py, f64>,
    tangents: PyReadonlyArray2<'py, f64>,
    proj_scale: PyReadonlyArray1<'py, f64>,
    grad_x_hat: PyReadonlyArray2<'py, f64>,
    curvature: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let atoms_p = atoms_projected.as_array().to_owned();
    let gates_owned = gates.as_array().to_owned();
    let v_owned = v.as_array().to_owned();
    let tangents_owned = tangents.as_array().to_owned();
    let proj_scale_owned = proj_scale.as_array().to_owned();
    let grad_owned = grad_x_hat.as_array().to_owned();
    let (gg, ga) = detach_geometry_result(py, "poincare_lorentz_decode_backward", move || {
        let cache = gam::geometry::poincare::TangentDecodeCache {
            atoms_projected: atoms_p,
            gates: gates_owned,
            v: v_owned,
            tangents: tangents_owned,
            proj_scale: proj_scale_owned,
            curvature,
        };
        poincare_lorentz_decode_backward_impl(&cache, grad_owned.view())
    })?;
    Ok((gg.into_pyarray(py).unbind(), ga.into_pyarray(py).unbind()))
}

#[pyfunction(signature = (values, base, coordinates, reference = -1))]
fn response_geometry_simplex_log_map<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: String,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_simplex_log_map", move || {
        let coord = gam::geometry::simplex::parse_simplex_coord(&coordinates)?;
        gam::geometry::simplex::simplex_log_map(arr.view(), base_owned.view(), coord, reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (tangent, base, coordinates, reference = -1))]
fn response_geometry_simplex_exp_map<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: String,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_simplex_exp_map", move || {
        let coord = gam::geometry::simplex::parse_simplex_coord(&coordinates)?;
        gam::geometry::simplex::simplex_exp_map(t_owned.view(), base_owned.view(), coord, reference)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_sphere_log_map<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_sphere_log_map", move || {
        gam::geometry::sphere::response_sphere_log_map(arr.view(), base_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_sphere_exp_map<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_sphere_exp_map", move || {
        gam::geometry::sphere::response_sphere_exp_map(t_owned.view(), base_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_sphere_normalize_base<'py>(
    py: Python<'py>,
    base: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let owned = base.as_array().to_owned();
    let normalized = py.detach(move || rg_normalize_sphere_base(owned.view()));
    let normalized = normalized.map_err(PyValueError::new_err)?;
    Ok(normalized.into_pyarray(py).unbind())
}

/// Consolidated response-geometry log map. Owns geometry-kind routing,
/// coordinate resolution, and base-point selection (intrinsic Fréchet mean when
/// `base` is `None`) so the Python wrapper marshals arrays only. Returns
/// `(tangent, base_point, resolved_coordinate_label)`.
#[pyfunction(signature = (values, geometry, base=None, coordinates=None, reference=-1))]
fn response_geometry_log_map<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    geometry: String,
    base: Option<PyReadonlyArray1<'py, f64>>,
    coordinates: Option<String>,
    reference: isize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>, String)> {
    let arr = values.as_array().to_owned();
    let base_owned = base.as_ref().map(|b| b.as_array().to_owned());
    let (tangent, base_point, coord_label) =
        detach_py_result(py, "response_geometry_log_map", move || {
            rg_log_map_dispatch(
                arr.view(),
                &geometry,
                base_owned.as_ref().map(|b| b.view()),
                coordinates.as_deref(),
                reference,
            )
        })?;
    Ok((
        tangent.into_pyarray(py).unbind(),
        base_point.into_pyarray(py).unbind(),
        coord_label,
    ))
}

/// Consolidated response-geometry exponential map. Dispatches tangent
/// coordinates back to the response manifold given the geometry kind and the
/// (already resolved) coordinate label.
#[pyfunction(signature = (tangent, geometry, base, coordinates=None, reference=-1))]
fn response_geometry_exp_map<'py>(
    py: Python<'py>,
    tangent: PyReadonlyArray2<'py, f64>,
    geometry: String,
    base: PyReadonlyArray1<'py, f64>,
    coordinates: Option<String>,
    reference: isize,
) -> PyResult<Py<PyArray2<f64>>> {
    let t_owned = tangent.as_array().to_owned();
    let base_owned = base.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_exp_map", move || {
        rg_exp_map_dispatch(
            t_owned.view(),
            &geometry,
            base_owned.view(),
            coordinates.as_deref(),
            reference,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

/// Fit curvature as an estimand on a constant-curvature response geometry
/// (#944 stage 4 / #1104). The user requests `response_geometry="constant_
/// curvature(dim=d)"`; κ is NOT supplied — it is estimated from the manifold-
/// valued responses by the REML/evidence outer loop (the profiled Fréchet-
/// dispersion criterion), and the fit reports κ̂ with a profile-likelihood CI,
/// the geometry verdict, and the interior-point Wilks flatness test of κ = 0.
///
/// Returns the summary tuple
/// `(kappa_hat, ci_lo, ci_hi, lo_at_bound, hi_at_bound, verdict, lr_stat,
///   p_value, railed_at_resolution_limit, kappa_r2, characteristic_radius,
///   base_point)` for the response-geometry fit summary.
///
/// `railed_at_resolution_limit` (#1104): `true` when the cloud is curved BEYOND
/// what its spread can resolve (it fills the sphere), so κ̂ railed to the chart
/// conjugate cap — κ̂/`ci_hi` are then a LOWER BOUND on |κ|, an honest "curvature
/// exceeds chart-resolvable range at this scale", NOT a resolved point estimate.
/// `kappa_r2 = κ̂·r²` is the scale-FREE invariant the cloud determines (invariant
/// under `y ↦ αy`); `characteristic_radius = r` is the κ=0 doubled-gauge spread it
/// is dimensionless to. Read these (not the scale-dependent κ̂ alone) when the data
/// have been arbitrarily rescaled — e.g. unit-normalised OLMo activations.
#[pyfunction(signature = (values, geometry, level=0.95))]
#[allow(clippy::type_complexity)]
fn response_geometry_fit_curvature<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    geometry: String,
    level: f64,
) -> PyResult<(
    f64,
    f64,
    f64,
    bool,
    bool,
    String,
    f64,
    f64,
    bool,
    f64,
    f64,
    Py<PyArray1<f64>>,
)> {
    let arr = values.as_array().to_owned();
    let fit = detach_py_result(py, "response_geometry_fit_curvature", move || {
        let manifold =
            gam::geometry::response_geometry::ResponseManifold::parse(&geometry, arr.ncols())?;
        let dim = match manifold {
            gam::geometry::response_geometry::ResponseManifold::ConstantCurvature {
                dim,
                ..
            } => dim,
            other => {
                return Err(format!(
                    "response_geometry_fit_curvature requires a constant_curvature geometry; \
                     got {}",
                    other.canonical_label()
                ));
            }
        };
        gam::geometry::response_geometry::fit_response_curvature(
            arr.view(),
            dim,
            level,
            1.0e-12,
            256,
        )
    })?;
    let verdict = match fit.profile_ci.verdict {
        gam::geometry::CurvatureVerdict::Spherical => "spherical",
        gam::geometry::CurvatureVerdict::Hyperbolic => "hyperbolic",
        gam::geometry::CurvatureVerdict::Flat => "flat",
    }
    .to_string();
    Ok((
        fit.kappa_hat,
        fit.profile_ci.ci_lo,
        fit.profile_ci.ci_hi,
        fit.profile_ci.lo_at_bound,
        fit.profile_ci.hi_at_bound,
        verdict,
        fit.flatness.lr_stat,
        fit.flatness.p_value,
        fit.railed_at_resolution_limit,
        fit.kappa_r2,
        fit.characteristic_radius,
        fit.base.into_pyarray(py).unbind(),
    ))
}

#[pyfunction]
fn sinkhorn_circular_cost<'py>(py: Python<'py>, m: usize) -> PyResult<Py<PyArray2<f64>>> {
    let out = py.detach(move || sinkhorn_circular_cost_impl(m));
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sinkhorn_euclidean_cost<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let owned = points.as_array().to_owned();
    let out = detach_py_result(py, "sinkhorn_euclidean_cost", move || {
        sinkhorn_euclidean_cost_impl(owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sinkhorn_geodesic_sphere_cost<'py>(
    py: Python<'py>,
    directions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let owned = directions.as_array().to_owned();
    let out = detach_py_result(py, "sinkhorn_geodesic_sphere_cost", move || {
        sinkhorn_geodesic_sphere_cost_impl(owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sinkhorn_barycenter_forward<'py>(
    py: Python<'py>,
    atoms: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    cost: PyReadonlyArray2<'py, f64>,
    eps: f64,
    n_iter: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let atoms_owned = atoms.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let cost_owned = cost.as_array().to_owned();
    let out = detach_py_result(py, "sinkhorn_barycenter_forward", move || {
        sinkhorn_barycenter_impl(
            atoms_owned.view(),
            weights_owned.view(),
            cost_owned.view(),
            eps,
            n_iter,
        )
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sinkhorn_barycenter_vjp<'py>(
    py: Python<'py>,
    atoms: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    cost: PyReadonlyArray2<'py, f64>,
    eps: f64,
    n_iter: usize,
    cotangent: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<f64>>)> {
    let atoms_owned = atoms.as_array().to_owned();
    let weights_owned = weights.as_array().to_owned();
    let cost_owned = cost.as_array().to_owned();
    let cot_owned = cotangent.as_array().to_owned();
    let result = detach_py_result(py, "sinkhorn_barycenter_vjp", move || {
        sinkhorn_barycenter_vjp_impl(
            atoms_owned.view(),
            weights_owned.view(),
            cost_owned.view(),
            eps,
            n_iter,
            cot_owned.view(),
        )
    })?;
    let d_atoms = result.d_atoms.into_pyarray(py).unbind();
    let d_weights = result.d_weights.into_pyarray(py).unbind();
    Ok((d_atoms, d_weights))
}

#[pyfunction]
fn numerics_sigmoid_stable<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let owned = x.as_array().to_owned();
    let out = py.detach(move || owned.mapv(sigmoid_stable));
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn numerics_inverse_softplus<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let owned = x.as_array().to_owned();
    let out = py.detach(move || owned.mapv(inverse_softplus_scalar));
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn response_geometry_normalize_fisher_rao<'py>(
    py: Python<'py>,
    value: PyReadonlyArrayDyn<'py, f64>,
    n_rows: usize,
    dim: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    let arr = value.as_array().to_owned();
    let out = detach_py_result(py, "response_geometry_normalize_fisher_rao", move || {
        gam::inference::fisher_rao::normalize_fisher_rao_blocks(arr.view(), n_rows, dim)
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (values, weights = None, tol = 1.0e-12, max_iter = 256))]
fn sphere_frechet_mean<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let arr = values.as_array().to_owned();
    let w_owned = weights.as_ref().map(|w| w.as_array().to_owned());
    let mean = detach_py_result(py, "sphere_frechet_mean", move || {
        gam::geometry::sphere::sphere_frechet_mean(
            arr.view(),
            w_owned.as_ref().map(|w| w.view()),
            tol,
            max_iter,
        )
    })?;
    Ok(Array1::from(mean).into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho<'py>(
    py: Python<'py>,
    group: String,
    g: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let g_owned = g.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho", move || {
        gam::geometry::lie_so::rho(group.as_str(), g_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_aux_enabled(aux: Option<String>) -> bool {
    aux.is_some()
}

#[pyfunction]
fn equivariant_rho_so2<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let theta_owned = theta.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so2", move || {
        Ok(gam::geometry::lie_so::rho_so2(theta_owned.view()))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so2_jvp<'py>(
    py: Python<'py>,
    theta: PyReadonlyArray1<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let theta_owned = theta.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so2_jvp", move || {
        Ok(gam::geometry::lie_so::rho_so2_jvp(theta_owned.view()))
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so3<'py>(
    py: Python<'py>,
    omega: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let omega_owned = omega.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so3", move || {
        gam::geometry::lie_so::rho_so3(omega_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn equivariant_rho_so3_jvp<'py>(
    py: Python<'py>,
    omega: PyReadonlyArray2<'py, f64>,
    domega: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray3<f64>>> {
    let o_owned = omega.as_array().to_owned();
    let d_owned = domega.as_array().to_owned();
    let out = detach_py_result(py, "equivariant_rho_so3_jvp", move || {
        gam::geometry::lie_so::rho_so3_jvp(o_owned.view(), d_owned.view())
    })?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (aux_values, theta, d_aux, weight = 1.0))]
fn equivariant_gauge_companion_loss<'py>(
    py: Python<'py>,
    aux_values: Option<PyReadonlyArray2<'py, f64>>,
    theta: PyReadonlyArray2<'py, f64>,
    d_aux: usize,
    weight: f64,
) -> PyResult<f64> {
    let Some(aux_values) = aux_values else {
        return Ok(0.0);
    };
    let aux_owned = aux_values.as_array().to_owned();
    let theta_owned = theta.as_array().to_owned();
    detach_py_result(py, "equivariant_gauge_companion_loss", move || {
        gam::terms::equivariant_penalty::gauge_companion_loss(
            aux_owned.view(),
            theta_owned.view(),
            d_aux,
            weight,
        )
    })
}

#[pyclass(module = "gam_pyffi._rust", name = "SparsityPenalty")]
struct SparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    kind: String,
    #[pyo3(get, set)]
    weight: PyObject,
    #[pyo3(get, set)]
    eps: f64,
    #[pyo3(get, set)]
    eps_weight: String,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl SparsityPenalty {
    #[new]
    #[pyo3(signature = (kind = "smooth_l1".to_string(), weight = None, eps = 1.0e-3, eps_weight = "fixed".to_string(), *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        kind: String,
        weight: Option<&Bound<'_, PyAny>>,
        eps: f64,
        eps_weight: String,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let weight = py_object_or_string_default(py, weight, "auto");
        let target = py_object_or_string_default(py, target, "t");
        validate_sparsity_weight(py, weight.bind(py), "SparsityPenalty")?;
        if !matches!(kind.as_str(), "smooth_l1" | "hoyer" | "log") {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.kind must be one of 'smooth_l1' | 'hoyer' | 'log', got {kind:?}"
            )));
        }
        if kind == "hoyer" && eps_weight == "auto" {
            return Err(PyValueError::new_err(
                "SparsityPenalty(kind='hoyer'): Hoyer has no smoothing scale, so eps_weight='auto' is not meaningful.",
            ));
        }
        if eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.eps must be > 0, got {eps}"
            )));
        }
        if !matches!(eps_weight.as_str(), "auto" | "fixed") {
            return Err(PyValueError::new_err(format!(
                "SparsityPenalty.eps_weight must be 'auto' or 'fixed', got {eps_weight:?}"
            )));
        }
        Ok(Self {
            target,
            kind,
            weight,
            eps,
            eps_weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("sparsity_kind", &self.kind)?;
        payload.set_item("weight", self.weight.bind(py))?;
        payload.set_item("eps", self.eps)?;
        payload.set_item("eps_weight", &self.eps_weight)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SparsityPenalty(kind={}, weight={}, eps={}, eps_weight={}, target={}, weight_schedule={})",
            self.kind.as_str(),
            py_repr(py, self.weight.bind(py))?,
            self.eps,
            self.eps_weight.as_str(),
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn validate_sparsity_weight(py: Python<'_>, weight: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    if let Ok(value) = weight.extract::<String>() {
        if value == "auto" {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "{name}.weight: only 'auto' is accepted as a string, got {value:?}"
        )));
    }
    if let Ok(value) = weight.extract::<f64>() {
        if value > 0.0 {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "{name}.weight must be > 0, got {}",
            py_repr(py, weight)?
        )));
    }
    Err(PyTypeError::new_err(format!(
        "{name}.weight must be 'auto' or a positive float, got {}",
        weight.get_type().name()?
    )))
}

fn py_object_or_string_default(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
    default: &str,
) -> PyObject {
    match value {
        Some(value) => value.clone().unbind(),
        None => pyo3::types::PyString::new(py, default).into_any().unbind(),
    }
}

fn target_descriptor(py: Python<'_>, target: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    if target.is_instance_of::<PyString>() || target.extract::<isize>().is_ok() {
        return Ok(target.clone().unbind());
    }
    if let Ok(name) = target.getattr("name") {
        return Ok(name.str()?.into_pyobject(py)?.unbind().into());
    }
    Err(PyTypeError::new_err(
        "analytic penalty target must be a latent block name, latent block index, or object exposing a 'name' attribute",
    ))
}

/// Eagerly validate that an analytic-penalty `target` keyword argument is well
/// formed at construction time, rather than deferring the check until
/// `to_rust_descriptor` is invoked. This lets users see configuration errors
/// at the moment they create the penalty object, with the offending class
/// named in the message so the source of the bad target is obvious.
fn validate_target_eager(
    py: Python<'_>,
    class_name: &str,
    target: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let Some(target) = target else {
        return Ok(());
    };
    if target.is_none() {
        return Ok(());
    }
    if target.is_instance_of::<PyString>() || target.extract::<isize>().is_ok() {
        return Ok(());
    }
    if let Ok(name) = target.getattr("name") {
        // Require that `.name` actually resolves to a string-coercible value.
        name.str()?.into_pyobject(py)?;
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "{class_name}.target must be a latent block name (str), index (int), or object exposing a 'name' attribute; got {}",
        target.get_type().name()?
    )))
}

fn py_repr(_py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    value.repr()?.extract()
}

#[pyclass(module = "gam_pyffi._rust", name = "ARDPenalty")]
struct ARDPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl ARDPenalty {
    #[new]
    #[pyo3(signature = (*, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        validate_target_eager(py, "ARDPenalty", target)?;
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "ard";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ARDPenalty(target={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "TopKActivationPenalty")]
struct PyTopKActivationPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    k: i64,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl PyTopKActivationPenalty {
    #[new]
    #[pyo3(signature = (k, weight = 1.0, *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        k: &Bound<'_, PyAny>,
        weight: f64,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let k = k.call_method0("__index__")?.extract::<i64>()?;
        if k <= 0 {
            return Err(PyValueError::new_err(format!(
                "TopKActivationPenalty.k must be > 0, got {k}"
            )));
        }
        if !(weight > 0.0) {
            return Err(PyValueError::new_err(format!(
                "TopKActivationPenalty.weight must be finite and > 0, got {weight}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            k,
            weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "topk_activation";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("k", self.k)?;
        payload.set_item("weight", self.weight)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "TopKActivationPenalty(k={}, weight={}, target={}, weight_schedule={})",
            self.k,
            self.weight,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "JumpReLUPenalty")]
struct JumpReLUPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    thresholds: Vec<f64>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl JumpReLUPenalty {
    #[new]
    #[pyo3(signature = (thresholds, weight = 1.0, smoothing_eps = 1.0e-3, *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        thresholds: &Bound<'_, PyAny>,
        weight: f64,
        smoothing_eps: f64,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let numpy = py.import("numpy")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "float")?;
        let thresholds_array = numpy
            .call_method("asarray", (thresholds,), Some(&kwargs))?
            .call_method1("reshape", (-1,))?;
        let thresholds = thresholds_array
            .call_method0("tolist")?
            .extract::<Vec<f64>>()?;
        if thresholds.is_empty() {
            return Err(PyValueError::new_err(
                "JumpReLUPenalty.thresholds must be non-empty",
            ));
        }
        if thresholds
            .iter()
            .any(|threshold| !threshold.is_finite() || *threshold <= 0.0)
        {
            return Err(PyValueError::new_err(
                "JumpReLUPenalty.thresholds must be finite and > 0",
            ));
        }
        if !(weight > 0.0) {
            return Err(PyValueError::new_err(format!(
                "JumpReLUPenalty.weight must be finite and > 0, got {weight}"
            )));
        }
        if !(smoothing_eps > 0.0) {
            return Err(PyValueError::new_err(format!(
                "JumpReLUPenalty.smoothing_eps must be finite and > 0, got {smoothing_eps}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            thresholds,
            weight,
            smoothing_eps,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "jumprelu";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("thresholds", self.thresholds.clone())?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "JumpReLUPenalty(thresholds={:?}, weight={}, smoothing_eps={}, target={}, weight_schedule={})",
            self.thresholds,
            self.weight,
            self.smoothing_eps,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn topk_weight_schedule_descriptor(
    _py: Python<'_>,
    schedule: &Option<PyObject>,
) -> PyResult<Option<PyObject>> {
    let Some(schedule) = schedule else {
        return Ok(None);
    };
    let bound = schedule.bind(_py);
    if bound.hasattr("to_rust_descriptor")? {
        return Ok(Some(
            bound.call_method0("to_rust_descriptor")?.unbind().into(),
        ));
    }
    if let Ok(mapping) = bound.cast::<PyDict>() {
        return Ok(Some(mapping.copy()?.unbind().into()));
    }
    Err(PyTypeError::new_err(
        "weight_schedule must be ScalarWeightSchedule, a mapping, or None",
    ))
}

fn aux_conditional_prior_float_array<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "float")?;
    numpy.call_method("asarray", (value,), Some(&kwargs))
}

fn validate_aux_conditional_prior_lambda(
    lambda_per_row: &Bound<'_, PyAny>,
    weight: f64,
    n_eff: i64,
) -> PyResult<()> {
    if !(weight > 0.0) {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.weight must be > 0, got {weight}"
        )));
    }
    if n_eff <= 0 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.n_eff must be > 0, got {n_eff}"
        )));
    }

    let array = lambda_per_row.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let view = array.as_array();
    let shape = view.shape();
    if view.ndim() != 3 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row must have shape (N, d, d), got ndim={}",
            view.ndim()
        )));
    }

    let n_obs = shape[0];
    let rows = shape[1];
    let cols = shape[2];
    if n_obs as i64 != n_eff {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row first dimension must equal n_eff={n_eff}, got {n_obs}"
        )));
    }
    if rows == 0 || cols == 0 || rows != cols {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row must have square non-empty row matrices, got shape ({n_obs}, {rows}, {cols})"
        )));
    }
    if !view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "AuxConditionalPriorPenalty.lambda_per_row must be finite",
        ));
    }

    let mut max_asym = 0.0_f64;
    for obs in 0..n_obs {
        for row in 0..rows {
            for col in 0..cols {
                let asym = (view[IxDyn(&[obs, row, col])] - view[IxDyn(&[obs, col, row])]).abs();
                if asym > max_asym {
                    max_asym = asym;
                }
            }
        }
    }
    if max_asym >= 1.0e-10 {
        return Err(PyValueError::new_err(format!(
            "AuxConditionalPriorPenalty.lambda_per_row matrices must be symmetric within 1e-10; max asymmetry is {max_asym:.3e}"
        )));
    }
    Ok(())
}

#[pyclass(module = "gam_pyffi._rust", name = "BlockSparsityPenalty")]
struct BlockSparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    groups: Vec<Vec<usize>>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl BlockSparsityPenalty {
    #[new]
    #[pyo3(signature = (groups, weight, n_eff, smoothing_eps = 1e-6, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        groups: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: i64,
        smoothing_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let groups = block_sparsity_coerce_groups(groups)?;
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        if smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            groups,
            weight,
            n_eff,
            smoothing_eps,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "block_sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("groups", self.groups.clone())?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "BlockSparsityPenalty(groups={:?}, weight={}, n_eff={}, smoothing_eps={}, learnable={}, target={}, weight_schedule={})",
            self.groups,
            self.weight,
            self.n_eff,
            self.smoothing_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn block_sparsity_groups_type_error() -> PyErr {
    PyTypeError::new_err("BlockSparsityPenalty.groups must be a sequence of integer sequences")
}

fn block_sparsity_coerce_groups(groups: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
    let group_iter = groups
        .try_iter()
        .map_err(|_| block_sparsity_groups_type_error())?;
    let mut coerced = Vec::new();
    for group in group_iter {
        let group = group.map_err(|_| block_sparsity_groups_type_error())?;
        let axis_iter = group
            .try_iter()
            .map_err(|_| block_sparsity_groups_type_error())?;
        let mut axes = Vec::new();
        for axis in axis_iter {
            let axis = axis.map_err(|_| block_sparsity_groups_type_error())?;
            axes.push(
                axis.call_method0("__index__")
                    .and_then(|value| value.extract::<isize>())
                    .map_err(|_| block_sparsity_groups_type_error())?,
            );
        }
        coerced.push(axes);
    }
    if coerced.is_empty() {
        return Err(PyValueError::new_err(
            "BlockSparsityPenalty.groups must not be empty",
        ));
    }

    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(coerced.len());
    for (group_idx, group) in coerced.into_iter().enumerate() {
        if group.is_empty() {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.groups[{group_idx}] must not be empty"
            )));
        }
        let mut out_group = Vec::with_capacity(group.len());
        for axis in group {
            if axis < 0 {
                return Err(PyValueError::new_err(format!(
                    "BlockSparsityPenalty.groups entries must be non-negative; got {axis}"
                )));
            }
            let axis = axis as usize;
            if !seen.insert(axis) {
                return Err(PyValueError::new_err(format!(
                    "BlockSparsityPenalty.groups axis {axis} appears more than once"
                )));
            }
            out_group.push(axis);
        }
        out.push(out_group);
    }
    let max_axis = seen.iter().next_back().copied().unwrap_or(0);
    for axis in 0..=max_axis {
        if !seen.contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "BlockSparsityPenalty.groups must partition contiguous axes from 0; missing axis {axis}"
            )));
        }
    }
    Ok(out)
}

#[pyclass(module = "gam_pyffi._rust", name = "SoftmaxAssignmentSparsityPenalty")]
struct SoftmaxAssignmentSparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    k_atoms: i64,
    #[pyo3(get, set)]
    temperature: f64,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl SoftmaxAssignmentSparsityPenalty {
    #[new]
    #[pyo3(signature = (k_atoms, temperature = 1.0, *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        k_atoms: &Bound<'_, PyAny>,
        temperature: f64,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let builtins = PyModule::import(py, "builtins")?;
        let k_atoms = builtins
            .getattr("int")?
            .call1((k_atoms,))?
            .extract::<i64>()?;
        if k_atoms <= 0 {
            return Err(PyValueError::new_err(format!(
                "SoftmaxAssignmentSparsityPenalty.k_atoms must be > 0, got {k_atoms}"
            )));
        }
        if temperature <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "SoftmaxAssignmentSparsityPenalty.temperature must be > 0, got {temperature}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            k_atoms,
            temperature,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "softmax_assignment_sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("k_atoms", self.k_atoms)?;
        payload.set_item("temperature", self.temperature)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn _to_rust_payload(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_rust_descriptor(py)
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SoftmaxAssignmentSparsityPenalty(target={}, k_atoms={}, temperature={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            self.k_atoms,
            self.temperature,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "IsometryPenalty")]
struct IsometryPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: PyObject,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl IsometryPenalty {
    #[new]
    #[pyo3(signature = (weight = None, *, target = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        weight: Option<&Bound<'_, PyAny>>,
        target: Option<&Bound<'_, PyAny>>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let weight = py_object_or_string_default(py, weight, "auto");
        let target = py_object_or_string_default(py, target, "t");
        validate_sparsity_weight(py, weight.bind(py), "IsometryPenalty")?;
        Ok(Self {
            target,
            weight,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "isometry";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight.bind(py))?;
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "IsometryPenalty(weight={}, target={}, weight_schedule={})",
            py_repr(py, self.weight.bind(py))?,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "IBPAssignmentPenalty")]
struct PyIBPAssignmentPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    k_max: i64,
    #[pyo3(get, set)]
    alpha: f64,
    #[pyo3(get, set)]
    tau: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get, set)]
    temperature_schedule: Option<PyObject>,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl PyIBPAssignmentPenalty {
    #[new]
    #[pyo3(signature = (k_max, alpha = 1.0, tau = 1.0, learnable = false, *, target = None, temperature_schedule = None, weight_schedule = None))]
    fn new(
        py: Python<'_>,
        k_max: &Bound<'_, PyAny>,
        alpha: f64,
        tau: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
        temperature_schedule: Option<PyObject>,
        weight_schedule: Option<PyObject>,
    ) -> PyResult<Self> {
        let builtins = PyModule::import(py, "builtins")?;
        let k_max = builtins.getattr("int")?.call1((k_max,))?.extract::<i64>()?;
        if k_max <= 0 {
            return Err(PyValueError::new_err(format!(
                "IBPAssignmentPenalty.k_max must be > 0, got {k_max}"
            )));
        }
        if alpha <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "IBPAssignmentPenalty.alpha must be > 0, got {alpha}"
            )));
        }
        if tau <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "IBPAssignmentPenalty.tau must be > 0, got {tau}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            k_max,
            alpha,
            tau,
            learnable,
            temperature_schedule,
            weight_schedule,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "ibp_assignment";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("k_max", self.k_max)?;
        payload.set_item("alpha", self.alpha)?;
        payload.set_item("tau", self.tau)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = &self.temperature_schedule {
            payload.set_item("temperature_schedule", schedule.bind(py))?;
        }
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn _to_rust_payload(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_rust_descriptor(py)
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "IBPAssignmentPenalty(target={}, k_max={}, alpha={}, tau={}, learnable={}, temperature_schedule={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            self.k_max,
            self.alpha,
            self.tau,
            self.learnable,
            match &self.temperature_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            },
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn total_variation_difference_op_type_error() -> PyErr {
    PyTypeError::new_err(
        "TotalVariationPenalty.difference_op must be 'forward_1d' or a sequence of (from_row, to_row) edges",
    )
}

fn total_variation_coerce_edges(edges: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, i64)>> {
    let edge_iter = edges
        .try_iter()
        .map_err(|_| total_variation_difference_op_type_error())?;
    let mut out = Vec::new();
    for edge in edge_iter {
        let edge = edge.map_err(|_| total_variation_difference_op_type_error())?;
        let mut endpoints = edge
            .try_iter()
            .map_err(|_| total_variation_difference_op_type_error())?;
        let a = endpoints
            .next()
            .ok_or_else(total_variation_difference_op_type_error)?
            .map_err(|_| total_variation_difference_op_type_error())?
            .call_method0("__index__")
            .and_then(|value| value.extract::<i64>())
            .map_err(|_| total_variation_difference_op_type_error())?;
        let b = endpoints
            .next()
            .ok_or_else(total_variation_difference_op_type_error)?
            .map_err(|_| total_variation_difference_op_type_error())?
            .call_method0("__index__")
            .and_then(|value| value.extract::<i64>())
            .map_err(|_| total_variation_difference_op_type_error())?;
        if endpoints.next().is_some() {
            return Err(total_variation_difference_op_type_error());
        }
        out.push((a, b));
    }
    Ok(out)
}

#[pyclass(module = "gam_pyffi._rust", name = "TotalVariationPenalty")]
struct TotalVariationPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    difference_op: PyObject,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get, set)]
    _edges: Option<Vec<(i64, i64)>>,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl TotalVariationPenalty {
    #[new]
    #[pyo3(signature = (weight, n_eff, difference_op = None, smoothing_eps = 1.0e-6, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        weight: f64,
        n_eff: i64,
        difference_op: Option<&Bound<'_, PyAny>>,
        smoothing_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "TotalVariationPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "TotalVariationPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        if smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "TotalVariationPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }

        let difference_op = py_object_or_string_default(py, difference_op, "forward_1d");
        let target = py_object_or_string_default(py, target, "t");
        let bound_difference_op = difference_op.bind(py);
        let edges = if let Ok(op) = bound_difference_op.extract::<String>() {
            if op != "forward_1d" {
                return Err(PyValueError::new_err(format!(
                    "TotalVariationPenalty.difference_op string must be 'forward_1d', got {op:?}"
                )));
            }
            None
        } else {
            let parsed_edges = total_variation_coerce_edges(bound_difference_op)?;
            if parsed_edges.is_empty() {
                return Err(PyValueError::new_err(
                    "TotalVariationPenalty graph edges must not be empty",
                ));
            }
            for (a, b) in &parsed_edges {
                if *a < 0 || *b < 0 || *a >= n_eff || *b >= n_eff {
                    return Err(PyValueError::new_err(format!(
                        "TotalVariationPenalty graph edges must be within [0, n_eff); got ({a}, {b}) for n_eff={n_eff}"
                    )));
                }
                if a == b {
                    return Err(PyValueError::new_err(format!(
                        "TotalVariationPenalty graph edge ({a}, {b}) is self-referential"
                    )));
                }
            }
            Some(parsed_edges)
        };

        Ok(Self {
            target,
            weight,
            n_eff,
            difference_op,
            smoothing_eps,
            learnable,
            _edges: edges,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "total_variation";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(edges) = &self._edges {
            payload.set_item("difference_op", "graph_edges")?;
            payload.set_item("edges", edges)?;
        } else {
            payload.set_item("difference_op", "forward_1d")?;
        }
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "TotalVariationPenalty(weight={}, n_eff={}, difference_op={}, smoothing_eps={}, learnable={}, target={}, weight_schedule={})",
            self.weight,
            self.n_eff,
            py_repr(py, self.difference_op.bind(py))?,
            self.smoothing_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

fn validate_parametric_aux_conditional_prior(
    aux: &Bound<'_, PyAny>,
    alpha_init: &Bound<'_, PyAny>,
    beta_init: &Bound<'_, PyAny>,
    mu_init: &Bound<'_, PyAny>,
    weight: f64,
    n_eff: i64,
) -> PyResult<()> {
    if weight <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.weight must be > 0, got {weight}"
        )));
    }
    if n_eff <= 0 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.n_eff must be > 0, got {n_eff}"
        )));
    }

    let aux_array = aux.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let alpha_array = alpha_init.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let beta_array = beta_init.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let mu_array = mu_init.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let aux_view = aux_array.as_array();
    let alpha_view = alpha_array.as_array();
    let beta_view = beta_array.as_array();
    let mu_view = mu_array.as_array();

    if aux_view.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.aux must have shape (N, du), got ndim={}",
            aux_view.ndim()
        )));
    }
    if alpha_view.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.alpha_init must have shape (d,), got ndim={}",
            alpha_view.ndim()
        )));
    }
    if beta_view.ndim() != 1 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.beta_init must have shape (d,), got ndim={}",
            beta_view.ndim()
        )));
    }
    if mu_view.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.mu_init must have shape (d, du), got ndim={}",
            mu_view.ndim()
        )));
    }

    let aux_shape = aux_view.shape();
    let alpha_shape = alpha_view.shape();
    let beta_shape = beta_view.shape();
    let mu_shape = mu_view.shape();
    let n_obs = aux_shape[0];
    let aux_dim = aux_shape[1];
    let latent_dim = alpha_shape[0];
    if n_obs as i64 != n_eff {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.aux first dimension must equal n_eff={n_eff}, got {n_obs}"
        )));
    }
    if aux_dim == 0 {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.aux must have du > 0",
        ));
    }
    if latent_dim == 0 {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.alpha_init must be non-empty",
        ));
    }
    if beta_shape[0] != latent_dim {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.beta_init shape must match alpha_init shape ({latent_dim},), got ({},)",
            beta_shape[0]
        )));
    }
    if mu_shape[0] != latent_dim || mu_shape[1] != aux_dim {
        return Err(PyValueError::new_err(format!(
            "ParametricAuxConditionalPriorPenalty.mu_init must have shape ({latent_dim}, {aux_dim}), got ({}, {})",
            mu_shape[0], mu_shape[1]
        )));
    }
    if !aux_view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.aux must be finite",
        ));
    }
    if !alpha_view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.alpha_init must be finite",
        ));
    }
    if !beta_view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.beta_init must be finite",
        ));
    }
    if !mu_view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.mu_init must be finite",
        ));
    }
    if !alpha_view.iter().all(|value| *value > 0.0) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.alpha_init must be > 0",
        ));
    }
    if !beta_view.iter().all(|value| *value > 0.0) {
        return Err(PyValueError::new_err(
            "ParametricAuxConditionalPriorPenalty.beta_init must be > 0",
        ));
    }
    Ok(())
}

fn inverse_softplus_scalar(value: f64) -> f64 {
    if value <= 0.0 || value.is_nan() {
        f64::NAN
    } else if value > 30.0 {
        value + (-(-value).exp()).ln_1p()
    } else {
        value.exp_m1().ln()
    }
}

#[pyclass(
    module = "gam_pyffi._rust",
    name = "ParametricAuxConditionalPriorPenalty"
)]
struct ParametricAuxConditionalPriorPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    aux: PyObject,
    #[pyo3(get, set)]
    alpha_init: PyObject,
    #[pyo3(get, set)]
    beta_init: PyObject,
    #[pyo3(get, set)]
    mu_init: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl ParametricAuxConditionalPriorPenalty {
    #[new]
    #[pyo3(signature = (aux, alpha_init, beta_init, mu_init, weight, n_eff, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        aux: &Bound<'_, PyAny>,
        alpha_init: &Bound<'_, PyAny>,
        beta_init: &Bound<'_, PyAny>,
        mu_init: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: i64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let aux = aux_conditional_prior_float_array(py, aux)?;
        let alpha_init = aux_conditional_prior_float_array(py, alpha_init)?;
        let beta_init = aux_conditional_prior_float_array(py, beta_init)?;
        let mu_init = aux_conditional_prior_float_array(py, mu_init)?;
        validate_parametric_aux_conditional_prior(
            &aux,
            &alpha_init,
            &beta_init,
            &mu_init,
            weight,
            n_eff,
        )?;
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            aux: aux.unbind(),
            alpha_init: alpha_init.unbind(),
            beta_init: beta_init.unbind(),
            mu_init: mu_init.unbind(),
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "parametric_aux_conditional_prior";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let aux_array = self.aux.bind(py).extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let alpha_array = self
            .alpha_init
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let beta_array = self
            .beta_init
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let mu_array = self
            .mu_init
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let aux_view = aux_array.as_array();
        let alpha_view = alpha_array.as_array();
        let beta_view = beta_array.as_array();
        let mu_view = mu_array.as_array();

        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item(
            "aux",
            PyList::new(py, aux_view.iter().copied().collect::<Vec<_>>())?,
        )?;
        payload.set_item("aux_shape", PyList::new(py, aux_view.shape())?)?;
        payload.set_item(
            "log_alpha",
            PyList::new(
                py,
                alpha_view
                    .iter()
                    .map(|value| value.ln())
                    .collect::<Vec<_>>(),
            )?,
        )?;
        payload.set_item(
            "raw_beta",
            PyList::new(
                py,
                beta_view
                    .iter()
                    .map(|value| inverse_softplus_scalar(*value))
                    .collect::<Vec<_>>(),
            )?,
        )?;
        payload.set_item(
            "mu",
            PyList::new(py, mu_view.iter().copied().collect::<Vec<_>>())?,
        )?;
        payload.set_item("mu_shape", PyList::new(py, mu_view.shape())?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ParametricAuxConditionalPriorPenalty(aux={}, alpha_init={}, beta_init={}, mu_init={}, weight={}, n_eff={}, learnable={}, target={}, weight_schedule={})",
            py_repr(py, self.aux.bind(py))?,
            py_repr(py, self.alpha_init.bind(py))?,
            py_repr(py, self.beta_init.bind(py))?,
            py_repr(py, self.mu_init.bind(py))?,
            self.weight,
            self.n_eff,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }

    /// Evaluate the parametric iVAE-style row-precision prior value and
    /// gradient at latent block `t` of shape `(n_eff, latent_dim)`. Returns
    /// `(value, grad)` with `grad` having the same shape as `t`.
    #[pyo3(signature = (t))]
    fn value_grad<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>)> {
        let view = t.as_array();
        let (n_rows, latent_dim) = view.dim();
        if n_rows as i64 != self.n_eff {
            return Err(PyValueError::new_err(format!(
                "ParametricAuxConditionalPriorPenalty.value_grad: t first dim {n_rows} must match n_eff={}",
                self.n_eff
            )));
        }
        if latent_dim == 0 {
            return Err(PyValueError::new_err(
                "ParametricAuxConditionalPriorPenalty.value_grad: t must have latent_dim > 0",
            ));
        }
        let aux_array = self.aux.bind(py).extract::<PyReadonlyArray2<'_, f64>>()?;
        let aux_owned: Array2<f64> = aux_array.as_array().to_owned();
        let alpha_array = self
            .alpha_init
            .bind(py)
            .extract::<PyReadonlyArray1<'_, f64>>()?;
        let beta_array = self
            .beta_init
            .bind(py)
            .extract::<PyReadonlyArray1<'_, f64>>()?;
        let mu_array = self
            .mu_init
            .bind(py)
            .extract::<PyReadonlyArray2<'_, f64>>()?;
        let log_alpha: Array1<f64> = alpha_array.as_array().iter().map(|v| v.ln()).collect();
        let raw_beta: Array1<f64> = beta_array
            .as_array()
            .iter()
            .map(|v| inverse_softplus_scalar(*v))
            .collect();
        let mu_owned: Array2<f64> = mu_array.as_array().to_owned();
        let slice = PsiSlice::full(n_rows * latent_dim, Some(latent_dim));
        let penalty = ParametricRowPrecisionPriorPenalty::new(
            slice,
            aux_owned,
            log_alpha,
            raw_beta,
            mu_owned,
            self.weight,
            n_rows,
            false,
        )
        .map_err(py_value_error)?;
        let flat: Array1<f64> = view.iter().copied().collect();
        let rho = Array1::<f64>::zeros(0);
        let value = penalty.value(flat.view(), rho.view());
        let grad_flat = penalty.grad_target(flat.view(), rho.view());
        let grad = grad_flat
            .into_shape_with_order((n_rows, latent_dim))
            .map_err(|err| {
                py_value_error(format!(
                    "ParametricAuxConditionalPriorPenalty.value_grad: gradient reshape failed: {err}"
                ))
            })?;
        Ok((value, grad.into_pyarray(py).unbind()))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "OrthogonalityPenalty")]
struct OrthogonalityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl OrthogonalityPenalty {
    #[new]
    #[pyo3(signature = (weight, n_eff, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        weight: f64,
        n_eff: i64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "OrthogonalityPenalty.weight must be > 0, got {weight:?}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "OrthogonalityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "orthogonality";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "OrthogonalityPenalty(weight={:?}, n_eff={}, learnable={}, target={}, weight_schedule={})",
            self.weight,
            self.n_eff,
            if self.learnable { "True" } else { "False" },
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "ScadMcpPenalty")]
struct ScadMcpPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    gamma: f64,
    #[pyo3(get, set)]
    variant: String,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl ScadMcpPenalty {
    #[new]
    #[pyo3(signature = (weight, n_eff, gamma = None, variant = "mcp".to_string(), smoothing_eps = 1.0e-6, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        weight: f64,
        n_eff: i64,
        gamma: Option<f64>,
        variant: String,
        smoothing_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let gamma = match gamma {
            Some(value) => value,
            None => match variant.as_str() {
                "mcp" => 2.5,
                "scad" => 3.7,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "ScadMcpPenalty.variant must be 'mcp' or 'scad', got {variant:?}"
                    )));
                }
            },
        };
        let penalty = Self {
            target: py_object_or_string_default(py, target, "t"),
            weight,
            n_eff,
            gamma,
            variant,
            smoothing_eps,
            learnable,
            weight_schedule: None,
        };
        penalty.validate()?;
        Ok(penalty)
    }

    #[classattr]
    const KIND_TAG: &'static str = "scad_mcp";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("gamma", self.gamma)?;
        payload.set_item("variant", &self.variant)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = &self.weight_schedule {
            payload.set_item("weight_schedule", schedule.bind(py))?;
        }
        Ok(payload.into())
    }

    fn _to_rust_payload(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.to_rust_descriptor(py)
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ScadMcpPenalty(weight={}, n_eff={}, gamma={}, variant={}, smoothing_eps={}, learnable={}, target={}, weight_schedule={})",
            self.weight,
            self.n_eff,
            self.gamma,
            self.variant.as_str(),
            self.smoothing_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

impl ScadMcpPenalty {
    fn validate(&self) -> PyResult<()> {
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.weight must be > 0, got {}",
                self.weight
            )));
        }
        if self.n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.n_eff must be > 0, got {}",
                self.n_eff
            )));
        }
        if !self.gamma.is_finite() || self.gamma <= 1.0 {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.gamma must be > 1, got {}",
                self.gamma
            )));
        }
        if !matches!(self.variant.as_str(), "mcp" | "scad") {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.variant must be 'mcp' or 'scad', got {:?}",
                self.variant
            )));
        }
        if !self.smoothing_eps.is_finite() || self.smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "ScadMcpPenalty.smoothing_eps must be > 0, got {}",
                self.smoothing_eps
            )));
        }
        Ok(())
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "IvaeRidgeMeanGauge")]
struct IvaeRidgeMeanGauge {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    aux: PyObject,
    #[pyo3(get, set)]
    ridge_eps: f64,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl IvaeRidgeMeanGauge {
    #[new]
    #[pyo3(signature = (aux, weight, n_eff, ridge_eps = 1.0e-6, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        aux: &Bound<'_, PyAny>,
        weight: &Bound<'_, PyAny>,
        n_eff: &Bound<'_, PyAny>,
        ridge_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let builtins = py.import("builtins")?;
        let weight = builtins
            .getattr("float")?
            .call1((weight,))?
            .extract::<f64>()?;
        let n_eff = builtins.getattr("int")?.call1((n_eff,))?.extract::<i64>()?;
        let aux = aux_conditional_prior_float_array(py, aux)?;
        validate_ivae_ridge_mean_gauge_aux(&aux, weight, n_eff, ridge_eps)?;
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            aux: aux.unbind(),
            ridge_eps,
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "ivae_ridge_mean_gauge";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;

        let aux = aux_conditional_prior_float_array(py, self.aux.bind(py))?;
        let array = aux.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let view = array.as_array();
        payload.set_item(
            "aux",
            PyList::new(py, view.iter().copied().collect::<Vec<_>>())?,
        )?;
        payload.set_item("aux_shape", PyList::new(py, view.shape())?)?;
        payload.set_item("ridge_eps", self.ridge_eps)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "IvaeRidgeMeanGauge(aux={}, weight={}, n_eff={}, ridge_eps={}, learnable={}, target={}, weight_schedule={})",
            py_repr(py, self.aux.bind(py))?,
            self.weight,
            self.n_eff,
            self.ridge_eps,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }

    /// Evaluate the iVAE ridge conditional-mean gauge penalty value and
    /// gradient at latent block `t` of shape `(n_eff, latent_dim)`. Returns
    /// `(value, grad)` with `grad` having the same shape as `t`.
    #[pyo3(signature = (t))]
    fn value_grad<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>)> {
        let view = t.as_array();
        let (n_rows, latent_dim) = view.dim();
        if n_rows as i64 != self.n_eff {
            return Err(PyValueError::new_err(format!(
                "IvaeRidgeMeanGauge.value_grad: t first dim {n_rows} must match n_eff={}",
                self.n_eff
            )));
        }
        if latent_dim == 0 {
            return Err(PyValueError::new_err(
                "IvaeRidgeMeanGauge.value_grad: t must have latent_dim > 0",
            ));
        }
        let aux_bound = aux_conditional_prior_float_array(py, self.aux.bind(py))?;
        let aux_array = aux_bound.extract::<PyReadonlyArray2<'_, f64>>()?;
        let aux_owned: Array2<f64> = aux_array.as_array().to_owned();
        let slice = PsiSlice::full(n_rows * latent_dim, Some(latent_dim));
        let penalty = IvaeRidgeMeanGaugePenalty::new(
            slice,
            aux_owned,
            self.ridge_eps,
            self.weight,
            n_rows,
            false,
        )
        .map_err(py_value_error)?;
        let flat: Array1<f64> = view.iter().copied().collect();
        let rho = Array1::<f64>::zeros(0);
        let value = penalty.value(flat.view(), rho.view());
        let grad_flat = penalty.grad_target(flat.view(), rho.view());
        let grad = grad_flat
            .into_shape_with_order((n_rows, latent_dim))
            .map_err(|err| {
                py_value_error(format!(
                    "IvaeRidgeMeanGauge.value_grad: gradient reshape failed: {err}"
                ))
            })?;
        Ok((value, grad.into_pyarray(py).unbind()))
    }
}

fn validate_ivae_ridge_mean_gauge_aux(
    aux: &Bound<'_, PyAny>,
    weight: f64,
    n_eff: i64,
    ridge_eps: f64,
) -> PyResult<()> {
    if !(weight > 0.0) {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.weight must be > 0, got {weight}"
        )));
    }
    if n_eff <= 0 {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.n_eff must be > 0, got {n_eff}"
        )));
    }
    if !ridge_eps.is_finite() || ridge_eps <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.ridge_eps must be finite and > 0, got {ridge_eps}"
        )));
    }

    let array = aux.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
    let view = array.as_array();
    let shape = view.shape();
    if view.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.aux must have shape (N, q), got ndim={}",
            view.ndim()
        )));
    }

    let n_obs = shape[0];
    let aux_dim = shape[1];
    if n_obs as i64 != n_eff {
        return Err(PyValueError::new_err(format!(
            "IvaeRidgeMeanGauge.aux first dimension must equal n_eff={n_eff}, got {n_obs}"
        )));
    }
    if aux_dim == 0 {
        return Err(PyValueError::new_err(
            "IvaeRidgeMeanGauge.aux must have q > 0",
        ));
    }
    if !view.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "IvaeRidgeMeanGauge.aux must be finite",
        ));
    }
    Ok(())
}

#[pyclass(module = "gam_pyffi._rust", name = "MechanismSparsityPenalty")]
struct MechanismSparsityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    feature_groups: Vec<Vec<usize>>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    n_eff: f64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl MechanismSparsityPenalty {
    #[new]
    #[pyo3(
        signature = (
            feature_groups,
            weight,
            n_eff,
            smoothing_eps = 1.0e-6,
            learnable = false,
            *,
            target = None
        ),
        text_signature = "(feature_groups, weight, n_eff, smoothing_eps=1e-06, learnable=False, *, target=None)"
    )]
    fn new(
        py: Python<'_>,
        feature_groups: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: f64,
        smoothing_eps: f64,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "MechanismSparsityPenalty.weight must be > 0, got {weight}"
            )));
        }
        if smoothing_eps <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "MechanismSparsityPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }
        if !n_eff.is_finite() || n_eff <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "MechanismSparsityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            feature_groups: coerce_mechanism_feature_groups(feature_groups)?,
            weight,
            smoothing_eps,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "mechanism_sparsity";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        set_mechanism_target_descriptor(py, &payload, &self.target)?;
        payload.set_item("feature_groups", self.feature_groups.clone())?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = &self.weight_schedule {
            let schedule = mechanism_weight_schedule_descriptor(py, schedule)?;
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.unbind().into_any())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let target_repr = self.target.bind(py).repr()?.extract::<String>()?;
        let schedule_repr = match &self.weight_schedule {
            Some(schedule) => schedule.bind(py).repr()?.extract::<String>()?,
            None => "None".to_string(),
        };
        let learnable = if self.learnable { "True" } else { "False" };
        Ok(format!(
            "MechanismSparsityPenalty(target={target_repr}, feature_groups={:?}, weight={}, smoothing_eps={}, n_eff={}, learnable={learnable}, weight_schedule={schedule_repr})",
            self.feature_groups, self.weight, self.smoothing_eps, self.n_eff
        ))
    }

    /// Evaluate the group-lasso mechanism-sparsity penalty value and gradient
    /// at decoder weight matrix `t` of shape `(d_latent, p_features)`.
    /// Returns `(value, grad)` with `grad` having the same shape as `t`.
    #[pyo3(signature = (t))]
    fn value_grad<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>)> {
        let view = t.as_array();
        let (d_latent, p_features) = view.dim();
        if d_latent == 0 || p_features == 0 {
            return Err(PyValueError::new_err(
                "MechanismSparsityPenalty.value_grad: t must have non-zero dimensions",
            ));
        }
        let total = d_latent * p_features;
        let slice = PsiSlice::full(total, Some(d_latent));
        let penalty = CoreMechanismSparsityPenalty::new(
            slice,
            self.feature_groups.clone(),
            self.weight,
            self.smoothing_eps,
            self.n_eff,
            false,
        )
        .map_err(py_value_error)?;
        let flat: Array1<f64> = view.iter().copied().collect();
        let rho = Array1::<f64>::zeros(0);
        let value = penalty.value(flat.view(), rho.view());
        let grad_flat = penalty.grad_target(flat.view(), rho.view());
        let grad = grad_flat
            .into_shape_with_order((d_latent, p_features))
            .map_err(|err| {
                py_value_error(format!(
                    "MechanismSparsityPenalty.value_grad: gradient reshape failed: {err}"
                ))
            })?;
        Ok((value, grad.into_pyarray(py).unbind()))
    }
}

fn coerce_mechanism_feature_groups(feature_groups: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
    let groups_iter = feature_groups.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences",
        )
    })?;
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut max_feature = None::<usize>;
    for (group_idx, raw_group) in groups_iter.enumerate() {
        let raw_group = raw_group?;
        let features_iter = raw_group.try_iter().map_err(|_| {
            PyTypeError::new_err(
                "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences",
            )
        })?;
        let mut group = Vec::new();
        for raw_feature in features_iter {
            let raw_feature = raw_feature?;
            let indexed = raw_feature.call_method0("__index__").map_err(|_| {
                PyTypeError::new_err(
                    "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences",
                )
            })?;
            let feature = indexed.extract::<isize>().map_err(|_| {
                PyTypeError::new_err(
                    "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences",
                )
            })?;
            if feature < 0 {
                return Err(PyValueError::new_err(format!(
                    "MechanismSparsityPenalty.feature_groups entries must be non-negative; got {feature}"
                )));
            }
            let feature = usize::try_from(feature).map_err(|_| {
                PyValueError::new_err(format!(
                    "MechanismSparsityPenalty.feature_groups entries must be non-negative; got {feature}"
                ))
            })?;
            if !seen.insert(feature) {
                return Err(PyValueError::new_err(format!(
                    "MechanismSparsityPenalty.feature_groups feature {feature} appears more than once"
                )));
            }
            max_feature = Some(max_feature.map_or(feature, |current| current.max(feature)));
            group.push(feature);
        }
        if group.is_empty() {
            return Err(PyValueError::new_err(format!(
                "MechanismSparsityPenalty.feature_groups[{group_idx}] must not be empty"
            )));
        }
        out.push(group);
    }
    if out.is_empty() {
        return Err(PyValueError::new_err(
            "MechanismSparsityPenalty.feature_groups must not be empty",
        ));
    }
    if let Some(max_feature) = max_feature {
        for feature in 0..=max_feature {
            if !seen.contains(&feature) {
                return Err(PyValueError::new_err(format!(
                    "MechanismSparsityPenalty.feature_groups must partition contiguous features from 0; missing feature {feature}"
                )));
            }
        }
    }
    Ok(out)
}

fn set_mechanism_target_descriptor(
    py: Python<'_>,
    payload: &Bound<'_, PyDict>,
    target: &PyObject,
) -> PyResult<()> {
    let target = target.bind(py);
    if let Ok(target_name) = target.extract::<String>() {
        payload.set_item("target", target_name)?;
        return Ok(());
    }
    if let Ok(target_index) = target.extract::<isize>() {
        payload.set_item("target", target_index)?;
        return Ok(());
    }
    if let Ok(name) = target.getattr("name") {
        payload.set_item("target", name.str()?.to_str()?)?;
        return Ok(());
    }
    Err(PyTypeError::new_err(
        "analytic penalty target must be a latent block name, latent block index, or object exposing a 'name' attribute",
    ))
}

fn mechanism_weight_schedule_descriptor(py: Python<'_>, schedule: &PyObject) -> PyResult<PyObject> {
    let schedule = schedule.bind(py);
    if schedule.hasattr("to_rust_descriptor")? {
        return Ok(schedule.call_method0("to_rust_descriptor")?.unbind());
    }
    if schedule.cast::<PyDict>().is_ok() {
        return Ok(schedule.clone().unbind());
    }
    Err(PyTypeError::new_err(
        "weight_schedule must be ScalarWeightSchedule, a mapping, or None",
    ))
}

#[pyclass(module = "gam_pyffi._rust", name = "BlockOrthogonalityPenalty")]
struct BlockOrthogonalityPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    groups: Vec<Vec<i64>>,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

fn block_orthogonality_groups(groups: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<i64>>> {
    let group_iter = groups.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "BlockOrthogonalityPenalty.groups must be a sequence of integer sequences",
        )
    })?;
    let mut coerced = Vec::new();
    for group in group_iter {
        let group = group.map_err(|_| {
            PyTypeError::new_err(
                "BlockOrthogonalityPenalty.groups must be a sequence of integer sequences",
            )
        })?;
        let axis_iter = group.try_iter().map_err(|_| {
            PyTypeError::new_err(
                "BlockOrthogonalityPenalty.groups must be a sequence of integer sequences",
            )
        })?;
        let mut axes = Vec::new();
        for axis in axis_iter {
            let axis = axis.map_err(|_| {
                PyTypeError::new_err(
                    "BlockOrthogonalityPenalty.groups must be a sequence of integer sequences",
                )
            })?;
            axes.push(axis.call_method0("__index__")?.extract::<i64>()?);
        }
        coerced.push(axes);
    }
    if coerced.len() < 2 {
        return Err(PyValueError::new_err(
            "BlockOrthogonalityPenalty.groups must contain at least two groups",
        ));
    }
    let mut seen = BTreeSet::new();
    for (group_idx, group) in coerced.iter().enumerate() {
        if group.is_empty() {
            return Err(PyValueError::new_err(format!(
                "BlockOrthogonalityPenalty.groups[{group_idx}] must not be empty"
            )));
        }
        for &axis in group {
            if axis < 0 {
                return Err(PyValueError::new_err(format!(
                    "BlockOrthogonalityPenalty.groups entries must be non-negative; got {axis}"
                )));
            }
            if !seen.insert(axis) {
                return Err(PyValueError::new_err(format!(
                    "BlockOrthogonalityPenalty.groups axis {axis} appears more than once"
                )));
            }
        }
    }
    let max_axis = seen.iter().next_back().copied().unwrap_or(0);
    for axis in 0..=max_axis {
        if !seen.contains(&axis) {
            return Err(PyValueError::new_err(format!(
                "BlockOrthogonalityPenalty.groups must partition contiguous axes from 0; missing axis {axis}"
            )));
        }
    }
    Ok(coerced)
}

fn block_orthogonality_target_descriptor(target: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    if target.is_instance_of::<PyString>() || target.extract::<isize>().is_ok() {
        return Ok(target.clone().unbind());
    }
    if target.hasattr("to_rust_descriptor")? {
        return Ok(target.call_method0("to_rust_descriptor")?.unbind().into());
    }
    Err(PyTypeError::new_err(
        "target must be a string, integer, or object with to_rust_descriptor()",
    ))
}

#[pymethods]
impl BlockOrthogonalityPenalty {
    #[new]
    #[pyo3(signature = (groups, weight, n_eff, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        groups: &Bound<'_, PyAny>,
        weight: f64,
        n_eff: &Bound<'_, PyAny>,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let groups = block_orthogonality_groups(groups)?;
        let n_eff = n_eff.call_method0("__index__")?.extract::<i64>()?;
        if weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "BlockOrthogonalityPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "BlockOrthogonalityPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            groups,
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "block_orthogonality";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item(
            "target",
            block_orthogonality_target_descriptor(self.target.bind(py))?,
        )?;
        payload.set_item("groups", &self.groups)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "BlockOrthogonalityPenalty(target={}, groups={:?}, weight={}, n_eff={}, learnable={}, weight_schedule={})",
            py_repr(py, self.target.bind(py))?,
            self.groups,
            self.weight,
            self.n_eff,
            self.learnable,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

// ---------------------------------------------------------------------------
// SheafConsistencyPenalty — cellular-sheaf consistency loss.
// ---------------------------------------------------------------------------

#[pyclass(module = "gam_pyffi._rust", name = "SheafConsistencyPenalty")]
struct SheafConsistencyPenalty {
    inner: CoreSheafConsistencyPenalty,
    #[pyo3(get)]
    weight: f64,
    #[pyo3(get)]
    target: PyObject,
}

fn sheaf_extract_edges(edges: &Bound<'_, PyAny>) -> PyResult<Vec<(usize, usize)>> {
    let iter = edges.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "SheafConsistencyPenalty.edges must be a sequence of (u, v) integer pairs",
        )
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item.map_err(|_| {
            PyTypeError::new_err(
                "SheafConsistencyPenalty.edges must be a sequence of (u, v) integer pairs",
            )
        })?;
        let pair = item.try_iter().map_err(|_| {
            PyTypeError::new_err(
                "SheafConsistencyPenalty.edges entries must be (u, v) sequences of length 2",
            )
        })?;
        let pair_vec: Vec<i64> = pair
            .map(|x| {
                x.and_then(|v| v.call_method0("__index__")?.extract::<i64>())
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "SheafConsistencyPenalty.edges entries must contain integers",
                        )
                    })
            })
            .collect::<PyResult<_>>()?;
        if pair_vec.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.edges entries must have length 2, got {}",
                pair_vec.len()
            )));
        }
        if pair_vec[0] < 0 || pair_vec[1] < 0 {
            return Err(PyValueError::new_err(
                "SheafConsistencyPenalty.edges entries must be non-negative",
            ));
        }
        out.push((pair_vec[0] as usize, pair_vec[1] as usize));
    }
    Ok(out)
}

fn sheaf_extract_array2(obj: &Bound<'_, PyAny>, label: &str) -> PyResult<Array2<f64>> {
    let arr = obj.cast::<PyArray2<f64>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "SheafConsistencyPenalty: {label} must be a 2-D numpy.ndarray of float64"
        ))
    })?;
    let readonly = arr.readonly();
    Ok(readonly.as_array().to_owned())
}

fn sheaf_extract_restrictions(
    restrictions: &Bound<'_, PyAny>,
) -> PyResult<Vec<CoreEdgeRestriction>> {
    let iter = restrictions.try_iter().map_err(|_| {
        PyTypeError::new_err("SheafConsistencyPenalty.restriction_ops must be a sequence")
    })?;
    let mut out = Vec::new();
    for (e, item) in iter.enumerate() {
        let item = item.map_err(|_| {
            PyTypeError::new_err(
                "SheafConsistencyPenalty.restriction_ops entries must be numpy arrays or (W_uv, W_vu) tuples",
            )
        })?;
        // Try direct 2-D array first → single-restriction edge.
        if item.cast::<PyArray2<f64>>().is_ok() {
            let r = sheaf_extract_array2(&item, &format!("restriction_ops[{e}]"))?;
            out.push(CoreEdgeRestriction::single(r));
            continue;
        }
        // Otherwise expect a (W_uv, W_vu) pair.
        let pair = item.try_iter().map_err(|_| {
            PyTypeError::new_err(format!(
                "SheafConsistencyPenalty.restriction_ops[{e}] must be a 2-D ndarray or a (W_uv, W_vu) tuple"
            ))
        })?;
        let parts: Vec<Bound<'_, PyAny>> = pair.collect::<PyResult<_>>()?;
        if parts.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.restriction_ops[{e}] tuple must have length 2, got {}",
                parts.len()
            )));
        }
        let r_uv = sheaf_extract_array2(&parts[0], &format!("restriction_ops[{e}].W_uv"))?;
        // Allow None for the second slot via Python's None object.
        if parts[1].is_none() {
            out.push(CoreEdgeRestriction::single(r_uv));
        } else {
            let r_vu = sheaf_extract_array2(&parts[1], &format!("restriction_ops[{e}].W_vu"))?;
            out.push(CoreEdgeRestriction::paired(r_uv, r_vu));
        }
    }
    Ok(out)
}

fn sheaf_extract_stalk_dims(stalk_dims: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let iter = stalk_dims.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "SheafConsistencyPenalty.stalk_dims must be a sequence of positive integers",
        )
    })?;
    let mut out = Vec::new();
    for item in iter {
        let item = item.map_err(|_| {
            PyTypeError::new_err("SheafConsistencyPenalty.stalk_dims entries must be integers")
        })?;
        let d = item.call_method0("__index__")?.extract::<i64>()?;
        if d <= 0 {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.stalk_dims entries must be > 0, got {d}"
            )));
        }
        out.push(d as usize);
    }
    Ok(out)
}

#[pymethods]
impl SheafConsistencyPenalty {
    #[new]
    #[pyo3(signature = (edges, restriction_ops, stalk_dims, weight = 1.0, *, target = None))]
    fn new(
        py: Python<'_>,
        edges: &Bound<'_, PyAny>,
        restriction_ops: &Bound<'_, PyAny>,
        stalk_dims: &Bound<'_, PyAny>,
        weight: f64,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if !(weight.is_finite() && weight > 0.0) {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.weight must be > 0, got {weight}"
            )));
        }
        let edges = sheaf_extract_edges(edges)?;
        let restrictions = sheaf_extract_restrictions(restriction_ops)?;
        let stalk_dims = sheaf_extract_stalk_dims(stalk_dims)?;
        let inner = CoreSheafConsistencyPenalty::new(edges, restrictions, weight, stalk_dims)
            .map_err(PyValueError::new_err)?;
        let target_obj = py_object_or_string_default(py, target, "z");
        Ok(Self {
            inner,
            weight,
            target: target_obj,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "sheaf_consistency";

    #[getter]
    fn total_dim(&self) -> usize {
        self.inner.total_dim()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    #[getter]
    fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    #[getter]
    fn stalk_dims(&self) -> Vec<usize> {
        self.inner.stalk_dims().to_vec()
    }

    fn __call__<'py>(&self, s: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        let view = s.as_array();
        if view.len() != self.inner.total_dim() {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty: input length {} != total stalk dim {}",
                view.len(),
                self.inner.total_dim()
            )));
        }
        Ok(self.inner.value(view))
    }

    fn value<'py>(&self, s: PyReadonlyArray1<'py, f64>) -> PyResult<f64> {
        self.__call__(s)
    }

    fn gradient<'py>(
        &self,
        py: Python<'py>,
        s: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let view = s.as_array();
        if view.len() != self.inner.total_dim() {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.gradient: input length {} != total stalk dim {}",
                view.len(),
                self.inner.total_dim()
            )));
        }
        let g = self.inner.gradient(view);
        Ok(g.into_pyarray(py))
    }

    fn hessian_diag<'py>(
        &self,
        py: Python<'py>,
        s: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let view = s.as_array();
        if view.len() != self.inner.total_dim() {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.hessian_diag: input length {} != total stalk dim {}",
                view.len(),
                self.inner.total_dim()
            )));
        }
        let h = self.inner.hessian_diag(view);
        Ok(h.into_pyarray(py))
    }

    fn hvp<'py>(
        &self,
        py: Python<'py>,
        s: PyReadonlyArray1<'py, f64>,
        v: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let sv = s.as_array();
        let vv = v.as_array();
        if sv.len() != self.inner.total_dim() || vv.len() != self.inner.total_dim() {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.hvp: input lengths ({}, {}) != total stalk dim {}",
                sv.len(),
                vv.len(),
                self.inner.total_dim()
            )));
        }
        let hv = self.inner.hvp(sv, vv);
        Ok(hv.into_pyarray(py))
    }

    #[pyo3(signature = (tol = 1e-8))]
    fn harmonic_modes(&self, tol: f64) -> PyResult<usize> {
        if !(tol.is_finite() && tol >= 0.0) {
            return Err(PyValueError::new_err(format!(
                "SheafConsistencyPenalty.harmonic_modes: tol must be finite and >= 0, got {tol}"
            )));
        }
        Ok(self.inner.harmonic_modes(tol))
    }

    fn __repr__(&self) -> String {
        format!(
            "SheafConsistencyPenalty(K={}, edges={}, total_dim={}, weight={})",
            self.inner.num_vertices(),
            self.inner.num_edges(),
            self.inner.total_dim(),
            self.weight,
        )
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "AuxConditionalPriorPenalty")]
struct AuxConditionalPriorPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    lambda_per_row: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl AuxConditionalPriorPenalty {
    #[new]
    #[pyo3(signature = (lambda_per_row, weight, n_eff, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        lambda_per_row: &Bound<'_, PyAny>,
        weight: &Bound<'_, PyAny>,
        n_eff: &Bound<'_, PyAny>,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let builtins = py.import("builtins")?;
        let weight = builtins
            .getattr("float")?
            .call1((weight,))?
            .extract::<f64>()?;
        let n_eff = builtins.getattr("int")?.call1((n_eff,))?.extract::<i64>()?;
        let lambda_per_row = aux_conditional_prior_float_array(py, lambda_per_row)?;
        validate_aux_conditional_prior_lambda(&lambda_per_row, weight, n_eff)?;
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            lambda_per_row: lambda_per_row.unbind(),
            weight,
            n_eff,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "aux_conditional_prior";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;

        let array = self
            .lambda_per_row
            .bind(py)
            .extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let view = array.as_array();
        let flattened = view.iter().copied().collect::<Vec<_>>();
        payload.set_item("lambda_per_row", PyList::new(py, flattened)?)?;
        payload.set_item("lambda_per_row_shape", PyList::new(py, view.shape())?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "AuxConditionalPriorPenalty(lambda_per_row={}, weight={}, n_eff={}, learnable={}, target={}, weight_schedule={})",
            py_repr(py, self.lambda_per_row.bind(py))?,
            self.weight,
            self.n_eff,
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }

    /// Evaluate the fixed-precomputed row-precision prior value and gradient
    /// at latent block `t` of shape `(n_eff, latent_dim)`. Returns
    /// `(value, grad)` with `grad` having the same shape as `t`.
    #[pyo3(signature = (t))]
    fn value_grad<'py>(
        &self,
        py: Python<'py>,
        t: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(f64, Py<PyArray2<f64>>)> {
        let view = t.as_array();
        let (n_rows, latent_dim) = view.dim();
        if n_rows as i64 != self.n_eff {
            return Err(PyValueError::new_err(format!(
                "AuxConditionalPriorPenalty.value_grad: t first dim {n_rows} must match n_eff={}",
                self.n_eff
            )));
        }
        if latent_dim == 0 {
            return Err(PyValueError::new_err(
                "AuxConditionalPriorPenalty.value_grad: t must have latent_dim > 0",
            ));
        }
        let lambda_array = self
            .lambda_per_row
            .bind(py)
            .extract::<PyReadonlyArray3<'_, f64>>()?;
        let lambda_owned: Array3<f64> = lambda_array.as_array().to_owned();
        let slice = PsiSlice::full(n_rows * latent_dim, Some(latent_dim));
        let n_eff_usize = usize::try_from(self.n_eff).map_err(|_| {
            py_value_error(format!(
                "AuxConditionalPriorPenalty.value_grad: n_eff={} does not fit in usize",
                self.n_eff
            ))
        })?;
        let penalty =
            RowPrecisionPriorPenalty::new(slice, lambda_owned, self.weight, n_eff_usize, false)
                .map_err(py_value_error)?;
        let flat: Array1<f64> = view.iter().copied().collect();
        let rho = Array1::<f64>::zeros(0);
        let value = penalty.value(flat.view(), rho.view());
        let grad_flat = penalty.grad_target(flat.view(), rho.view());
        let grad = grad_flat
            .into_shape_with_order((n_rows, latent_dim))
            .map_err(|err| {
                py_value_error(format!(
                    "AuxConditionalPriorPenalty.value_grad: gradient reshape failed: {err}"
                ))
            })?;
        Ok((value, grad.into_pyarray(py).unbind()))
    }
}

#[pyclass(module = "gam_pyffi._rust", name = "NuclearNormPenalty")]
struct NuclearNormPenalty {
    #[pyo3(get, set)]
    target: PyObject,
    #[pyo3(get, set)]
    weight: f64,
    #[pyo3(get, set)]
    n_eff: i64,
    #[pyo3(get, set)]
    smoothing_eps: f64,
    #[pyo3(get, set)]
    max_rank: Option<i64>,
    #[pyo3(get, set)]
    learnable: bool,
    #[pyo3(get)]
    weight_schedule: Option<PyObject>,
}

#[pymethods]
impl NuclearNormPenalty {
    #[new]
    #[pyo3(signature = (weight, n_eff, smoothing_eps = 1.0e-6, max_rank = None, learnable = false, *, target = None))]
    fn new(
        py: Python<'_>,
        weight: f64,
        n_eff: &Bound<'_, PyAny>,
        smoothing_eps: f64,
        max_rank: Option<&Bound<'_, PyAny>>,
        learnable: bool,
        target: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let n_eff = n_eff.call_method0("__index__")?.extract::<i64>()?;
        let max_rank = match max_rank {
            Some(rank) => Some(rank.call_method0("__index__")?.extract::<i64>()?),
            None => None,
        };
        if !weight.is_finite() || weight <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "NuclearNormPenalty.weight must be > 0, got {weight}"
            )));
        }
        if n_eff <= 0 {
            return Err(PyValueError::new_err(format!(
                "NuclearNormPenalty.n_eff must be > 0, got {n_eff}"
            )));
        }
        if !(smoothing_eps > 0.0) {
            return Err(PyValueError::new_err(format!(
                "NuclearNormPenalty.smoothing_eps must be > 0, got {smoothing_eps}"
            )));
        }
        if let Some(rank) = max_rank {
            if rank <= 0 {
                return Err(PyValueError::new_err(format!(
                    "NuclearNormPenalty.max_rank must be > 0, got {rank}"
                )));
            }
        }
        Ok(Self {
            target: py_object_or_string_default(py, target, "t"),
            weight,
            n_eff,
            smoothing_eps,
            max_rank,
            learnable,
            weight_schedule: None,
        })
    }

    #[classattr]
    const KIND_TAG: &'static str = "nuclear_norm";

    fn to_rust_descriptor(&self, py: Python<'_>) -> PyResult<PyObject> {
        let payload = PyDict::new(py);
        payload.set_item("kind", Self::KIND_TAG)?;
        payload.set_item("target", target_descriptor(py, self.target.bind(py))?)?;
        payload.set_item("weight", self.weight)?;
        payload.set_item("n_eff", self.n_eff)?;
        payload.set_item("smoothing_eps", self.smoothing_eps)?;
        payload.set_item("max_rank", self.max_rank)?;
        payload.set_item("learnable", self.learnable)?;
        if let Some(schedule) = topk_weight_schedule_descriptor(py, &self.weight_schedule)? {
            payload.set_item("weight_schedule", schedule)?;
        }
        Ok(payload.into())
    }

    fn set_weight_schedule<'py>(
        mut slf: PyRefMut<'py, Self>,
        schedule: PyObject,
    ) -> PyRefMut<'py, Self> {
        slf.weight_schedule = Some(schedule);
        slf
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "NuclearNormPenalty(weight={}, n_eff={}, smoothing_eps={}, max_rank={}, learnable={}, target={}, weight_schedule={})",
            self.weight,
            self.n_eff,
            self.smoothing_eps,
            match self.max_rank {
                Some(rank) => rank.to_string(),
                None => "None".to_string(),
            },
            self.learnable,
            py_repr(py, self.target.bind(py))?,
            match &self.weight_schedule {
                Some(schedule) => py_repr(py, schedule.bind(py))?,
                None => "None".to_string(),
            }
        ))
    }
}

/// Identifiability theorem precondition checks (Principle (f)).
///
/// Caller serialises a `gam::inference::identifiability_precondition_checks::FitSummary` as
/// JSON; Rust does all numerical work (min std, faer-SVD rank, decoder
/// zero-fraction, activation variance bounds) and returns a JSON array of
/// per-theorem results. This is the *single source of truth* — Python's
/// `gamfit.identifiability.check`, the CLI, and any future R/Julia binding
/// all consume this same FFI.
#[pyfunction]
fn identifiability_check_json(input: &str) -> PyResult<String> {
    gam::inference::identifiability_precondition_checks::identifiability_check_json(input)
        .map_err(py_value_error)
}

#[pymodule(name = "_rust", gil_used = false)]
fn rust_extension(module: &Bound<'_, PyModule>) -> PyResult<()> {
    gam::init_parallelism();
    // Install the same stderr logger used by the CLI so long-running Rust
    // solver phases (including survival marginal-slope joint-Newton cycles)
    // are visible from Python without requiring a separate shell.
    gam::visualizer::init_logging();
    // Background process monitor: emits a `[process-monitor] elapsed=… rss=…`
    // line every 60s for the life of the process, so silent stretches
    // inside long PIRLS line-searches still surface a process-alive
    // signal with current memory footprint. Unconditional — does not
    // depend on any family pushing a tracked scope.
    gam::process_monitor::start();
    module.add("__doc__", "PyO3 boundary for the gam Rust engine.")?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // gamfit exception hierarchy (see the comment block at the
    // `create_exception!` definitions above and issue #343). Registering
    // the classes here makes them addressable as `gam._rust.GamError`,
    // `gam._rust.RemlConvergenceError`, etc.; `gamfit/_exceptions.py`
    // re-exports each one under its public `gamfit.*` name so the class
    // identity caught by `pytest.raises(gamfit.RemlConvergenceError)`
    // and constructed by `RemlConvergenceError::new_err(...)` on the
    // Rust side is exactly the same object.
    module.add("GamError", module.py().get_type::<GamError>())?;
    module.add("FormulaError", module.py().get_type::<FormulaError>())?;
    module.add(
        "ColumnNotFoundError",
        module.py().get_type::<ColumnNotFoundError>(),
    )?;
    module.add(
        "SchemaMismatchError",
        module.py().get_type::<SchemaMismatchError>(),
    )?;
    module.add("PredictionError", module.py().get_type::<PredictionError>())?;
    module.add("BasisError", module.py().get_type::<BasisError>())?;
    module.add(
        "LinearSystemSolveError",
        module.py().get_type::<LinearSystemSolveError>(),
    )?;
    module.add(
        "EigendecompositionError",
        module.py().get_type::<EigendecompositionError>(),
    )?;
    module.add(
        "PenaltySpectrumError",
        module.py().get_type::<PenaltySpectrumError>(),
    )?;
    module.add(
        "ParameterConstraintError",
        module.py().get_type::<ParameterConstraintError>(),
    )?;
    module.add(
        "PirlsConvergenceError",
        module.py().get_type::<PirlsConvergenceError>(),
    )?;
    module.add(
        "PerfectSeparationError",
        module.py().get_type::<PerfectSeparationError>(),
    )?;
    module.add(
        "HessianNotPositiveDefiniteError",
        module.py().get_type::<HessianNotPositiveDefiniteError>(),
    )?;
    module.add(
        "RemlConvergenceError",
        module.py().get_type::<RemlConvergenceError>(),
    )?;
    module.add(
        "GradientUnavailableError",
        module.py().get_type::<GradientUnavailableError>(),
    )?;
    module.add("LayoutError", module.py().get_type::<LayoutError>())?;
    module.add(
        "ModelOverparameterizedError",
        module.py().get_type::<ModelOverparameterizedError>(),
    )?;
    module.add(
        "IllConditionedError",
        module.py().get_type::<IllConditionedError>(),
    )?;
    module.add(
        "InvalidInputError",
        module.py().get_type::<InvalidInputError>(),
    )?;
    module.add(
        "MonotoneRootError",
        module.py().get_type::<MonotoneRootError>(),
    )?;
    module.add("CalibratorError", module.py().get_type::<CalibratorError>())?;
    module.add(
        "InvalidSpecificationError",
        module.py().get_type::<InvalidSpecificationError>(),
    )?;
    // Remaining engine error enum subclasses (issue #343 follow-up).
    module.add("GeometryError", module.py().get_type::<GeometryError>())?;
    module.add(
        "MatrixMaterializationError",
        module.py().get_type::<MatrixMaterializationError>(),
    )?;
    module.add("GpuError", module.py().get_type::<GpuError>())?;
    module.add(
        "LinearAlgebraError",
        module.py().get_type::<LinearAlgebraError>(),
    )?;
    module.add("MatrixError", module.py().get_type::<MatrixError>())?;
    module.add("CacheStoreError", module.py().get_type::<CacheStoreError>())?;
    module.add("SmoothError", module.py().get_type::<SmoothError>())?;
    module.add("ArrowSchurError", module.py().get_type::<ArrowSchurError>())?;
    module.add(
        "OuterStrategyError",
        module.py().get_type::<OuterStrategyError>(),
    )?;
    module.add(
        "TermBuilderError",
        module.py().get_type::<TermBuilderError>(),
    )?;
    module.add(
        "CorrectedCovarianceError",
        module.py().get_type::<CorrectedCovarianceError>(),
    )?;
    module.add(
        "PredictInputError",
        module.py().get_type::<PredictInputError>(),
    )?;
    module.add("HmcError", module.py().get_type::<HmcError>())?;
    module.add("AloError", module.py().get_type::<AloError>())?;
    module.add("SurvivalError", module.py().get_type::<SurvivalError>())?;
    module.add(
        "CubicCellKernelError",
        module.py().get_type::<CubicCellKernelError>(),
    )?;
    module.add(
        "SurvivalConstructionError",
        module.py().get_type::<SurvivalConstructionError>(),
    )?;
    module.add(
        "TransformationNormalError",
        module.py().get_type::<TransformationNormalError>(),
    )?;
    module.add(
        "CustomFamilyError",
        module.py().get_type::<CustomFamilyError>(),
    )?;
    module.add("GamlssError", module.py().get_type::<GamlssError>())?;
    module.add(
        "SurvivalMarginalSlopeError",
        module.py().get_type::<SurvivalMarginalSlopeError>(),
    )?;
    module.add(
        "LatentSurvivalError",
        module.py().get_type::<LatentSurvivalError>(),
    )?;
    module.add(
        "SurvivalPredictError",
        module.py().get_type::<SurvivalPredictError>(),
    )?;
    module.add(
        "DeviationRuntimeError",
        module.py().get_type::<DeviationRuntimeError>(),
    )?;
    module.add("DataError", module.py().get_type::<DataError>())?;
    module.add(
        "FittedModelError",
        module.py().get_type::<FittedModelError>(),
    )?;
    module.add(
        "LognormalKernelError",
        module.py().get_type::<LognormalKernelError>(),
    )?;
    module.add(
        "ScaleDesignError",
        module.py().get_type::<ScaleDesignError>(),
    )?;
    module.add(
        "IdentifiabilityCompilerError",
        module.py().get_type::<IdentifiabilityCompilerError>(),
    )?;
    module.add(
        "JointPenaltyError",
        module.py().get_type::<JointPenaltyError>(),
    )?;
    module.add(
        "SurvivalLocationScaleError",
        module.py().get_type::<SurvivalLocationScaleError>(),
    )?;
    module.add(
        "MapUniquenessError",
        module.py().get_type::<MapUniquenessError>(),
    )?;
    module.add(
        "UnsupportedLinkError",
        module.py().get_type::<UnsupportedLinkError>(),
    )?;
    module.add(
        "InvalidConfigurationError",
        module.py().get_type::<InvalidConfigurationError>(),
    )?;
    module.add(
        "MissingDependencyError",
        module.py().get_type::<MissingDependencyError>(),
    )?;
    module.add(
        "IntegrationError",
        module.py().get_type::<IntegrationError>(),
    )?;

    // #773: `create_exception!` stamps every gamfit exception with
    // `__module__ = "_rust"`, but the compiled extension is importable only as
    // `gamfit._rust` (see `gamfit._binding.rust_module`). Pickle records a class
    // by `(__module__, __qualname__)` and reconstructs it with `import _rust;
    // getattr(...)`, which fails — so any `_rust.*` exception raised inside a
    // `ProcessPoolExecutor` worker is masked by an opaque `PicklingError` that
    // hides the real failure and takes down the whole pool. Repoint each
    // exception class at its true importable module so the type round-trips
    // through pickle. Walking the module dict for `GamError` subclasses keeps
    // this correct as exceptions are added — no parallel name list to drift.
    {
        let gam_error = module.py().get_type::<GamError>();
        for (_name, value) in module.dict().iter() {
            let Ok(ty) = value.cast::<PyType>() else {
                continue;
            };
            if ty.is_subclass(gam_error.as_any())? {
                ty.as_any().setattr("__module__", "gamfit._rust")?;
            }
        }
    }

    module.add_class::<EuclideanManifold>()?;
    module.add_class::<CircleManifold>()?;
    module.add_class::<SphereManifold>()?;
    module.add_class::<TorusManifold>()?;
    module.add_class::<GrassmannManifold>()?;
    module.add_class::<StiefelManifold>()?;
    module.add_class::<SpdManifold>()?;
    module.add_class::<ProductManifold>()?;
    module.add_function(wrap_pyfunction!(fit_penalized_multinomial_pyfunc, module)?)?;
    module.add_function(wrap_pyfunction!(fit_multinomial_formula_pyfunc, module)?)?;
    module.add_function(wrap_pyfunction!(
        predict_multinomial_formula_pyfunc,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(multinomial_model_metadata_pyfunc, module)?)?;
    module.add_function(wrap_pyfunction!(
        predict_multinomial_intervals_pyfunc,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        multinomial_smooth_significance_pyfunc,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sklearn_fit_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(identifiability_check_json, module)?)?;
    module.add_function(wrap_pyfunction!(interpolate_survival_surface, module)?)?;
    module.add_function(wrap_pyfunction!(interpolate_rows, module)?)?;
    module.add_function(wrap_pyfunction!(survival_chunk_iter_collect, module)?)?;
    module.add_function(wrap_pyfunction!(write_survival_csv, module)?)?;
    module.add_function(wrap_pyfunction!(survival_coerce_times, module)?)?;
    module.add_function(wrap_pyfunction!(survival_parameters_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(survival_collect_chunks, module)?)?;
    module.add_function(wrap_pyfunction!(hazard_from_cumulative, module)?)?;
    module.add_function(wrap_pyfunction!(survival_cumulative_from_survival, module)?)?;
    module.add_function(wrap_pyfunction!(survival_block, module)?)?;
    module.add_function(wrap_pyfunction!(survival_block_hazard, module)?)?;
    module.add_function(wrap_pyfunction!(survival_ffi_surface, module)?)?;
    module.add_function(wrap_pyfunction!(numeric_matrix_validate, module)?)?;
    module.add_function(wrap_pyfunction!(numeric_matrix_f64, module)?)?;
    module.add_function(wrap_pyfunction!(marginal_slope_clip_probabilities, module)?)?;
    module.add_function(wrap_pyfunction!(
        transformation_normal_z_from_columns,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(column_stack_f64, module)?)?;
    module.add_function(wrap_pyfunction!(
        survival_prediction_payload_from_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(flat_to_matrix_f64, module)?)?;
    module.add_function(wrap_pyfunction!(vec_to_array1_f64, module)?)?;
    module.add_function(wrap_pyfunction!(extract_row_ids, module)?)?;
    module.add_function(wrap_pyfunction!(default_survival_time_grid, module)?)?;
    module.add_function(wrap_pyfunction!(torch_from_fitted, module)?)?;
    module.add_function(wrap_pyfunction!(fit_table, module)?)?;
    module.add_function(wrap_pyfunction!(fit_array, module)?)?;
    module.add_function(wrap_pyfunction!(load_model, module)?)?;
    module.add_function(wrap_pyfunction!(bayes_factor_log_diff, module)?)?;
    module.add_function(wrap_pyfunction!(saved_model_payload_string, module)?)?;
    module.add_function(wrap_pyfunction!(
        required_saved_model_payload_string,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_extend_group_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(extend_model_with_group, module)?)?;
    module.add_function(wrap_pyfunction!(validate_formula_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        apply_shape_constraints_to_formula,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        formula_validation_supported_by_python_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(formula_validation_repr_json, module)?)?;
    module.add_function(wrap_pyfunction!(formula_validation_html_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_predict_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(build_model_predict_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(predict_table_conformal, module)?)?;
    // #1054: exact Gaussian jackknife+ conformal intervals (no calibration fold).
    module.add_function(wrap_pyfunction!(predict_table_jackknife_plus, module)?)?;
    // #1098: exact Gaussian full-conformal prediction set (no calibration fold).
    module.add_function(wrap_pyfunction!(predict_table_full_conformal, module)?)?;
    // #1057: posterior-predictive replicate sampling from the fitted model.
    module.add_function(wrap_pyfunction!(generative_replicates, module)?)?;
    module.add_function(wrap_pyfunction!(predict_array, module)?)?;
    module.add_function(wrap_pyfunction!(competing_risks_cif, module)?)?;
    module.add_function(wrap_pyfunction!(
        competing_risks_prediction_payload_from_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        competing_risks_cif_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(build_sample_payload_json, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_analytic_penalty_registry_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sample_table, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_table, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_table_dense, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_array, module)?)?;
    module.add_function(wrap_pyfunction!(
        build_difference_smooth_request_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(bspline_basis, module)?)?;
    module.add_function(wrap_pyfunction!(bspline_basis_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(periodic_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(periodic_spline_curve_basis, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis_with_jets, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis, module)?)?;
    module.add_function(wrap_pyfunction!(matern_basis, module)?)?;
    module.add_function(wrap_pyfunction!(smoothness_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_function_norm_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_operator_penalties, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis_with_centers, module)?)?;
    module.add_function(wrap_pyfunction!(
        sphere_select_farthest_point_centers,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sphere_chart_basis_with_jet, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis_jet, module)?)?;
    module.add_function(wrap_pyfunction!(sphere_basis_jet_with_centers, module)?)?;
    module.add_function(wrap_pyfunction!(thin_plate_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(auto_knots_1d, module)?)?;
    module.add_function(wrap_pyfunction!(auto_centers_1d, module)?)?;
    module.add_function(wrap_pyfunction!(_block_diag, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_weighted_ridge_array, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_weighted_ridge_batch, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_score, module)?)?;
    module.add_function(wrap_pyfunction!(skip_transcoder_reml_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(skip_transcoder_select_reml, module)?)?;
    module.add_function(wrap_pyfunction!(tierney_kadane_normalized_score, module)?)?;
    module.add_function(wrap_pyfunction!(torch_smooth_dispatch_key, module)?)?;
    module.add_function(wrap_pyfunction!(assemble_candidate_formula, module)?)?;
    module.add_function(wrap_pyfunction!(ordered_prediction_columns, module)?)?;
    module.add_function(wrap_pyfunction!(rank_topology_candidates, module)?)?;
    module.add_function(wrap_pyfunction!(stacking_weights_from_log_density, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_score, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_score_raw, module)?)?;
    module.add_function(wrap_pyfunction!(extract_reml_edf, module)?)?;
    module.add_function(wrap_pyfunction!(compare_reml_fits, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_backward, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_formula_table, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_blocks_forward, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_blocks_orthogonal_forward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_blocks_backward, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_with_constraints_forward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_with_constraints_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_batched, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_batched_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_positions, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_batched,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_batched_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_latent, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_optimize_latent, module)?)?;
    module.add_function(wrap_pyfunction!(register_analytic_penalties, module)?)?;
    module.add_function(wrap_pyfunction!(analytic_penalty_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(analytic_penalty_hvp, module)?)?;
    module.add_function(wrap_pyfunction!(gumbel_schedule_tau, module)?)?;
    module.add_function(wrap_pyfunction!(sae_ibp_map_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(jumprelu_gate_value_grad, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_penalty_value, module)?)?;
    module.add_function(wrap_pyfunction!(riemannian_retract, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_exp_map_vjp, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_metric_tensor, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_dimension, module)?)?;
    module.add_function(wrap_pyfunction!(manifold_ambient_dimension, module)?)?;
    module.add_function(wrap_pyfunction!(periodic_harmonic_basis, module)?)?;
    module.add_function(wrap_pyfunction!(
        periodic_harmonic_basis_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sphere_frechet_mean, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_closure, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_clr, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_alr, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_inverse_alr, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_simplex_frechet_mean,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(poincare_mobius_add, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_distance, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_project_into_ball, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_log_origin, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_exp_origin, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_conformal_factor, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_to_lorentz, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_from_lorentz, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_lorentz_log_origin, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_lorentz_exp_origin, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_tangent_decode_forward, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_tangent_decode_backward, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_lorentz_decode_forward, module)?)?;
    module.add_function(wrap_pyfunction!(poincare_lorentz_decode_backward, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_simplex_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_simplex_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_sphere_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_sphere_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_log_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_exp_map, module)?)?;
    module.add_function(wrap_pyfunction!(response_geometry_fit_curvature, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_sphere_normalize_base,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_circular_cost, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_euclidean_cost, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_geodesic_sphere_cost, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_barycenter_forward, module)?)?;
    module.add_function(wrap_pyfunction!(sinkhorn_barycenter_vjp, module)?)?;
    module.add_function(wrap_pyfunction!(numerics_sigmoid_stable, module)?)?;
    module.add_function(wrap_pyfunction!(numerics_inverse_softplus, module)?)?;
    module.add_function(wrap_pyfunction!(
        response_geometry_normalize_fisher_rao,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_aux_enabled, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so2, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so2_jvp, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so3, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_rho_so3_jvp, module)?)?;
    module.add_function(wrap_pyfunction!(equivariant_gauge_companion_loss, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit_ibp, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_fit_minimal, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_predict_oos, module)?)?;
    module.add_function(wrap_pyfunction!(sae_steer_delta, module)?)?;
    module.add_function(wrap_pyfunction!(sae_manifold_reconstruction_r2, module)?)?;
    module.add_function(wrap_pyfunction!(sae_streaming_plan, module)?)?;
    module.add_function(wrap_pyfunction!(layer_transport_fit, module)?)?;
    module.add_function(wrap_pyfunction!(layer_transport_ladder, module)?)?;
    module.add_function(wrap_pyfunction!(sae_checkpoint_dynamics, module)?)?;
    inference_instruments::register(module)?;
    module.add_function(wrap_pyfunction!(sae_manifold_assignment_summary, module)?)?;
    module.add_function(wrap_pyfunction!(gated_sae_decode, module)?)?;
    module.add_function(wrap_pyfunction!(interchange_decode_forward, module)?)?;
    module.add_function(wrap_pyfunction!(interchange_decode_backward, module)?)?;
    module.add_function(wrap_pyfunction!(interchange_swap_forward, module)?)?;
    module.add_function(wrap_pyfunction!(interchange_swap_backward, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_latent_backward, module)?)?;
    module.add_function(wrap_pyfunction!(glm_reml_fit_latent, module)?)?;
    module.add_function(wrap_pyfunction!(glm_reml_fit_latent_backward, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_predict_bands_table, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_eta_bands, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_credible_interval, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_coefficient_names_json, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_samples_summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(posterior_trace_selection_json, module)?)?;
    module.add_function(wrap_pyfunction!(apply_inverse_link_array, module)?)?;
    module.add_function(wrap_pyfunction!(summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(curvature_inference_json, module)?)?;
    module.add_function(wrap_pyfunction!(smooth_term_lr_inference_json, module)?)?;
    module.add_function(wrap_pyfunction!(model_debiased_functional_json, module)?)?;
    module.add_function(wrap_pyfunction!(summary_payload_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(smoothing_parameters_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(model_group_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(model_deployment_extensions, module)?)?;
    module.add_function(wrap_pyfunction!(model_evidence, module)?)?;
    module.add_function(wrap_pyfunction!(summary_repr, module)?)?;
    module.add_function(wrap_pyfunction!(summary_html, module)?)?;
    module.add_function(wrap_pyfunction!(coefficient_state_json, module)?)?;
    module.add_function(wrap_pyfunction!(term_blocks_for_model, module)?)?;
    module.add_function(wrap_pyfunction!(difference_smooth_json, module)?)?;
    module.add_function(wrap_pyfunction!(difference_smooth_rows, module)?)?;
    module.add_function(wrap_pyfunction!(
        cross_fit_shared_precision_groups_json,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(check_json, module)?)?;
    module.add_function(wrap_pyfunction!(check_payload_from_model, module)?)?;
    module.add_function(wrap_pyfunction!(report_html, module)?)?;
    module.add_function(wrap_pyfunction!(compute_residuals, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(benchmark_prediction_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(auc_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(brier_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(log_loss_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(nagelkerke_r2_from_predictions, module)?)?;
    module.add_function(wrap_pyfunction!(make_folds_indices, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_log_loss_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_prediction_scores_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(zscore_train_test_arrays, module)?)?;
    module.add_function(wrap_pyfunction!(classification_metrics, module)?)?;
    module.add_function(wrap_pyfunction!(survival_concordance, module)?)?;
    module.add_function(wrap_pyfunction!(survival_score_grid_from_times, module)?)?;
    module.add_function(wrap_pyfunction!(repeat_survival_curve, module)?)?;
    module.add_function(wrap_pyfunction!(survival_null_curve_from_train, module)?)?;
    module.add_function(wrap_pyfunction!(
        survival_matrix_from_risk_calibration,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        survival_lifted_metrics_from_predictions,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(synthetic_binomial_columns, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_geo_disease_columns, module)?)?;
    module.add_function(wrap_pyfunction!(
        synthetic_continuous_order_columns,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        synthetic_thread3_admixture_cliff_columns,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(synthetic_geo_disease_eas_columns, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_papuan_oce_columns, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_hgdp_pc_panel_columns, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_geo_subpop_response, module)?)?;
    module.add_function(wrap_pyfunction!(synthetic_geo_latlon_response, module)?)?;
    module.add_function(wrap_pyfunction!(thread3_cliff_gradient_magnitude, module)?)?;
    // LatentCoord input-location derivative helpers (one per basis kind).
    // The Duchon descriptor differentiates the *built* design (kernel-nullspace
    // projection + polynomial columns + amplification) via `duchon_basis_with_jets`,
    // not the raw centerwise radial kernel, so no Duchon raw-radial FFI is
    // registered here. The Matérn descriptor has no nullspace projection /
    // polynomial tail, so its input-location jet and Hessian are the full
    // metric-aware kernel derivatives (anisotropy threaded through, matching the
    // forward design exactly — issue #437); the assembly lives in Rust, Python
    // only contracts the returned tensors with upstream gradients.
    module.add_function(wrap_pyfunction!(matern_input_location_first_jet, module)?)?;
    module.add_function(wrap_pyfunction!(matern_input_location_hessian, module)?)?;
    module.add_function(wrap_pyfunction!(matern_basis_gradient_streaming, module)?)?;
    module.add_function(wrap_pyfunction!(
        sphere_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        periodic_bspline_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        bspline_tensor_input_location_first_derivative,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(mechanism_sparsity_jacobian, module)?)?;
    module.add_function(wrap_pyfunction!(conditional_prior_ivae, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_aux_richness, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_jacobian_sparsity, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_anchor_consistency, module)?)?;
    module.add_function(wrap_pyfunction!(diagnostics_concat_decoder_blocks, module)?)?;
    module.add_function(wrap_pyfunction!(partial_supervision_solve, module)?)?;
    module.add_function(wrap_pyfunction!(thin_svd_scores, module)?)?;
    module.add_function(wrap_pyfunction!(linear_dictionary_fit, module)?)?;
    module.add_function(wrap_pyfunction!(
        identifiable_factor_select_weights_array,
        module
    )?)?;
    module.add_class::<IsometryPenalty>()?;
    module.add_class::<SparsityPenalty>()?;
    module.add_class::<PyTopKActivationPenalty>()?;
    module.add_class::<JumpReLUPenalty>()?;
    module.add_class::<ARDPenalty>()?;
    module.add_class::<AuxConditionalPriorPenalty>()?;
    module.add_class::<IvaeRidgeMeanGauge>()?;
    module.add_class::<ParametricAuxConditionalPriorPenalty>()?;
    module.add_class::<BlockSparsityPenalty>()?;
    module.add_class::<BlockOrthogonalityPenalty>()?;
    module.add_class::<OrthogonalityPenalty>()?;
    module.add_class::<SheafConsistencyPenalty>()?;
    module.add_class::<SoftmaxAssignmentSparsityPenalty>()?;
    module.add_class::<PyIBPAssignmentPenalty>()?;
    module.add_class::<ScadMcpPenalty>()?;
    module.add_class::<TotalVariationPenalty>()?;
    module.add_class::<NuclearNormPenalty>()?;
    module.add_class::<MechanismSparsityPenalty>()?;
    Ok(())
}

/// Smoothed column-2-norm penalty on a decoder weight matrix W of shape
/// `(d_obs, k_latent)`. Returns `(value, grad)` where `grad` has the same
/// shape as `W`.
#[pyfunction(signature = (weight, epsilon, w))]
fn mechanism_sparsity_jacobian<'py>(
    py: Python<'py>,
    weight: f64,
    epsilon: f64,
    w: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, Py<PyArray2<f64>>)> {
    let pen = gam::terms::sae::identifiability::MechanismSparsityJacobian::new(weight, epsilon)
        .map_err(py_value_error)?;
    let (value, grad) = pen.value_and_grad(w.as_array());
    Ok((value, grad.into_pyarray(py).unbind()))
}

/// iVAE conditional-Gaussian log-prior. Given per-row `(mean, scale)` arrays
/// of shape `(n_rows, latent_dim)`, returns the negative log-prior value and
/// its gradient w.r.t. `t` (same shape as `t`).
#[pyfunction(signature = (weight, t, mean, scale))]
fn conditional_prior_ivae<'py>(
    py: Python<'py>,
    weight: f64,
    t: PyReadonlyArray2<'py, f64>,
    mean: PyReadonlyArray2<'py, f64>,
    scale: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, Py<PyArray2<f64>>)> {
    let pen = gam::sae_identifiability::ConditionalPriorIvae::new(
        mean.as_array().to_owned(),
        scale.as_array().to_owned(),
        weight,
    )
    .map_err(py_value_error)?;
    let (value, grad) = pen.value_and_grad(t.as_array());
    Ok((value, grad.into_pyarray(py).unbind()))
}

/// iVAE auxiliary-richness metrics (Khemakhem et al. 2020 Theorem 1).
///
/// Returns a dict with keys:
/// `aux_observed` (bool), `n_nonfinite_aux` (int), `aux_dim` (int),
/// `latent_dim` (int), `n_rows` (int), `constant_columns` (list[int]),
/// `aux_is_discrete` (bool), `n_distinct_levels` (int),
/// `jacobian_rank` (int), `jacobian_rank_estimated` (bool).
#[pyfunction(signature = (aux, latents))]
fn diagnostics_aux_richness<'py>(
    py: Python<'py>,
    aux: PyReadonlyArray2<'py, f64>,
    latents: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyDict>> {
    let m =
        gam::identifiability_diagnostics::aux_richness_metrics(aux.as_array(), latents.as_array());
    let dict = PyDict::new(py);
    dict.set_item("aux_observed", m.aux_observed)?;
    dict.set_item("n_nonfinite_aux", m.n_nonfinite_aux)?;
    dict.set_item("aux_dim", m.aux_dim)?;
    dict.set_item("latent_dim", m.latent_dim)?;
    dict.set_item("n_rows", m.n_rows)?;
    dict.set_item("constant_columns", m.constant_columns)?;
    dict.set_item("aux_is_discrete", m.aux_is_discrete)?;
    dict.set_item("n_distinct_levels", m.n_distinct_levels)?;
    // Map usize::MAX sentinel to Python None when rank not estimated.
    if m.jacobian_rank_estimated {
        dict.set_item("jacobian_rank", m.jacobian_rank)?;
    } else {
        dict.set_item("jacobian_rank", py.None())?;
    }
    dict.set_item("jacobian_rank_estimated", m.jacobian_rank_estimated)?;
    Ok(dict.unbind())
}

/// Decoder-Jacobian sparsity metrics (Hyvarinen-Morioka 2017; Lachapelle 2024).
///
/// `jacobians_flat` has shape `(n_samples * P, latent_dim)`. Returns a dict
/// with keys: `n_samples`, `p_features`, `latent_dim`, `mean_sparsity`,
/// `max_abs`, `ranks` (list[int] of length n_samples).
#[pyfunction(signature = (jacobians_flat, n_samples, zero_threshold))]
fn diagnostics_jacobian_sparsity<'py>(
    py: Python<'py>,
    jacobians_flat: PyReadonlyArray2<'py, f64>,
    n_samples: usize,
    zero_threshold: f64,
) -> PyResult<Py<PyDict>> {
    if n_samples == 0 {
        return Err(py_value_error(
            "diagnostics_jacobian_sparsity: n_samples must be >= 1".into(),
        ));
    }
    let (rows, _) = jacobians_flat.as_array().dim();
    if rows % n_samples != 0 {
        return Err(py_value_error(format!(
            "diagnostics_jacobian_sparsity: rows {} not divisible by n_samples {}",
            rows, n_samples
        )));
    }
    let m = gam::identifiability_diagnostics::jacobian_sparsity_metrics(
        jacobians_flat.as_array(),
        n_samples,
        zero_threshold,
    );
    let dict = PyDict::new(py);
    dict.set_item("n_samples", m.n_samples)?;
    dict.set_item("p_features", m.p_features)?;
    dict.set_item("latent_dim", m.latent_dim)?;
    dict.set_item("mean_sparsity", m.mean_sparsity)?;
    dict.set_item("max_abs", m.max_abs)?;
    dict.set_item("ranks", m.ranks)?;
    Ok(dict.unbind())
}

/// Anchor-consistency metrics for a manifold-SAE assignment matrix.
///
/// `assignments` has shape `(N, K)`. Returns a dict with keys:
/// `n_rows`, `n_atoms`, `n_anchors`, `anchors_per_atom` (list[int] length K).
#[pyfunction(signature = (assignments, anchor_dominance))]
fn diagnostics_anchor_consistency<'py>(
    py: Python<'py>,
    assignments: PyReadonlyArray2<'py, f64>,
    anchor_dominance: f64,
) -> PyResult<Py<PyDict>> {
    if !(anchor_dominance > 0.0 && anchor_dominance <= 1.0) {
        return Err(py_value_error(format!(
            "diagnostics_anchor_consistency: anchor_dominance must be in (0, 1]; got {}",
            anchor_dominance
        )));
    }
    let m = gam::identifiability_diagnostics::anchor_consistency_metrics(
        assignments.as_array(),
        anchor_dominance,
    );
    let dict = PyDict::new(py);
    dict.set_item("n_rows", m.n_rows)?;
    dict.set_item("n_atoms", m.n_atoms)?;
    dict.set_item("n_anchors", m.n_anchors)?;
    dict.set_item("anchors_per_atom", m.anchors_per_atom)?;
    Ok(dict.unbind())
}

/// Concatenate a list of `(basis_size_k, P)` per-atom decoder blocks into a
/// single `(P, sum_k basis_size_k)` Jacobian matrix. Saves a numpy
/// concatenation in the Python diagnostics dispatcher.
#[pyfunction(signature = (blocks))]
fn diagnostics_concat_decoder_blocks<'py>(
    py: Python<'py>,
    blocks: Vec<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let views: Vec<_> = blocks.iter().map(|b| b.as_array()).collect();
    let out =
        gam::identifiability_diagnostics::concat_decoder_blocks(&views).map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

/// Generic 2D log-λ weight-selection driver.
///
/// Takes a precomputed `(G1, G2)` RSS grid and a matching penalty grid
/// together with the two 1D weight grids, computes the Laplace-style log
/// marginal-likelihood proxy on every cell, and returns the maximising
/// cell with deterministic tie-breaking. Useful for any two-penalty model
/// (identifiable-factor recipe, double-penalty smooths, IBP + sparsity).
///
/// Returns a dict with keys: ``best_i``, ``best_j``, ``best_lam1``,
/// ``best_lam2``, ``best_evidence``, ``evidence_grid``.
#[pyfunction(signature = (rss_grid, penalty_grid, lam1_grid, lam2_grid, n_obs))]
fn identifiable_factor_select_weights_array<'py>(
    py: Python<'py>,
    rss_grid: PyReadonlyArray2<'py, f64>,
    penalty_grid: PyReadonlyArray2<'py, f64>,
    lam1_grid: PyReadonlyArray1<'py, f64>,
    lam2_grid: PyReadonlyArray1<'py, f64>,
    n_obs: i64,
) -> PyResult<Py<PyDict>> {
    if n_obs <= 0 {
        return Err(py_value_error(format!(
            "identifiable_factor_select_weights_array: n_obs must be > 0, got {n_obs}"
        )));
    }
    let res = gam::sae_identifiability::identifiable_factor_select_weights(
        rss_grid.as_array(),
        penalty_grid.as_array(),
        lam1_grid.as_array(),
        lam2_grid.as_array(),
        n_obs as usize,
    )
    .map_err(py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("best_i", res.best_i)?;
    out.set_item("best_j", res.best_j)?;
    out.set_item("best_lam1", res.best_lam1)?;
    out.set_item("best_lam2", res.best_lam2)?;
    out.set_item("best_evidence", res.best_evidence)?;
    out.set_item("evidence_grid", res.evidence_grid.into_pyarray(py))?;
    Ok(out.unbind())
}

/// Partial-supervision gauge-fix solver.
///
/// Single-call FFI for the supervised + free latent-block gauge fix used by
/// `gamfit.examples.partial_supervision` (and reusable from the CLI / R /
/// Julia bindings). All linear-algebra work — orthogonal Procrustes via
/// SVD, anchor least-squares via SVD pseudo-inverse, soft-L2 ridge map via
/// symmetric eigendecomposition with a REML grid, and the orthogonal-
/// complement projection via thin QR — runs in Rust through the faer
/// bridge.
///
/// Parameters
/// ----------
/// t_sup, aux : (N, d_sup) f64 arrays
///     Initial supervised block and the auxiliary target it aligns to.
/// t_free : (N, d_free) f64 array
///     Free block; ``d_free`` may be zero.
/// method : {"procrustes", "anchor", "soft_l2"}
///     Supervised-block alignment method.
/// anchor_idx : Vec<i64>
///     Anchor rows for the ``anchor`` method (ignored otherwise).
/// free_constraint : {"orthogonal_to_sup", "none"}
///     Decorrelation rule for the free block.
///
/// Returns a dict with keys: ``t_supervised``, ``t_free``,
/// ``alignment_score``, ``selected_weight``, ``map_r``, ``map_a``,
/// ``map_b``. Values not produced by the chosen method are returned as
/// Python ``None``.
#[pyfunction(signature = (t_sup, aux, t_free, method, anchor_idx, free_constraint))]
fn partial_supervision_solve<'py>(
    py: Python<'py>,
    t_sup: PyReadonlyArray2<'py, f64>,
    aux: PyReadonlyArray2<'py, f64>,
    t_free: PyReadonlyArray2<'py, f64>,
    method: &str,
    anchor_idx: Vec<i64>,
    free_constraint: &str,
) -> PyResult<Py<PyDict>> {
    let method_enum = match method {
        "procrustes" => gam::sae_identifiability::PartialSupervisionSupMethod::Procrustes,
        "anchor" => gam::sae_identifiability::PartialSupervisionSupMethod::Anchor,
        "soft_l2" => gam::sae_identifiability::PartialSupervisionSupMethod::SoftL2,
        other => {
            return Err(py_value_error(format!(
                "partial_supervision_solve: method must be 'procrustes', 'anchor' or \
                 'soft_l2'; got {other:?}"
            )));
        }
    };
    let free_enum = match free_constraint {
        "orthogonal_to_sup" => {
            gam::sae_identifiability::PartialSupervisionFreeConstraint::OrthogonalToSup
        }
        "none" => gam::sae_identifiability::PartialSupervisionFreeConstraint::None,
        other => {
            return Err(py_value_error(format!(
                "partial_supervision_solve: free_constraint must be 'orthogonal_to_sup' or \
                 'none'; got {other:?}"
            )));
        }
    };
    let mut anchor_usize: Vec<usize> = Vec::with_capacity(anchor_idx.len());
    for idx in &anchor_idx {
        if *idx < 0 {
            return Err(py_value_error(format!(
                "partial_supervision_solve: anchor_idx entries must be non-negative; got {idx}"
            )));
        }
        anchor_usize.push(*idx as usize);
    }
    let result = gam::sae_identifiability::partial_supervision_solve(
        t_sup.as_array(),
        aux.as_array(),
        t_free.as_array(),
        method_enum,
        &anchor_usize,
        free_enum,
    )
    .map_err(py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("t_supervised", result.t_supervised.into_pyarray(py))?;
    out.set_item("t_free", result.t_free.into_pyarray(py))?;
    out.set_item("alignment_score", result.alignment_score)?;
    match result.selected_weight {
        Some(v) => out.set_item("selected_weight", v)?,
        None => out.set_item("selected_weight", py.None())?,
    }
    match result.map_r {
        Some(arr) => out.set_item("map_r", arr.into_pyarray(py))?,
        None => out.set_item("map_r", py.None())?,
    }
    match result.map_a {
        Some(arr) => out.set_item("map_a", arr.into_pyarray(py))?,
        None => out.set_item("map_a", py.None())?,
    }
    match result.map_b {
        Some(arr) => out.set_item("map_b", arr.into_pyarray(py))?,
        None => out.set_item("map_b", py.None())?,
    }
    Ok(out.unbind())
}

/// Column-centred thin-SVD scores for the leading `k` components.
///
/// Returns `U[:, :k] * Σ[:k]` of `X − mean(X, axis=0)`, computed via the
/// faer SVD bridge. Used by the partial-supervision recipe to seed an
/// initial latent block when the caller does not provide one.
#[pyfunction(signature = (x, k))]
fn thin_svd_scores<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: i64,
) -> PyResult<Py<PyArray2<f64>>> {
    if k < 0 {
        return Err(py_value_error(format!(
            "thin_svd_scores: k must be non-negative, got {k}"
        )));
    }
    let out = gam::sae_identifiability::thin_svd_scores(x.as_array(), k as usize)
        .map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (
    x,
    k,
    max_iter = 30,
    top_k = 1,
    assignment = "top_k",
    temperature = 0.25,
    code_ridge = 1.0e-8,
    tolerance = 1.0e-7
))]
fn linear_dictionary_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: usize,
    max_iter: usize,
    top_k: usize,
    assignment: &str,
    temperature: f64,
    code_ridge: f64,
    tolerance: f64,
) -> PyResult<Py<PyDict>> {
    let x_values = x.as_array().to_owned();
    let assignment_kind = LinearDictionaryAssignment::parse(assignment).map_err(py_value_error)?;
    let config = LinearDictionaryConfig {
        n_atoms: k,
        max_iter,
        top_k,
        assignment: assignment_kind,
        temperature,
        code_ridge,
        tolerance,
    };
    let fit = detach_py_result(py, "linear_dictionary_fit", move || {
        fit_linear_dictionary(x_values.view(), &config)
    })?;
    let out = PyDict::new(py);
    out.set_item("atoms", fit.atoms.into_pyarray(py))?;
    out.set_item("assignments", fit.assignments.into_pyarray(py))?;
    out.set_item("fitted", fit.fitted.into_pyarray(py))?;
    out.set_item("lambdas", fit.lambdas.into_pyarray(py))?;
    out.set_item("reml_scores", fit.reml_scores.into_pyarray(py))?;
    out.set_item("explained_variance", fit.explained_variance)?;
    out.set_item("iterations", fit.iterations)?;
    out.set_item("converged", fit.converged)?;
    out.set_item("assignment", fit.assignment.as_str())?;
    out.set_item("top_k", fit.top_k)?;
    Ok(out.unbind())
}

fn inject_scalar_fisher_rao_weight(
    dataset: &mut EncodedDataset,
    fit_config: &mut FitConfig,
    fisher_rao_w: ArrayView3<'_, f64>,
) -> Result<(), String> {
    let n = dataset.values.nrows();
    if fisher_rao_w.shape() != [n, 1, 1] {
        return Err(format!(
            "fit_table fisher_rao_w requires scalar blocks of shape ({n}, 1, 1); got {:?}",
            fisher_rao_w.shape()
        ));
    }
    let weight_name = "__gamfit_fisher_rao_weight".to_string();
    if dataset.headers.iter().any(|name| name == &weight_name) {
        return Err(format!(
            "reserved fisher_rao_w column already exists: {weight_name}"
        ));
    }
    let col_map = dataset.column_map();
    let base_weight_col = fit_config
        .weight_column
        .as_ref()
        .map(|name| {
            col_map
                .get(name)
                .copied()
                .ok_or_else(|| format!("weights column {name:?} is not present"))
        })
        .transpose()?;
    let mut combined = Array1::<f64>::zeros(n);
    for row in 0..n {
        let mut value = fisher_rao_w[[row, 0, 0]];
        if let Some(col) = base_weight_col {
            value *= dataset.values[[row, col]];
        }
        if !(value.is_finite() && value >= 0.0) {
            return Err(format!(
                "fisher_rao_w scalar row {row} produced invalid non-negative weight {value}"
            ));
        }
        combined[row] = value;
    }
    let old_cols = dataset.values.ncols();
    let mut values = Array2::<f64>::zeros((n, old_cols + 1));
    values.slice_mut(s![.., ..old_cols]).assign(&dataset.values);
    values.column_mut(old_cols).assign(&combined);
    dataset.values = values;
    dataset.headers.push(weight_name.clone());
    dataset.schema.columns.push(SchemaColumn {
        name: weight_name.clone(),
        kind: ColumnKindTag::Continuous,
        levels: Vec::new(),
    });
    dataset.column_kinds.push(ColumnKindTag::Continuous);
    fit_config.weight_column = Some(weight_name);
    Ok(())
}

fn fit_dataset_impl(
    mut dataset: EncodedDataset,
    formula: String,
    config_json: Option<&str>,
    fisher_rao_w: Option<ArrayView3<'_, f64>>,
) -> Result<Vec<u8>, WorkflowError> {
    // Always-on progress for the Python bindings: every gamfit fit call
    // installs a visualizer session so the [OUTER step] log stream is
    // accompanied by the `Workflow: Fit | step=N | objective=… (best=…,
    // Δ=±…) | |grad|=…` lane and the `Descent: ▇▆▄▃▂▁` sparkline. The
    // session goes through DumbVisualizer in Python land (stdout/stderr
    // are virtually never TTYs under the gamfit import), so output is
    // throttled stderr writes — safe for notebooks and batch scripts.
    // Session is owned by this function so its Drop runs (clearing the
    // static ACTIVE_FEED) before the result is handed back to Python.
    let mut progress = gam::visualizer::VisualizerSession::new(true);
    progress.set_stage("fit", "optimizing penalized likelihood");
    progress.start_workflow_open_ended("Fit");
    let (mut fit_config, training_table_kind) = parse_fit_config(config_json)?;
    if let Some(w) = fisher_rao_w {
        inject_scalar_fisher_rao_weight(&mut dataset, &mut fit_config, w)?;
    }
    // Calibrated marginal-slope chain (#461): when a CTN Stage-1 recipe is present
    // (config.ctn_stage1), the marginal-slope materializer cross-fits the CTN and
    // produces the calibrated `z` out-of-fold — no z_column is needed and no
    // Stage-1 pre-fit / synthetic column round-trip is performed here. The recipe
    // rides on fit_config straight into materialize.
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let request = materialized.request;

    let mut payload = match request {
        FitRequest::Standard(standard_request) => {
            // Exact O(n) spline-scan fast path (#1030/#1034): a single 1-D
            // Gaussian cubic smooth is the penalized cubic-spline problem the
            // state-space scan solves exactly — route through it and persist
            // the smoother state instead of the dense fit. Detection is
            // structural; every other shape falls through to the dense fit
            // below. Mirrors the CLI run_fit path so CLI and FFI saves agree.
            if let Some(inputs) = gam::spline_scan_fast_path(&standard_request) {
                let scan = gam::solver::spline_scan::fit_spline_scan(
                    &inputs.x,
                    &inputs.y,
                    &inputs.w,
                    inputs.order,
                )
                .map_err(|reason| gam::WorkflowError::IntegrationFailed { reason })?;
                let feature_col = match &standard_request.spec.smooth_terms[0].basis {
                    gam::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => *feature_col,
                    _ => {
                        return Err(gam::WorkflowError::SchemaMismatch {
                            reason: "spline-scan detection accepted a non-1D basis".to_string(),
                        });
                    }
                };
                let feature_column =
                    dataset.headers.get(feature_col).cloned().ok_or_else(|| {
                        gam::WorkflowError::SchemaMismatch {
                            reason: format!(
                                "spline-scan feature column {feature_col} has no header"
                            ),
                        }
                    })?;
                let mut scan_payload =
                    gam::inference::model_payload_builders::assemble_spline_scan_payload(
                        formula,
                        feature_column,
                        &scan,
                        dataset.schema.clone(),
                        dataset.headers.clone(),
                        dataset.feature_ranges(),
                    );
                scan_payload.group_metadata = fit_config.group_metadata.clone();
                scan_payload.training_table_kind = training_table_kind;
                let model = FittedModel::from_payload(scan_payload);
                return serde_json::to_vec(&model).map_err(|err| {
                    gam::WorkflowError::IntegrationFailed {
                        reason: format!("failed to serialize model: {err}"),
                    }
                });
            }
            let family = standard_request.family.clone();
            let fit_result = fit_model(FitRequest::Standard(standard_request))?;
            let standard_result = match fit_result {
                FitResult::Standard(standard_result) => standard_result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the standard workflow to return a standard fit result"
                            .to_string(),
                    });
                }
            };
            let saved_fit = standard_result.fit.clone();
            build_standard_payload(
                formula,
                &dataset,
                &fit_config,
                family,
                &saved_fit,
                &standard_result.design,
                standard_result.resolvedspec,
                standard_result.adaptive_diagnostics,
                standard_result.wiggle_knots.map(|knots| knots.to_vec()),
                standard_result.wiggle_degree,
            )?
        }
        FitRequest::TransformationNormal(tn_request) => {
            let fit_result = fit_model(FitRequest::TransformationNormal(tn_request))?;
            let tn_result = match fit_result {
                FitResult::TransformationNormal(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the transformation-normal workflow to return a transformation-normal fit result"
                            .to_string(),
                    });
                }
            };
            build_transformation_normal_ffi_payload(formula, &dataset, &fit_config, tn_result)?
        }
        FitRequest::BernoulliMarginalSlope(ms_request) => {
            let base_link = ms_request.spec.base_link.clone();
            let frailty = ms_request.spec.frailty.clone();
            let fit_result = fit_model(FitRequest::BernoulliMarginalSlope(ms_request))?;
            let ms_result = match fit_result {
                FitResult::BernoulliMarginalSlope(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the bernoulli marginal-slope workflow to return a marginal-slope fit result"
                            .to_string(),
                    });
                }
            };
            build_bernoulli_marginal_slope_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                base_link,
                frailty,
                ms_result,
            )?
        }
        FitRequest::SurvivalMarginalSlope(ms_request) => {
            let frailty = ms_request.spec.frailty.clone();
            let fit_result = fit_model(FitRequest::SurvivalMarginalSlope(ms_request))?;
            let ms_result = match fit_result {
                FitResult::SurvivalMarginalSlope(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival marginal-slope workflow to return a survival marginal-slope fit result"
                            .to_string(),
                    });
                }
            };
            build_survival_marginal_slope_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                frailty,
                ms_result,
            )?
        }
        FitRequest::GaussianLocationScale(ls_request) => {
            let fit_result = fit_model(FitRequest::GaussianLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::GaussianLocationScale(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the gaussian location-scale workflow to return a gaussian location-scale fit result"
                            .to_string(),
                    });
                }
            };
            // Persist the response standardization factor the fit applied so
            // prediction reconstructs the σ floor at `response_scale·0.01`,
            // keeping predictive σ response-scale-equivariant (#884). The fit
            // already mapped the log-σ `exp(η)` term to raw units via the
            // `+ln(response_scale)` intercept shift; only the additive floor
            // still needs the factor at reconstruction time.
            let response_scale = ls_result.response_scale;
            build_gaussian_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                ls_result,
                response_scale,
            )?
        }
        FitRequest::BinomialLocationScale(ls_request) => {
            let weights = ls_request.spec.weights.clone();
            let link_kind = ls_request.spec.link_kind.clone();
            let fit_result = fit_model(FitRequest::BinomialLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::BinomialLocationScale(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the binomial location-scale workflow to return a binomial location-scale fit result"
                            .to_string(),
                    });
                }
            };
            build_binomial_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                link_kind,
                &weights,
                ls_result,
            )?
        }
        FitRequest::SurvivalLocationScale(ls_request) => {
            let weights = ls_request.spec.weights.clone();
            let fit_result = fit_model(FitRequest::SurvivalLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::SurvivalLocationScale(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival location-scale workflow to return a survival location-scale fit result"
                            .to_string(),
                    });
                }
            };
            build_survival_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                &weights,
                ls_result,
            )?
        }
        FitRequest::SurvivalTransformation(rp_request) => {
            let fit_result = fit_model(FitRequest::SurvivalTransformation(rp_request))?;
            let rp_result = match fit_result {
                FitResult::SurvivalTransformation(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival transformation workflow to return a survival transformation fit result"
                            .to_string(),
                    });
                }
            };
            build_survival_transformation_ffi_payload(formula, &dataset, &fit_config, rp_result)?
        }
        FitRequest::LatentSurvival(lat_request) => {
            let frailty = lat_request.frailty.clone();
            let fit_result = fit_model(FitRequest::LatentSurvival(lat_request))?;
            let lat_result = match fit_result {
                FitResult::LatentSurvival(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the latent survival workflow to return a latent survival fit result"
                            .to_string(),
                    });
                }
            };
            build_latent_survival_ffi_payload(formula, &dataset, &fit_config, frailty, lat_result)?
        }
        FitRequest::LatentBinary(lat_request) => {
            let frailty = lat_request.frailty.clone();
            let fit_result = fit_model(FitRequest::LatentBinary(lat_request))?;
            let lat_result = match fit_result {
                FitResult::LatentBinary(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the latent binary workflow to return a latent binary fit result"
                            .to_string(),
                    });
                }
            };
            build_latent_binary_ffi_payload(formula, &dataset, &fit_config, frailty, lat_result)?
        }
        FitRequest::DispersionLocationScale(ls_request) => {
            // Genuine-dispersion location-scale family (#913): NB / Gamma / Beta
            // / Tweedie mean families whose `noise_formula` models the
            // overdispersion channel. Magic-detected upstream from a
            // `noise_formula` on one of those families; the FFI freezes the mean
            // and log-precision specs and persists them via the same shared
            // location-scale assembler the CLI uses.
            let kind = ls_request.spec.kind;
            let fit_result = fit_model(FitRequest::DispersionLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::DispersionLocationScale(result) => result,
                _ => {
                    return Err(gam::WorkflowError::SchemaMismatch {
                        reason: "python binding expected the dispersion location-scale workflow to return a dispersion location-scale fit result"
                            .to_string(),
                    });
                }
            };
            build_dispersion_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                kind,
                ls_result,
            )?
        }
    };
    payload.group_metadata = fit_config.group_metadata.clone();
    payload.training_table_kind = training_table_kind;
    let model = FittedModel::from_payload(payload);
    serde_json::to_vec(&model).map_err(|err| gam::WorkflowError::IntegrationFailed {
        reason: format!("failed to serialize model: {err}"),
    })
}

fn load_model_impl(model_bytes: &[u8]) -> Result<FittedModel, String> {
    let model: FittedModel = serde_json::from_slice(model_bytes)
        .map_err(|err| format!("failed to parse model json: {err}"))?;
    model.validate_for_persistence()?;
    model.validate_numeric_finiteness()?;
    Ok(model)
}

fn extend_model_with_group_impl(model_bytes: &[u8], request_json: &str) -> Result<Vec<u8>, String> {
    let mut model = load_model_impl(model_bytes)?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "extend_with_group currently supports standard GAM models only; got '{}'",
            prediction_model_class_label(&model)
        ));
    }
    if model.has_link_wiggle() {
        return Err("extend_with_group does not support link-wiggle models".to_string());
    }
    let request: PyExtendGroupRequest = serde_json::from_str(request_json)
        .map_err(|err| format!("failed to parse extend_with_group request: {err}"))?;
    let PyExtendGroupRequest {
        kind,
        name,
        term,
        column,
        level,
        levels,
        metadata,
        prior,
    } = request;
    let kind = kind
        .as_deref()
        .unwrap_or("random-effect-level")
        .replace('_', "-");
    if kind != "random-effect-level" {
        return Err(format!(
            "extend_with_group supports kind='random-effect-level'; got '{kind}'"
        ));
    }
    let mut levels = levels.unwrap_or_default();
    if let Some(level) = level {
        levels.push(level);
    }
    if levels.is_empty() {
        return Err("extend_with_group requires level or levels".to_string());
    }
    let term = match term.or(column) {
        Some(term) => term,
        None => {
            let payload = model.payload();
            let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
                "extend_with_group requires saved resolved_termspec; refit".to_string()
            })?;
            if spec.random_effect_terms.len() == 1 {
                spec.random_effect_terms[0].name.clone()
            } else {
                return Err(
                    "extend_with_group requires term when the model has zero or multiple group terms"
                        .to_string(),
                );
            }
        }
    };

    for level in levels {
        extend_model_with_random_effect_level(
            &mut model,
            term.as_str(),
            name.as_deref(),
            level,
            metadata.clone(),
            prior.clone(),
        )?;
    }
    model.validate_for_persistence()?;
    model.validate_numeric_finiteness()?;
    serde_json::to_vec(&model).map_err(|err| format!("failed to serialize extended model: {err}"))
}

fn extend_model_with_random_effect_level(
    model: &mut FittedModel,
    term_name: &str,
    requested_name: Option<&str>,
    level: serde_json::Value,
    metadata: Option<serde_json::Value>,
    prior: Option<serde_json::Value>,
) -> Result<(), String> {
    let payload: &mut FittedModelPayload = &mut *model;
    let (term_idx, feature_col, penalty_index) = {
        let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
            "extend_with_group requires saved resolved_termspec; refit".to_string()
        })?;
        let term_idx = spec
            .random_effect_terms
            .iter()
            .position(|term| term.name == term_name)
            .ok_or_else(|| format!("extend_with_group unknown random-effect term '{term_name}'"))?;
        (
            term_idx,
            spec.random_effect_terms[term_idx].feature_col,
            random_effect_penalty_index(spec, term_idx),
        )
    };
    let schema = payload
        .data_schema
        .as_mut()
        .ok_or_else(|| "extend_with_group requires saved data_schema; refit".to_string())?;
    let schema_col = schema.columns.get_mut(feature_col).ok_or_else(|| {
        format!(
            "extend_with_group term '{term_name}' feature column {feature_col} out of saved schema bounds"
        )
    })?;
    let (level_bits, encoded_value) = level_bits_for_extension(schema_col, &level)?;
    {
        let spec = payload.resolved_termspec.as_ref().ok_or_else(|| {
            "extend_with_group requires saved resolved_termspec; refit".to_string()
        })?;
        let levels = spec.random_effect_terms[term_idx]
            .frozen_levels
            .as_ref()
            .ok_or_else(|| {
                format!(
                    "extend_with_group term '{term_name}' is not frozen; refit with persisted metadata"
                )
            })?;
        if levels.contains(&level_bits) {
            return Err(format!(
                "extend_with_group level {} already exists for random-effect term '{term_name}'",
                compact_json(&level)
            ));
        }
    }
    if payload.deployment_extensions.iter().any(|extension| {
        extension.kind == "random-effect-level"
            && extension.term == term_name
            && extension.level_bits == level_bits
    }) {
        return Err(format!(
            "extend_with_group level {} is already deployed for random-effect term '{term_name}'",
            compact_json(&level)
        ));
    }
    let coefficient_index = payload
        .fit_result
        .as_ref()
        .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?
        .beta
        .len();
    let (coefficient_mean, supplied_variance) = extension_prior_parameters(prior.as_ref())?;
    let coefficient_variance = match supplied_variance {
        Some(variance) => variance,
        None => {
            let fit = payload
                .fit_result
                .as_ref()
                .ok_or_else(|| "extend_with_group requires saved fit_result; refit".to_string())?;
            let lambda = fit
                .lambdas
                .get(penalty_index)
                .copied()
                .filter(|lambda| lambda.is_finite() && *lambda > 0.0)
                .ok_or_else(|| {
                    format!(
                        "extend_with_group term '{term_name}' has no finite positive prior lambda"
                    )
                })?;
            // The unseen-level default prior is the fitted random-effect
            // variance component `σ_b² = φ̂ / λ` (mgcv's `λ = φ̂ / σ_b²`
            // convention), NOT the scale-free `1 / λ`. `φ̂` is the residual
            // dispersion that scales every predict-time covariance: `1` for
            // fixed-scale families (Poisson/Binomial — where `φ̂/λ` collapses
            // to the old `1/λ`), but `σ̂²` for Gaussian and the estimated
            // dispersion for Gamma/Tweedie/NB. Omitting `φ̂` made the prior
            // (and any deployment interval built from it) wrong by `1/φ̂` and,
            // for an estimated scale, not response-scale equivariant. See #674.
            let phi = fit.dispersion_phi();
            if !(phi.is_finite() && phi > 0.0) {
                return Err(format!(
                    "extend_with_group term '{term_name}' has a non-finite or non-positive \
                     dispersion (φ̂ = {phi}); cannot form the default prior variance"
                ));
            }
            phi / lambda
        }
    };
    extend_training_feature_range(
        payload.training_feature_ranges.as_mut(),
        feature_col,
        encoded_value,
    );
    insert_coefficient_into_saved_fit(
        payload.fit_result.as_mut(),
        coefficient_index,
        coefficient_mean,
        coefficient_variance,
    )?;
    insert_coefficient_into_saved_fit(
        payload.unified.as_mut(),
        coefficient_index,
        coefficient_mean,
        coefficient_variance,
    )?;

    let extension_name = requested_name
        .map(str::to_string)
        .unwrap_or_else(|| format!("{term_name}:{}", compact_json(&level)));
    if let Some(metadata_value) = metadata.clone() {
        let group_metadata = payload.group_metadata.get_or_insert_with(BTreeMap::new);
        group_metadata.insert(extension_name.clone(), metadata_value.clone());
    }
    payload
        .deployment_extensions
        .push(SavedDeploymentExtension {
            name: extension_name,
            kind: "random-effect-level".to_string(),
            term: term_name.to_string(),
            level,
            level_bits,
            coefficient_index,
            coefficient_mean,
            coefficient_variance,
            metadata,
            prior,
        });
    Ok(())
}

fn level_bits_for_extension(
    schema_col: &mut SchemaColumn,
    level: &serde_json::Value,
) -> Result<(u64, f64), String> {
    match schema_col.kind {
        ColumnKindTag::Categorical => {
            let label = match level {
                serde_json::Value::String(s) => s.clone(),
                other => compact_json(other),
            };
            if schema_col.levels.iter().any(|existing| existing == &label) {
                return Err(format!(
                    "extend_with_group categorical level '{label}' already exists in column '{}'",
                    schema_col.name
                ));
            }
            let encoded = schema_col.levels.len() as f64;
            schema_col.levels.push(label);
            Ok((encoded.to_bits(), encoded))
        }
        ColumnKindTag::Continuous | ColumnKindTag::Binary => {
            let value = json_level_to_f64(level)?;
            Ok((value.to_bits(), value))
        }
    }
}

fn json_level_to_f64(value: &serde_json::Value) -> Result<f64, String> {
    let out = match value {
        serde_json::Value::Number(n) => n
            .as_f64()
            .ok_or_else(|| format!("extend_with_group level {n} is not representable as f64"))?,
        serde_json::Value::String(s) => s
            .parse::<f64>()
            .map_err(|_| format!("extend_with_group level '{s}' is not numeric"))?,
        other => {
            return Err(format!(
                "extend_with_group numeric random-effect levels must be numbers or numeric strings; got {}",
                compact_json(other)
            ));
        }
    };
    if !out.is_finite() {
        return Err(format!(
            "extend_with_group random-effect level must be finite; got {out}"
        ));
    }
    Ok(out)
}

fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

fn random_effect_penalty_index(spec: &TermCollectionSpec, term_idx: usize) -> usize {
    usize::from(spec.linear_terms.iter().any(|term| term.double_penalty)) + term_idx
}

fn extension_prior_parameters(
    prior: Option<&serde_json::Value>,
) -> Result<(f64, Option<f64>), String> {
    let Some(value) = prior else {
        return Ok((0.0, None));
    };
    if value.is_null() {
        return Ok((0.0, None));
    }
    let parsed: PyExtensionPrior = serde_json::from_value(value.clone())
        .map_err(|err| format!("failed to parse extend_with_group prior: {err}"))?;
    let mean = parsed.mean.or(parsed.mu).unwrap_or(0.0);
    if !mean.is_finite() {
        return Err(format!(
            "extend_with_group prior mean must be finite; got {mean}"
        ));
    }
    let variance = match (parsed.variance, parsed.precision) {
        (Some(variance), _) => {
            if !(variance.is_finite() && variance > 0.0) {
                return Err(format!(
                    "extend_with_group prior variance must be finite and positive; got {variance}"
                ));
            }
            Some(variance)
        }
        (None, Some(precision)) => {
            if !(precision.is_finite() && precision > 0.0) {
                return Err(format!(
                    "extend_with_group prior precision must be finite and positive; got {precision}"
                ));
            }
            Some(1.0 / precision)
        }
        (None, None) => None,
    };
    Ok((mean, variance))
}

fn extend_training_feature_range(
    ranges: Option<&mut Vec<(f64, f64)>>,
    feature_col: usize,
    value: f64,
) {
    if let Some(ranges) = ranges
        && let Some((lo, hi)) = ranges.get_mut(feature_col)
    {
        if value.is_finite() {
            *lo = (*lo).min(value);
            *hi = (*hi).max(value);
        }
    }
}

fn insert_coefficient_into_saved_fit(
    fit: Option<&mut gam::estimate::UnifiedFitResult>,
    index: usize,
    value: f64,
    variance: f64,
) -> Result<(), String> {
    let Some(fit) = fit else {
        return Ok(());
    };
    if !(variance.is_finite() && variance > 0.0) {
        return Err(format!(
            "extend_with_group coefficient variance must be finite and positive; got {variance}"
        ));
    }
    if index > fit.beta.len() {
        return Err(format!(
            "extend_with_group coefficient index {index} exceeds fit coefficient length {}",
            fit.beta.len()
        ));
    }
    fit.beta = insert_array1(&fit.beta, index, value);
    let block_idx = fit
        .blocks
        .iter()
        .position(|block| block.role == BlockRole::Mean)
        .unwrap_or(0);
    if block_idx >= fit.blocks.len() {
        return Err("extend_with_group saved fit has no coefficient blocks".to_string());
    }
    if index > fit.blocks[block_idx].beta.len() {
        return Err(format!(
            "extend_with_group coefficient index {index} exceeds mean block length {}",
            fit.blocks[block_idx].beta.len()
        ));
    }
    fit.blocks[block_idx].beta = insert_array1(&fit.blocks[block_idx].beta, index, value);
    // No-refit posterior algebra for a deployment-only block:
    //
    // The fitted posterior precision for the original coefficients is H_old.
    // Extending with a new random-effect coefficient b and no likelihood
    // refit contributes only its Gaussian prior,
    //
    //   -log p(b) = 1/2 (b - mu)' (lambda_new S_new) (b - mu) + const.
    //
    // Since no old likelihood rows or old penalties are recomputed, the joint
    // precision is blockdiag(H_old, lambda_new S_new).  Therefore the
    // conditional covariance is blockdiag(V_old, S_new^{-1}/lambda_new).  The
    // current API extends one iid random-effect coordinate at a time, so
    // S_new = [1] and `variance` is exactly 1/lambda_new, or the caller's
    // supplied scalar prior covariance.
    if let Some(cov) = fit.covariance_conditional.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    if let Some(cov) = fit.covariance_corrected.as_mut() {
        *cov = insert_symmetric_array2(cov, index, variance)?;
    }
    let variance_diag = variance;
    let precision_diag = 1.0 / variance_diag;
    if let Some(inference) = fit.inference.as_mut() {
        // Boundary adapter: `penalized_hessian` is the `UnscaledPrecision`
        // newtype; unwrap for the `insert_symmetric_array2` helper and wrap
        // the result back on assignment.
        inference.penalized_hessian = insert_symmetric_array2(
            inference.penalized_hessian.as_array(),
            index,
            precision_diag,
        )?
        .into();
        if let Some(cov) = inference.beta_covariance.as_mut() {
            // `beta_covariance` is the `PhiScaledCovariance` newtype.
            *cov = insert_symmetric_array2(cov.as_array(), index, variance_diag)?.into();
        }
        if let Some(se) = inference.beta_standard_errors.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
        }
        if let Some(cov) = inference.beta_covariance_corrected.as_mut() {
            *cov = insert_symmetric_array2(cov, index, variance_diag)?;
        }
        if let Some(se) = inference.beta_standard_errors_corrected.as_mut() {
            *se = insert_array1(se, index, variance_diag.sqrt());
        }
        if let Some(cov) = inference.beta_covariance_frequentist.as_mut() {
            *cov = insert_symmetric_array2(cov, index, 0.0)?;
        }
        if let Some(influence) = inference.coefficient_influence.as_mut() {
            *influence = insert_symmetric_array2(influence, index, 0.0)?;
        }
        if let Some(correction) = inference.smoothing_correction.as_mut() {
            *correction = insert_symmetric_array2(correction, index, 0.0)?;
        }
        if let Some(qs) = inference.reparam_qs.as_mut() {
            *qs = insert_symmetric_array2(qs, index, 1.0)?;
        }
        if let Some(bias) = inference.bias_correction_beta.as_mut() {
            *bias = insert_array1(bias, index, 0.0);
        }
    }
    if let Some(geometry) = fit.geometry.as_mut() {
        geometry.penalized_hessian =
            insert_symmetric_array2(geometry.penalized_hessian.as_array(), index, precision_diag)?
                .into();
    }
    Ok(())
}

fn insert_array1(values: &Array1<f64>, index: usize, value: f64) -> Array1<f64> {
    let mut out = Vec::<f64>::with_capacity(values.len() + 1);
    out.extend(values.iter().take(index).copied());
    out.push(value);
    out.extend(values.iter().skip(index).copied());
    Array1::from_vec(out)
}

fn insert_symmetric_array2(
    matrix: &Array2<f64>,
    index: usize,
    diagonal: f64,
) -> Result<Array2<f64>, String> {
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "extend_with_group expected square matrix, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }
    if index > matrix.nrows() {
        return Err(format!(
            "extend_with_group matrix insert index {index} exceeds dimension {}",
            matrix.nrows()
        ));
    }
    let n = matrix.nrows();
    let mut out = Array2::<f64>::zeros((n + 1, n + 1));
    for old_i in 0..n {
        let new_i = if old_i < index { old_i } else { old_i + 1 };
        for old_j in 0..n {
            let new_j = if old_j < index { old_j } else { old_j + 1 };
            out[[new_i, new_j]] = matrix[[old_i, old_j]];
        }
    }
    out[[index, index]] = diagonal;
    Ok(out)
}

fn validate_formula_json_impl(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<&str>,
) -> Result<String, String> {
    let mut dataset = dataset_with_inferred_schema(headers, rows)?;
    let (mut fit_config, _training_table_kind) = parse_fit_config(config_json)?;
    // Calibrated marginal-slope chain (#461): validation is purely structural and
    // must stay cheap — it must NOT cross-fit Stage-1. When a CTN Stage-1 recipe
    // is present, strip it and stand in a zero-valued placeholder dose column so
    // the Stage-2 marginal-slope structure validates without any Stage-1 fit or
    // cross-fit. The real fit produces `z` out-of-fold and needs no such column.
    if fit_config.ctn_stage1.is_some() {
        const VALIDATION_PLACEHOLDER_Z: &str = "__gam_validation_ctn_stage1_z";
        fit_config.ctn_stage1 = None;
        if dataset
            .headers
            .iter()
            .any(|name| name == VALIDATION_PLACEHOLDER_Z)
        {
            return Err(format!(
                "reserved validation column '{VALIDATION_PLACEHOLDER_Z}' already exists in the \
                 input data; rename it before validating the calibrated chain"
            ));
        }
        let n = dataset.values.nrows();
        let old_cols = dataset.values.ncols();
        let mut values = Array2::<f64>::zeros((n, old_cols + 1));
        values.slice_mut(s![.., ..old_cols]).assign(&dataset.values);
        dataset.values = values;
        dataset.headers.push(VALIDATION_PLACEHOLDER_Z.to_string());
        dataset
            .column_kinds
            .push(gam::inference::model::ColumnKindTag::Continuous);
        dataset
            .schema
            .columns
            .push(gam::inference::model::SchemaColumn {
                name: VALIDATION_PLACEHOLDER_Z.to_string(),
                kind: gam::inference::model::ColumnKindTag::Continuous,
                levels: Vec::new(),
            });
        fit_config.z_column = Some(VALIDATION_PLACEHOLDER_Z.to_string());
    }
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let (family_name, model_class, supported_by_python) = request_metadata(&materialized.request);
    let response_column = response_column_name(&formula);
    let payload = ValidationPayload {
        formula,
        family_name: family_name.to_string(),
        model_class: model_class.to_string(),
        response_column,
        columns: dataset.headers.clone(),
        n_rows: dataset.values.nrows(),
        n_columns: dataset.values.ncols(),
        supported_by_python,
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize validation payload: {err}"))
}

fn parse_formula_validation_payload_json(
    payload_json: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    let payload: serde_json::Value = serde_json::from_str(payload_json)
        .map_err(|err| format!("failed to parse FormulaValidation payload: {err}"))?;
    match payload {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err("FormulaValidation payload must be a JSON object".to_string()),
    }
}

fn json_payload_truthy(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => {
            value.as_f64().map(|number| number != 0.0).unwrap_or(true)
        }
        serde_json::Value::String(value) => !value.is_empty(),
        serde_json::Value::Array(value) => !value.is_empty(),
        serde_json::Value::Object(value) => !value.is_empty(),
    }
}

fn python_repr_json_value(value: Option<&serde_json::Value>) -> String {
    match value {
        None | Some(serde_json::Value::Null) => "None".to_string(),
        Some(serde_json::Value::Bool(value)) => {
            if *value {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Some(serde_json::Value::Number(value)) => value.to_string(),
        Some(serde_json::Value::String(value)) => python_repr_string(value),
        Some(value @ serde_json::Value::Array(_)) | Some(value @ serde_json::Value::Object(_)) => {
            python_str_json_value(value)
        }
    }
}

fn python_repr_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('\'');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => {
                out.push_str(&format!("\\x{:02x}", ch as u32));
            }
            ch => out.push(ch),
        }
    }
    out.push('\'');
    out
}

fn python_str_json_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "None".to_string(),
        serde_json::Value::Bool(value) => {
            if *value {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => value.clone(),
        serde_json::Value::Array(values) => {
            let inner = values
                .iter()
                .map(|value| match value {
                    serde_json::Value::String(value) => python_repr_string(value),
                    value => python_str_json_value(value),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
        serde_json::Value::Object(values) => {
            let inner = values
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}: {}",
                        python_repr_string(key),
                        match value {
                            serde_json::Value::String(value) => python_repr_string(value),
                            value => python_str_json_value(value),
                        }
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
    }
}

fn escape_html(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            ch => out.push(ch),
        }
    }
    out
}

fn predict_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<&str>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let model_class = model.predict_model_class();
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
    predict_dataset_impl(&model, model_class, dataset, options_json)
}

fn predict_array_impl(
    model_bytes: &[u8],
    x: ArrayView2<'_, f64>,
    options_json: Option<&str>,
) -> Result<Array2<f64>, String> {
    let model = load_model_impl(model_bytes)?;
    let model_class = model.predict_model_class();
    let dataset = dataset_from_x_array_with_model_schema(&model, x)?;
    if matches!(model_class, PredictModelClass::Survival) {
        return Err("predict_array does not support survival prediction payloads".to_string());
    }
    let options = parse_predict_options(options_json)?;
    let columns = predict_columns(&model, dataset, &options)?;
    columns_to_array(columns)
}

fn predict_dataset_impl(
    model: &FittedModel,
    model_class: PredictModelClass,
    dataset: EncodedDataset,
    options_json: Option<&str>,
) -> Result<String, String> {
    let options = parse_predict_options(options_json)?;
    if matches!(model_class, PredictModelClass::Survival) {
        return predict_table_survival(model, &dataset, &options);
    }
    let columns = predict_columns(model, dataset, &options)?;
    serde_json::to_string(&PredictionPayload {
        columns,
        model_class: prediction_model_class_label(model),
        family: family_link_kind(&model_likelihood_spec(model)).to_string(),
        // The plain dataset predict path returns the model-based credible /
        // predictive band (or no interval at all); the jackknife+ provenance
        // tag is only attached by the dedicated full-conformal predict entry.
        interval_method: None,
    })
    .map_err(|err| format!("failed to serialize prediction payload: {err}"))
}

fn predict_columns(
    model: &FittedModel,
    dataset: EncodedDataset,
    options: &PyPredictOptions,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let col_map = dataset.column_map();
    // Spline-scan saved model (#1030/#1034): replay the exact Gaussian bridge
    // per row (identity link, η == mean). No design reconstruction, no
    // predictor. SE/intervals come from the exact posterior variance.
    if let Some((feature_column, fit)) = model.saved_spline_scan().map_err(String::from)? {
        let col = *col_map.get(feature_column).ok_or_else(|| {
            format!("prediction data is missing the model's feature column '{feature_column}'")
        })?;
        let n = dataset.values.nrows();
        let mut eta = Vec::with_capacity(n);
        let mut se = Vec::with_capacity(n);
        for (i, &x) in dataset.values.column(col).iter().enumerate() {
            let (m, v) = fit
                .predict(x)
                .map_err(|e| format!("spline-scan predict failed at row {i}: {e}"))?;
            eta.push(m);
            se.push(v.max(0.0).sqrt());
        }
        let mut columns = BTreeMap::<String, Vec<f64>>::new();
        columns.insert("linear_predictor".to_string(), eta.clone());
        columns.insert("mean".to_string(), eta.clone());
        if let Some(confidence_level) = options.interval {
            let z = gam::probability::standard_normal_quantile(0.5 + confidence_level * 0.5)?;
            let lower: Vec<f64> = eta.iter().zip(&se).map(|(m, s)| m - z * s).collect();
            let upper: Vec<f64> = eta.iter().zip(&se).map(|(m, s)| m + z * s).collect();
            // Observation (predictive) interval (#1047): the scan IS the exact
            // Gaussian smoothing-spline posterior, so the response-scale
            // predictive variance is `Var(f(x*)) + σ²` where `Var(f(x*)) = se²`
            // is the off-knot posterior variance from the bridge and
            // `fit.sigma2` is the profiled Gaussian observation variance. The
            // confidence band uses `se` alone; the observation band inflates by
            // σ². Identity link means no response-scale transform is needed.
            if options.observation_interval.unwrap_or(false) {
                let obs_se: Vec<f64> = se
                    .iter()
                    .map(|s| (s * s + fit.sigma2).max(0.0).sqrt())
                    .collect();
                let obs_lower: Vec<f64> = eta.iter().zip(&obs_se).map(|(m, s)| m - z * s).collect();
                let obs_upper: Vec<f64> = eta.iter().zip(&obs_se).map(|(m, s)| m + z * s).collect();
                columns.insert("observation_lower".to_string(), obs_lower);
                columns.insert("observation_upper".to_string(), obs_upper);
            }
            columns.insert("std_error".to_string(), se);
            columns.insert("mean_lower".to_string(), lower);
            columns.insert("mean_upper".to_string(), upper);
        }
        return Ok(columns);
    }
    let offset = resolve_offset_column(&dataset, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&dataset, &col_map, model.noise_offset_column.as_deref())?;
    let predict_input = build_predict_input_for_model(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;

    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    // SPEC: the posterior mean `E[g⁻¹(Xβ)]` is *always* the default point
    // estimate (never the plug-in mode `g⁻¹(Xβ̂)`). The integral is only
    // observably distinct from the plug-in when the inverse link is curved
    // over the posterior's support, so `FittedModel::prediction_uses_posterior_mean`
    // — the single predicate shared with the CLI's `gam predict` — selects the
    // posterior-mean path for exactly those models and keeps the cheaper,
    // exact plug-in for effectively-linear ones (identity-link Gaussian, …).
    // Reported point estimates therefore match the CLI default row-for-row.
    let uses_posterior_mean = model.prediction_uses_posterior_mean();
    // Issue #342 — single uncertainty knob: `interval` is the only switch.
    // `Some(level)` means "quantify uncertainty at this coverage and return
    // SE + CI bounds"; `None` means "point predictions only". The earlier
    // overlapping `with_uncertainty` boolean has been removed: it was
    // strictly weaker than `interval` (SE is a strict subset of
    // SE + bounds), and supporting both forced callers to learn two
    // partially-redundant flags. Migration: `with_uncertainty=True` →
    // `interval=0.95`.
    //
    // Issue #398: the point prediction is a property of the model and the
    // inputs, never of whether an interval was requested — `interval` only
    // *adds* std_error/mean_lower/mean_upper columns and never shifts `mean`
    // or `linear_predictor`. That invariant holds on both axes here: the
    // posterior-mean vs plug-in choice is driven solely by the model
    // (`uses_posterior_mean`), and within each axis the interval and
    // no-interval branches report the identical point.
    //
    // User-facing column names follow issue #310: the engine's internal labels
    // ("eta", "effective_se", "effective_variance") are renamed at the FFI
    // boundary to linear_predictor / std_error, and effective_variance is
    // dropped (== std_error ** 2). The internal Rust fields keep their
    // theoretic names because they describe the math object.
    match (options.interval, uses_posterior_mean) {
        (Some(confidence_level), true) => {
            // Curved inverse link + interval: the canonical posterior-mean path
            // returns the η-scale SE and the inverse-link-transformed credible
            // bounds in one pass, on top of the same posterior-mean point as the
            // no-interval branch. The `covariance_mode` and `observation_interval`
            // knobs are threaded through here exactly as the sibling
            // full-uncertainty arm does for the plug-in families: previously this
            // arm ignored both, so binomial intervals silently dropped the
            // observation band (#811) and the smoothing-parameter correction
            // (#812). The posterior-mean *point* stays conditional regardless of
            // mode (issue #398); only the reported uncertainty responds.
            let covariance_mode = parse_covariance_mode(options.covariance_mode.as_deref())?
                .unwrap_or(
                    gam::predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
                );
            let posterior_options = gam::predict::PosteriorMeanOptions {
                confidence_level: Some(confidence_level),
                covariance_mode,
                include_observation_interval: options.observation_interval.unwrap_or(false),
            };
            let prediction = predictor
                .predict_posterior_mean(&predict_input, &fit, &posterior_options)
                .map_err(|err| {
                    format!("posterior-mean prediction with uncertainty failed: {err}")
                })?;
            let (mean_lower, mean_upper) = prediction
                .mean_lower
                .zip(prediction.mean_upper)
                .ok_or_else(|| {
                    "posterior-mean prediction did not return confidence bounds".to_string()
                })?;
            columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
            columns.insert("mean".to_string(), prediction.mean.to_vec());
            columns.insert(
                "std_error".to_string(),
                prediction.eta_standard_error.to_vec(),
            );
            columns.insert("mean_lower".to_string(), mean_lower.to_vec());
            columns.insert("mean_upper".to_string(), mean_upper.to_vec());
            // Observation (prediction) interval: present only when the request
            // was made AND the family exposes a conditional response variance
            // (Binomial `p(1−p)`). Emitting separate columns keeps the standard
            // schema untouched when off and never overwrites the credible bounds.
            if let (Some(obs_lower), Some(obs_upper)) =
                (prediction.observation_lower, prediction.observation_upper)
            {
                columns.insert("observation_lower".to_string(), obs_lower.to_vec());
                columns.insert("observation_upper".to_string(), obs_upper.to_vec());
            }
        }
        (Some(confidence_level), false) => {
            // Effectively-linear model + interval: plug-in == posterior mean, so
            // the delta-method full-uncertainty path reports that same point and
            // only widens the interval for smoothing uncertainty.
            // `apply_bias_correction: false` keeps the point equal to the plain
            // plug-in branch: recentring η by X·H⁻¹Sβ̂ would silently shift `mean`
            // the moment an interval was requested (violating issue #398), is
            // empirically worse against truth, and is inconsistent with the
            // link-wiggle path that never bias-corrects; bias-aware coverage is
            // already supplied by the smoothing-corrected covariance.
            //
            // CLI<->Python parity: `covariance_mode` (default smoothing-preferred,
            // matching the prior hardcode) and `observation_interval` are now
            // user-selectable, mirroring `gam predict --covariance-mode` and the
            // engine's `includeobservation_interval` switch.
            let covariance_mode = parse_covariance_mode(options.covariance_mode.as_deref())?
                .unwrap_or(
                    gam::predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
                );
            let includeobservation_interval = options.observation_interval.unwrap_or(false);
            let uncertainty_options = gam::predict::PredictUncertaintyOptions {
                confidence_level,
                covariance_mode,
                mean_interval_method: gam::predict::MeanIntervalMethod::TransformEta,
                includeobservation_interval,
                apply_bias_correction: false,
                ..gam::predict::PredictUncertaintyOptions::default()
            };
            let prediction = predictor
                .predict_full_uncertainty(&predict_input, &fit, &uncertainty_options)
                .map_err(|err| format!("prediction with uncertainty failed: {err}"))?;
            columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
            columns.insert("mean".to_string(), prediction.mean.to_vec());
            columns.insert(
                "std_error".to_string(),
                prediction.eta_standard_error.to_vec(),
            );
            columns.insert("mean_lower".to_string(), prediction.mean_lower.to_vec());
            columns.insert("mean_upper".to_string(), prediction.mean_upper.to_vec());
            // Observation (prediction) interval: only present when the family
            // and the `observation_interval` request both support it. Emitting
            // it as separate columns keeps the standard schema untouched when
            // off and never overwrites the credible `mean_lower`/`mean_upper`.
            if let (Some(obs_lower), Some(obs_upper)) =
                (prediction.observation_lower, prediction.observation_upper)
            {
                columns.insert("observation_lower".to_string(), obs_lower.to_vec());
                columns.insert("observation_upper".to_string(), obs_upper.to_vec());
            }
        }
        (None, true) => {
            let prediction = predictor
                .predict_posterior_mean(
                    &predict_input,
                    &fit,
                    &gam::predict::PosteriorMeanOptions::point_only(),
                )
                .map_err(|err| format!("posterior-mean prediction failed: {err}"))?;
            columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
            columns.insert("mean".to_string(), prediction.mean.to_vec());
        }
        (None, false) => {
            let prediction = predictor
                .predict_plugin_response(&predict_input)
                .map_err(|err| format!("prediction failed: {err}"))?;
            columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
            columns.insert("mean".to_string(), prediction.mean.to_vec());
        }
    }

    // Issue #365 (secondary defect): location-scale / GAMLSS families fit a
    // per-observation distribution scale (e.g. Gaussian `sigma = exp(noise η)`)
    // that was previously unreachable from Python — `predict_columns` only ever
    // emitted the location block (`mean` / `linear_predictor`), so a user could
    // fit a smooth `noise_formula` but never retrieve the scale function it
    // learned. The scale is a model property independent of the uncertainty
    // knob, so it is emitted in both branches. Families without a response-side
    // scale (single-block GLMs, binomial location-scale, survival, …) return
    // `None` from `predict_noise_scale` and contribute no column, leaving the
    // standard prediction schema untouched.
    if let Some(noise_scale) = predictor
        .predict_noise_scale(&predict_input)
        .map_err(|err| format!("noise-scale prediction failed: {err}"))?
    {
        columns.insert("noise_scale".to_string(), noise_scale.to_vec());
    }

    Ok(columns)
}

/// Build the held-out calibration fold needed by the conformal calibrator: a
/// `PredictInput` over the calibration design (so the model's own predict
/// engine produces `μ̂(x_cal)` and `s(x_cal)` at exactly those points,
/// identically to the test path) and the calibration response `y_cal`. The
/// response column is resolved from the saved formula and must be present in
/// the calibration dataset (calibration is *labeled* held-out data, unlike a
/// predict batch). The fold carries its own design and may be of ANY size,
/// independent of the training set — it is never bound to the training rows.
fn conformal_calibration_fold(
    model: &FittedModel,
    fit: &gam::estimate::UnifiedFitResult,
    calibration: EncodedDataset,
) -> Result<(gam::predict::PredictInput, Array1<f64>), String> {
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "conformal calibration currently supports only standard GAM models; got '{}'",
            prediction_model_class_label(model)
        ));
    }
    let col_map = calibration.column_map();
    let offset = resolve_offset_column(&calibration, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&calibration, &col_map, model.noise_offset_column.as_deref())?;
    let response_name = response_column_name(&model.payload().formula).ok_or_else(|| {
        "conformal calibration: could not resolve the response column from the saved formula"
            .to_string()
    })?;
    let response_col = *col_map.get(&response_name).ok_or_else(|| {
        format!(
            "conformal calibration data must contain the response column '{response_name}' \
             (calibration is held-out labeled data)"
        )
    })?;
    let y = calibration.values.column(response_col).to_owned();
    // Build the calibration-fold predict input the same way the test batch is
    // built, so the predict engine yields μ̂(x_cal) and s(x_cal) from exactly
    // the same source used at test time.
    let cal_input = build_predict_input_for_model(
        model,
        calibration.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let design_cols = cal_input.design.ncols();
    if design_cols != fit.beta.len() {
        return Err(format!(
            "conformal calibration design has {} columns but the fit has {} coefficients",
            design_cols,
            fit.beta.len()
        ));
    }
    Ok((cal_input, y))
}

/// Conformal-calibrated prediction columns. Runs the model-based full-
/// uncertainty predictor on the test `dataset` (honouring `covariance_mode` /
/// `observation_interval`), then replaces the response-scale `mean_lower` /
/// `mean_upper` with the split-conformal interval calibrated from the supplied
/// held-out `calibration` fold at the level in `options.conformal_level`.
fn predict_columns_conformal(
    model: &FittedModel,
    dataset: EncodedDataset,
    calibration: EncodedDataset,
    options: &PyPredictOptions,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let Some(level) = options.conformal_level else {
        return Err("conformal prediction requires conformal_level in (0, 1)".to_string());
    };
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "conformal prediction currently supports only standard GAM models; got '{}'",
            prediction_model_class_label(model)
        ));
    }
    let col_map = dataset.column_map();
    let offset = resolve_offset_column(&dataset, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&dataset, &col_map, model.noise_offset_column.as_deref())?;
    let predict_input = build_predict_input_for_model(
        model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let family = model_likelihood_spec(model);

    let covariance_mode = parse_covariance_mode(options.covariance_mode.as_deref())?
        .unwrap_or(gam::predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred);
    let uncertainty_options = gam::predict::PredictUncertaintyOptions {
        confidence_level: level,
        covariance_mode,
        mean_interval_method: gam::predict::MeanIntervalMethod::TransformEta,
        includeobservation_interval: options.observation_interval.unwrap_or(false),
        apply_bias_correction: false,
        conformal_level: Some(level),
        ..gam::predict::PredictUncertaintyOptions::default()
    };

    let (cal_input, cal_y) = conformal_calibration_fold(model, &fit, calibration)?;
    let calibration_fold = gam::predict::ConformalCalibrationFold {
        input: cal_input,
        y: cal_y.view(),
    };
    let prediction = gam::predict::predict_full_uncertainty_conformal(
        predictor.as_ref(),
        &predict_input,
        &fit,
        &family,
        &uncertainty_options,
        &calibration_fold,
    )
    .map_err(|err| format!("conformal prediction failed: {err}"))?;

    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    columns.insert("linear_predictor".to_string(), prediction.eta.to_vec());
    columns.insert("mean".to_string(), prediction.mean.to_vec());
    columns.insert(
        "std_error".to_string(),
        prediction.eta_standard_error.to_vec(),
    );
    // mean_lower / mean_upper now carry the distribution-free conformal bounds.
    columns.insert("mean_lower".to_string(), prediction.mean_lower.to_vec());
    columns.insert("mean_upper".to_string(), prediction.mean_upper.to_vec());
    if let (Some(obs_lower), Some(obs_upper)) =
        (prediction.observation_lower, prediction.observation_upper)
    {
        columns.insert("observation_lower".to_string(), obs_lower.to_vec());
        columns.insert("observation_upper".to_string(), obs_upper.to_vec());
    }
    Ok(columns)
}

fn predict_table_conformal_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    calibration_headers: Vec<String>,
    calibration_rows: Vec<Vec<String>>,
    conformal_level: f64,
    options_json: Option<&str>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    // Split-conformal calibration is built on the dense predictor + fit_result,
    // neither of which a scan-routed model carries. Point and posterior-interval
    // prediction (the scan-aware predict path) work for these models, so direct
    // users there rather than failing with the cryptic missing-resolved_termspec
    // error (#1046).
    if let Some(scan) = scan_introspection(&model)? {
        return Err(format!(
            "{} is fit by the exact O(n) state-space spline scan, which does not \
             carry the dense predictor split-conformal calibration needs. Use \
             predict(..., interval=<level>) for posterior intervals (scan-aware), \
             or refit with double_penalty=true for conformal intervals.",
            scan_smooth_label(&scan)
        ));
    }
    let mut options = parse_predict_options(options_json)?;
    if !(conformal_level.is_finite() && conformal_level > 0.0 && conformal_level < 1.0) {
        return Err(format!(
            "conformal_level must be in (0, 1), got {conformal_level}"
        ));
    }
    options.conformal_level = Some(conformal_level);
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
    let calibration = dataset_with_model_schema(&model, &calibration_headers, &calibration_rows)?;
    drop(calibration_rows);
    drop(calibration_headers);
    let columns = predict_columns_conformal(&model, dataset, calibration, &options)?;
    serde_json::to_string(&PredictionPayload {
        columns,
        model_class: prediction_model_class_label(&model),
        family: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        interval_method: Some(
            "split-conformal (distribution-free, finite-sample marginal coverage)".to_string(),
        ),
    })
    .map_err(|err| format!("failed to serialize conformal prediction payload: {err}"))
}

/// #1054 Exact Gaussian jackknife+ conformal intervals — no calibration fold.
///
/// Reads the `GaussianJackknifePlusStats` precomputed at fit time (only
/// available for Gaussian-identity, unit-weight, offset-free models without a
/// link wiggle), builds the test design from the saved `resolved_termspec`,
/// and calls `stats.interval(x_*, alpha)` per test row. Returns the same
/// column schema as the model-based predict path so Python can shape-route
/// through the existing machinery.
///
/// Falls back with a clear error when the model is ineligible (non-Gaussian
/// family, scan-routed model, link wiggle, weighted training data, or an older
/// serialised payload that pre-dates the jackknife+ precomputation).
fn predict_table_jackknife_plus_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    conformal_level: f64,
) -> Result<String, String> {
    if !(conformal_level > 0.0 && conformal_level < 1.0) {
        return Err(format!(
            "conformal_level must be in (0, 1), got {conformal_level}"
        ));
    }
    let model = load_model_impl(model_bytes)?;
    // Reject the scan path immediately — it never builds a dense design, so the
    // jackknife+ stats are absent and the termspec-based design reconstruction
    // below would not apply.
    if scan_introspection(&model)
        .map_err(String::from)?
        .is_some()
    {
        return Err(
            "jackknife+ conformal intervals require a penalised-spline (B-spline) model; \
             this model was fit by the exact O(n) state-space scan. Refit with \
             double_penalty=true to obtain the standard model that carries jackknife+ stats."
                .to_string(),
        );
    }
    let stats = model.gaussian_jackknife_plus.as_ref().ok_or_else(|| {
        "jackknife+ conformal intervals require a Gaussian-identity GLM trained without \
         prior weights, offsets, or a link wiggle. This model is ineligible (non-Gaussian \
         family, weighted data, offset, link wiggle, or an older serialised payload). \
         Use Model.predict_conformal(calibration=...) for split-conformal intervals on \
         arbitrary families."
            .to_string()
    })?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "jackknife+ conformal prediction supports only standard GAM models; got '{}'",
            prediction_model_class_label(&model)
        ));
    }
    // Build test design via the frozen resolved_termspec so column ordering
    // and spline knots are identical to the training design the stats were
    // computed from.
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    let col_map = dataset.column_map();
    let spec = gam::survival_predict::resolve_termspec_for_prediction(
        &model.resolved_termspec,
        model.training_headers.as_ref(),
        &col_map,
        "resolved_termspec",
    )?;
    let design = gam::smooth::build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("jackknife+ conformal: failed to build test design: {err}"))?;
    let x_test = design
        .design
        .try_to_dense_by_chunks("jackknife+ conformal test design")?;
    let n_test = x_test.nrows();
    if x_test.ncols() != stats.p() {
        return Err(format!(
            "jackknife+ conformal: test design has {} columns but the stored stats have p={}; \
             the model may need to be refit",
            x_test.ncols(),
            stats.p()
        ));
    }
    // Plug-in mean (= beta-hat @ x_star for the Gaussian-identity model); we
    // read it from the stored stats' beta directly to avoid touching the
    // predictor stack.
    // The jackknife+ interval for the Barber et al. (2021) construction carries
    // the guarantee P(Y_* ∈ Ĉ_α) ≥ 1 − 2α. To deliver ≥conformal_level
    // coverage to the user (so that predict(interval='conformal', conformal_level=0.9)
    // genuinely gives 90% marginal coverage), set α = (1 − conformal_level) / 2.
    // Using α = 1 − conformal_level would yield only 1 − 2(1 − level) = 2·level − 1
    // coverage (80% at level=0.9), mismatching the advertised guarantee.
    let alpha = (1.0 - conformal_level) / 2.0;
    let mut mean_vec = Vec::with_capacity(n_test);
    let mut lower_vec = Vec::with_capacity(n_test);
    let mut upper_vec = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let x_star = x_test.row(i).to_owned();
        let iv = stats
            .interval(&x_star, alpha)
            .map_err(|e| format!("jackknife+ conformal at row {i}: {e}"))?;
        mean_vec.push(x_star.dot(stats.beta()));
        lower_vec.push(iv.lo);
        upper_vec.push(iv.hi);
    }
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    // Identity link: linear_predictor == mean.
    columns.insert("linear_predictor".to_string(), mean_vec.clone());
    columns.insert("mean".to_string(), mean_vec);
    columns.insert("mean_lower".to_string(), lower_vec);
    columns.insert("mean_upper".to_string(), upper_vec);
    serde_json::to_string(&PredictionPayload {
        columns,
        model_class: prediction_model_class_label(&model),
        family: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        interval_method: Some(format!(
            "jackknife+ (distribution-free, finite-sample ≥{:.0}% coverage; \
             Barber et al. 2021)",
            conformal_level * 100.0
        )),
    })
    .map_err(|err| format!("failed to serialize jackknife+ prediction payload: {err}"))
}

/// #1098 EXACT Gaussian full-conformal prediction set — no calibration fold.
///
/// Reads the `ExactFullConformalSubstrate` precomputed at fit time (only
/// available for Gaussian-identity, unit-weight, offset-free models without a
/// link wiggle), rebuilds the test design from the saved `resolved_termspec`,
/// and calls `substrate.interval(x_*, alpha)` per test row — one Cholesky each,
/// zero refits. The exact set is a union of intervals; the returned
/// `mean_lower`/`mean_upper` are its outer envelope (a superset, inheriting the
/// finite-sample coverage). A `frozen_rho_certified` column reports the Layer-3
/// self-diagnostic per row (1.0 accepted / 0.0 refused).
///
/// `alpha = 1 − conformal_level` (the full-conformal set `C_α` has marginal
/// coverage `≥ 1 − α`, so `conformal_level = 1 − α` directly; unlike jackknife+
/// there is no factor of two).
///
/// Falls back with a clear error when the model is ineligible (non-Gaussian
/// family, scan-routed model, link wiggle, weighted training data, or an older
/// serialised payload that pre-dates the substrate).
fn predict_table_full_conformal_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    conformal_level: f64,
) -> Result<String, String> {
    if !(conformal_level > 0.0 && conformal_level < 1.0) {
        return Err(format!(
            "conformal_level must be in (0, 1), got {conformal_level}"
        ));
    }
    let model = load_model_impl(model_bytes)?;
    if scan_introspection(&model)
        .map_err(String::from)?
        .is_some()
    {
        return Err(
            "exact full-conformal intervals require a penalised-spline (B-spline) model; \
             this model was fit by the exact O(n) state-space scan. Refit with \
             double_penalty=true to obtain the standard model that carries the substrate."
                .to_string(),
        );
    }
    let substrate = model.full_conformal.as_ref().ok_or_else(|| {
        "exact full-conformal intervals require a Gaussian-identity GLM trained without \
         prior weights, offsets, or a link wiggle. This model is ineligible (non-Gaussian \
         family, weighted data, offset, link wiggle, or an older serialised payload). \
         Use Model.predict_conformal(calibration=...) for split-conformal intervals on \
         arbitrary families."
            .to_string()
    })?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "exact full-conformal prediction supports only standard GAM models; got '{}'",
            prediction_model_class_label(&model)
        ));
    }
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    let col_map = dataset.column_map();
    let spec = gam::survival_predict::resolve_termspec_for_prediction(
        &model.resolved_termspec,
        model.training_headers.as_ref(),
        &col_map,
        "resolved_termspec",
    )?;
    let design = gam::smooth::build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("full conformal: failed to build test design: {err}"))?;
    let x_test = design
        .design
        .try_to_dense_by_chunks("full conformal test design")?;
    let n_test = x_test.nrows();
    if x_test.ncols() != substrate.p() {
        return Err(format!(
            "full conformal: test design has {} columns but the stored substrate has p={}; \
             the model may need to be refit",
            x_test.ncols(),
            substrate.p()
        ));
    }
    // The full-conformal set C_α covers Y_* with marginal probability ≥ 1 − α,
    // so the user's conformal_level maps directly to α = 1 − conformal_level
    // (no factor-of-two as in the jackknife+ ≥ 1 − 2α guarantee).
    let alpha = 1.0 - conformal_level;
    let mut mean_vec = Vec::with_capacity(n_test);
    let mut lower_vec = Vec::with_capacity(n_test);
    let mut upper_vec = Vec::with_capacity(n_test);
    let mut certified_vec = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let x_star = x_test.row(i).to_owned();
        let iv = substrate
            .interval(&x_star, alpha)
            .map_err(|e| format!("full conformal at row {i}: {e}"))?;
        // Report the envelope centre as the point summary when finite; an
        // unbounded honest envelope reports its finite endpoint.
        let point = if iv.lo.is_finite() && iv.hi.is_finite() {
            0.5 * (iv.lo + iv.hi)
        } else if iv.lo.is_finite() {
            iv.lo
        } else {
            iv.hi
        };
        mean_vec.push(point);
        lower_vec.push(iv.lo);
        upper_vec.push(iv.hi);
        certified_vec.push(if iv.frozen_rho_certified { 1.0 } else { 0.0 });
    }
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    columns.insert("linear_predictor".to_string(), mean_vec.clone());
    columns.insert("mean".to_string(), mean_vec);
    columns.insert("mean_lower".to_string(), lower_vec);
    columns.insert("mean_upper".to_string(), upper_vec);
    columns.insert("frozen_rho_certified".to_string(), certified_vec);
    serde_json::to_string(&PredictionPayload {
        columns,
        model_class: prediction_model_class_label(&model),
        family: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        interval_method: Some(format!(
            "exact full-conformal (distribution-free, finite-sample ≥{:.0}% coverage; \
             #942 Layer 1 + frozen-ρ certificate)",
            conformal_level * 100.0
        )),
    })
    .map_err(|err| format!("failed to serialize full-conformal prediction payload: {err}"))
}

/// Distribution-free EXACT full-conformal prediction intervals — no held-out
/// calibration fold required (#1098 / #942 Layer 1).
///
/// Routes `predict(interval='full_conformal')` for Gaussian-identity models to
/// the `ExactFullConformalSubstrate` precomputed at fit time, giving the EXACT
/// distribution-free set with finite-sample ≥`conformal_level` marginal
/// coverage. Returns the same column JSON as `predict_table` plus a
/// `frozen_rho_certified` column carrying the Layer-3 self-diagnostic.
///
/// Raises a descriptive Python exception for ineligible models (non-Gaussian,
/// weighted, scan-routed, …) directing the user to `predict_conformal`.
#[pyfunction(signature = (model_bytes, headers, rows, conformal_level=0.9))]
fn predict_table_full_conformal(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    conformal_level: f64,
) -> PyResult<String> {
    detach_py_result(py, "predict_table_full_conformal", move || {
        predict_table_full_conformal_impl(&model_bytes, headers, rows, conformal_level)
    })
}

/// Distribution-free jackknife+ conformal prediction intervals — no held-out
/// calibration fold required (#1054 / #942).
///
/// Auto-routes `predict(interval='conformal')` for Gaussian-identity models to
/// the `GaussianJackknifePlusStats` precomputed at fit time, giving
/// finite-sample ≥`conformal_level` marginal coverage (Barber et al. 2021).
/// Returns the same column JSON as `predict_table` (with `linear_predictor`,
/// `mean`, `mean_lower`, `mean_upper`) so the Python shaper is unchanged.
///
/// Raises a descriptive Python exception for ineligible models (non-Gaussian,
/// weighted, scan-routed, …) directing the user to `predict_conformal`.
#[pyfunction(signature = (model_bytes, headers, rows, conformal_level=0.9))]
fn predict_table_jackknife_plus(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    conformal_level: f64,
) -> PyResult<String> {
    detach_py_result(py, "predict_table_jackknife_plus", move || {
        predict_table_jackknife_plus_impl(&model_bytes, headers, rows, conformal_level)
    })
}

/// #1057 Posterior-predictive replicate sampling — `model.sample_replicates`.
///
/// Draws `n_draws` synthetic response vectors from the fitted predictive
/// distribution via `gam::inference::generative`:
///   1. Run the plug-in predictor to get `mean` (response scale).
///   2. Derive the `NoiseModel` from the saved `LikelihoodSpec` + fitted
///      dispersion (`standard_deviation` / `likelihood_scale`).
///   3. Call `sampleobservation_replicates` with a seeded `StdRng`.
///
/// Returns a row-major flat `Vec<f64>` of shape `(n_draws, n_rows)` plus the
/// two dimensions, so the Python side can reshape into a numpy array without
/// a copy. Survives any family supported by `NoiseModel::from_likelihood`
/// (Gaussian, Poisson, Bernoulli, Gamma, Beta, NegBin, Tweedie). Survival /
/// transformation-normal / scan-routed models are rejected early with a clear
/// error — they are not Gauss GLMs and `generativespec_from_predict` does not
/// cover them.
#[pyfunction]
fn generative_replicates(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    n_draws: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let result = py.detach(|| {
        generative_replicates_impl(&model_bytes, headers, rows, n_draws, seed)
    });
    match result {
        Ok((flat, n_rows)) => {
            let arr = ndarray::Array2::<f64>::from_shape_vec((n_draws, n_rows), flat)
                .map_err(|e| py_value_error(format!("generative_replicates reshape: {e}")))?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        }
        Err(e) => Err(py_value_error(e)),
    }
}

fn generative_replicates_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    n_draws: usize,
    seed: u64,
) -> Result<(Vec<f64>, usize), String> {
    use gam::inference::generative::{generativespec_from_predict, sampleobservation_replicates};
    use rand::SeedableRng;
    let model = load_model_impl(model_bytes)?;
    // Only standard GAM models have a dense predictor + UnifiedFitResult.
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "sample_replicates supports only standard GAM models; got '{}'. \
             Use the appropriate posterior sampling method for this model class.",
            prediction_model_class_label(&model)
        ));
    }
    // Scan-routed models: no dense predictor, no family-based noise model.
    if scan_introspection(&model)
        .map_err(String::from)?
        .is_some()
    {
        return Err(
            "sample_replicates is not yet supported for exact O(n) scan models; \
             refit with double_penalty=true to obtain the standard B-spline model."
                .to_string(),
        );
    }
    let family = model_likelihood_spec(&model);
    // Reject families generativespec_from_predict cannot handle (custom /
    // survival-parametric / latent-cloglog). They require a different noise
    // model path not yet covered by the built-in generative engine.
    match &family.response {
        gam::types::ResponseFamily::Gaussian
        | gam::types::ResponseFamily::Binomial
        | gam::types::ResponseFamily::Poisson
        | gam::types::ResponseFamily::NegativeBinomial { .. }
        | gam::types::ResponseFamily::Beta { .. }
        | gam::types::ResponseFamily::Gamma
        | gam::types::ResponseFamily::Tweedie { .. } => {}
        other => {
            return Err(format!(
                "sample_replicates does not yet support the '{}' family; \
                 supported families: gaussian, binomial, poisson, negbin, \
                 beta, gamma, tweedie",
                format!("{other:?}")
            ));
        }
    }
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    let col_map = dataset.column_map();
    let offset = resolve_offset_column(&dataset, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&dataset, &col_map, model.noise_offset_column.as_deref())?;
    let predict_input = build_predict_input_for_model(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let prediction = predictor
        .predict_plugin_response(&predict_input)
        .map_err(|e| format!("generative_replicates: prediction failed: {e}"))?;
    let n_rows = prediction.mean.len();
    // Extract the fitted dispersion through the SINGLE canonical picker that the
    // CLI `gam generate` path also uses. This crate previously carried its own
    // inline copy of the mapping, and its NB arm returned the *seed* theta
    // (`Some(*theta)`) instead of `likelihood_scale.negbin_theta()` — so
    // `Model.sample_replicates` drew Negative-Binomial counts at theta = 1
    // regardless of the fitted overdispersion (the live remnant of #1124 in the
    // Python path). Routing through `gam::generative::family_noise_parameter`
    // keeps the supported families and the interpretation of every dispersion
    // parameter identical across the CLI and Python front-ends — the whole point
    // of unifying the picker — so this class of bug cannot diverge again.
    let gaussian_scale = gam::generative::family_noise_parameter(
        fit.likelihood_scale,
        fit.standard_deviation,
        &family,
    );
    // Build the generative specification (mean + noise model).
    let spec = generativespec_from_predict(prediction, family, gaussian_scale)
    .map_err(|e| format!("generative_replicates: spec error: {e}"))?;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let draws = sampleobservation_replicates(&spec, n_draws, &mut rng)
        .map_err(|e| format!("generative_replicates: sampling failed: {e}"))?;
    Ok((draws.into_raw_vec_and_offset().0, n_rows))
}

fn columns_to_array(columns: BTreeMap<String, Vec<f64>>) -> Result<Array2<f64>, String> {
    let ordered = ordered_prediction_column_values(&columns);
    let n_cols = ordered.len();
    let n_rows = ordered.first().map(|values| values.len()).unwrap_or(0);
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for (j, values) in ordered.into_iter().enumerate() {
        if values.len() != n_rows {
            return Err("prediction columns have inconsistent lengths".to_string());
        }
        for (i, value) in values.into_iter().enumerate() {
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

fn ordered_prediction_column_values(columns: &BTreeMap<String, Vec<f64>>) -> Vec<Vec<f64>> {
    // Single source of truth for the user-facing column order; the numpy
    // (Array2) path and the Python dict path (`ordered_prediction_columns`)
    // must agree, so both read `PREFERRED_PREDICTION_COLUMNS`.
    let mut out = Vec::<Vec<f64>>::new();
    let mut seen = BTreeSet::<&str>::new();
    for key in PREFERRED_PREDICTION_COLUMNS.iter().copied() {
        if let Some(values) = columns.get(key) {
            out.push(values.clone());
            seen.insert(key);
        }
    }
    for (key, values) in columns {
        if !seen.contains(key.as_str()) {
            out.push(values.clone());
        }
    }
    out
}

fn parse_sample_options(options_json: Option<&str>) -> Result<PySampleOptions, String> {
    match options_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str(raw)
            .map_err(|err| format!("failed to parse sample options json: {err}")),
        _ => Ok(PySampleOptions::default()),
    }
}

fn resolve_nuts_config(model: &FittedModel, options: PySampleOptions) -> NutsConfig {
    // Mirror the CLI's adaptive sizing so Python users get sensible defaults
    // without having to think about chain/warmup counts: NUTS samples needed
    // grow with the coefficient count, so we anchor on the saved beta length.
    let n_base_params = model
        .fit_result
        .as_ref()
        .map(|fr| fr.beta.len())
        .unwrap_or(0);
    let adaptive = NutsConfig::for_dimension(n_base_params);
    NutsConfig {
        n_samples: options.samples.unwrap_or(adaptive.n_samples),
        nwarmup: options.warmup.unwrap_or(adaptive.nwarmup),
        n_chains: options.chains.unwrap_or(adaptive.n_chains),
        target_accept: options.target_accept.unwrap_or(adaptive.target_accept),
        seed: options.seed.unwrap_or(adaptive.seed),
    }
}

#[derive(Serialize)]
struct DesignMatrixPayload {
    x_flat: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    family_kind: String,
    model_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    beta: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_flat: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    covariance_n: Option<usize>,
}

#[derive(Deserialize)]
struct DesignMatrixDensePayload {
    x_flat: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
}

fn design_matrix_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_with_model_schema(&model, &headers, &rows)?;
    drop(rows);
    drop(headers);
    design_matrix_dataset_impl(&model, dataset)
}

fn design_matrix_payload_to_dense(raw: &str) -> Result<Array2<f64>, String> {
    let payload: DesignMatrixDensePayload = serde_json::from_str(raw)
        .map_err(|err| format!("failed to parse design matrix payload: {err}"))?;
    let expected_len = payload
        .n_rows
        .checked_mul(payload.n_cols)
        .ok_or_else(|| "design matrix payload shape overflow".to_string())?;
    if payload.x_flat.len() != expected_len {
        return Err(format!(
            "design matrix FFI payload shape mismatch: got {} floats, expected {} * {}",
            payload.x_flat.len(),
            payload.n_rows,
            payload.n_cols
        ));
    }
    Array2::from_shape_vec((payload.n_rows, payload.n_cols), payload.x_flat)
        .map_err(|err| format!("failed to reshape design matrix payload: {err}"))
}

fn design_matrix_array_impl(
    model_bytes: &[u8],
    x: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_from_x_array_with_model_schema(&model, x)?;
    design_matrix_dense(&model, dataset)
}

fn design_matrix_dataset_impl(
    model: &FittedModel,
    dataset: EncodedDataset,
) -> Result<String, String> {
    let dense = design_matrix_dense(model, dataset)?;
    let n_rows = dense.nrows();
    let n_cols = dense.ncols();
    // Row-major flatten matches numpy's default reshape order.
    let x_flat: Vec<f64> = dense.iter().copied().collect();
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let covariance = fit
        .beta_covariance_corrected()
        .or_else(|| fit.beta_covariance());
    let covariance_n = covariance.map(|c| c.nrows());
    let covariance_flat = covariance.map(|c| c.iter().copied().collect::<Vec<_>>());
    let payload = DesignMatrixPayload {
        x_flat,
        n_rows,
        n_cols,
        family_kind: family_link_kind(&model_likelihood_spec(&model)).to_string(),
        model_class: prediction_model_class_label(model),
        beta: Some(fit.beta.to_vec()),
        covariance_flat,
        covariance_n,
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize design matrix payload: {err}"))
}

fn design_matrix_dense(
    model: &FittedModel,
    dataset: EncodedDataset,
) -> Result<Array2<f64>, String> {
    // A scan-routed model never materializes a dense B-spline design — the
    // exact O(n) state-space smoother is the whole point — so there is no model
    // matrix to export. Replace the cryptic "missing resolved_termspec" error
    // with a precise, actionable one (#1046).
    if let Some(scan) = scan_introspection(model)? {
        return Err(format!(
            "{} is fit by the exact O(n) state-space spline scan, which does not \
             build a dense design matrix; design_matrix() is unavailable for it. \
             Refit with double_penalty=true if you need the explicit B-spline \
             model matrix.",
            scan_smooth_label(&scan)
        ));
    }
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "design_matrix currently supports only standard GAM models; got '{}'. \
             For other classes use Model.predict / posterior.predict, which dispatch \
             through the saved-model predictor.",
            prediction_model_class_label(model)
        ));
    }
    if model.has_link_wiggle() {
        return Err(
            "design_matrix does not yet support link-wiggle models because the \
             linear predictor is q0 + B(q0)·theta, not a single X·beta product. \
             Use Model.predict for these models."
                .to_string(),
        );
    }
    let col_map = dataset.column_map();
    let training_headers = model.training_headers.as_ref();
    let spec = gam::survival_predict::resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        &col_map,
        "resolved_termspec",
    )?;
    let design = gam::smooth::build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("failed to build design matrix: {err}"))?;
    let dense = design
        .design
        .try_to_dense_by_chunks("design_matrix prediction design")?;
    append_deployment_extension_columns(
        model.payload(),
        dataset.values.view(),
        &col_map,
        training_headers,
        dense,
    )
    .map_err(|err| err.to_string())
}

#[derive(Serialize)]
struct PosteriorPredictPayload {
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    model_class: String,
    family_kind: String,
}

fn eta_bands_from_matrix(
    eta: ArrayView2<'_, f64>,
    family_kind: &str,
    level: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    posterior_bands::eta_bands_from_matrix(eta, family_kind, level)
}

fn posterior_credible_interval_impl(
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
    level: f64,
) -> Result<Vec<f64>, String> {
    gam::inference::posterior::credible_interval(&samples_flat, n_draws, n_coeffs, level)
}

#[derive(Deserialize)]
struct PosteriorCoefficientNamesRequest {
    coefficient_names: Option<serde_json::Value>,
    n_coeffs: usize,
}

fn coefficient_name_from_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(name) => name.clone(),
        other => other.to_string(),
    }
}

fn default_coefficient_names(n_coeffs: usize) -> Vec<String> {
    (0..n_coeffs).map(|j| format!("beta_{j}")).collect()
}
