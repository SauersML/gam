//! Isometric-log-ratio (ILR) charts and Aitchison compositional geometry, with
//! per-row analytic Jacobians ("jets") for the torch autograd bridge.
//!
//! The centered-log-ratio (CLR), additive-log-ratio (ALR), closure, and simplex
//! Fréchet/log/exp primitives already live in [`super::simplex`]. This module
//! adds the pieces the torch `gamfit.torch.geometry` surface needs on top of
//! them — the ILR chart (Helmert/pivot orthonormal contrast basis), its inverse,
//! the ALR Aitchison Gram, and **value + row-wise Jacobian** kernels for every
//! chart the torch autograd `Function`s differentiate through.
//!
//! ## Why the Jacobians live here
//!
//! The former `torch/geometry.py` computed these charts with elementwise torch
//! ops so torch autograd produced the gradients implicitly. Migrating the value
//! math to Rust would break that tape, so each gradient-carrying chart is paired
//! with a `*_jet` returning the closed-form per-row Jacobian
//! `J[row, out, in] = ∂output_out / ∂input_in`. The torch `Function.backward`
//! contracts `grad_output` with `J` (`einsum('nod,no->nd', J, g)`), reproducing
//! exactly what torch autograd would have computed through the elementwise form.
//!
//! Every Jacobian is transcribed from the closed-form derivative of the chart:
//!
//! * **CLR** `clr(x)_j = ln x_j − mean_k ln x_k` (closure-invariant), so
//!   `∂clr_j/∂x_i = (δ_ij − 1/d)/x_i`.
//! * **ILR** `ilr(x) = clr(x)·V` for the Helmert basis `V` (`(d, d−1)`,
//!   sum-zero columns), so `∂ilr_m/∂x_i = V[i,m]/x_i` (the `Σ_j V[j,m]` term
//!   vanishes because each contrast column sums to zero).
//! * **inverse-ILR** `p = softmax(z·Vᵀ)`, so
//!   `∂p_j/∂z_i = p_j (V[j,i] − Σ_k p_k V[k,i])`.
//! * **ALR** `alr(x)_m = ln x_{k_m} − ln x_ref`, so
//!   `∂alr_m/∂x_i = δ_{i,k_m}/x_{k_m} − δ_{i,ref}/x_ref`.
//! * **inverse-ALR** `p = softmax(l)` with `l_ref = 0`, `l_{k_m} = z_m`, so
//!   `∂p_j/∂z_m = p_j (δ_{j,k_m} − p_{k_m})`.
//! * **sphere exp** `y = cos(r) b + (sin r / r) z'`, `z' = (I − bbᵀ)z`,
//!   `r = ‖z'‖` (already unit, the ambient renorm is a no-op), so
//!   `∂y_j/∂z_i = (z'_i/r)[−sin(r) b_j + f'(r) z'_j] + (sin r / r)(δ_ij − b_j b_i)`
//!   with `f'(r) = cos(r)/r − sin(r)/r²`.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

use super::simplex;

/// Coordinate system for the torch simplex log/exp maps: centered log-ratio
/// (`Clr`, `d`-dim), additive log-ratio (`Alr`, `(d-1)`-dim relative to a
/// reference part), or isometric log-ratio (`Ilr`, `(d-1)`-dim, the Helmert
/// contrast chart that is Euclidean-isometric to Aitchison geometry).
///
/// This mirrors [`simplex::SimplexCoord`] but adds `Ilr` and maps the bare
/// label `"simplex"` to `Ilr` — the torch surface's default and reference-free
/// isometric chart (the NumPy `_response_geometry` surface maps `"simplex"` to
/// `Clr`; that historical convention is preserved on its own FFI).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LogRatioCoord {
    Clr,
    Alr,
    Ilr,
}

/// Parse a torch simplex coordinate label. `"simplex"`/`"ilr"` → ILR (the
/// reference-free isometric chart), `"clr"` → CLR, `"alr"` → ALR.
pub fn parse_log_ratio_coord(coordinates: &str) -> Result<LogRatioCoord, String> {
    match coordinates.to_ascii_lowercase().as_str() {
        "simplex" | "ilr" => Ok(LogRatioCoord::Ilr),
        "clr" => Ok(LogRatioCoord::Clr),
        "alr" => Ok(LogRatioCoord::Alr),
        other => Err(format!(
            "simplex coordinates must be 'ilr', 'clr', or 'alr'; got {other:?}"
        )),
    }
}

/// Wrap a (possibly negative) reference index into `0..d`.
fn resolve_reference(reference: isize, d: usize) -> usize {
    let d_i = d as isize;
    let mut r = reference % d_i;
    if r < 0 {
        r += d_i;
    }
    r as usize
}

/// Helmert (pivot-coordinate) orthonormal contrast basis `V` of shape
/// `(d, d-1)`. Its columns are an orthonormal basis of the sum-zero hyperplane
/// (the CLR subspace); `ilr(x) = clr(x)·V` and `clr = ilr·Vᵀ`.
pub fn helmert_ilr_basis(d: usize) -> Result<Array2<f64>, String> {
    if d < 2 {
        return Err("ILR basis requires at least two parts".to_string());
    }
    let mut v = Array2::<f64>::zeros((d, d - 1));
    for i in 1..d {
        let scale = ((i as f64) / (i as f64 + 1.0)).sqrt();
        let inv_i = 1.0 / (i as f64);
        for k in 0..i {
            v[[k, i - 1]] = scale * inv_i;
        }
        v[[i, i - 1]] = -scale;
    }
    Ok(v)
}

/// Isometric log-ratio coordinates: `ilr(x) = clr(x)·V` for the Helmert basis
/// `V`, mapping a `d`-part composition to `d-1` Euclidean coordinates whose
/// Euclidean distance equals the Aitchison distance on the simplex.
pub fn ilr(values: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let clr = simplex::clr(values)?;
    let d = clr.ncols();
    let v = helmert_ilr_basis(d)?;
    Ok(clr.dot(&v))
}

/// Inverse ILR: map `(d-1)` ILR coordinates back to the simplex via
/// `p = softmax(z·Vᵀ)` (numerically stabilized by the row max).
pub fn inverse_ilr(coords: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let (n, dm1) = coords.dim();
    if dm1 == 0 {
        return Err("ILR coordinates require at least one column".to_string());
    }
    if let Some(((row, col), value)) = coords.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "ILR coordinates must contain only finite values; got {value} at ({row}, {col})"
        ));
    }
    let d = dm1 + 1;
    let v = helmert_ilr_basis(d)?;
    let clr = coords.dot(&v.t());
    let mut out = Array2::<f64>::zeros((n, d));
    for row in 0..n {
        let mut max_v = f64::NEG_INFINITY;
        for col in 0..d {
            if clr[[row, col]] > max_v {
                max_v = clr[[row, col]];
            }
        }
        let mut total = 0.0_f64;
        for col in 0..d {
            let e = (clr[[row, col]] - max_v).exp();
            out[[row, col]] = e;
            total += e;
        }
        for col in 0..d {
            out[[row, col]] /= total;
        }
    }
    Ok(out)
}

/// Aitchison Gram matrix `G = I_{d-1} − (1/d)·11ᵀ` for ALR coordinates. The ALR
/// chart is NOT isometric to Aitchison geometry: the Aitchison inner product in
/// ALR coordinates is `⟨u, v⟩ = uᵀ G v` with this `(d-1)×(d-1)` Gram (`G = I`
/// only after the ILR reparameterization).
pub fn aitchison_metric(d: usize) -> Result<Array2<f64>, String> {
    if d < 2 {
        return Err("Aitchison metric requires at least two parts".to_string());
    }
    let inv = 1.0 / (d as f64);
    let mut g = Array2::<f64>::zeros((d - 1, d - 1));
    for i in 0..d - 1 {
        for j in 0..d - 1 {
            let kron = if i == j { 1.0 } else { 0.0 };
            g[[i, j]] = kron - inv;
        }
    }
    Ok(g)
}

// ─────────────────────────── chart jets (value + Jacobian) ──────────────────
//
// Every jet returns `(value, jac)` where `jac[row, out, in] = ∂value_out /
// ∂input_in` for the corresponding row. Positivity/finiteness are validated by
// the reused [`super::simplex`] value kernels, so the reciprocals below are safe.

/// CLR value and per-row Jacobian `J[row,j,i] = (δ_ij − 1/d)/x_i`.
pub fn clr_jet(values: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
    let value = simplex::clr(values)?;
    let (n, d) = values.dim();
    let inv = 1.0 / (d as f64);
    let mut jac = Array3::<f64>::zeros((n, d, d));
    for row in 0..n {
        for i in 0..d {
            let recip = 1.0 / values[[row, i]];
            for j in 0..d {
                let kron = if i == j { 1.0 } else { 0.0 };
                jac[[row, j, i]] = (kron - inv) * recip;
            }
        }
    }
    Ok((value, jac))
}

/// ILR value and per-row Jacobian `J[row,m,i] = V[i,m]/x_i`.
pub fn ilr_jet(values: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>), String> {
    let value = simplex::clr(values)?;
    let d = value.ncols();
    let v = helmert_ilr_basis(d)?;
    let ilr_value = value.dot(&v);
    let (n, _) = values.dim();
    let dm1 = d - 1;
    let mut jac = Array3::<f64>::zeros((n, dm1, d));
    for row in 0..n {
        for i in 0..d {
            let recip = 1.0 / values[[row, i]];
            for m in 0..dm1 {
                jac[[row, m, i]] = v[[i, m]] * recip;
            }
        }
    }
    Ok((ilr_value, jac))
}

/// inverse-ILR value and per-row Jacobian
/// `J[row,j,i] = p_j (V[j,i] − Σ_k p_k V[k,i])`.
pub fn inverse_ilr_jet(
    coords: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    let value = inverse_ilr(coords)?;
    let (n, dm1) = coords.dim();
    let d = dm1 + 1;
    let v = helmert_ilr_basis(d)?;
    let mut jac = Array3::<f64>::zeros((n, d, dm1));
    for row in 0..n {
        for i in 0..dm1 {
            let mut wbar = 0.0_f64;
            for k in 0..d {
                wbar += value[[row, k]] * v[[k, i]];
            }
            for j in 0..d {
                jac[[row, j, i]] = value[[row, j]] * (v[[j, i]] - wbar);
            }
        }
    }
    Ok((value, jac))
}

/// ALR value and per-row Jacobian
/// `J[row,m,i] = δ_{i,k_m}/x_{k_m} − δ_{i,ref}/x_ref`.
pub fn alr_jet(
    values: ArrayView2<'_, f64>,
    reference: isize,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    let value = simplex::alr(values, reference)?;
    let (n, d) = values.dim();
    let ref_idx = resolve_reference(reference, d);
    let keep: Vec<usize> = (0..d).filter(|&c| c != ref_idx).collect();
    let dm1 = d - 1;
    let mut jac = Array3::<f64>::zeros((n, dm1, d));
    for row in 0..n {
        let recip_ref = 1.0 / values[[row, ref_idx]];
        for m in 0..dm1 {
            let km = keep[m];
            jac[[row, m, km]] += 1.0 / values[[row, km]];
            jac[[row, m, ref_idx]] -= recip_ref;
        }
    }
    Ok((value, jac))
}

/// inverse-ALR value and per-row Jacobian
/// `J[row,j,m] = p_j (δ_{j,k_m} − p_{k_m})`.
pub fn inverse_alr_jet(
    coords: ArrayView2<'_, f64>,
    reference: isize,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    let value = simplex::inverse_alr(coords, reference)?;
    let (n, dm1) = coords.dim();
    let d = dm1 + 1;
    let ref_idx = resolve_reference(reference, d);
    let keep: Vec<usize> = (0..d).filter(|&c| c != ref_idx).collect();
    let mut jac = Array3::<f64>::zeros((n, d, dm1));
    for row in 0..n {
        for m in 0..dm1 {
            let km = keep[m];
            let pkm = value[[row, km]];
            for j in 0..d {
                let kron = if j == km { 1.0 } else { 0.0 };
                jac[[row, j, m]] = value[[row, j]] * (kron - pkm);
            }
        }
    }
    Ok((value, jac))
}

/// Reshape a base point vector into a `(1, d)` row for the chart kernels.
fn base_row(base: ArrayView1<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((1, base.len()), |(_, j)| base[j])
}

/// Subtract a single chart row `base_chart[0, ·]` from every row of `value`.
fn subtract_base_row(mut value: Array2<f64>, base_chart: &Array2<f64>) -> Array2<f64> {
    let (n, m) = value.dim();
    for row in 0..n {
        for col in 0..m {
            value[[row, col]] -= base_chart[[0, col]];
        }
    }
    value
}

/// Simplex log map at `base` in the chosen chart, returning the tangent value
/// and its per-row Jacobian **with respect to `values`** (the base point is
/// treated as a constant, matching the torch autograd `Function` that only
/// differentiates the primary tensor argument).
pub fn simplex_log_map_jet(
    values: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
    coord: LogRatioCoord,
    reference: isize,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    if values.ncols() != base.len() {
        return Err("simplex values and base point have different dimensions".to_string());
    }
    let base2 = base_row(base);
    match coord {
        LogRatioCoord::Clr => {
            let (value, jac) = clr_jet(values)?;
            let base_clr = simplex::clr(base2.view())?;
            Ok((subtract_base_row(value, &base_clr), jac))
        }
        LogRatioCoord::Alr => {
            let (value, jac) = alr_jet(values, reference)?;
            let base_alr = simplex::alr(base2.view(), reference)?;
            Ok((subtract_base_row(value, &base_alr), jac))
        }
        LogRatioCoord::Ilr => {
            let (value, jac) = ilr_jet(values)?;
            let base_ilr = ilr(base2.view())?;
            Ok((subtract_base_row(value, &base_ilr), jac))
        }
    }
}

/// Simplex exp map from tangent coordinates back to the simplex at `base`,
/// returning the composition value and its per-row Jacobian **with respect to
/// `tangent`** (the base point is a constant). Inverts [`simplex_log_map_jet`].
pub fn simplex_exp_map_jet(
    tangent: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
    coord: LogRatioCoord,
    reference: isize,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    if let Some(((row, col), value)) = tangent.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(format!(
            "simplex exp map tangent must contain only finite values; got {value} at ({row}, {col})"
        ));
    }
    let base2 = base_row(base);
    let base_comp = simplex::closure(base2.view())?;
    let d = base_comp.ncols();
    let (n, tcols) = tangent.dim();
    match coord {
        LogRatioCoord::Clr => {
            if tcols != d {
                return Err("CLR tangent dimension must equal simplex dimension".to_string());
            }
            // Base must be strictly positive to take its log.
            for &value in base_comp.iter() {
                if value <= 0.0 {
                    return Err("simplex exp map require strictly positive simplex values".to_string());
                }
            }
            let mut log_base = Array1::<f64>::zeros(d);
            for col in 0..d {
                log_base[col] = base_comp[[0, col]].ln();
            }
            let mut value = Array2::<f64>::zeros((n, d));
            let mut jac = Array3::<f64>::zeros((n, d, d));
            for row in 0..n {
                let mut max_v = f64::NEG_INFINITY;
                for col in 0..d {
                    let lg = log_base[col] + tangent[[row, col]];
                    value[[row, col]] = lg;
                    if lg > max_v {
                        max_v = lg;
                    }
                }
                let mut total = 0.0_f64;
                for col in 0..d {
                    let e = (value[[row, col]] - max_v).exp();
                    value[[row, col]] = e;
                    total += e;
                }
                for col in 0..d {
                    value[[row, col]] /= total;
                }
                // softmax Jacobian J[j,i] = p_j (δ_ij − p_i)
                for i in 0..d {
                    let pi = value[[row, i]];
                    for j in 0..d {
                        let kron = if i == j { 1.0 } else { 0.0 };
                        jac[[row, j, i]] = value[[row, j]] * (kron - pi);
                    }
                }
            }
            Ok((value, jac))
        }
        LogRatioCoord::Alr => {
            if tcols + 1 != d {
                return Err("ALR tangent dimension must be simplex dimension minus one".to_string());
            }
            let base_alr = simplex::alr(base2.view(), reference)?;
            let dm1 = d - 1;
            let mut shifted = Array2::<f64>::zeros((n, dm1));
            for row in 0..n {
                for col in 0..dm1 {
                    shifted[[row, col]] = base_alr[[0, col]] + tangent[[row, col]];
                }
            }
            inverse_alr_jet(shifted.view(), reference)
        }
        LogRatioCoord::Ilr => {
            if tcols + 1 != d {
                return Err("ILR tangent dimension must be simplex dimension minus one".to_string());
            }
            let base_ilr = ilr(base2.view())?;
            let dm1 = d - 1;
            let mut shifted = Array2::<f64>::zeros((n, dm1));
            for row in 0..n {
                for col in 0..dm1 {
                    shifted[[row, col]] = base_ilr[[0, col]] + tangent[[row, col]];
                }
            }
            inverse_ilr_jet(shifted.view())
        }
    }
}

/// Normalize a base point onto the unit sphere, erroring on a zero-norm base.
fn normalize_sphere_base(base: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
    let mut norm_sq = 0.0_f64;
    for &v in base.iter() {
        if !v.is_finite() {
            return Err("spherical base point must contain only finite values".to_string());
        }
        norm_sq += v * v;
    }
    let norm = norm_sq.sqrt();
    if !(norm > 0.0) {
        return Err("spherical base point must have non-zero norm".to_string());
    }
    Ok(base.mapv(|v| v / norm))
}

/// Sphere exp map value (via [`super::sphere::response_sphere_exp_map`]) and its
/// per-row Jacobian **with respect to `tangent`**. The curved geodesic step is
/// already unit-norm (the ambient renormalization is a numerical no-op), so the
/// analytic Jacobian is
/// `∂y_j/∂z_i = (z'_i/r)[−sin(r) b_j + f'(r) z'_j] + (sin r / r)(δ_ij − b_j b_i)`
/// with `z' = (I − bbᵀ)z`, `r = ‖z'‖`, `f'(r) = cos(r)/r − sin(r)/r²`. Near
/// `r → 0` the tangent-projection limit `∂y_j/∂z_i → δ_ij − b_j b_i` is used.
pub fn sphere_exp_map_jet(
    tangent: ArrayView2<'_, f64>,
    base: ArrayView1<'_, f64>,
) -> Result<(Array2<f64>, Array3<f64>), String> {
    let value = super::sphere::response_sphere_exp_map(tangent, base)?;
    let b = normalize_sphere_base(base)?;
    let (n, d) = tangent.dim();
    if d != b.len() {
        return Err("spherical tangent and base point have different dimensions".to_string());
    }
    let mut jac = Array3::<f64>::zeros((n, d, d));
    for row in 0..n {
        // Project tangent orthogonal to the base: z' = z − (z·b) b.
        let mut z_dot_b = 0.0_f64;
        for c in 0..d {
            z_dot_b += tangent[[row, c]] * b[c];
        }
        let mut zp = vec![0.0_f64; d];
        let mut r_sq = 0.0_f64;
        for c in 0..d {
            let v = tangent[[row, c]] - z_dot_b * b[c];
            zp[c] = v;
            r_sq += v * v;
        }
        let r = r_sq.sqrt();
        if r < 1.0e-12 {
            // z' ≈ 0: the geodesic step reduces to the tangent projection P.
            for j in 0..d {
                for i in 0..d {
                    let kron = if i == j { 1.0 } else { 0.0 };
                    jac[[row, j, i]] = kron - b[j] * b[i];
                }
            }
        } else {
            let sin_r = r.sin();
            let cos_r = r.cos();
            let f = sin_r / r;
            let fp = cos_r / r - sin_r / (r * r);
            let inv_r = 1.0 / r;
            for j in 0..d {
                let head = -sin_r * b[j] + fp * zp[j];
                for i in 0..d {
                    let kron = if i == j { 1.0 } else { 0.0 };
                    jac[[row, j, i]] = zp[i] * inv_r * head + f * (kron - b[j] * b[i]);
                }
            }
        }
    }
    Ok((value, jac))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn norm_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let mut s = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            s += (x - y) * (x - y);
        }
        s.sqrt()
    }

    // ── ILR forward: isometry to Aitchison (CLR) distance (issue #626) ────────

    #[test]
    fn ilr_distance_equals_clr_distance() {
        let x = array![[0.1_f64, 0.2, 0.3, 0.4]];
        let y = array![[0.4_f64, 0.3, 0.2, 0.1]];
        let zx = ilr(x.view()).unwrap();
        let zy = ilr(y.view()).unwrap();
        let cx = simplex::clr(x.view()).unwrap();
        let cy = simplex::clr(y.view()).unwrap();
        let ilr_d = norm_diff(&zx, &zy);
        let clr_d = norm_diff(&cx, &cy);
        assert!(
            (ilr_d - clr_d).abs() < 1e-12,
            "ILR must be isometric to CLR/Aitchison: {ilr_d} vs {clr_d}"
        );
        assert!(ilr_d > 1e-3);
    }

    #[test]
    fn ilr_has_d_minus_one_columns() {
        let x = array![[1.0_f64, 2.0, 3.0, 4.0, 5.0]];
        let z = ilr(x.view()).unwrap();
        assert_eq!(z.ncols(), 4);
    }

    // ── inverse_ilr round-trip ────────────────────────────────────────────────

    #[test]
    fn inverse_ilr_recovers_closed_composition() {
        let x = array![[0.2_f64, 0.5, 0.3], [0.1, 0.1, 0.8]];
        let closed = simplex::closure(x.view()).unwrap();
        let recovered = inverse_ilr(ilr(x.view()).unwrap().view()).unwrap();
        assert!(
            norm_diff(&recovered, &closed) < 1e-12,
            "inverse_ilr(ilr(x)) must recover the closed composition"
        );
    }

    // ── Aitchison Gram known value (D = 3) ────────────────────────────────────

    #[test]
    fn aitchison_metric_d3_is_known_non_identity() {
        let g = aitchison_metric(3).unwrap();
        let expected = array![[2.0 / 3.0, -1.0 / 3.0], [-1.0 / 3.0, 2.0 / 3.0]];
        assert!(norm_diff(&g, &expected) < 1e-12);
    }

    // ── Jacobian finite-difference checks ─────────────────────────────────────
    //
    // Each hand-derived jet is validated against a central finite difference of
    // the corresponding value kernel. A passing FD check proves the transcribed
    // closed-form derivative is correct to O(h²).

    const FD_H: f64 = 1e-6;

    /// Central-difference the (single-row) map `f` at `x` and compare against the
    /// analytic per-row Jacobian `jac[out, in]`.
    fn check_jac_fd<F>(x: &Array2<f64>, jac: &Array3<f64>, mut f: F)
    where
        F: FnMut(&Array2<f64>) -> Array2<f64>,
    {
        let din = x.ncols();
        let base = f(x);
        let dout = base.ncols();
        for i in 0..din {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[[0, i]] += FD_H;
            xm[[0, i]] -= FD_H;
            let fp = f(&xp);
            let fm = f(&xm);
            for o in 0..dout {
                let fd = (fp[[0, o]] - fm[[0, o]]) / (2.0 * FD_H);
                let an = jac[[0, o, i]];
                assert!(
                    (fd - an).abs() < 1e-6,
                    "jac[{o},{i}] analytic {an} vs FD {fd}"
                );
            }
        }
    }

    #[test]
    fn clr_jet_matches_finite_difference() {
        let x = array![[0.2_f64, 0.5, 0.3, 0.9]];
        let (_, jac) = clr_jet(x.view()).unwrap();
        check_jac_fd(&x, &jac, |v| simplex::clr(v.view()).unwrap());
    }

    #[test]
    fn ilr_jet_matches_finite_difference() {
        let x = array![[0.2_f64, 0.5, 0.3, 0.9]];
        let (_, jac) = ilr_jet(x.view()).unwrap();
        check_jac_fd(&x, &jac, |v| ilr(v.view()).unwrap());
    }

    #[test]
    fn inverse_ilr_jet_matches_finite_difference() {
        let z = array![[0.4_f64, -0.2, 0.7]];
        let (_, jac) = inverse_ilr_jet(z.view()).unwrap();
        check_jac_fd(&z, &jac, |v| inverse_ilr(v.view()).unwrap());
    }

    #[test]
    fn alr_jet_matches_finite_difference() {
        let x = array![[0.2_f64, 0.5, 0.3, 0.9]];
        for reference in [-1_isize, 0, 1, 2] {
            let (_, jac) = alr_jet(x.view(), reference).unwrap();
            check_jac_fd(&x, &jac, |v| simplex::alr(v.view(), reference).unwrap());
        }
    }

    #[test]
    fn inverse_alr_jet_matches_finite_difference() {
        let z = array![[0.4_f64, -0.2, 0.7]];
        for reference in [-1_isize, 0, 1] {
            let (_, jac) = inverse_alr_jet(z.view(), reference).unwrap();
            check_jac_fd(&z, &jac, |v| simplex::inverse_alr(v.view(), reference).unwrap());
        }
    }

    #[test]
    fn simplex_log_map_jet_matches_finite_difference_all_charts() {
        let x = array![[0.15_f64, 0.35, 0.2, 0.3]];
        let base = array![0.25_f64, 0.25, 0.25, 0.25];
        for coord in [LogRatioCoord::Clr, LogRatioCoord::Alr, LogRatioCoord::Ilr] {
            let (_, jac) = simplex_log_map_jet(x.view(), base.view(), coord, -1).unwrap();
            check_jac_fd(&x, &jac, |v| {
                simplex_log_map_jet(v.view(), base.view(), coord, -1)
                    .unwrap()
                    .0
            });
        }
    }

    #[test]
    fn simplex_exp_map_jet_matches_finite_difference_all_charts() {
        let base = array![0.25_f64, 0.35, 0.15, 0.25];
        let tangent_full = array![[0.1_f64, -0.05, 0.02, -0.07]]; // CLR (d cols)
        let tangent_red = array![[0.1_f64, -0.05, 0.02]]; // ILR/ALR (d-1 cols)
        let (_, jac) = simplex_exp_map_jet(tangent_full.view(), base.view(), LogRatioCoord::Clr, -1)
            .unwrap();
        check_jac_fd(&tangent_full, &jac, |v| {
            simplex_exp_map_jet(v.view(), base.view(), LogRatioCoord::Clr, -1)
                .unwrap()
                .0
        });
        for coord in [LogRatioCoord::Alr, LogRatioCoord::Ilr] {
            let (_, jac) =
                simplex_exp_map_jet(tangent_red.view(), base.view(), coord, -1).unwrap();
            check_jac_fd(&tangent_red, &jac, |v| {
                simplex_exp_map_jet(v.view(), base.view(), coord, -1)
                    .unwrap()
                    .0
            });
        }
    }

    #[test]
    fn simplex_log_exp_jet_round_trip_ilr() {
        let x = array![[0.15_f64, 0.35, 0.2, 0.3]];
        let base = array![0.25_f64, 0.25, 0.25, 0.25];
        let (tangent, _) =
            simplex_log_map_jet(x.view(), base.view(), LogRatioCoord::Ilr, -1).unwrap();
        let (recovered, _) =
            simplex_exp_map_jet(tangent.view(), base.view(), LogRatioCoord::Ilr, -1).unwrap();
        let closed = simplex::closure(x.view()).unwrap();
        assert!(
            norm_diff(&recovered, &closed) < 1e-12,
            "ILR log/exp jet round-trip must recover the closed input"
        );
    }

    #[test]
    fn sphere_exp_map_jet_matches_finite_difference() {
        let base = array![0.0_f64, 0.0, 1.0];
        let tangent = array![[0.05_f64, -0.03, 0.0]];
        let (_, jac) = sphere_exp_map_jet(tangent.view(), base.view()).unwrap();
        check_jac_fd(&tangent, &jac, |v| {
            super::super::sphere::response_sphere_exp_map(v.view(), base.view()).unwrap()
        });
    }

    /// Pinned-output parity with the pre-migration Python twins.
    ///
    /// The `expected_*` literals below are the outputs of the ORIGINAL
    /// pure-torch `gamfit/torch/geometry.py` (commit d4aa7b85e~1, f64) on these
    /// deterministic fixtures, generated before the torch surface was reduced
    /// to FFI marshalling. Where a NumPy twin existed
    /// (`gamfit._response_geometry`, already Rust-routed) the generator also
    /// asserted torch-vs-numpy agreement at ≤ 4.5e-16, so one set of literals
    /// pins BOTH pre-migration Python surfaces. Tolerance 1e-12.
    #[test]
    fn pinned_python_twin_parity_fixtures() {
        let x = array![
            [0.2_f64, 0.5, 0.3],
            [0.1, 0.1, 0.8],
            [0.25, 0.4, 0.35]
        ];
        let base = array![0.25_f64, 0.35, 0.40];
        let w = array![0.2_f64, 0.3, 0.5];
        let z2 = array![[0.4_f64, -0.2], [0.1, 0.7], [-0.3, 0.05]];
        let z3 = array![
            [0.1_f64, -0.05, -0.05],
            [0.2, 0.1, -0.3],
            [0.0, 0.05, -0.05]
        ];
        let sphere_base = array![0.0_f64, 0.0, 1.0];
        let sphere_tangent = array![[0.05_f64, -0.03, 0.0], [0.3, 0.4, 0.0], [-0.2, 0.1, 0.0]];
        let sphere_values = array![[0.6_f64, 0.8, 0.0], [0.0, 0.6, 0.8], [1.0, 0.0, 0.0]];

        let expected_clr = array![
            [-0.44058527999410635_f64, 0.47570545188004865, -0.035120171885942186],
            [-0.6931471805599452, -0.6931471805599452, 1.3862943611198906],
            [-0.2688252886223159, 0.20117834062341966, 0.06764694799889681]
        ];
        let expected_ilr = array![
            [-0.6479153900465996_f64, 0.0430132503996988],
            [-2.3655649367851632e-17, -1.6978569090206654],
            [-0.3323427534219476, -0.08285025262694215]
        ];
        let expected_inv_ilr_roundtrip = array![
            [0.2_f64, 0.5, 0.29999999999999993],
            [0.10000000000000002, 0.10000000000000002, 0.8],
            [0.25, 0.4, 0.3499999999999999]
        ];
        let expected_alr = array![
            [-0.4054651081081643_f64, 0.5108256237659907],
            [-2.0794415416798357, -2.0794415416798357],
            [-0.3364722366212129, 0.13353139262452277]
        ];
        let expected_inv_alr = array![
            [0.45062670595568977_f64, 0.24730917976320385, 0.3020641142811064],
            [0.2683154674734019, 0.48890265771885366, 0.24278187480774446],
            [0.2653275510048439, 0.37651771737869627, 0.3581547316164598]
        ];
        let expected_gram = array![
            [0.6666666666666667_f64, -0.3333333333333333],
            [-0.3333333333333333, 0.6666666666666667]
        ];
        let expected_frechet =
            array![0.20350911397742996_f64, 0.3091945687314347, 0.4872963172911353];
        let expected_log_ilr = array![
            [-0.4099935898507355_f64, 0.28940539131330195],
            [0.2379218001958641, -1.4514647681070623],
            [-0.09442095322608346, 0.163541888286661]
        ];
        let expected_log_clr = array![
            [-0.1717599913717902_f64, 0.40805850388115206, -0.23629851250936162],
            [-0.42432189193762904, -0.7607941285588418, 1.185116020496471],
            [2.220446049250313e-16, 0.13353139262452307, -0.13353139262452263]
        ];
        let expected_log_alr = array![
            [0.0645385211375713_f64, 0.6443570163905135],
            [-1.6094379124341, -1.945910149055313],
            [0.13353139262452268, 0.2670627852490455]
        ];
        let expected_exp_ilr = array![
            [0.29979044135614885_f64, 0.23838106660410166, 0.4618284920397494],
            [0.351135641326721, 0.4267607158491462, 0.2221036428241327],
            [0.19998199702310523, 0.42793172119833767, 0.3720862817785571]
        ];
        let expected_exp_clr = array![
            [0.2791639875514708_f64, 0.3363901391426469, 0.3844458733058823],
            [0.30890688767827745, 0.3913147149244714, 0.29977839739725126],
            [0.25039144858679213, 0.36852100975062485, 0.381087541662583]
        ];
        let expected_exp_alr = array![
            [0.35200752444440936_f64, 0.27046015557084196, 0.3775323199847487],
            [0.20005176581886308, 0.5103253169698176, 0.2896229172113192],
            [0.19430799370114754, 0.38603063561098605, 0.4196613706878664]
        ];
        let expected_sphere_log = array![
            [0.9424777960769379_f64, 1.2566370614359172, 0.0],
            [0.0, 0.6435011087932844, 0.0],
            [1.5707963267948966, 0.0, 0.0]
        ];
        let expected_sphere_exp = array![
            [0.04997167148294343_f64, -0.029983002889766058, 0.9983004816120811],
            [0.2876553231625218, 0.3835404308833624, 0.8775825618903728],
            [-0.19833749504312567, 0.09916874752156284, 0.9751039932104795]
        ];

        let tol = 1e-12_f64;
        let assert_close = |label: &str, got: &Array2<f64>, want: &Array2<f64>| {
            assert_eq!(got.dim(), want.dim(), "{label}: shape mismatch");
            let mut max_d = 0.0_f64;
            for (g, w) in got.iter().zip(want.iter()) {
                let d = (g - w).abs();
                if d > max_d {
                    max_d = d;
                }
            }
            assert!(
                max_d <= tol,
                "{label}: max |rust − pinned-python| = {max_d} exceeds {tol}"
            );
        };

        assert_close("clr", &simplex::clr(x.view()).unwrap(), &expected_clr);
        let ilr_out = ilr(x.view()).unwrap();
        assert_close("ilr", &ilr_out, &expected_ilr);
        assert_close(
            "inverse_ilr(ilr(x))",
            &inverse_ilr(ilr_out.view()).unwrap(),
            &expected_inv_ilr_roundtrip,
        );
        assert_close("alr", &simplex::alr(x.view(), -1).unwrap(), &expected_alr);
        assert_close(
            "inverse_alr",
            &simplex::inverse_alr(z2.view(), -1).unwrap(),
            &expected_inv_alr,
        );
        assert_close(
            "aitchison_metric",
            &aitchison_metric(3).unwrap(),
            &expected_gram,
        );
        let frechet = simplex::simplex_frechet_mean(x.view(), Some(w.view())).unwrap();
        for (col, want) in expected_frechet.iter().enumerate() {
            assert!(
                (frechet[col] - want).abs() <= tol,
                "frechet[{col}]: {} vs pinned {want}",
                frechet[col]
            );
        }
        for (label, coord, want) in [
            ("log_map ilr", LogRatioCoord::Ilr, &expected_log_ilr),
            ("log_map clr", LogRatioCoord::Clr, &expected_log_clr),
            ("log_map alr", LogRatioCoord::Alr, &expected_log_alr),
        ] {
            let (value, _) = simplex_log_map_jet(x.view(), base.view(), coord, -1).unwrap();
            assert_close(label, &value, want);
        }
        for (label, coord, tangent, want) in [
            ("exp_map ilr", LogRatioCoord::Ilr, &z2, &expected_exp_ilr),
            ("exp_map clr", LogRatioCoord::Clr, &z3, &expected_exp_clr),
            ("exp_map alr", LogRatioCoord::Alr, &z2, &expected_exp_alr),
        ] {
            let (value, _) = simplex_exp_map_jet(tangent.view(), base.view(), coord, -1).unwrap();
            assert_close(label, &value, want);
        }
        assert_close(
            "sphere_log_map",
            &super::super::sphere::response_sphere_log_map(
                sphere_values.view(),
                sphere_base.view(),
            )
            .unwrap(),
            &expected_sphere_log,
        );
        let (sphere_exp, _) =
            sphere_exp_map_jet(sphere_tangent.view(), sphere_base.view()).unwrap();
        assert_close("sphere_exp_map", &sphere_exp, &expected_sphere_exp);
    }

    #[test]
    fn parse_log_ratio_coord_maps_simplex_to_ilr() {
        assert_eq!(parse_log_ratio_coord("simplex").unwrap(), LogRatioCoord::Ilr);
        assert_eq!(parse_log_ratio_coord("ilr").unwrap(), LogRatioCoord::Ilr);
        assert_eq!(parse_log_ratio_coord("clr").unwrap(), LogRatioCoord::Clr);
        assert_eq!(parse_log_ratio_coord("ALR").unwrap(), LogRatioCoord::Alr);
        assert!(parse_log_ratio_coord("pca").is_err());
    }
}
