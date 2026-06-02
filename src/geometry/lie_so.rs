//! Special-orthogonal group representations `SO(2)` / `SO(3)` and their tangent
//! (Jacobian-vector-product) maps, used by the equivariant-term machinery.
//!
//! `SO(2)` is parameterised by a scalar angle `θ` → 2×2 rotation; `SO(3)` by a
//! body-frame rotation vector `ω ∈ ℝ³` via the closed-form Rodrigues exponential
//! `exp([ω]×)`. The JVPs are the exact directional derivatives of those maps.
//! This is the single source of truth shared by the CLI/core and the
//! `equivariant_rho*` FFI shims.

use ndarray::{Array1, Array3, ArrayD, ArrayView1, ArrayView2, ArrayViewD, IxDyn};

/// `SO(2)` representation: one 2×2 rotation matrix per input angle.
pub fn rho_so2(theta: ArrayView1<'_, f64>) -> Array3<f64> {
    let n = theta.len();
    let mut out = Array3::<f64>::zeros((n, 2, 2));
    for (i, &t) in theta.iter().enumerate() {
        let (s, c) = t.sin_cos();
        out[[i, 0, 0]] = c;
        out[[i, 0, 1]] = -s;
        out[[i, 1, 0]] = s;
        out[[i, 1, 1]] = c;
    }
    out
}

/// Derivative of [`rho_so2`] with respect to the angle (its JVP per unit `dθ`).
pub fn rho_so2_jvp(theta: ArrayView1<'_, f64>) -> Array3<f64> {
    let n = theta.len();
    let mut out = Array3::<f64>::zeros((n, 2, 2));
    for (i, &t) in theta.iter().enumerate() {
        let (s, c) = t.sin_cos();
        out[[i, 0, 0]] = -s;
        out[[i, 0, 1]] = -c;
        out[[i, 1, 0]] = c;
        out[[i, 1, 1]] = -s;
    }
    out
}

/// Closed-form `SO(3)` exponential `exp([ω]×)` (Rodrigues) for a single
/// rotation vector, returned as a row-major 3×3 array.
pub fn rho_so3_single(ox: f64, oy: f64, oz: f64) -> [[f64; 3]; 3] {
    let angle = (ox * ox + oy * oy + oz * oz).sqrt().max(1.0e-12);
    let ax = ox / angle;
    let ay = oy / angle;
    let az = oz / angle;
    let k = [[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]];
    let mut kk = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut acc = 0.0;
            for r in 0..3 {
                acc += k[i][r] * k[r][j];
            }
            kk[i][j] = acc;
        }
    }
    let s = angle.sin();
    let one_minus_c = 1.0 - angle.cos();
    let mut out = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            out[i][j] = id + s * k[i][j] + one_minus_c * kk[i][j];
        }
    }
    out
}

/// `SO(3)` representation: one 3×3 rotation matrix per `(N, 3)` rotation vector.
pub fn rho_so3(omega: ArrayView2<'_, f64>) -> Result<Array3<f64>, String> {
    if omega.ncols() != 3 {
        return Err(format!(
            "SO(3) rep input must have shape (N, 3); got {}",
            omega.ncols()
        ));
    }
    let n = omega.nrows();
    let mut out = Array3::<f64>::zeros((n, 3, 3));
    for row in 0..n {
        let r = rho_so3_single(omega[[row, 0]], omega[[row, 1]], omega[[row, 2]]);
        for i in 0..3 {
            for j in 0..3 {
                out[[row, i, j]] = r[i][j];
            }
        }
    }
    Ok(out)
}

/// Closed-form right-Jacobian of the SO(3) exponential map (a.k.a. the
/// Jacobian of `exp([ω]×)` with respect to `ω` in body-frame coordinates):
///
/// ```text
/// J_r(ω) = I − ((1 − cos θ)/θ) · K + ((θ − sin θ)/θ) · K²,    θ = ‖ω‖, K = [ω/θ]×
/// ```
///
/// Used by [`rho_so3_jvp`] to push the input perturbation `dω`
/// through the tangent map of the body-frame parametrisation before
/// forming the `[·]×` matrix that left-multiplies `R(ω)`. Pre-fix the
/// JVP used the raw `[dω]×` (i.e. assumed `J_r = I`), giving the correct
/// directional derivative only when `dω ∥ ω` or `ω = 0`.
///
/// Small-θ expansion (Taylor): `A = (1 − cos θ)/θ = θ/2 − θ³/24 + …` and
/// `B = (θ − sin θ)/θ = θ²/6 − θ⁴/120 + …`. For θ ≤ a small cutoff we
/// use the second-order polynomial `J_r ≈ I − ½[ω]× + (1/6)[ω]×²`, which
/// agrees with the exact expression to relative O(θ⁴) and avoids the
/// 0/0 in `A/θ`, `B/θ`.
pub fn so3_right_jacobian_times_vec(
    ox: f64,
    oy: f64,
    oz: f64,
    vx: f64,
    vy: f64,
    vz: f64,
) -> [f64; 3] {
    let theta2 = ox * ox + oy * oy + oz * oz;
    let theta = theta2.sqrt();
    // [ω]× · v = ω × v
    let ox_v = [oy * vz - oz * vy, oz * vx - ox * vz, ox * vy - oy * vx];
    // [ω]×² · v = ω × (ω × v) = ω(ω·v) − v‖ω‖²
    let omega_dot_v = ox * vx + oy * vy + oz * vz;
    let ox2_v = [
        ox * omega_dot_v - theta2 * vx,
        oy * omega_dot_v - theta2 * vy,
        oz * omega_dot_v - theta2 * vz,
    ];
    // Coefficient of [ω]×/‖ω‖ = (1−cos θ)/θ  →  scaled to [ω]× factor is −(1−cos θ)/θ²
    // Coefficient of [ω]×²/‖ω‖² = (θ−sin θ)/θ  →  scaled to [ω]×² factor is  (θ−sin θ)/θ³
    let (alpha, beta) = if theta < 1.0e-6 {
        // Taylor series of −A/θ = −(1 − cos θ)/θ² and B/θ² = (θ − sin θ)/θ³,
        // expressed as power series in θ². Truncating at O(θ²) keeps relative
        // error below 1e-13 for θ < 1e-3.
        // −A/θ = −1/2 + θ²/24 − θ⁴/720 + …
        // B/θ² = 1/6 − θ²/120 + θ⁴/5040 − …
        (
            -0.5 + theta2 / 24.0 - theta2 * theta2 / 720.0,
            1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        )
    } else {
        let s = theta.sin();
        let c = theta.cos();
        (-(1.0 - c) / theta2, (theta - s) / (theta2 * theta))
    };
    [
        vx + alpha * ox_v[0] + beta * ox2_v[0],
        vy + alpha * ox_v[1] + beta * ox2_v[1],
        vz + alpha * ox_v[2] + beta * ox2_v[2],
    ]
}

/// JVP of [`rho_so3`]: the directional derivative `R(ω) · [J_r(ω)·dω]×` of the
/// `SO(3)` exponential along `dω`, one 3×3 block per `(N, 3)` row.
pub fn rho_so3_jvp(
    omega: ArrayView2<'_, f64>,
    domega: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, String> {
    if omega.ncols() != 3 || domega.ncols() != 3 {
        return Err("SO(3) rep JVP requires (N, 3) inputs".to_string());
    }
    if omega.nrows() != domega.nrows() {
        return Err("SO(3) rep JVP omega/domega must agree in row count".to_string());
    }
    let n = omega.nrows();
    let mut out = Array3::<f64>::zeros((n, 3, 3));
    for row in 0..n {
        let ox = omega[[row, 0]];
        let oy = omega[[row, 1]];
        let oz = omega[[row, 2]];
        let rg = rho_so3_single(ox, oy, oz);
        // The directional derivative of exp([ω + t·dω]×) at t=0 is
        // R(ω) · [J_r(ω) · dω]×. Using the raw `dω` (i.e. assuming
        // J_r = I) is correct only when dω is parallel to ω; for any
        // perpendicular component the JVP picks up errors that scale
        // with ‖ω‖ and reach ~1 for ‖ω‖ near π/2.
        let jr_dw = so3_right_jacobian_times_vec(
            ox,
            oy,
            oz,
            domega[[row, 0]],
            domega[[row, 1]],
            domega[[row, 2]],
        );
        let sx = jr_dw[0];
        let sy = jr_dw[1];
        let sz = jr_dw[2];
        let kd = [[0.0, -sz, sy], [sz, 0.0, -sx], [-sy, sx, 0.0]];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for r in 0..3 {
                    acc += rg[i][r] * kd[r][j];
                }
                out[[row, i, j]] = acc;
            }
        }
    }
    Ok(out)
}

/// Equivariant group representation `ρ(g)`: dispatch by group name and return a
/// stack of representation matrices broadcast over the leading axes of `g`.
/// `SO2` appends a 2×2 block per angle; `SO3` a 3×3 block per rotation vector
/// (last axis length 3); `R1`/`Trivial` the 1×1 identity.
pub fn rho(group: &str, g: ArrayViewD<'_, f64>) -> Result<ArrayD<f64>, String> {
    match group {
        "SO2" => {
            let mut out_shape = g.shape().to_vec();
            out_shape.push(2);
            out_shape.push(2);
            let flat = Array1::from_vec(g.iter().copied().collect());
            rho_so2(flat.view())
                .into_shape_with_order(IxDyn(&out_shape))
                .map_err(|err| format!("failed to reshape SO(2) representation: {err}"))
        }
        "SO3" => {
            let shape = g.shape().to_vec();
            if shape.last().copied() != Some(3) {
                return Err("SO(3) rep input requires last axis of length 3".to_string());
            }
            let n = g.len() / 3;
            let flat = g
                .to_owned()
                .into_shape_with_order((n, 3))
                .map_err(|err| format!("failed to flatten SO(3) representation input: {err}"))?;
            let mut out_shape = shape[..shape.len() - 1].to_vec();
            out_shape.push(3);
            out_shape.push(3);
            rho_so3(flat.view())?
                .into_shape_with_order(IxDyn(&out_shape))
                .map_err(|err| format!("failed to reshape SO(3) representation: {err}"))
        }
        "R1" => {
            let mut out_shape = g.shape().to_vec();
            out_shape.push(1);
            out_shape.push(1);
            Ok(ArrayD::<f64>::from_elem(IxDyn(&out_shape), 1.0))
        }
        "Trivial" => {
            let shape = g.shape();
            let mut out_shape = if shape.is_empty() {
                Vec::new()
            } else {
                shape[..shape.len() - 1].to_vec()
            };
            out_shape.push(1);
            out_shape.push(1);
            Ok(ArrayD::<f64>::from_elem(IxDyn(&out_shape), 1.0))
        }
        other => Err(format!("unknown equivariant group {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for issue #388: the SO(3) JVP must include the right-Jacobian
    /// factor, so its output matches `R(ω) · [J_r(ω)·dω]×` rather than
    /// `R(ω) · [dω]×`. Verify the JVP matches a 4-point central
    /// finite-difference of `exp([ω + t·dω]×)` at machine-FD precision.
    #[test]
    fn so3_jvp_matches_finite_difference() {
        // Independent expm via series; the same routine that drives the
        // closed-form forward `rho_so3_single`. Using it for the FD ground
        // truth keeps the test self-contained (no SciPy dependency) while
        // staying agnostic to the JVP formula under test.
        fn expm_hat(wx: f64, wy: f64, wz: f64) -> [[f64; 3]; 3] {
            rho_so3_single(wx, wy, wz)
        }
        fn fd_central_4pt(
            wx: f64,
            wy: f64,
            wz: f64,
            dx: f64,
            dy: f64,
            dz: f64,
            h: f64,
        ) -> [[f64; 3]; 3] {
            let r_p2 = expm_hat(wx + 2.0 * h * dx, wy + 2.0 * h * dy, wz + 2.0 * h * dz);
            let r_p1 = expm_hat(wx + h * dx, wy + h * dy, wz + h * dz);
            let r_m1 = expm_hat(wx - h * dx, wy - h * dy, wz - h * dz);
            let r_m2 = expm_hat(wx - 2.0 * h * dx, wy - 2.0 * h * dy, wz - 2.0 * h * dz);
            let mut out = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j] = (-r_p2[i][j] + 8.0 * r_p1[i][j] - 8.0 * r_m1[i][j] + r_m2[i][j])
                        / (12.0 * h);
                }
            }
            out
        }
        // Three deterministic (ω, dω) pairs chosen so dω has a non-trivial
        // component perpendicular to ω (the regime the missing J_r breaks).
        let cases: [([f64; 3], [f64; 3]); 3] = [
            ([0.7, -0.4, 0.3], [1.1, 0.5, -0.8]),
            ([0.2, 0.9, -0.5], [-0.6, 1.2, 0.4]),
            ([-1.0, 0.3, 0.6], [0.5, -0.7, 0.9]),
        ];
        let omega = ndarray::Array2::from_shape_vec(
            (cases.len(), 3),
            cases
                .iter()
                .flat_map(|(w, _)| w.iter().copied())
                .collect::<Vec<_>>(),
        )
        .expect("omega array");
        let domega = ndarray::Array2::from_shape_vec(
            (cases.len(), 3),
            cases
                .iter()
                .flat_map(|(_, dw)| dw.iter().copied())
                .collect::<Vec<_>>(),
        )
        .expect("domega array");
        let jvp = rho_so3_jvp(omega.view(), domega.view())
            .expect("SO(3) JVP must succeed on (N,3) input");
        for (row, (w, dw)) in cases.iter().enumerate() {
            let fd = fd_central_4pt(w[0], w[1], w[2], dw[0], dw[1], dw[2], 1.0e-4);
            let mut max_err = 0.0_f64;
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (jvp[[row, i, j]] - fd[i][j]).abs();
                    if diff > max_err {
                        max_err = diff;
                    }
                }
            }
            assert!(
                max_err < 1.0e-7,
                "row {row}: SO(3) JVP - 4pt-FD has max |err| = {max_err:.3e} (omega={w:?}, domega={dw:?})"
            );
        }
    }

    /// Sanity: when dω ∥ ω (or ω = 0) the right Jacobian collapses to identity,
    /// so the JVP equals R · [dω]×. Verify that boundary case is unchanged.
    #[test]
    fn so3_jvp_parallel_direction_unaffected_by_right_jacobian_fix() {
        // ω || dω: J_r(ω)·dω = dω because [ω]×·ω = 0 and [ω]²×·ω = 0.
        let omega = ndarray::Array2::from_shape_vec((1, 3), vec![0.3, -0.6, 0.4]).expect("omega");
        let domega = {
            let scale = 1.7_f64;
            ndarray::Array2::from_shape_vec((1, 3), vec![scale * 0.3, scale * -0.6, scale * 0.4])
                .expect("domega")
        };
        let jvp = rho_so3_jvp(omega.view(), domega.view()).expect("JVP");
        // Expected: R(ω) · [dω]× (pre-fix and post-fix agree for parallel dω).
        let rg = rho_so3_single(omega[[0, 0]], omega[[0, 1]], omega[[0, 2]]);
        let dx = domega[[0, 0]];
        let dy = domega[[0, 1]];
        let dz = domega[[0, 2]];
        let kd = [[0.0, -dz, dy], [dz, 0.0, -dx], [-dy, dx, 0.0]];
        let mut expected = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for r in 0..3 {
                    expected[i][j] += rg[i][r] * kd[r][j];
                }
            }
        }
        let mut max_err = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let diff = (jvp[[0, i, j]] - expected[i][j]).abs();
                if diff > max_err {
                    max_err = diff;
                }
            }
        }
        assert!(
            max_err < 1.0e-13,
            "parallel-dω boundary case shifted: max err = {max_err:.3e}"
        );
    }
}
