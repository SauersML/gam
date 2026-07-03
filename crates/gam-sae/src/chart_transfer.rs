//! Pulled-back chart-to-chart transfer operators for frozen model components.
//!
//! The torch lane owns component JVPs (`J_F(x) v`); Rust owns the chart-frame
//! algebra and density aggregation:
//! `A_kj(x) = (J_k^T J_k)^{-1} J_k^T J_F(x) J_j(x)`.

use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, s};

/// Token-level pulled-back transfer plus density moments for one component/atom pair.
#[derive(Clone, Debug)]
pub struct ChartTransferReport {
    /// Density-weighted mean operator, shape `(d_out, d_in)`.
    pub mean: Array2<f64>,
    /// Density-weighted elementwise token variance around [`Self::mean`].
    pub variance: Array2<f64>,
    /// Per-token operators, shape `(n_tokens, d_out, d_in)`.
    pub token_operators: Array3<f64>,
    /// Effective number of tokens under the supplied non-negative weights.
    pub effective_n: f64,
}

/// Equivariance/transport diagnostics for a square transfer operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferCertificate {
    /// Frobenius norm of `A^T A - I`; zero means isometric transport.
    pub transport_defect: f64,
    /// Frobenius norm of the Lie-generator commutator `A G_in - G_out A`.
    pub equivariance_defect: f64,
}

/// One coordinate-valued graph edge `(atom_j, layer_l) -> (atom_k, layer_l+1)`.
#[derive(Clone, Debug)]
pub struct CoordinateAttributionEdge {
    pub component: String,
    pub input_atom: usize,
    pub output_atom: usize,
    pub operator: Array2<f64>,
    pub variance: Array2<f64>,
}

/// Compute one pulled-back operator from an output chart jet and ambient JVPs.
pub fn pulled_back_operator(
    output_chart_jet: ArrayView2<'_, f64>,
    ambient_jvp_input_chart: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let p = output_chart_jet.nrows();
    let d_out = output_chart_jet.ncols();
    if ambient_jvp_input_chart.nrows() != p {
        return Err(format!(
            "ambient shape mismatch: output jet has {p} rows but JVP has {}",
            ambient_jvp_input_chart.nrows()
        ));
    }
    if d_out == 0 || d_out > 2 {
        return Err(format!(
            "chart transfer currently supports 1D/2D output atoms, got d_out={d_out}"
        ));
    }
    if ambient_jvp_input_chart.ncols() == 0 || ambient_jvp_input_chart.ncols() > 2 {
        return Err(format!(
            "chart transfer currently supports 1D/2D input atoms, got d_in={}",
            ambient_jvp_input_chart.ncols()
        ));
    }
    ensure_finite(output_chart_jet, "output chart jet")?;
    ensure_finite(ambient_jvp_input_chart, "ambient JVP")?;
    let gram = output_chart_jet.t().dot(&output_chart_jet);
    let rhs = output_chart_jet.t().dot(&ambient_jvp_input_chart);
    solve_spd_1_or_2(gram.view(), rhs.view())
}

/// Aggregate token operators with optional density weights.
pub fn aggregate_pulled_back_operators(
    output_chart_jets: ArrayView3<'_, f64>,
    ambient_jvps: ArrayView3<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<ChartTransferReport, String> {
    let n = output_chart_jets.len_of(Axis(0));
    if ambient_jvps.len_of(Axis(0)) != n
        || ambient_jvps.len_of(Axis(1)) != output_chart_jets.len_of(Axis(1))
    {
        return Err("token/ambient dimensions of chart jets and JVPs must match".to_string());
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(format!(
            "weights length {} does not match token count {n}",
            w.len()
        ));
    }
    let d_out = output_chart_jets.len_of(Axis(2));
    let d_in = ambient_jvps.len_of(Axis(2));
    let mut ops = Array3::<f64>::zeros((n, d_out, d_in));
    let mut weight_sum = 0.0;
    let mut weight_sq_sum = 0.0;
    let mut mean = Array2::<f64>::zeros((d_out, d_in));
    for i in 0..n {
        let w = weights.as_ref().map_or(1.0, |ws| ws[i]);
        if !w.is_finite() || w < 0.0 {
            return Err(format!(
                "weights must be finite and non-negative; got {w} at token {i}"
            ));
        }
        let op = pulled_back_operator(
            output_chart_jets.slice(s![i, .., ..]),
            ambient_jvps.slice(s![i, .., ..]),
        )?;
        ops.slice_mut(s![i, .., ..]).assign(&op);
        mean.scaled_add(w, &op);
        weight_sum += w;
        weight_sq_sum += w * w;
    }
    if weight_sum <= 0.0 {
        return Err("at least one token must have positive weight".to_string());
    }
    mean.mapv_inplace(|x| x / weight_sum);
    let mut variance = Array2::<f64>::zeros((d_out, d_in));
    for i in 0..n {
        let w = weights.as_ref().map_or(1.0, |ws| ws[i]);
        let diff = &ops.slice(s![i, .., ..]) - &mean;
        variance.scaled_add(w, &diff.mapv(|x| x * x));
    }
    variance.mapv_inplace(|x| x / weight_sum);
    Ok(ChartTransferReport {
        mean,
        variance,
        token_operators: ops,
        effective_n: weight_sum * weight_sum / weight_sq_sum,
    })
}

/// Compute transport and Lie-equivariance defects for a square operator.
pub fn certify_square_transfer(
    operator: ArrayView2<'_, f64>,
    input_generator: ArrayView2<'_, f64>,
    output_generator: ArrayView2<'_, f64>,
) -> Result<TransferCertificate, String> {
    let d = operator.nrows();
    if operator.ncols() != d || input_generator.dim() != (d, d) || output_generator.dim() != (d, d)
    {
        return Err("operator and generators must be square with matching dimensions".to_string());
    }
    ensure_finite(operator, "operator")?;
    ensure_finite(input_generator, "input generator")?;
    ensure_finite(output_generator, "output generator")?;
    let mut metric = operator.t().dot(&operator);
    for i in 0..d {
        metric[[i, i]] -= 1.0;
    }
    let comm = operator.dot(&input_generator) - output_generator.dot(&operator);
    Ok(TransferCertificate {
        transport_defect: frob(metric.view()),
        equivariance_defect: frob(comm.view()),
    })
}

/// The SO(2) polar-factor angle of a 2×2 operator with positive determinant.
///
/// The polar decomposition `A = R·S` (rotation × SPD stretch) has the unique
/// rotation maximising `tr(Rᵀ A)`; for 2×2 that angle is
/// `atan2(a₁₀ − a₀₁, a₀₀ + a₁₁)`. A non-positive determinant means the
/// operator reflects (or collapses) the chart rather than rotating it, so no
/// rotation angle exists and the call errs — a consumer must report the
/// orientation flip, not silently fold it into an angle.
pub fn so2_polar_angle(operator: ArrayView2<'_, f64>) -> Result<f64, String> {
    if operator.dim() != (2, 2) {
        return Err(format!(
            "SO(2) polar angle requires a 2x2 operator, got {:?}",
            operator.dim()
        ));
    }
    ensure_finite(operator, "operator")?;
    let det = operator[[0, 0]] * operator[[1, 1]] - operator[[0, 1]] * operator[[1, 0]];
    if !(det > 0.0) {
        return Err(format!(
            "SO(2) polar angle requires positive determinant (rotation, not reflection/collapse); got det={det}"
        ));
    }
    Ok((operator[[1, 0]] - operator[[0, 1]]).atan2(operator[[0, 0]] + operator[[1, 1]]))
}

/// Density-weighted circular mean and standard error of the per-token SO(2)
/// polar angles of a stack of 2×2 operators.
///
/// Returns `(mean_angle, se)`. The mean direction comes from the weighted
/// resultant `(C, S) = Σ wᵢ (cos θᵢ, sin θᵢ)`; the SE is the circular standard
/// deviation `√(−2 ln R̄)` (with `R̄` the mean resultant length) divided by
/// `√n_eff`, where `n_eff` is Kish's effective sample size under the weights.
/// Any token whose operator is a reflection/collapse propagates the
/// [`so2_polar_angle`] error — a mixed rotation/reflection population has no
/// honest mean angle.
pub fn rotation_angle_band(
    token_operators: ArrayView3<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<(f64, f64), String> {
    let n = token_operators.len_of(Axis(0));
    if n == 0 {
        return Err("rotation angle band requires at least one token operator".to_string());
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(format!(
            "weights length {} does not match token count {n}",
            w.len()
        ));
    }
    let mut c = 0.0_f64;
    let mut s_sum = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    let mut weight_sq_sum = 0.0_f64;
    for i in 0..n {
        let w = weights.as_ref().map_or(1.0, |ws| ws[i]);
        if !w.is_finite() || w < 0.0 {
            return Err(format!(
                "weights must be finite and non-negative; got {w} at token {i}"
            ));
        }
        let angle = so2_polar_angle(token_operators.slice(s![i, .., ..]))
            .map_err(|e| format!("token {i}: {e}"))?;
        c += w * angle.cos();
        s_sum += w * angle.sin();
        weight_sum += w;
        weight_sq_sum += w * w;
    }
    if weight_sum <= 0.0 {
        return Err("at least one token must have positive weight".to_string());
    }
    let mean_angle = s_sum.atan2(c);
    let resultant = (c * c + s_sum * s_sum).sqrt() / weight_sum;
    let effective_n = weight_sum * weight_sum / weight_sq_sum;
    // Degenerate resultant (angles spread over the whole circle): the circular
    // std diverges; report it honestly as infinite rather than clamping.
    let circular_sd = if resultant > 0.0 {
        (-2.0 * resultant.ln()).max(0.0).sqrt()
    } else {
        f64::INFINITY
    };
    Ok((mean_angle, circular_sd / effective_n.sqrt()))
}

fn solve_spd_1_or_2(
    gram: ArrayView2<'_, f64>,
    rhs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    match gram.nrows() {
        1 => {
            let g = gram[[0, 0]];
            if g <= 0.0 || !g.is_finite() {
                return Err("singular output chart metric".to_string());
            }
            Ok(rhs.mapv(|x| x / g))
        }
        2 => {
            let (a, b, c) = (gram[[0, 0]], gram[[0, 1]], gram[[1, 1]]);
            let det = a * c - b * b;
            let scale = (a.abs() * c.abs()).max(b.abs() * b.abs()).max(1.0);
            if !det.is_finite() || det <= f64::EPSILON.sqrt() * scale {
                return Err("singular output chart metric".to_string());
            }
            let mut out = Array2::<f64>::zeros(rhs.dim());
            for col in 0..rhs.ncols() {
                let r0 = rhs[[0, col]];
                let r1 = rhs[[1, col]];
                out[[0, col]] = (c * r0 - b * r1) / det;
                out[[1, col]] = (-b * r0 + a * r1) / det;
            }
            Ok(out)
        }
        d => Err(format!(
            "chart transfer currently supports 1D/2D output atoms, got d_out={d}"
        )),
    }
}

fn ensure_finite(a: ArrayView2<'_, f64>, name: &str) -> Result<(), String> {
    if a.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(format!("{name} contains non-finite values"))
    }
}

fn frob(a: ArrayView2<'_, f64>) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, array};

    #[test]
    fn pullback_recovers_rotation_in_nonorthonormal_output_frame() {
        let angle = std::f64::consts::FRAC_PI_2;
        let rotation = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
        let output_jet = array![[2.0, 0.0], [0.0, 0.5], [1.0, 1.0]];
        let ambient = output_jet.dot(&rotation);
        let op = pulled_back_operator(output_jet.view(), ambient.view()).unwrap();
        assert!((&op - &rotation).iter().all(|x| x.abs() < 1.0e-12));
    }

    #[test]
    fn aggregation_reports_density_mean_and_token_variance() {
        let mut jets = Array3::<f64>::zeros((2, 2, 2));
        jets.slice_mut(s![0, .., ..])
            .assign(&array![[1.0, 0.0], [0.0, 1.0]]);
        jets.slice_mut(s![1, .., ..])
            .assign(&array![[1.0, 0.0], [0.0, 1.0]]);
        let mut jvps = Array3::<f64>::zeros((2, 2, 2));
        jvps.slice_mut(s![0, .., ..])
            .assign(&array![[1.0, 0.0], [0.0, 1.0]]);
        jvps.slice_mut(s![1, .., ..])
            .assign(&array![[0.0, -1.0], [1.0, 0.0]]);
        let report = aggregate_pulled_back_operators(
            jets.view(),
            jvps.view(),
            Some(array![1.0, 3.0].view()),
        )
        .unwrap();
        assert!((report.mean[[0, 0]] - 0.25).abs() < 1.0e-12);
        assert!((report.mean[[0, 1]] + 0.75).abs() < 1.0e-12);
        assert!(report.variance[[0, 0]] > 0.0);
        assert!((report.effective_n - 1.6).abs() < 1.0e-12);
    }

    #[test]
    fn so2_polar_angle_recovers_rotation_through_stretch() {
        let angle = 0.7_f64;
        let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
        // A = R·S with an anisotropic SPD stretch: the polar angle must ignore S.
        let stretch = array![[2.0, 0.3], [0.3, 0.9]];
        let a = rot.dot(&stretch);
        let got = so2_polar_angle(a.view()).unwrap();
        assert!((got - angle).abs() < 1.0e-12, "got {got}, want {angle}");
    }

    #[test]
    fn so2_polar_angle_rejects_reflection() {
        let reflection = array![[1.0, 0.0], [0.0, -1.0]];
        assert!(so2_polar_angle(reflection.view()).is_err());
    }

    #[test]
    fn rotation_angle_band_weights_and_wraps() {
        // Two rotations straddling the ±π seam: circular mean must wrap, not
        // average to zero.
        let mut ops = Array3::<f64>::zeros((2, 2, 2));
        for (i, angle) in [std::f64::consts::PI - 0.1, -std::f64::consts::PI + 0.1]
            .into_iter()
            .enumerate()
        {
            ops.slice_mut(s![i, .., ..])
                .assign(&array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]]);
        }
        let (mean, se) = rotation_angle_band(ops.view(), None).unwrap();
        assert!(
            (mean.abs() - std::f64::consts::PI).abs() < 1.0e-9,
            "seam-straddling mean should sit at ±π, got {mean}"
        );
        assert!(se.is_finite() && se > 0.0);

        // Fully weighted onto one token, the band collapses to that angle.
        let (mean_one, se_one) =
            rotation_angle_band(ops.view(), Some(array![1.0, 0.0].view())).unwrap();
        assert!((mean_one - (std::f64::consts::PI - 0.1)).abs() < 1.0e-9);
        assert!(se_one.abs() < 1.0e-9);
    }

    #[test]
    fn certificate_accepts_circle_rotation_generator() {
        let rot = array![[0.0, -1.0], [1.0, 0.0]];
        let generator = array![[0.0, -1.0], [1.0, 0.0]];
        let cert = certify_square_transfer(rot.view(), generator.view(), generator.view()).unwrap();
        assert!(cert.transport_defect < 1.0e-12);
        assert!(cert.equivariance_defect < 1.0e-12);
    }
}
