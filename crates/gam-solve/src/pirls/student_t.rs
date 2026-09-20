//! Scaled Student-t response on the identity link.
//!
//! With `r = y − η`, `A = νσ²` and `D = A + r²`, one row contributes
//!
//! ```text
//!   ℓ(η; σ, ν) = lgΓ((ν+1)/2) − lgΓ(ν/2) − ½ ln(πA) − ((ν+1)/2)·ln(1 + r²/A)
//! ```
//!
//! to the log-likelihood. The density is not log-concave in `η`: the observed
//! information `W_obs = (ν+1)(A − r²)/D²` turns negative on rows with
//! `r² > A`, which is exactly how the family discounts outliers. The inner
//! solve therefore carries two curvatures:
//!
//! * the EM (scale-mixture) weight `W_EM = (ν+1)/D` with working response
//!   `z = y`, which is positive everywhere and satisfies `W_EM·(z − η) = u`,
//!   the exact score. It is the Fisher slot: the quadratic it defines
//!   majorizes the negative log-likelihood, so a step on it never increases
//!   the penalized deviance;
//! * the observed information `W_obs`, which is the exact Hessian of the
//!   penalized objective and the curvature the Laplace approximation needs.
//!
//! `σ` and `ν` are outer LAML hyperparameters, searched on `θ = (ln σ, ln ν)`.
//! Everything the outer gradient and Hessian need at fixed β — the θ-partials
//! of `ℓ`, of the score `u = ∂ℓ/∂η`, of `W_obs`, and of `∂W_obs/∂η` — is
//! written here in closed form, in the dimensionless ratios
//! `s = A/D ∈ (0, 1]` and `t = r²/D = 1 − s`.

use super::*;

/// The `(σ, ν)`-dependent constants of one Student-t likelihood evaluation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StudentTScale {
    nu: f64,
    /// `A = νσ²`, the squared scale of `r` in the density kernel.
    a: f64,
    /// `½[ψ((ν+1)/2) − ψ(ν/2)]`.
    half_digamma_gap: f64,
    /// `¼[ψ'((ν+1)/2) − ψ'(ν/2)]`.
    quarter_trigamma_gap: f64,
}

/// One row's residual geometry at a fixed `(σ, ν)`.
#[derive(Clone, Copy, Debug)]
struct StudentTRow {
    r: f64,
    /// `D = A + r²`.
    d: f64,
    /// `s = A/D`.
    s: f64,
    /// `t = r²/D`.
    t: f64,
    /// `ln(1 + r²/A)`.
    log1p_q: f64,
}

/// Fixed-β θ-partials of one row, `θ = (ln σ, ln ν)`, unweighted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StudentTThetaJet {
    /// `∂ℓ/∂θ_i`.
    pub(crate) log_likelihood: [f64; 2],
    /// `∂²ℓ/∂θ_i∂θ_j`.
    pub(crate) log_likelihood2: [[f64; 2]; 2],
    /// `∂u/∂θ_i`, `u = ∂ℓ/∂η`.
    pub(crate) score: [f64; 2],
    /// `∂²u/∂θ_i∂θ_j`.
    pub(crate) score2: [[f64; 2]; 2],
    /// `∂W_obs/∂θ_i`.
    pub(crate) weight: [f64; 2],
    /// `∂²W_obs/∂θ_i∂θ_j`.
    pub(crate) weight2: [[f64; 2]; 2],
    /// `∂²W_obs/∂θ_i∂η`.
    pub(crate) weight_eta: [f64; 2],
}

impl StudentTThetaJet {
    /// Multiply every partial by the row's prior weight.
    pub(crate) fn scale_by(&mut self, w: f64) {
        for i in 0..2 {
            self.log_likelihood[i] *= w;
            self.score[i] *= w;
            self.weight[i] *= w;
            self.weight_eta[i] *= w;
            for j in 0..2 {
                self.log_likelihood2[i][j] *= w;
                self.score2[i][j] *= w;
                self.weight2[i][j] *= w;
            }
        }
    }
}

/// Partials of a row quantity `f(A, ν)` in the scale-free form
/// `[A f_A, f_ν, A² f_AA, A f_Aν, f_νν]`.
#[derive(Clone, Copy)]
struct ScaledPartials {
    a: f64,
    nu: f64,
    aa: f64,
    a_nu: f64,
    nu_nu: f64,
}

impl ScaledPartials {
    /// Chain to `θ = (ln σ, ln ν)`: `A = νσ²` gives `∂A/∂θ = (2A, A)`,
    /// `∂ν/∂θ = (0, ν)`, `∂²A/∂θ∂θ = [[4A, 2A], [2A, A]]`, `∂²ν/∂θ₂² = ν`.
    fn first(self, nu: f64) -> [f64; 2] {
        [2.0 * self.a, self.a + nu * self.nu]
    }

    fn second(self, nu: f64) -> [[f64; 2]; 2] {
        let s00 = 4.0 * self.aa + 4.0 * self.a;
        let s01 = 2.0 * self.aa + 2.0 * nu * self.a_nu + 2.0 * self.a;
        let s11 = self.aa + 2.0 * nu * self.a_nu + nu * nu * self.nu_nu + self.a + nu * self.nu;
        [[s00, s01], [s01, s11]]
    }
}

impl StudentTScale {
    pub(crate) fn new(sigma: f64, nu: f64) -> Result<Self, EstimationError> {
        let a = student_t_kernel_scale(sigma, nu)?;
        let log_normalizer = student_t_log_normalizer_at(nu, a);
        // The half-shift gaps are formed in closed form: as `ν → ∞` each is
        // `O(ν^{−k−1})` against polygammas of size `ln ν` or `ν^{−k}`, and their
        // direct difference keeps only seven digits of the ν-score at `ν ≈ 5·10⁷`.
        let [digamma_gap, trigamma_gap, ..] =
            gam_math::special::polygamma_half_shift_gap_stack(0.5 * nu, 2);
        let half_digamma_gap = 0.5 * digamma_gap;
        let quarter_trigamma_gap = 0.25 * trigamma_gap;
        let scale = Self {
            nu,
            a,
            half_digamma_gap,
            quarter_trigamma_gap,
        };
        if !(log_normalizer.is_finite()
            && half_digamma_gap.is_finite()
            && quarter_trigamma_gap.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "Student-t likelihood constants are not representable at sigma={sigma}, nu={nu}: {scale:?}"
            )));
        }
        Ok(scale)
    }

    pub(crate) fn from_likelihood(likelihood: &GlmLikelihoodSpec) -> Result<Self, EstimationError> {
        match likelihood.student_t_parameters() {
            Some(Ok((sigma, nu))) => Self::new(sigma, nu),
            Some(Err(error)) => Err(EstimationError::InvalidInput(error.to_string())),
            None => Err(EstimationError::InvalidInput(format!(
                "Student-t row geometry requested for the {} family",
                likelihood.spec.response.name()
            ))),
        }
    }

    fn row(&self, row: usize, y: f64, eta: f64) -> Result<StudentTRow, EstimationError> {
        student_t_row(row, y, eta, self.a)
    }

    /// `(W_EM, ∂W_EM/∂η, ∂²W_EM/∂η²)` with `W_EM = (ν+1)/D`.
    fn em_weight_jet(&self, geometry: &StudentTRow) -> (f64, f64, f64) {
        let nu1 = self.nu + 1.0;
        let d = geometry.d;
        (
            nu1 / d,
            2.0 * nu1 * geometry.r / (d * d),
            2.0 * nu1 * (3.0 * geometry.t - geometry.s) / (d * d),
        )
    }

    /// `(W_obs, ∂W_obs/∂η, ∂²W_obs/∂η²)` with `W_obs = −∂²ℓ/∂η²`.
    pub(crate) fn observed_weight_jet(
        &self,
        row: usize,
        y: f64,
        eta: f64,
    ) -> Result<(f64, f64, f64), EstimationError> {
        let geometry = self.row(row, y, eta)?;
        let nu1 = self.nu + 1.0;
        let (d, r, s, t) = (geometry.d, geometry.r, geometry.s, geometry.t);
        Ok((
            nu1 * (s - t) / d,
            2.0 * nu1 * r * (3.0 * s - t) / (d * d),
            -6.0 * nu1 * (s * s - 6.0 * s * t + t * t) / (d * d),
        ))
    }

    /// Fixed-β θ-partials of the row log-likelihood, score, observed weight
    /// and `∂W_obs/∂η`.
    pub(crate) fn theta_jet(
        &self,
        row: usize,
        y: f64,
        eta: f64,
    ) -> Result<StudentTThetaJet, EstimationError> {
        let geometry = self.row(row, y, eta)?;
        let nu = self.nu;
        let nu1 = nu + 1.0;
        let (d, r, s, t) = (geometry.d, geometry.r, geometry.s, geometry.t);

        let log_likelihood = ScaledPartials {
            // `ν − (ν+1)s = (ν+1)t − 1` and `(ν+1)s² − ν = 1 − (ν+1)t(1+s)`,
            // written without the `O(ν)` cancellation of the left-hand sides.
            a: 0.5 * (nu1 * t - 1.0),
            nu: self.half_digamma_gap - 0.5 * geometry.log1p_q,
            aa: 0.5 * (1.0 - nu1 * t * (1.0 + s)),
            a_nu: 0.5 * t,
            nu_nu: self.quarter_trigamma_gap,
        };
        let u = nu1 * r / d;
        let score = ScaledPartials {
            a: -u * s,
            nu: r / d,
            aa: 2.0 * u * s * s,
            a_nu: -s * r / d,
            nu_nu: 0.0,
        };
        let weight = ScaledPartials {
            a: nu1 * (3.0 * t - s) * s / d,
            nu: (s - t) / d,
            aa: nu1 * (2.0 * s - 10.0 * t) * s * s / d,
            a_nu: (3.0 * t - s) * s / d,
            nu_nu: 0.0,
        };
        let weight_eta = ScaledPartials {
            a: 12.0 * nu1 * r * (t - s) * s / (d * d),
            nu: 2.0 * r * (3.0 * s - t) / (d * d),
            aa: 0.0,
            a_nu: 0.0,
            nu_nu: 0.0,
        };
        let jet = StudentTThetaJet {
            log_likelihood: log_likelihood.first(nu),
            log_likelihood2: log_likelihood.second(nu),
            score: score.first(nu),
            score2: score.second(nu),
            weight: weight.first(nu),
            weight2: weight.second(nu),
            weight_eta: weight_eta.first(nu),
        };
        let finite = jet
            .log_likelihood
            .iter()
            .chain(jet.score.iter())
            .chain(jet.weight.iter())
            .chain(jet.weight_eta.iter())
            .chain(jet.log_likelihood2.iter().flatten())
            .chain(jet.score2.iter().flatten())
            .chain(jet.weight2.iter().flatten())
            .all(|value| value.is_finite());
        if !finite {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                row,
                "Student-t hyperparameter jet",
                eta,
                r,
            ));
        }
        Ok(jet)
    }
}

/// `A = νσ²` for a validated `(σ, ν)`.
fn student_t_kernel_scale(sigma: f64, nu: f64) -> Result<f64, EstimationError> {
    let a = nu * sigma * sigma;
    if sigma.is_finite() && sigma > 0.0 && nu.is_finite() && nu > 0.0 && a.is_finite() && a > 0.0
    {
        Ok(a)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "Student-t likelihood requires finite positive sigma and nu with representable \
             nu*sigma^2; got sigma={sigma}, nu={nu}"
        )))
    }
}

/// `lgΓ((ν+1)/2) − lgΓ(ν/2) − ½ ln(πA)`, with the gamma ratio in its Stirling
/// difference form: two `lgΓ` of size `ν ln ν` would leave an absolute error of
/// `ε·ν ln ν` per row.
fn student_t_log_normalizer_at(nu: f64, a: f64) -> f64 {
    log_gamma_large_ratio(0.5 * nu, 0.5) - 0.5 * (std::f64::consts::PI * a).ln()
}

/// The Student-t row log-likelihood at `r = 0`, the part of `ℓ` the
/// half-deviance leaves out. It depends on `(σ, ν)`, which are LAML
/// hyperparameters, so the marginal likelihood must carry it.
pub(crate) fn student_t_log_normalizer(sigma: f64, nu: f64) -> Result<f64, EstimationError> {
    let value = student_t_log_normalizer_at(nu, student_t_kernel_scale(sigma, nu)?);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "Student-t log normalizer is not representable at sigma={sigma}, nu={nu}"
        )))
    }
}

/// Residual geometry of one row at kernel scale `A`.
fn student_t_row(row: usize, y: f64, eta: f64, a: f64) -> Result<StudentTRow, EstimationError> {
    if !y.is_finite() {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "Student-t response",
            eta,
            y,
        ));
    }
    if !eta.is_finite() {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "linear predictor",
            eta,
            eta,
        ));
    }
    let r = y - eta;
    let r2 = r * r;
    let d = a + r2;
    let q = r2 / a;
    let log1p_q = q.ln_1p();
    if !(r.is_finite() && d.is_finite() && log1p_q.is_finite()) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "Student-t residual",
            eta,
            r,
        ));
    }
    Ok(StudentTRow {
        r,
        d,
        s: a / d,
        t: r2 / d,
        log1p_q,
    })
}

/// Weighted half-deviance `w·½(ν+1)·ln(1 + r²/A)` and its η-derivative
/// `−w·(ν+1)·r/D` for one row. Only the kernel is needed, so none of the
/// ν special functions of [`StudentTScale`] are evaluated.
pub(crate) fn student_t_half_deviance_and_eta_score(
    row: usize,
    y: f64,
    eta: f64,
    sigma: f64,
    nu: f64,
    weight: f64,
) -> Result<(f64, f64), EstimationError> {
    let geometry = student_t_row(row, y, eta, student_t_kernel_scale(sigma, nu)?)?;
    let half = weight * 0.5 * (nu + 1.0) * geometry.log1p_q;
    let score = -weight * (nu + 1.0) * geometry.r / geometry.d;
    if !(half.is_finite() && score.is_finite()) {
        return Err(EstimationError::pirls_row_geometry_unrepresentable(
            row,
            "Student-t half-deviance",
            eta,
            half,
        ));
    }
    Ok((half, score))
}

/// Intercept-only location of the Student-t model at fixed `(σ, ν)`: a
/// stationary point of `Σ wᵢ·½(ν+1)·ln(1 + (yᵢ − m)²/A)` reached by the EM
/// (iteratively reweighted mean) map `m ← Σ wᵢ Wᵢ yᵢ / Σ wᵢ Wᵢ`,
/// `Wᵢ = (ν+1)/(A + (yᵢ − m)²)`, started at the weighted median.
///
/// Each EM step minimizes a quadratic majorizer of the objective, so the
/// objective is non-increasing along the sequence; iteration stops at the
/// first step that no longer strictly decreases it, which is where the
/// sequence has reached its floating-point fixed point.
pub(crate) fn student_t_null_location(
    y: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
    sigma: f64,
    nu: f64,
) -> Result<f64, EstimationError> {
    let a = student_t_kernel_scale(sigma, nu)?;
    let mut rows: Vec<(f64, f64)> = Vec::with_capacity(y.len());
    for i in 0..y.len() {
        let w = priorweights[i];
        if !(w.is_finite() && w >= 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "Student-t null model: invalid prior weight at row {i}: {w}"
            )));
        }
        if !y[i].is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "Student-t null model: invalid response at row {i}: {}",
                y[i]
            )));
        }
        if w > 0.0 {
            rows.push((y[i], w));
        }
    }
    if rows.is_empty() {
        return Err(EstimationError::InvalidInput(
            "Student-t null model has no positively weighted rows".to_string(),
        ));
    }
    let mut location = weighted_median(&mut rows);
    let objective = |m: f64| -> f64 {
        rows.iter()
            .map(|&(value, w)| {
                let r = value - m;
                w * (r * r / a).ln_1p()
            })
            .sum()
    };
    let mut current = objective(location);
    loop {
        let (mut numerator, mut denominator) = (0.0, 0.0);
        for &(value, w) in &rows {
            let r = value - location;
            let weight = w / (a + r * r);
            numerator += weight * value;
            denominator += weight;
        }
        let next = numerator / denominator;
        if !next.is_finite() {
            return Err(EstimationError::InvalidInput(format!(
                "Student-t null location EM step is not representable: {next}"
            )));
        }
        let candidate = objective(next);
        if !(candidate < current) {
            return Ok(location);
        }
        location = next;
        current = candidate;
    }
}

/// The lower weighted median of `rows = [(value, weight)]` (all weights
/// positive, at least one row); sorts `rows` by value.
fn weighted_median(rows: &mut [(f64, f64)]) -> f64 {
    rows.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
    let total: f64 = rows.iter().map(|&(_, w)| w).sum();
    let mut cumulative = 0.0;
    for &(value, w) in rows.iter() {
        cumulative += w;
        if cumulative >= 0.5 * total {
            return value;
        }
    }
    rows[rows.len() - 1].0
}

/// The reference scale `s₀` of the outer `log σ` coordinate: the weighted
/// median absolute deviation of `y − offset` about its weighted median.
///
/// The outer search runs on `ln(σ/s₀)` and `ln ν`, and seeds both at zero, the
/// centre of their derived precision box. At that seed `ν = 1`, the Cauchy
/// member of the family, whose scale is exactly the median absolute deviation
/// about the median, so `σ = s₀` is the consistent null-model scale there, with
/// no calibration constant. Standardizing by `s₀` also makes the outer
/// coordinate, its box, and every seed equivariant under `y ↦ a·y`.
///
/// A zero median absolute deviation (more than half of the weighted mass at
/// one value) leaves the scale without a reference; that is refused typed.
pub(crate) fn student_t_reference_scale(
    y: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    priorweights: ArrayView1<f64>,
) -> Result<f64, EstimationError> {
    let mut rows: Vec<(f64, f64)> = Vec::with_capacity(y.len());
    for i in 0..y.len() {
        let (value, w) = (y[i] - offset[i], priorweights[i]);
        if !(w.is_finite() && w >= 0.0 && value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "Student-t reference scale: invalid row {i} (y - offset = {value}, weight = {w})"
            )));
        }
        if w > 0.0 {
            rows.push((value, w));
        }
    }
    if rows.is_empty() {
        return Err(EstimationError::InvalidInput(
            "Student-t fit has no positively weighted rows".to_string(),
        ));
    }
    let centre = weighted_median(&mut rows);
    for row in rows.iter_mut() {
        row.0 = (row.0 - centre).abs();
    }
    let scale = weighted_median(&mut rows);
    if !(scale.is_finite() && scale > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "Student-t scale has no reference: the weighted median absolute deviation of \
             y - offset is {scale} (more than half of the weighted responses share one value)"
        )));
    }
    Ok(scale)
}

/// Working state for the Student-t identity model: the EM weight
/// `prior·(ν+1)/D` with working response `z = y`, so that
/// `W·(z − η) = prior·u` is the exact score, and its η-derivatives in the
/// curvature buffers.
pub(crate) fn write_student_t_working_state(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    priorweights: ArrayView1<f64>,
    scale: &StudentTScale,
    mu: &mut Array1<f64>,
    weights: &mut Array1<f64>,
    z: &mut Array1<f64>,
    derivatives: Option<WorkingDerivativeBuffersMut<'_>>,
) -> Result<(), EstimationError> {
    let n = eta.len();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let prior = priorweights[i];
        if !(prior.is_finite() && prior >= 0.0) {
            return Err(EstimationError::pirls_row_geometry_unrepresentable(
                i,
                "prior weight",
                eta[i],
                prior,
            ));
        }
        if !eta[i].is_finite() {
            return Err(EstimationError::InverseLinkDomainViolation {
                link: "standard identity inverse link",
                eta: eta[i],
                lower: -f64::MAX,
                upper: f64::MAX,
            });
        }
        rows.push(if prior == 0.0 {
            None
        } else {
            let geometry = scale.row(i, y[i], eta[i])?;
            Some(scale.em_weight_jet(&geometry))
        });
    }
    let mut derivatives = derivatives;
    for (i, jet) in rows.into_iter().enumerate() {
        mu[i] = eta[i];
        let (w, c, d, z_i) = match jet {
            None => (0.0, 0.0, 0.0, eta[i]),
            Some((w, c, d)) => {
                let prior = priorweights[i];
                (prior * w, prior * c, prior * d, y[i])
            }
        };
        weights[i] = w;
        z[i] = z_i;
        if let Some(derivs) = derivatives.as_mut() {
            derivs.c[i] = c;
            derivs.d[i] = d;
        }
    }
    if let Some(derivs) = derivatives {
        derivs.dmu_deta.fill(1.0);
        derivs.d2mu_deta2.fill(0.0);
        derivs.d3mu_deta3.fill(0.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    impl StudentTScale {
        /// `½(ν+1)·ln(1 + r²/A)`: the row's half-deviance against the saturated
        /// fit `η = y`.
        fn half_deviance(&self, row: usize, y: f64, eta: f64) -> Result<f64, EstimationError> {
            Ok(0.5 * (self.nu + 1.0) * self.row(row, y, eta)?.log1p_q)
        }

        /// `u = ∂ℓ/∂η = (ν+1)·r/D`.
        fn score(&self, row: usize, y: f64, eta: f64) -> Result<f64, EstimationError> {
            let geometry = self.row(row, y, eta)?;
            Ok((self.nu + 1.0) * geometry.r / geometry.d)
        }

        /// The row log-likelihood at `r = 0`.
        fn log_normalizer(&self) -> f64 {
            student_t_log_normalizer_at(self.nu, self.a)
        }
    }

    fn loglik(y: f64, eta: f64, log_sigma: f64, log_nu: f64) -> f64 {
        let scale = StudentTScale::new(log_sigma.exp(), log_nu.exp()).unwrap();
        scale.log_normalizer() - scale.half_deviance(0, y, eta).unwrap()
    }

    fn score(y: f64, eta: f64, log_sigma: f64, log_nu: f64) -> f64 {
        StudentTScale::new(log_sigma.exp(), log_nu.exp())
            .unwrap()
            .score(0, y, eta)
            .unwrap()
    }

    fn observed(y: f64, eta: f64, log_sigma: f64, log_nu: f64) -> (f64, f64, f64) {
        StudentTScale::new(log_sigma.exp(), log_nu.exp())
            .unwrap()
            .observed_weight_jet(0, y, eta)
            .unwrap()
    }

    fn jet(y: f64, eta: f64, log_sigma: f64, log_nu: f64) -> StudentTThetaJet {
        StudentTScale::new(log_sigma.exp(), log_nu.exp())
            .unwrap()
            .theta_jet(0, y, eta)
            .unwrap()
    }

    fn assert_close(label: &str, analytic: f64, fd: f64) {
        let tol = 1e-6 * (1.0 + analytic.abs().max(fd.abs()));
        assert!(
            (analytic - fd).abs() <= tol,
            "{label}: analytic={analytic:e} fd={fd:e}"
        );
    }

    /// Central differences in `η` and in `θ = (ln σ, ln ν)` reproduce every
    /// closed form the PIRLS and LAML paths consume, on rows inside and far
    /// outside the log-concave band `r² < A`.
    #[test]
    fn student_t_row_derivatives_match_finite_differences() {
        let h = 1e-5;
        for &(y, eta, log_sigma, log_nu) in &[
            (0.3, -0.1, -0.7, 0.4),
            (4.0, 0.2, -1.2, 1.1),
            (-7.5, 1.0, 0.3, -0.6),
            (0.01, 0.0, 0.0, 2.5),
        ] {
            let fd_eta = |f: &dyn Fn(f64) -> f64| (f(eta + h) - f(eta - h)) / (2.0 * h);
            assert_close(
                "score",
                score(y, eta, log_sigma, log_nu),
                fd_eta(&|e| loglik(y, e, log_sigma, log_nu)),
            );
            let (w, c, d) = observed(y, eta, log_sigma, log_nu);
            assert_close(
                "observed weight",
                w,
                -fd_eta(&|e| score(y, e, log_sigma, log_nu)),
            );
            assert_close(
                "observed c",
                c,
                fd_eta(&|e| observed(y, e, log_sigma, log_nu).0),
            );
            assert_close(
                "observed d",
                d,
                fd_eta(&|e| observed(y, e, log_sigma, log_nu).1),
            );

            let theta = [log_sigma, log_nu];
            let at = |shift: [f64; 2]| (theta[0] + shift[0], theta[1] + shift[1]);
            let fd_theta = |i: usize, f: &dyn Fn(f64, f64) -> f64| {
                let mut plus = [0.0; 2];
                plus[i] = h;
                let mut minus = [0.0; 2];
                minus[i] = -h;
                let (sp, np) = at(plus);
                let (sm, nm) = at(minus);
                (f(sp, np) - f(sm, nm)) / (2.0 * h)
            };
            let analytic = jet(y, eta, log_sigma, log_nu);
            for i in 0..2 {
                assert_close(
                    "dl/dtheta",
                    analytic.log_likelihood[i],
                    fd_theta(i, &|s, n| loglik(y, eta, s, n)),
                );
                assert_close(
                    "du/dtheta",
                    analytic.score[i],
                    fd_theta(i, &|s, n| score(y, eta, s, n)),
                );
                assert_close(
                    "dW/dtheta",
                    analytic.weight[i],
                    fd_theta(i, &|s, n| observed(y, eta, s, n).0),
                );
                assert_close(
                    "d2W/dtheta deta",
                    analytic.weight_eta[i],
                    fd_theta(i, &|s, n| observed(y, eta, s, n).1),
                );
                for j in 0..2 {
                    assert_close(
                        "d2l/dtheta2",
                        analytic.log_likelihood2[i][j],
                        fd_theta(j, &|s, n| jet(y, eta, s, n).log_likelihood[i]),
                    );
                    assert_close(
                        "d2u/dtheta2",
                        analytic.score2[i][j],
                        fd_theta(j, &|s, n| jet(y, eta, s, n).score[i]),
                    );
                    assert_close(
                        "d2W/dtheta2",
                        analytic.weight2[i][j],
                        fd_theta(j, &|s, n| jet(y, eta, s, n).weight[i]),
                    );
                }
            }
        }
    }

    /// The EM working state reproduces the exact score, `W·(z − η) = u`, and
    /// its curvature buffers are the η-derivatives of that weight.
    #[test]
    fn student_t_em_working_state_carries_the_exact_score() {
        let scale = StudentTScale::new(0.4, 2.5).unwrap();
        let y = ndarray::array![0.2, 3.0, -1.5];
        let prior = ndarray::array![1.0, 0.5, 2.0];
        let weights_at = |eta: &Array1<f64>| {
            let n = eta.len();
            let (mut mu, mut w, mut z) = (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
            let (mut c, mut d) = (Array1::zeros(n), Array1::zeros(n));
            let (mut d1, mut d2, mut d3) = (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
            write_student_t_working_state(
                y.view(),
                eta,
                prior.view(),
                &scale,
                &mut mu,
                &mut w,
                &mut z,
                Some(WorkingDerivativeBuffersMut {
                    c: &mut c,
                    d: &mut d,
                    dmu_deta: &mut d1,
                    d2mu_deta2: &mut d2,
                    d3mu_deta3: &mut d3,
                }),
            )
            .unwrap();
            (w, z, c, d)
        };
        let eta = ndarray::array![0.0, 0.1, -0.2];
        let (w, z, c, d) = weights_at(&eta);
        let h = 1e-5;
        for i in 0..eta.len() {
            assert_close(
                "EM score",
                w[i] * (z[i] - eta[i]),
                prior[i] * scale.score(i, y[i], eta[i]).unwrap(),
            );
            let mut plus = eta.clone();
            plus[i] += h;
            let mut minus = eta.clone();
            minus[i] -= h;
            let (wp, _, cp, _) = weights_at(&plus);
            let (wm, _, cm, _) = weights_at(&minus);
            assert_close("EM c", c[i], (wp[i] - wm[i]) / (2.0 * h));
            assert_close("EM d", d[i], (cp[i] - cm[i]) / (2.0 * h));
        }
    }

    /// Deep in the Gaussian limit, at the `ln ν = 17.70` where the Gaussian-data
    /// fit stalled, the row log-likelihood and its θ-partials match 80-digit
    /// `mpmath` values of the closed form (differentiated there) to rounding.
    /// The ν-score is `O(1/ν) = 8·10⁻⁹` here; differencing two digammas of size
    /// `ln ν` left it a `7·10⁻⁸` bias per row, and two `lgΓ` of size `ν ln ν`
    /// left the value a `10⁻⁷` error.
    #[test]
    fn student_t_row_jet_is_exact_in_the_gaussian_limit() {
        let (y, eta, log_sigma, log_nu) = (1.3, 0.4, -0.35, 17.700267812895927);
        let tol = 1e-14;
        let value = loglik(y, eta, log_sigma, log_nu);
        let reference_value = -1.384508387959885914128259;
        assert!(
            (value - reference_value).abs() <= tol,
            "log-likelihood {value:+.17e} against {reference_value:+.17e}"
        );
        let jet = jet(y, eta, log_sigma, log_nu);
        let checks = [
            ("dl/dln_sigma", jet.log_likelihood[0], 0.6311396718924339480361508),
            ("dl/dln_nu", jet.log_likelihood[1], 8.229670179425737530035979e-9),
            ("d2l/dln_sigma2", jet.log_likelihood2[0][0], -3.262279234418535433308766),
            ("d2l/dln_sigma dln_nu", jet.log_likelihood2[0][1], 2.115865133319476116838475e-8),
            ("d2l/dln_nu2", jet.log_likelihood2[1][1], -8.229670228554728920032822e-9),
        ];
        for (label, analytic, reference) in checks {
            assert!(
                (analytic - reference).abs() <= tol,
                "{label}: {analytic:+.17e} against {reference:+.17e}"
            );
        }
    }

    /// As `ν → ∞` the Student-t row log-likelihood, score and observed weight
    /// converge to the Gaussian ones at `φ = σ²`.
    #[test]
    fn student_t_row_tends_to_gaussian_at_large_nu() {
        let sigma: f64 = 0.7;
        let (y, eta) = (1.3, 0.4);
        let r: f64 = y - eta;
        let gaussian_loglik = -0.5 * (2.0 * std::f64::consts::PI * sigma * sigma).ln()
            - 0.5 * r * r / (sigma * sigma);
        let gaussian_score = r / (sigma * sigma);
        let gaussian_weight = 1.0 / (sigma * sigma);
        let mut previous_gap = f64::INFINITY;
        for &nu in &[1e2, 1e4, 1e6] {
            let scale = StudentTScale::new(sigma, nu).unwrap();
            let loglik = scale.log_normalizer() - scale.half_deviance(0, y, eta).unwrap();
            let gap = (loglik - gaussian_loglik).abs();
            assert!(gap < previous_gap, "log-likelihood gap must shrink with nu");
            previous_gap = gap;
            let tol = 10.0 / nu;
            assert!(gap < tol, "nu={nu}: loglik gap {gap:e}");
            assert!((scale.score(0, y, eta).unwrap() - gaussian_score).abs() < tol);
            assert!(
                (scale.observed_weight_jet(0, y, eta).unwrap().0 - gaussian_weight).abs() < tol
            );
        }
    }
}
