//! Harmonic fits for making attention heads legible on chart coordinates.
//!
//! The QK part is fit two ways:
//! - a stationary circulant kernel depending only on `t_q - t_k`;
//! - a separable low-harmonic surface on `(t_q, t_k)` for heads whose score is
//!   not well described by phase difference alone.
//!
//! Both fits are ordinary least squares in a fixed harmonic basis. The module
//! does not choose harmonics by search; callers provide the maximum harmonic
//! they want to inspect.

use ndarray::ArrayView2;

const TWO_PI: f64 = std::f64::consts::PI * 2.0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HarmonicBasisKind {
    Constant,
    Cos,
    Sin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarmonicBasisTerm {
    pub harmonic: usize,
    pub kind: HarmonicBasisKind,
}

#[derive(Clone, Debug)]
pub struct HarmonicCoefficient {
    pub harmonic: usize,
    pub cos: f64,
    pub sin: f64,
    pub amplitude: f64,
}

#[derive(Clone, Debug)]
pub struct HarmonicContent {
    pub harmonic: usize,
    pub cos: f64,
    pub sin: f64,
    pub amplitude: f64,
    pub amplitude_fraction: f64,
}

#[derive(Clone, Debug)]
pub struct StationaryKernelFit {
    pub intercept: f64,
    pub harmonics: Vec<HarmonicCoefficient>,
    pub r2: f64,
    pub sse: f64,
    pub sst: f64,
}

#[derive(Clone, Debug)]
pub struct SeparableKernelFit {
    pub max_harmonic: usize,
    pub basis_terms: Vec<HarmonicBasisTerm>,
    pub coefficients_row_major: Vec<f64>,
    pub r2: f64,
    pub sse: f64,
    pub sst: f64,
}

#[derive(Clone, Debug)]
pub struct AttentionKernelFit {
    pub stationary: StationaryKernelFit,
    pub separable: SeparableKernelFit,
    pub stationary_r2_gap: f64,
    pub is_stationary: bool,
}

#[derive(Clone, Debug)]
pub struct AttentionKernelReport {
    pub stationary_r2: f64,
    pub separable_r2: f64,
    pub stationary_r2_gap: f64,
    pub is_stationary: bool,
    pub dominant_stationary_harmonic: Option<HarmonicCoefficient>,
    pub stationary_harmonic_content: Vec<HarmonicContent>,
}

#[derive(Clone, Debug)]
pub struct CoordinateMapFit {
    pub intercept: f64,
    pub harmonics: Vec<HarmonicCoefficient>,
    pub r2: f64,
    pub sse: f64,
    pub sst: f64,
}

impl StationaryKernelFit {
    pub fn dominant_harmonic(&self) -> Option<&HarmonicCoefficient> {
        self.harmonics
            .iter()
            .max_by(|left, right| left.amplitude.total_cmp(&right.amplitude))
    }

    pub fn predict(&self, query_t: f64, key_t: f64) -> f64 {
        let mut out = self.intercept;
        let delta = query_t - key_t;
        for coefficient in &self.harmonics {
            let angle = TWO_PI * coefficient.harmonic as f64 * delta;
            out += coefficient.cos * angle.cos() + coefficient.sin * angle.sin();
        }
        out
    }
}

impl SeparableKernelFit {
    pub fn coefficient(&self, query_basis: usize, key_basis: usize) -> Option<f64> {
        let width = self.basis_terms.len();
        if query_basis >= width || key_basis >= width {
            return None;
        }
        Some(self.coefficients_row_major[query_basis * width + key_basis])
    }

    pub fn predict(&self, query_t: f64, key_t: f64) -> f64 {
        let query_basis = harmonic_basis_values(query_t, self.max_harmonic);
        let key_basis = harmonic_basis_values(key_t, self.max_harmonic);
        let width = self.basis_terms.len();
        let mut out = 0.0;
        for query_index in 0..width {
            for key_index in 0..width {
                out += self.coefficients_row_major[query_index * width + key_index]
                    * query_basis[query_index]
                    * key_basis[key_index];
            }
        }
        out
    }
}

impl AttentionKernelFit {
    pub fn report(&self) -> AttentionKernelReport {
        AttentionKernelReport {
            stationary_r2: self.stationary.r2,
            separable_r2: self.separable.r2,
            stationary_r2_gap: self.stationary_r2_gap,
            is_stationary: self.is_stationary,
            dominant_stationary_harmonic: self.stationary.dominant_harmonic().cloned(),
            stationary_harmonic_content: self.stationary.harmonic_content(),
        }
    }
}

impl CoordinateMapFit {
    pub fn dominant_harmonic(&self) -> Option<&HarmonicCoefficient> {
        self.harmonics
            .iter()
            .max_by(|left, right| left.amplitude.total_cmp(&right.amplitude))
    }

}

pub fn fit_attention_kernel(
    query_t: &[f64],
    key_t: &[f64],
    scores: ArrayView2<'_, f64>,
    max_harmonic: usize,
) -> Result<AttentionKernelFit, String> {
    validate_kernel_inputs(query_t, key_t, scores)?;
    let stationary = fit_stationary_kernel(query_t, key_t, scores, max_harmonic)?;
    let separable = fit_separable_kernel(query_t, key_t, scores, max_harmonic)?;
    let stationary_r2_gap = separable.r2 - stationary.r2;
    // Stationarity is a NESTED-model decision, not a raw training-R² tie at
    // machine epsilon. The separable `(t_q, t_k)` surface strictly CONTAINS the
    // stationary circulant kernel (set every off-diagonal query⊗key coefficient
    // to the circulant value), so under a truly stationary process the larger
    // model almost surely lowers in-sample SSE — a machine-epsilon R² gap always
    // fires, declaring even stationary heads non-stationary. Compare the two
    // nested Gaussian fits by BIC on their SSE with the models' own parameter
    // counts: `BIC = n·ln(SSE/n) + p·ln n`. The head is stationary unless the
    // separable surface reduces SSE by more than its extra parameters cost — the
    // same BIC nested-model comparison the structure search uses elsewhere.
    let n_obs = (query_t.len() * key_t.len()) as f64;
    let params_stationary = (1 + 2 * max_harmonic) as f64;
    let separable_width = 1 + 2 * max_harmonic;
    let params_separable = (separable_width * separable_width) as f64;
    let bic = |sse: f64, params: f64| -> f64 {
        let mean_sq = (sse / n_obs).max(f64::MIN_POSITIVE);
        n_obs * mean_sq.ln() + params * n_obs.ln()
    };
    let is_stationary =
        bic(stationary.sse, params_stationary) <= bic(separable.sse, params_separable);
    Ok(AttentionKernelFit {
        stationary,
        separable,
        stationary_r2_gap,
        is_stationary,
    })
}

pub fn fit_ov_coordinate_map(
    key_t: &[f64],
    delta_t: &[f64],
    max_harmonic: usize,
) -> Result<CoordinateMapFit, String> {
    if key_t.len() != delta_t.len() {
        return Err(format!(
            "fit_ov_coordinate_map: key_t length {} must equal delta_t length {}",
            key_t.len(),
            delta_t.len()
        ));
    }
    if key_t.is_empty() {
        return Err("fit_ov_coordinate_map requires at least one observation".to_string());
    }
    for index in 0..key_t.len() {
        assert_finite(key_t[index], "key coordinate")?;
        assert_finite(delta_t[index], "coordinate delta")?;
    }
    // The OV coordinate delta is a PHASE (turns, period 1): `delta` and
    // `delta + 1` are the same shift. A raw Euclidean least-squares of the
    // wrapped delta collapses seam-straddling pairs to their arithmetic midpoint
    // — `-0.49` and `+0.49` (nearly the same half-turn) average to `0`, the
    // antipode of the truth. Regress the SHORTEST-ARC representative instead:
    // unwrap each delta around the circular mean `μ = atan2(Σsin, Σcos)/2π`, i.e.
    // `δ̃ = δ − round(δ − μ)`, so the response is seam-invariant. For a delta map
    // localized within a half-turn (the OV shift case) unwrapping is the identity;
    // it only bites when the deltas straddle the seam, exactly where the raw fit
    // was antipode-biased.
    let (mut cos_sum, mut sin_sum) = (0.0_f64, 0.0_f64);
    for &delta in delta_t {
        let angle = TWO_PI * delta;
        cos_sum += angle.cos();
        sin_sum += angle.sin();
    }
    let circular_mean_turns = sin_sum.atan2(cos_sum) / TWO_PI;
    let unwrapped_delta: Vec<f64> = delta_t
        .iter()
        .map(|&delta| delta - (delta - circular_mean_turns).round())
        .collect();
    let parameter_count = 1 + 2 * max_harmonic;
    let mut normal = vec![0.0; parameter_count * parameter_count];
    let mut rhs = vec![0.0; parameter_count];
    let mut basis = vec![0.0; parameter_count];
    for index in 0..key_t.len() {
        coordinate_basis(key_t[index], max_harmonic, &mut basis);
        accumulate_normal_equation(&mut normal, &mut rhs, &basis, unwrapped_delta[index]);
    }
    let coefficients = solve_linear_system(normal, rhs, parameter_count)?;
    let (sse, sst) =
        coordinate_sums_of_squares(key_t, &unwrapped_delta, max_harmonic, &coefficients);
    Ok(CoordinateMapFit {
        intercept: coefficients[0],
        harmonics: harmonic_coefficients_from_regression(&coefficients, max_harmonic),
        r2: r_squared(sse, sst),
        sse,
        sst,
    })
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn stationary_single_harmonic_qk_fit_recovers_planted_phase_kernel() {
        let query_t: Vec<f64> = (0..24).map(|index| index as f64 / 24.0).collect();
        let key_t: Vec<f64> = (0..20).map(|index| (index as f64 + 0.25) / 20.0).collect();
        let mut scores = Array2::<f64>::zeros((query_t.len(), key_t.len()));
        for query_index in 0..query_t.len() {
            for key_index in 0..key_t.len() {
                let delta = query_t[query_index] - key_t[key_index];
                let deterministic_noise =
                    1.0e-5 * (TWO_PI * (3.0 * query_t[query_index] + 5.0 * key_t[key_index])).sin();
                scores[[query_index, key_index]] =
                    1.7 * (TWO_PI * delta).cos() + deterministic_noise;
            }
        }

        let fit = fit_attention_kernel(&query_t, &key_t, scores.view(), 3)
            .expect("stationary kernel fit should succeed");
        let dominant = fit
            .stationary
            .dominant_harmonic()
            .expect("stationary fit should report a dominant harmonic");

        assert_eq!(dominant.harmonic, 1);
        assert!(dominant.amplitude > 1.699);
        assert!(fit.stationary.r2 > 0.999_999_999);
        assert!(fit.is_stationary);

        let report = fit.report();
        let reported_dominant = report
            .dominant_stationary_harmonic
            .expect("report should carry the dominant harmonic");
        assert_eq!(reported_dominant.harmonic, 1);
        assert!(report.stationary_harmonic_content[0].amplitude_fraction > 0.999);
        assert!(report.stationary_r2 > 0.999_999_999);
        assert!(report.is_stationary);
    }

    #[test]
    fn separable_fit_beats_stationary_fit_for_nonstationary_head() {
        let query_t: Vec<f64> = (0..23).map(|index| index as f64 / 23.0).collect();
        let key_t: Vec<f64> = (0..29).map(|index| (index as f64 + 0.4) / 29.0).collect();
        let mut scores = Array2::<f64>::zeros((query_t.len(), key_t.len()));
        for query_index in 0..query_t.len() {
            for key_index in 0..key_t.len() {
                scores[[query_index, key_index]] =
                    (TWO_PI * query_t[query_index]).cos() * (TWO_PI * 2.0 * key_t[key_index]).sin();
            }
        }

        let fit = fit_attention_kernel(&query_t, &key_t, scores.view(), 2)
            .expect("nonstationary kernel fit should succeed");

        assert!(fit.separable.r2 > 0.999_999_999);
        assert!(
            fit.separable.r2 > fit.stationary.r2 + 0.5,
            "separable r2 {} should beat stationary r2 {}",
            fit.separable.r2,
            fit.stationary.r2
        );
        assert!(!fit.is_stationary);
    }

    #[test]
    fn ov_coordinate_map_fit_recovers_planted_shift() {
        let key_t: Vec<f64> = (0..31).map(|index| index as f64 / 31.0).collect();
        let delta_t: Vec<f64> = key_t
            .iter()
            .map(|t| 1.0 / 7.0 + 0.25 * (TWO_PI * *t).sin())
            .collect();

        let fit = fit_ov_coordinate_map(&key_t, &delta_t, 2)
            .expect("coordinate map harmonic fit should succeed");
        let dominant = fit
            .dominant_harmonic()
            .expect("coordinate map should report a dominant harmonic");

        assert_eq!(dominant.harmonic, 1);
        assert!((fit.intercept - 1.0 / 7.0).abs() < 1.0e-12);
        assert!((dominant.sin - 0.25).abs() < 1.0e-12);
        assert!(fit.r2 > 0.999_999_999);
    }

    #[test]
    fn ov_coordinate_map_unwraps_seam_straddling_half_turn() {
        // A half-turn (0.5) shift with mild key-dependent variation. In the
        // wrapped chart the deltas straddle the seam — some near +0.45, some near
        // −0.45 — so a raw Euclidean regression averages them to ≈0, the antipode.
        // The shortest-arc unwrapping around the circular mean recovers the true
        // ≈0.5 half-turn shift and fits the variation.
        let key_t: Vec<f64> = (0..40).map(|index| index as f64 / 40.0).collect();
        let delta_t: Vec<f64> = key_t
            .iter()
            .map(|t| {
                let raw = 0.5 + 0.1 * (TWO_PI * *t).cos();
                raw - raw.round() // wrap into (−0.5, 0.5]
            })
            .collect();
        let fit = fit_ov_coordinate_map(&key_t, &delta_t, 1).expect("ov fit");
        let recovered = fit.intercept.rem_euclid(1.0);
        assert!(
            (recovered - 0.5).abs() < 0.05,
            "circular unwrapping must recover the half-turn shift, not the antipode: got {recovered}"
        );
        assert!(
            fit.r2 > 0.99,
            "the unwrapped harmonic fit explains the key-dependent variation: r2={}",
            fit.r2
        );
    }
}
