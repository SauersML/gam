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

    pub fn harmonic_content(&self) -> Vec<HarmonicContent> {
        harmonic_content(&self.harmonics)
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

    pub fn harmonic_content(&self) -> Vec<HarmonicContent> {
        harmonic_content(&self.harmonics)
    }

    pub fn predict_delta(&self, key_t: f64) -> f64 {
        let mut out = self.intercept;
        for coefficient in &self.harmonics {
            let angle = TWO_PI * coefficient.harmonic as f64 * key_t;
            out += coefficient.cos * angle.cos() + coefficient.sin * angle.sin();
        }
        out
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
    let tolerance = f64::EPSILON.sqrt() * (1.0 + separable.r2.abs());
    let is_stationary = stationary_r2_gap <= tolerance;
    Ok(AttentionKernelFit {
        stationary,
        separable,
        stationary_r2_gap,
        is_stationary,
    })
}

pub fn fit_stationary_kernel(
    query_t: &[f64],
    key_t: &[f64],
    scores: ArrayView2<'_, f64>,
    max_harmonic: usize,
) -> Result<StationaryKernelFit, String> {
    validate_kernel_inputs(query_t, key_t, scores)?;
    let parameter_count = 1 + 2 * max_harmonic;
    let mut normal = vec![0.0; parameter_count * parameter_count];
    let mut rhs = vec![0.0; parameter_count];
    let mut basis = vec![0.0; parameter_count];
    for query_index in 0..query_t.len() {
        for key_index in 0..key_t.len() {
            stationary_basis(
                query_t[query_index],
                key_t[key_index],
                max_harmonic,
                &mut basis,
            );
            accumulate_normal_equation(
                &mut normal,
                &mut rhs,
                &basis,
                scores[[query_index, key_index]],
            );
        }
    }
    let coefficients = solve_linear_system(normal, rhs, parameter_count)?;
    let (sse, sst) =
        stationary_sums_of_squares(query_t, key_t, scores, max_harmonic, &coefficients);
    Ok(StationaryKernelFit {
        intercept: coefficients[0],
        harmonics: harmonic_coefficients_from_regression(&coefficients, max_harmonic),
        r2: r_squared(sse, sst),
        sse,
        sst,
    })
}

pub fn fit_separable_kernel(
    query_t: &[f64],
    key_t: &[f64],
    scores: ArrayView2<'_, f64>,
    max_harmonic: usize,
) -> Result<SeparableKernelFit, String> {
    validate_kernel_inputs(query_t, key_t, scores)?;
    let basis_terms = harmonic_basis_terms(max_harmonic);
    let basis_width = basis_terms.len();
    let parameter_count = basis_width * basis_width;
    let mut normal = vec![0.0; parameter_count * parameter_count];
    let mut rhs = vec![0.0; parameter_count];
    let mut row_basis = vec![0.0; parameter_count];
    for query_value in query_t {
        assert_finite(*query_value, "query coordinate")?;
    }
    for key_value in key_t {
        assert_finite(*key_value, "key coordinate")?;
    }
    for query_index in 0..query_t.len() {
        let query_basis = harmonic_basis_values(query_t[query_index], max_harmonic);
        for key_index in 0..key_t.len() {
            let key_basis = harmonic_basis_values(key_t[key_index], max_harmonic);
            fill_separable_basis(&query_basis, &key_basis, &mut row_basis);
            accumulate_normal_equation(
                &mut normal,
                &mut rhs,
                &row_basis,
                scores[[query_index, key_index]],
            );
        }
    }
    let coefficients = solve_linear_system(normal, rhs, parameter_count)?;
    let (sse, sst) = separable_sums_of_squares(query_t, key_t, scores, max_harmonic, &coefficients);
    Ok(SeparableKernelFit {
        max_harmonic,
        basis_terms,
        coefficients_row_major: coefficients,
        r2: r_squared(sse, sst),
        sse,
        sst,
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
    let parameter_count = 1 + 2 * max_harmonic;
    let mut normal = vec![0.0; parameter_count * parameter_count];
    let mut rhs = vec![0.0; parameter_count];
    let mut basis = vec![0.0; parameter_count];
    for index in 0..key_t.len() {
        assert_finite(key_t[index], "key coordinate")?;
        assert_finite(delta_t[index], "coordinate delta")?;
        coordinate_basis(key_t[index], max_harmonic, &mut basis);
        accumulate_normal_equation(&mut normal, &mut rhs, &basis, delta_t[index]);
    }
    let coefficients = solve_linear_system(normal, rhs, parameter_count)?;
    let (sse, sst) = coordinate_sums_of_squares(key_t, delta_t, max_harmonic, &coefficients);
    Ok(CoordinateMapFit {
        intercept: coefficients[0],
        harmonics: harmonic_coefficients_from_regression(&coefficients, max_harmonic),
        r2: r_squared(sse, sst),
        sse,
        sst,
    })
}

fn validate_kernel_inputs(
    query_t: &[f64],
    key_t: &[f64],
    scores: ArrayView2<'_, f64>,
) -> Result<(), String> {
    if query_t.is_empty() || key_t.is_empty() {
        return Err(
            "attention kernel fit requires non-empty query and key coordinates".to_string(),
        );
    }
    if scores.nrows() != query_t.len() || scores.ncols() != key_t.len() {
        return Err(format!(
            "attention kernel score shape {:?} must equal ({}, {})",
            scores.dim(),
            query_t.len(),
            key_t.len()
        ));
    }
    for query_value in query_t {
        assert_finite(*query_value, "query coordinate")?;
    }
    for key_value in key_t {
        assert_finite(*key_value, "key coordinate")?;
    }
    for score in scores.iter() {
        assert_finite(*score, "QK score")?;
    }
    Ok(())
}

fn stationary_basis(query_t: f64, key_t: f64, max_harmonic: usize, out: &mut [f64]) {
    out[0] = 1.0;
    let delta = query_t - key_t;
    for harmonic in 1..=max_harmonic {
        let angle = TWO_PI * harmonic as f64 * delta;
        let base = 1 + 2 * (harmonic - 1);
        out[base] = angle.cos();
        out[base + 1] = angle.sin();
    }
}

fn coordinate_basis(t: f64, max_harmonic: usize, out: &mut [f64]) {
    out[0] = 1.0;
    for harmonic in 1..=max_harmonic {
        let angle = TWO_PI * harmonic as f64 * t;
        let base = 1 + 2 * (harmonic - 1);
        out[base] = angle.cos();
        out[base + 1] = angle.sin();
    }
}

fn harmonic_basis_values(t: f64, max_harmonic: usize) -> Vec<f64> {
    let mut out = vec![0.0; 1 + 2 * max_harmonic];
    coordinate_basis(t, max_harmonic, &mut out);
    out
}

fn harmonic_basis_terms(max_harmonic: usize) -> Vec<HarmonicBasisTerm> {
    let mut out = Vec::with_capacity(1 + 2 * max_harmonic);
    out.push(HarmonicBasisTerm {
        harmonic: 0,
        kind: HarmonicBasisKind::Constant,
    });
    for harmonic in 1..=max_harmonic {
        out.push(HarmonicBasisTerm {
            harmonic,
            kind: HarmonicBasisKind::Cos,
        });
        out.push(HarmonicBasisTerm {
            harmonic,
            kind: HarmonicBasisKind::Sin,
        });
    }
    out
}

fn fill_separable_basis(query_basis: &[f64], key_basis: &[f64], out: &mut [f64]) {
    let width = query_basis.len();
    for query_index in 0..width {
        for key_index in 0..width {
            out[query_index * width + key_index] = query_basis[query_index] * key_basis[key_index];
        }
    }
}

fn accumulate_normal_equation(normal: &mut [f64], rhs: &mut [f64], basis: &[f64], y: f64) {
    let width = basis.len();
    for row in 0..width {
        rhs[row] += basis[row] * y;
        for col in 0..width {
            normal[row * width + col] += basis[row] * basis[col];
        }
    }
}

fn solve_linear_system(
    mut matrix: Vec<f64>,
    mut rhs: Vec<f64>,
    width: usize,
) -> Result<Vec<f64>, String> {
    let mut matrix_scale = 0.0_f64;
    for value in &matrix {
        matrix_scale = matrix_scale.max(value.abs());
    }
    let pivot_floor = f64::EPSILON * width.max(1) as f64 * matrix_scale.max(1.0);
    for col in 0..width {
        let mut pivot_row = col;
        let mut pivot_abs = matrix[col * width + col].abs();
        for candidate in (col + 1)..width {
            let candidate_abs = matrix[candidate * width + col].abs();
            if candidate_abs > pivot_abs {
                pivot_row = candidate;
                pivot_abs = candidate_abs;
            }
        }
        if pivot_abs <= pivot_floor {
            return Err(format!(
                "least-squares normal equation is rank deficient at column {col}; pivot {pivot_abs:e}"
            ));
        }
        if pivot_row != col {
            for swap_col in 0..width {
                matrix.swap(col * width + swap_col, pivot_row * width + swap_col);
            }
            rhs.swap(col, pivot_row);
        }
        let pivot = matrix[col * width + col];
        for row in (col + 1)..width {
            let factor = matrix[row * width + col] / pivot;
            matrix[row * width + col] = 0.0;
            for update_col in (col + 1)..width {
                matrix[row * width + update_col] -= factor * matrix[col * width + update_col];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut solution = vec![0.0; width];
    for row in (0..width).rev() {
        let mut residual = rhs[row];
        for col in (row + 1)..width {
            residual -= matrix[row * width + col] * solution[col];
        }
        solution[row] = residual / matrix[row * width + row];
    }
    Ok(solution)
}

fn stationary_sums_of_squares(
    query_t: &[f64],
    key_t: &[f64],
    scores: ArrayView2<'_, f64>,
    max_harmonic: usize,
    coefficients: &[f64],
) -> (f64, f64) {
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let mut basis = vec![0.0; coefficients.len()];
    let mut sse = 0.0;
    let mut sst = 0.0;
    for query_index in 0..query_t.len() {
        for key_index in 0..key_t.len() {
            stationary_basis(
                query_t[query_index],
                key_t[key_index],
                max_harmonic,
                &mut basis,
            );
            let prediction = dot(&basis, coefficients);
            let observed = scores[[query_index, key_index]];
            let residual = observed - prediction;
            let centered = observed - mean;
            sse += residual * residual;
            sst += centered * centered;
        }
    }
    (sse, sst)
}

fn separable_sums_of_squares(
    query_t: &[f64],
    key_t: &[f64],
    scores: ArrayView2<'_, f64>,
    max_harmonic: usize,
    coefficients: &[f64],
) -> (f64, f64) {
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let basis_width = 1 + 2 * max_harmonic;
    let mut row_basis = vec![0.0; coefficients.len()];
    // Precondition: the separable (query ⊗ key) harmonic design has
    // `basis_width²` columns, so the coefficient vector `fill_separable_basis`
    // writes into must match. Checked before the loop that indexes it, in every
    // build (not a debug-only invariant).
    assert_eq!(
        row_basis.len(),
        basis_width * basis_width,
        "separable harmonic coefficient length {} must equal basis_width² = {}",
        row_basis.len(),
        basis_width * basis_width,
    );
    let mut sse = 0.0;
    let mut sst = 0.0;
    for query_index in 0..query_t.len() {
        let query_basis = harmonic_basis_values(query_t[query_index], max_harmonic);
        for key_index in 0..key_t.len() {
            let key_basis = harmonic_basis_values(key_t[key_index], max_harmonic);
            fill_separable_basis(&query_basis, &key_basis, &mut row_basis);
            let prediction = dot(&row_basis, coefficients);
            let observed = scores[[query_index, key_index]];
            let residual = observed - prediction;
            let centered = observed - mean;
            sse += residual * residual;
            sst += centered * centered;
        }
    }
    (sse, sst)
}

fn coordinate_sums_of_squares(
    key_t: &[f64],
    delta_t: &[f64],
    max_harmonic: usize,
    coefficients: &[f64],
) -> (f64, f64) {
    let mean = delta_t.iter().sum::<f64>() / delta_t.len() as f64;
    let mut basis = vec![0.0; coefficients.len()];
    let mut sse = 0.0;
    let mut sst = 0.0;
    for index in 0..key_t.len() {
        coordinate_basis(key_t[index], max_harmonic, &mut basis);
        let prediction = dot(&basis, coefficients);
        let residual = delta_t[index] - prediction;
        let centered = delta_t[index] - mean;
        sse += residual * residual;
        sst += centered * centered;
    }
    (sse, sst)
}

fn harmonic_coefficients_from_regression(
    coefficients: &[f64],
    max_harmonic: usize,
) -> Vec<HarmonicCoefficient> {
    let mut out = Vec::with_capacity(max_harmonic);
    for harmonic in 1..=max_harmonic {
        let base = 1 + 2 * (harmonic - 1);
        let cos = coefficients[base];
        let sin = coefficients[base + 1];
        out.push(HarmonicCoefficient {
            harmonic,
            cos,
            sin,
            amplitude: cos.hypot(sin),
        });
    }
    out
}

fn harmonic_content(harmonics: &[HarmonicCoefficient]) -> Vec<HarmonicContent> {
    let total_amplitude: f64 = harmonics
        .iter()
        .map(|coefficient| coefficient.amplitude)
        .sum();
    harmonics
        .iter()
        .map(|coefficient| {
            let amplitude_fraction = if total_amplitude > 0.0 {
                coefficient.amplitude / total_amplitude
            } else {
                0.0
            };
            HarmonicContent {
                harmonic: coefficient.harmonic,
                cos: coefficient.cos,
                sin: coefficient.sin,
                amplitude: coefficient.amplitude,
                amplitude_fraction,
            }
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

fn r_squared(sse: f64, sst: f64) -> f64 {
    if sst > 0.0 {
        1.0 - sse / sst
    } else if sse == 0.0 {
        1.0
    } else {
        0.0
    }
}

fn assert_finite(value: f64, label: &str) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("{label} must be finite, got {value}"))
    }
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
}
