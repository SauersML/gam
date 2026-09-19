//! Harmonic fits for making attention heads legible on chart coordinates.
//!
//! The QK part is fit two ways:
//! - a stationary circulant kernel depending only on `t_q - t_k`;
//! - a separable low-harmonic surface on `(t_q, t_k)` for heads whose score is
//!   not well described by phase difference alone.
//!
//! Both QK fits are Gaussian REML smooths in the tensor Fourier basis of the flat
//! torus, penalized by the squared-Laplacian roughness `∫∫(Δg)²` of the fitted
//! surface itself. Every real tensor column is a Laplacian eigenfunction and the
//! columns are L²-orthogonal under Haar measure, so that penalty is diagonal in
//! the basis and its coefficient form equals the function penalty; only constants
//! are unpenalized. The stationary kernel is the same prior restricted to
//! functions of `t_q - t_k`, so the two models are nested, each with one smoothing
//! parameter and the constants as declared null space. Callers provide the
//! maximum harmonic they want to inspect; the roughness penalty, not a parameter
//! count, prices the harmonics the data do not support.
//!
//! The response is the executed pre-softmax score. Softmax, masking and the
//! positional encoding stay executed by the model; nothing here forms or
//! approximates them.
//!
//! The OV coordinate map is a least-squares harmonic fit of the phase shift.

use gam_solve::gaussian_reml::{GaussianRemlMultiResult, gaussian_reml_multi_closed_form};
use ndarray::{Array2, ArrayView2};

use crate::basis::{RealHarmonicComponent, SaeBasisEvaluator, TorusHarmonicEvaluator};
use crate::manifold::anisotropic_flat_product_torus_penalty;

const TWO_PI: f64 = std::f64::consts::PI * 2.0;

/// Constants are the only functions the squared-Laplacian roughness leaves
/// unpenalized, on the torus and on its stationary restriction alike.
const CONSTANT_NULL_SPACE_DIM: usize = 1;

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

/// Gaussian REML readout of one harmonic smooth at its REML-optimal smoothing
/// strength.
#[derive(Clone, Debug)]
pub struct KernelEvidence {
    pub lambda: f64,
    pub edf: f64,
    /// Profiled restricted negative log marginal likelihood in nats (lower is
    /// better), with the constants integrated out.
    pub reml_score: f64,
    /// Forward-error bound on `reml_score` (#2729); `None` when the producer
    /// accumulated no bound.
    pub reml_score_roundoff: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct StationaryKernelFit {
    pub intercept: f64,
    pub harmonics: Vec<HarmonicCoefficient>,
    pub r2: f64,
    pub sse: f64,
    pub sst: f64,
    pub evidence: KernelEvidence,
}

#[derive(Clone, Debug)]
pub struct SeparableKernelFit {
    pub max_harmonic: usize,
    pub basis_terms: Vec<HarmonicBasisTerm>,
    pub coefficients_row_major: Vec<f64>,
    pub r2: f64,
    pub sse: f64,
    pub sst: f64,
    pub evidence: KernelEvidence,
}

#[derive(Clone, Debug)]
pub struct AttentionKernelFit {
    pub stationary: StationaryKernelFit,
    pub separable: SeparableKernelFit,
    pub stationary_r2_gap: f64,
    /// `reml_score` of the stationary kernel minus that of the separable surface,
    /// in nats: the log Bayes factor of the separable surface over the stationary
    /// kernel.
    pub log_evidence_margin: f64,
    /// Forward-error bound on `log_evidence_margin`; `None` when either score
    /// carries no bound.
    pub log_evidence_resolution: Option<f64>,
    pub is_stationary: bool,
}

#[derive(Clone, Debug)]
pub struct AttentionKernelReport {
    pub stationary_r2: f64,
    pub separable_r2: f64,
    pub stationary_r2_gap: f64,
    pub log_evidence_margin: f64,
    pub log_evidence_resolution: Option<f64>,
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
        let query_basis = axis_harmonic_values(query_t, self.max_harmonic);
        let key_basis = axis_harmonic_values(key_t, self.max_harmonic);
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
            log_evidence_margin: self.log_evidence_margin,
            log_evidence_resolution: self.log_evidence_resolution,
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
    let surface = HarmonicSurface::new(query_t, key_t, scores, max_harmonic)?;
    let stationary = surface.fit_stationary()?;
    let separable = surface.fit_separable()?;
    let stationary_r2_gap = separable.r2 - stationary.r2;
    // Stationarity is a NESTED-model decision. Both smooths declare the constants
    // as null space, so they share the residual degrees of freedom `n − 1` and the
    // improper-prior constant, and each carries exactly one smoothing parameter.
    // Their REML scores are therefore restricted marginal likelihoods of the same
    // error contrasts, and the difference is a log Bayes factor: the separable
    // surface's extra harmonics pay for themselves through
    // `log|XᵀX + λS| − log|λS|₊`, not through a parameter count. The ONE shared λ
    // is essential. A separate λ on the non-stationary harmonics would contain the
    // stationary kernel as its `λ → ∞` boundary, so its maximized evidence could
    // never lose, and a stationary head would be flagged whenever that λ landed
    // inside (#1362). A margin inside the two scores' combined rounding keeps the
    // nested null.
    let log_evidence_margin = stationary.evidence.reml_score - separable.evidence.reml_score;
    let log_evidence_resolution = stationary
        .evidence
        .reml_score_roundoff
        .zip(separable.evidence.reml_score_roundoff)
        .map(|(stationary_bound, separable_bound)| {
            stationary_bound
                + separable_bound
                + gam_linalg::roundoff::UNIT_ROUNDOFF * log_evidence_margin.abs()
        });
    let is_stationary =
        !log_evidence_resolution.is_some_and(|resolution| log_evidence_margin > resolution);
    Ok(AttentionKernelFit {
        stationary,
        separable,
        stationary_r2_gap,
        log_evidence_margin,
        log_evidence_resolution,
        is_stationary,
    })
}

pub fn fit_stationary_kernel(
    query_t: &[f64],
    key_t: &[f64],
    scores: ArrayView2<'_, f64>,
    max_harmonic: usize,
) -> Result<StationaryKernelFit, String> {
    HarmonicSurface::new(query_t, key_t, scores, max_harmonic)?.fit_stationary()
}

pub fn fit_separable_kernel(
    query_t: &[f64],
    key_t: &[f64],
    scores: ArrayView2<'_, f64>,
    max_harmonic: usize,
) -> Result<SeparableKernelFit, String> {
    HarmonicSurface::new(query_t, key_t, scores, max_harmonic)?.fit_separable()
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

/// The QK observations laid out row-major over `(query, key)`, with the tensor
/// Fourier design of the flat torus and its squared-Laplacian roughness.
struct HarmonicSurface {
    max_harmonic: usize,
    response: Array2<f64>,
    separable_design: Array2<f64>,
    separable_penalty: Array2<f64>,
}

impl HarmonicSurface {
    fn new(
        query_t: &[f64],
        key_t: &[f64],
        scores: ArrayView2<'_, f64>,
        max_harmonic: usize,
    ) -> Result<Self, String> {
        validate_kernel_inputs(query_t, key_t, scores)?;
        let n_obs = query_t.len() * key_t.len();
        let mut coords = Array2::<f64>::zeros((n_obs, 2));
        let mut response = Array2::<f64>::zeros((n_obs, 1));
        for query_index in 0..query_t.len() {
            for key_index in 0..key_t.len() {
                let row = query_index * key_t.len() + key_index;
                coords[[row, 0]] = query_t[query_index];
                coords[[row, 1]] = key_t[key_index];
                response[[row, 0]] = scores[[query_index, key_index]];
            }
        }
        // Axis 0 is the query phase and varies slowest, so a design column's flat
        // index is `query_column · width + key_column`: the row-major layout of
        // `SeparableKernelFit::coefficients_row_major`.
        let separable_design = TorusHarmonicEvaluator::new(2, max_harmonic)?
            .evaluate(coords.view())?
            .0;
        let separable_penalty = anisotropic_flat_product_torus_penalty(max_harmonic, 1.0)?;
        Ok(Self {
            max_harmonic,
            response,
            separable_design,
            separable_penalty,
        })
    }

    fn fit_separable(&self) -> Result<SeparableKernelFit, String> {
        let fit = fit_harmonic_reml(
            self.separable_design.view(),
            self.response.view(),
            self.separable_penalty.view(),
            "separable attention kernel",
        )?;
        let (sse, sst) = sums_of_squares(self.response.view(), fit.fitted.view());
        Ok(SeparableKernelFit {
            max_harmonic: self.max_harmonic,
            basis_terms: harmonic_basis_terms(self.max_harmonic),
            coefficients_row_major: fit.coefficients.column(0).to_vec(),
            r2: r_squared(sse, sst),
            sse,
            sst,
            evidence: kernel_evidence(&fit),
        })
    }

    fn fit_stationary(&self) -> Result<StationaryKernelFit, String> {
        let embedding = stationary_embedding(self.max_harmonic)?;
        let design = self.separable_design.dot(&embedding);
        let penalty = embedding.t().dot(&self.separable_penalty.dot(&embedding));
        let fit = fit_harmonic_reml(
            design.view(),
            self.response.view(),
            penalty.view(),
            "stationary attention kernel",
        )?;
        let (sse, sst) = sums_of_squares(self.response.view(), fit.fitted.view());
        let coefficients = fit.coefficients.column(0);
        let mut intercept = 0.0;
        let mut harmonics: Vec<HarmonicCoefficient> = (1..=self.max_harmonic)
            .map(|harmonic| HarmonicCoefficient {
                harmonic,
                cos: 0.0,
                sin: 0.0,
                amplitude: 0.0,
            })
            .collect();
        for (column, component) in axis_components(self.max_harmonic).into_iter().enumerate() {
            match component {
                RealHarmonicComponent::Constant => intercept = coefficients[column],
                RealHarmonicComponent::Sine { harmonic } => {
                    harmonics[harmonic - 1].sin = coefficients[column];
                }
                RealHarmonicComponent::Cosine { harmonic } => {
                    harmonics[harmonic - 1].cos = coefficients[column];
                }
            }
        }
        for coefficient in &mut harmonics {
            coefficient.amplitude = coefficient.cos.hypot(coefficient.sin);
        }
        Ok(StationaryKernelFit {
            intercept,
            harmonics,
            r2: r_squared(sse, sst),
            sse,
            sst,
            evidence: kernel_evidence(&fit),
        })
    }
}

/// Exact embedding `T` of the stationary harmonics of `δ = t_q − t_k` in the
/// tensor columns: `sin 2πhδ = s_h⊗c_h − c_h⊗s_h`, `cos 2πhδ = c_h⊗c_h + s_h⊗s_h`,
/// and the constant to the constant. Column `j` of `T` is per-axis column `j` of
/// the evaluator's layout, so the stationary coefficients read in that order.
/// `Tᵀ S T` is the torus roughness restricted to functions of `δ`: `Δg = 2f″`, so
/// `∫∫(Δg)² = 4∫(f″)²`, the circle's squared-Laplacian roughness up to the scale
/// the smoothing strength absorbs.
fn stationary_embedding(max_harmonic: usize) -> Result<Array2<f64>, String> {
    let components = axis_components(max_harmonic);
    let width = components.len();
    let column_of = |wanted: RealHarmonicComponent| {
        components
            .iter()
            .position(|component| *component == wanted)
            .ok_or_else(|| format!("stationary embedding: per-axis layout has no {wanted:?} column"))
    };
    let mut embedding = Array2::<f64>::zeros((width * width, width));
    for (column, component) in components.iter().copied().enumerate() {
        match component {
            RealHarmonicComponent::Constant => {
                embedding[[column * width + column, column]] = 1.0;
            }
            RealHarmonicComponent::Sine { harmonic } => {
                let cosine = column_of(RealHarmonicComponent::Cosine { harmonic })?;
                embedding[[column * width + cosine, column]] = 1.0;
                embedding[[cosine * width + column, column]] = -1.0;
            }
            RealHarmonicComponent::Cosine { harmonic } => {
                let sine = column_of(RealHarmonicComponent::Sine { harmonic })?;
                embedding[[column * width + column, column]] = 1.0;
                embedding[[sine * width + sine, column]] = 1.0;
            }
        }
    }
    Ok(embedding)
}

fn fit_harmonic_reml(
    design: ArrayView2<'_, f64>,
    response: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    label: &str,
) -> Result<GaussianRemlMultiResult, String> {
    let fit = gaussian_reml_multi_closed_form(design, response, penalty, None, None)
        .map_err(|error| format!("{label}: Gaussian REML failed: {error}"))?;
    if fit.cache.nullity != CONSTANT_NULL_SPACE_DIM {
        return Err(format!(
            "{label}: the solver classified a penalty null space of dimension {}, but the \
             squared-Laplacian roughness declares only the constants ({CONSTANT_NULL_SPACE_DIM})",
            fit.cache.nullity
        ));
    }
    Ok(fit)
}

fn kernel_evidence(fit: &GaussianRemlMultiResult) -> KernelEvidence {
    KernelEvidence {
        lambda: fit.lambda,
        edf: fit.edf,
        reml_score: fit.reml_score,
        reml_score_roundoff: fit.reml_score_roundoff,
    }
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

/// Per-axis real Fourier components in the torus evaluator's column order.
fn axis_components(max_harmonic: usize) -> Vec<RealHarmonicComponent> {
    (0..1 + 2 * max_harmonic)
        .map(TorusHarmonicEvaluator::axis_component)
        .collect()
}

fn axis_harmonic_values(t: f64, max_harmonic: usize) -> Vec<f64> {
    axis_components(max_harmonic)
        .into_iter()
        .map(|component| match component {
            RealHarmonicComponent::Constant => 1.0,
            RealHarmonicComponent::Sine { harmonic } => (TWO_PI * harmonic as f64 * t).sin(),
            RealHarmonicComponent::Cosine { harmonic } => (TWO_PI * harmonic as f64 * t).cos(),
        })
        .collect()
}

fn harmonic_basis_terms(max_harmonic: usize) -> Vec<HarmonicBasisTerm> {
    axis_components(max_harmonic)
        .into_iter()
        .map(|component| match component {
            RealHarmonicComponent::Constant => HarmonicBasisTerm {
                harmonic: 0,
                kind: HarmonicBasisKind::Constant,
            },
            RealHarmonicComponent::Sine { harmonic } => HarmonicBasisTerm {
                harmonic,
                kind: HarmonicBasisKind::Sin,
            },
            RealHarmonicComponent::Cosine { harmonic } => HarmonicBasisTerm {
                harmonic,
                kind: HarmonicBasisKind::Cos,
            },
        })
        .collect()
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

fn sums_of_squares(response: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> (f64, f64) {
    let mean = response.iter().sum::<f64>() / response.len() as f64;
    let mut sse = 0.0;
    let mut sst = 0.0;
    for (observed, prediction) in response.iter().zip(fitted.iter()) {
        let residual = observed - prediction;
        let centered = observed - mean;
        sse += residual * residual;
        sst += centered * centered;
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
    use gam_linalg::utils::splitmix64;

    /// Standard normal draws: SplitMix64 uniforms through Box-Muller under a
    /// fixed seed.
    fn seeded_standard_normals(count: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut uniform = || (splitmix64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
        let mut draws = Vec::with_capacity(count);
        while draws.len() < count {
            // `1 − U[0, 1)` lies in `(0, 1]`, so the logarithm is finite.
            let radial = 1.0 - uniform();
            let angular = uniform();
            draws.push((-2.0 * radial.ln()).sqrt() * (TWO_PI * angular).cos());
        }
        draws
    }

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
        // `cos 2πt_q · sin 4πt_k` lies in the separable surface and is orthogonal to
        // every stationary harmonic on this grid. The seeded noise keeps the residual
        // resolvable: profiled REML refuses an exactly interpolated response (#2723).
        let query_t: Vec<f64> = (0..23).map(|index| index as f64 / 23.0).collect();
        let key_t: Vec<f64> = (0..29).map(|index| (index as f64 + 0.4) / 29.0).collect();
        let noise = seeded_standard_normals(query_t.len() * key_t.len(), 23 * 29);
        let mut scores = Array2::<f64>::zeros((query_t.len(), key_t.len()));
        for query_index in 0..query_t.len() {
            for key_index in 0..key_t.len() {
                scores[[query_index, key_index]] = (TWO_PI * query_t[query_index]).cos()
                    * (TWO_PI * 2.0 * key_t[key_index]).sin()
                    + 0.1 * noise[query_index * key_t.len() + key_index];
            }
        }

        let fit = fit_attention_kernel(&query_t, &key_t, scores.view(), 2)
            .expect("nonstationary kernel fit should succeed");

        assert!(
            fit.separable.r2 > fit.stationary.r2 + 0.5,
            "separable r2 {} should beat stationary r2 {}",
            fit.separable.r2,
            fit.stationary.r2
        );
        assert!(!fit.is_stationary);
    }

    #[test]
    fn planted_query_bias_is_detected_and_stationary_head_is_not_2946() {
        // A stationary head `cos 2π(t_q − t_k)` with seeded noise σ = 0.5 on a 24×24
        // phase grid, inspected up to harmonic 6: the separable surface has 169
        // columns and the stationary kernel 13. The planted non-stationary head
        // adds `0.6·cos 2πt_q`, a query-position bias that is no function of
        // `t_q − t_k`. A count charge `p·ln n` prices the 156 extra columns at
        // `156·ln 576 ≈ 992` against a least-squares gain `n·ln(SSE_S/SSE_N) ≈ 500`,
        // so it reports that head stationary. The REML evidence prices the unused
        // harmonics through their roughness instead of their count: measured on this
        // seed, the margin is +121 nats for the biased head and −31 nats for the
        // stationary one, each against a resolution below 1e-9 (#2946, job 1170868).
        let grid: Vec<f64> = (0..24).map(|index| index as f64 / 24.0).collect();
        let noise = seeded_standard_normals(grid.len() * grid.len(), 2946);
        let head = |bias: f64| {
            Array2::from_shape_fn((grid.len(), grid.len()), |(query_index, key_index)| {
                let query_t = grid[query_index];
                let key_t = grid[key_index];
                (TWO_PI * (query_t - key_t)).cos()
                    + bias * (TWO_PI * query_t).cos()
                    + 0.5 * noise[query_index * grid.len() + key_index]
            })
        };

        let biased = fit_attention_kernel(&grid, &grid, head(0.6).view(), 6)
            .expect("biased head fit should succeed");
        assert!(
            !biased.is_stationary,
            "the planted query bias must be detected: stationary sse {}, separable sse {}",
            biased.stationary.sse,
            biased.separable.sse
        );

        let stationary = fit_attention_kernel(&grid, &grid, head(0.0).view(), 6)
            .expect("stationary head fit should succeed");
        assert!(
            stationary.is_stationary,
            "a stationary head must not be flagged: stationary sse {}, separable sse {}",
            stationary.stationary.sse,
            stationary.separable.sse
        );
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
