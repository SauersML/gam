use super::fit::{
    DEFAULT_BLOCK_WIDTH, DenseBlockSource, HwePcaError, HwePcaModel, HweScaler, VariantBlockSource,
    apply_ld_weights,
};
use super::progress::{
    NoopProjectionProgress, ProjectionProgressObserver, ProjectionProgressStage,
};
use core::cmp::min;
use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatMut, Par, unzip, zip};
use std::error::Error;

pub struct HwePcaProjector<'model> {
    model: &'model HwePcaModel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroAlignmentAction {
    Zero,
    NaN,
}

impl Default for ZeroAlignmentAction {
    fn default() -> Self {
        Self::Zero
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectionOptions {
    pub missing_axis_renormalization: bool,
    pub return_alignment: bool,
    pub on_zero_alignment: ZeroAlignmentAction,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            missing_axis_renormalization: true,
            return_alignment: false,
            on_zero_alignment: ZeroAlignmentAction::NaN,
        }
    }
}

#[derive(Debug)]
pub struct ProjectionResult {
    pub scores: Mat<f64>,
    pub alignment: Option<Mat<f64>>,
}

impl HwePcaModel {
    pub fn projector(&self) -> HwePcaProjector<'_> {
        HwePcaProjector { model: self }
    }

    pub fn project_dense(
        &self,
        data: &[f64],
        n_samples: usize,
        n_variants: usize,
    ) -> Result<Mat<f64>, HwePcaError> {
        let mut source = DenseBlockSource::new(data, n_samples, n_variants)?;
        self.projector().project(&mut source)
    }

    pub fn project_with_options<S>(
        &self,
        source: &mut S,
        opts: &ProjectionOptions,
    ) -> Result<ProjectionResult, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let progress = NoopProjectionProgress::default();
        self.projector()
            .project_with_options_and_progress(source, opts, &progress)
    }
}
impl<'model> HwePcaProjector<'model> {
    pub fn project<S>(&self, source: &mut S) -> Result<Mat<f64>, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let options = ProjectionOptions::default();
        let result = self.project_with_options(source, &options)?;
        Ok(result.scores)
    }

    pub fn project_with_options<S>(
        &self,
        source: &mut S,
        opts: &ProjectionOptions,
    ) -> Result<ProjectionResult, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let progress = NoopProjectionProgress::default();
        self.project_with_options_and_progress(source, opts, &progress)
    }

    pub fn project_with_options_and_progress<S, P>(
        &self,
        source: &mut S,
        opts: &ProjectionOptions,
        progress: &P,
    ) -> Result<ProjectionResult, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
        P: ProjectionProgressObserver,
    {
        let n_samples = source.n_samples();
        let mut scores = Mat::zeros(n_samples, self.model.components());
        let mut alignment = if opts.return_alignment {
            Some(Mat::zeros(n_samples, self.model.components()))
        } else {
            None
        };

        self.project_into_with_options_and_progress(
            source,
            scores.as_mut(),
            opts,
            alignment.as_mut().map(|mat| mat.as_mut()),
            progress,
        )?;

        Ok(ProjectionResult { scores, alignment })
    }

    pub fn project_into<S>(
        &self,
        source: &mut S,
        scores: MatMut<'_, f64>,
    ) -> Result<(), HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let options = ProjectionOptions::default();
        let progress = NoopProjectionProgress::default();
        self.project_into_with_options_and_progress(source, scores, &options, None, &progress)
    }

    fn project_into_with_options_and_progress<S, P>(
        &self,
        source: &mut S,
        mut scores: MatMut<'_, f64>,
        opts: &ProjectionOptions,
        mut alignment_out: Option<MatMut<'_, f64>>,
        progress: &P,
    ) -> Result<(), HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
        P: ProjectionProgressObserver,
    {
        let n_samples = source.n_samples();
        let variant_hint = source.n_variants();
        let model_variants = self.model.n_variants();
        let expected_variants = if variant_hint == 0 {
            model_variants
        } else {
            variant_hint
        };
        let components = self.model.components();

        if opts.return_alignment && !opts.missing_axis_renormalization {
            return Err(HwePcaError::InvalidInput(
                "Alignment output requires missing axis renormalization",
            ));
        }

        if n_samples == 0 {
            return Err(HwePcaError::InvalidInput(
                "Projection requires at least one sample",
            ));
        }
        if variant_hint != 0 && variant_hint != model_variants {
            return Err(HwePcaError::InvalidInput(
                "Projection variant dimension must match fitted model",
            ));
        }
        if scores.nrows() != n_samples {
            return Err(HwePcaError::InvalidInput(
                "Projection output row count mismatch",
            ));
        }
        if scores.ncols() != components {
            return Err(HwePcaError::InvalidInput(
                "Projection output column count must equal number of components",
            ));
        }
        if let Some(ref alignment) = alignment_out {
            if alignment.nrows() != n_samples || alignment.ncols() != components {
                return Err(HwePcaError::InvalidInput(
                    "Alignment output shape must match projection output",
                ));
            }
        }

        scores.fill(0.0);

        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;

        let block_capacity =
            projection_block_capacity(self.model.n_samples(), n_samples, expected_variants);
        let elements = n_samples
            .checked_mul(block_capacity)
            .ok_or_else(|| HwePcaError::InvalidInput("Projection workspace size overflow"))?;
        let mut block_storage = vec![0.0f64; elements];
        let mut presence_storage = if opts.missing_axis_renormalization {
            vec![0.0f64; elements]
        } else {
            Vec::new()
        };
        let mut sq_loadings_storage = if opts.missing_axis_renormalization {
            vec![
                0.0f64;
                block_capacity.checked_mul(components).ok_or_else(|| {
                    HwePcaError::InvalidInput("Projection workspace size overflow")
                })?
            ]
        } else {
            Vec::new()
        };
        let scaler = self.model.scaler();
        let loadings = self.model.variant_loadings();
        let ld_weights = self.model.ld().map(|ld| ld.weights.as_slice());
        let mut processed = 0usize;
        let par = faer::get_global_parallelism();
        let mut alignment_r2 = if opts.missing_axis_renormalization {
            Some(Mat::zeros(n_samples, components))
        } else {
            None
        };

        progress.on_stage_start(ProjectionProgressStage::Projection, variant_hint);
        if variant_hint == 0 && expected_variants > 0 {
            progress.on_stage_total(ProjectionProgressStage::Projection, expected_variants);
        }

        loop {
            let filled = source
                .next_block_into(block_capacity, &mut block_storage)
                .map_err(|e| HwePcaError::Source(Box::new(e)))?;
            if filled == 0 {
                break;
            }
            if processed + filled > expected_variants {
                if variant_hint != 0 || processed + filled > model_variants {
                    return Err(HwePcaError::InvalidInput(
                        "VariantBlockSource returned more variants than reported",
                    ));
                }
            }

            let mut block = MatMut::from_column_major_slice_mut(
                &mut block_storage[..n_samples * filled],
                n_samples,
                filled,
            );

            if opts.missing_axis_renormalization {
                let mut presence_block = MatMut::from_column_major_slice_mut(
                    &mut presence_storage[..n_samples * filled],
                    n_samples,
                    filled,
                );
                scaler.standardize_block_with_mask(
                    block.as_mut(),
                    processed..processed + filled,
                    presence_block.as_mut(),
                    par,
                );

                if let Some(weights) = ld_weights {
                    apply_ld_weights(block.as_mut(), processed..processed + filled, weights);
                }

                let standardized = block.as_ref();
                let loadings_block = loadings.submatrix(processed, 0, filled, components);

                matmul(
                    scores.as_mut(),
                    Accum::Add,
                    standardized,
                    loadings_block,
                    1.0,
                    par,
                );

                let mut sq_block = MatMut::from_column_major_slice_mut(
                    &mut sq_loadings_storage[..filled * components],
                    filled,
                    components,
                );
                if let Some(weights) = ld_weights {
                    let start = processed.min(weights.len());
                    let end = (processed + filled).min(weights.len());
                    let weight_slice = &weights[start..end];
                    let mut row_idx = 0usize;
                    let weight_len = weight_slice.len();
                    zip!(sq_block.rb_mut(), loadings_block).for_each(|unzip!(sq, value)| {
                        let value = *value;
                        let weight_sq = if row_idx < weight_len {
                            let weight = weight_slice[row_idx];
                            weight * weight
                        } else {
                            1.0
                        };
                        *sq = weight_sq * value * value;
                        row_idx += 1;
                        if row_idx == filled {
                            row_idx = 0;
                        }
                    });
                } else {
                    zip!(sq_block.rb_mut(), loadings_block).for_each(|unzip!(sq, value)| {
                        let value = *value;
                        *sq = value * value;
                    });
                }

                if let Some(ref mut r2) = alignment_r2 {
                    let presence_block_ref = presence_block.as_ref();
                    let sq_block_ref = sq_block.as_ref();
                    matmul(
                        r2.as_mut(),
                        Accum::Add,
                        presence_block_ref,
                        sq_block_ref,
                        1.0,
                        par,
                    );
                }
            } else {
                standardize_projection_block(scaler, block.as_mut(), processed, filled, par);

                if let Some(weights) = ld_weights {
                    apply_ld_weights(block.as_mut(), processed..processed + filled, weights);
                }

                let standardized = block.as_ref();
                let loadings_block = loadings.submatrix(processed, 0, filled, components);

                matmul(
                    scores.as_mut(),
                    Accum::Add,
                    standardized,
                    loadings_block,
                    1.0,
                    par,
                );
            }

            processed += filled;
            progress.on_stage_advance(ProjectionProgressStage::Projection, processed);
        }

        if processed != expected_variants {
            if variant_hint != 0 {
                return Err(HwePcaError::InvalidInput(
                    "VariantBlockSource terminated early during projection",
                ));
            }
            if processed != model_variants {
                return Err(HwePcaError::InvalidInput(
                    "VariantBlockSource terminated early during projection",
                ));
            }
        }

        progress.on_stage_finish(ProjectionProgressStage::Projection);

        if let Some(r2) = alignment_r2 {
            let normalization = self.model.component_weighted_norms_sq();
            match alignment_out.as_mut() {
                Some(alignment_mat) => {
                    for col in 0..components {
                        let denom = normalization.get(col).copied().unwrap_or(0.0);
                        for row in 0..n_samples {
                            let mass = r2[(row, col)];
                            let score = &mut scores[(row, col)];
                            let align = &mut alignment_mat[(row, col)];
                            if mass > 0.0 && denom > 0.0 {
                                let norm = (mass / denom).sqrt();
                                *score /= norm;
                                *align = norm;
                            } else {
                                match opts.on_zero_alignment {
                                    ZeroAlignmentAction::Zero => {
                                        *score = 0.0;
                                    }
                                    ZeroAlignmentAction::NaN => {
                                        *score = f64::NAN;
                                    }
                                }
                                *align = 0.0;
                            }
                        }
                    }
                }
                None => {
                    for col in 0..components {
                        let denom = normalization.get(col).copied().unwrap_or(0.0);
                        for row in 0..n_samples {
                            let mass = r2[(row, col)];
                            let score = &mut scores[(row, col)];
                            if mass > 0.0 && denom > 0.0 {
                                let norm = (mass / denom).sqrt();
                                *score /= norm;
                            } else {
                                match opts.on_zero_alignment {
                                    ZeroAlignmentAction::Zero => {
                                        *score = 0.0;
                                    }
                                    ZeroAlignmentAction::NaN => {
                                        *score = f64::NAN;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else if alignment_out.is_some() {
            debug_assert!(
                opts.missing_axis_renormalization,
                "alignment_out should be None when renormalization is disabled",
            );
        }

        Ok(())
    }

    pub fn model(&self) -> &'model HwePcaModel {
        self.model
    }
}

fn standardize_projection_block(
    scaler: &HweScaler,
    block: MatMut<'_, f64>,
    offset: usize,
    filled: usize,
    par: Par,
) {
    scaler.standardize_block(block, offset..offset + filled, par);
}

fn projection_block_capacity(
    fitted_samples: usize,
    projected_samples: usize,
    n_variants: usize,
) -> usize {
    if n_variants == 0 {
        return 1;
    }
    let default = DEFAULT_BLOCK_WIDTH.max(1);
    let safe_reference = fitted_samples.checked_mul(default).unwrap_or(usize::MAX);
    let mut capacity = if projected_samples == 0 {
        default
    } else {
        safe_reference / projected_samples
    };
    if capacity == 0 {
        capacity = 1;
    }
    capacity = min(capacity, default);
    capacity = min(capacity, n_variants);
    capacity
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::sync::Arc;

    use super::super::fit::{FitOptions, LdConfig, LdWindow};
    use super::super::progress::NoopFitProgress;
    use serde_json::Value;

    const N_SAMPLES: usize = 3;
    const N_VARIANTS: usize = 4;
    const TOLERANCE: f64 = 1e-10;
    const TEST_COMPONENTS: usize = 2;

    fn sample_data() -> Vec<f64> {
        vec![
            0.0, 1.0, 2.0, // variant 0
            1.0, 2.0, 0.0, // variant 1
            2.0, 1.0, 0.0, // variant 2
            1.0, 0.0, 2.0, // variant 3
        ]
    }

    fn fit_example_model() -> HwePcaModel {
        let data = sample_data();
        let mut source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("source");
        HwePcaModel::fit_k(&mut source, TEST_COMPONENTS).expect("model fit")
    }

    fn fit_example_model_with_ld() -> HwePcaModel {
        let data = sample_data();
        let mut source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("source");
        let options = FitOptions {
            ld: Some(LdConfig {
                window: Some(LdWindow::Sites(3)),
                ridge: Some(1.0e-3),
                variant_keys: None,
            }),
        };
        let progress = Arc::new(NoopFitProgress::default());
        HwePcaModel::fit_k_with_options_and_progress(
            &mut source,
            TEST_COMPONENTS,
            &options,
            &progress,
        )
        .expect("model fit with ld")
    }

    fn model_with_zero_norm_component() -> HwePcaModel {
        let model = fit_example_model();
        let components = model.components();
        assert!(
            components >= 2,
            "example model should expose multiple components"
        );
        let zero_idx = components - 1;
        let mut value = serde_json::to_value(&model).expect("serialize model");

        if let Some(loadings) = value.get_mut("loadings").and_then(Value::as_object_mut) {
            let nrows = loadings
                .get("nrows")
                .and_then(Value::as_u64)
                .expect("loadings nrows") as usize;
            if let Some(data) = loadings.get_mut("data").and_then(Value::as_array_mut) {
                for row in 0..nrows {
                    let idx = zero_idx * nrows + row;
                    data[idx] = Value::from(0.0);
                }
            }
        }

        if let Some(scores) = value
            .get_mut("sample_scores")
            .and_then(Value::as_object_mut)
        {
            let nrows = scores
                .get("nrows")
                .and_then(Value::as_u64)
                .expect("scores nrows") as usize;
            if let Some(data) = scores.get_mut("data").and_then(Value::as_array_mut) {
                for row in 0..nrows {
                    let idx = zero_idx * nrows + row;
                    data[idx] = Value::from(0.0);
                }
            }
        }

        if let Some(singular_values) = value
            .get_mut("singular_values")
            .and_then(Value::as_array_mut)
        {
            singular_values[zero_idx] = Value::from(0.0);
        }
        if let Some(eigenvalues) = value.get_mut("eigenvalues").and_then(Value::as_array_mut) {
            if zero_idx < eigenvalues.len() {
                eigenvalues[zero_idx] = Value::from(0.0);
            }
        }
        if let Some(norms) = value
            .get_mut("component_weighted_norms_sq")
            .and_then(Value::as_array_mut)
        {
            norms[zero_idx] = Value::from(0.0);
        }

        serde_json::from_value(value).expect("deserialize mutated model")
    }

    fn recompute_component_norms_sq(model: &HwePcaModel) -> Vec<f64> {
        let loadings = model.variant_loadings();
        let ld_weights = model.ld().map(|ld| ld.weights.as_slice());
        let weights = ld_weights.unwrap_or(&[]);
        let n_weights = weights.len();
        let mut result = Vec::with_capacity(loadings.ncols());

        for col in 0..loadings.ncols() {
            let mut sum = 0.0f64;
            let mut compensation = 0.0f64;
            for row in 0..loadings.nrows() {
                let weight = if row < n_weights { weights[row] } else { 1.0 };
                let value = loadings[(row, col)];
                let weighted = weight * value;
                let square = weighted * weighted;
                let y = square - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }

            if !sum.is_finite() || sum < 0.0 {
                sum = 0.0;
            }
            result.push(sum);
        }

        result
    }

    #[test]
    fn cached_component_norms_match_manual_computation() {
        let model = fit_example_model();
        let manual = recompute_component_norms_sq(&model);
        let cached = model.component_weighted_norms_sq();
        assert_eq!(manual.len(), cached.len());
        for (idx, (expected, actual)) in manual.iter().zip(cached.iter()).enumerate() {
            assert!(
                (expected - actual).abs() <= 1e-12,
                "component {idx} mismatch: expected {expected}, cached {actual}",
            );
        }
    }

    #[test]
    fn cached_component_norms_match_with_ld_weights() {
        let model = fit_example_model_with_ld();
        let manual = recompute_component_norms_sq(&model);
        let cached = model.component_weighted_norms_sq();
        assert_eq!(manual.len(), cached.len());
        for (idx, (expected, actual)) in manual.iter().zip(cached.iter()).enumerate() {
            assert!(
                (expected - actual).abs() <= 1e-12,
                "component {idx} mismatch: expected {expected}, cached {actual}",
            );
        }
    }

    #[test]
    fn cached_component_norms_handle_zero_norm_components() {
        let model = model_with_zero_norm_component();
        let manual = recompute_component_norms_sq(&model);
        let cached = model.component_weighted_norms_sq();
        assert_eq!(manual.len(), cached.len());
        let mut saw_zero = false;
        for (idx, (expected, actual)) in manual.iter().zip(cached.iter()).enumerate() {
            if expected.abs() <= 1e-12 {
                saw_zero = true;
            }
            assert!(
                (expected - actual).abs() <= 1e-12,
                "component {idx} mismatch: expected {expected}, cached {actual}",
            );
        }
        assert!(saw_zero, "expected at least one near-zero norm component");
    }

    fn assert_mats_close(a: &Mat<f64>, b: &Mat<f64>, tol: f64) {
        assert_eq!(a.nrows(), b.nrows());
        assert_eq!(a.ncols(), b.ncols());
        for row in 0..a.nrows() {
            for col in 0..a.ncols() {
                let left = a[(row, col)];
                let right = b[(row, col)];
                if left.is_nan() && right.is_nan() {
                    continue;
                }
                assert!(
                    (left - right).abs() <= tol,
                    "mismatch at ({row}, {col}): {left} vs {right}"
                );
            }
        }
    }

    fn set_sample_to_nan(data: &mut [f64], sample_idx: usize) {
        for variant in 0..N_VARIANTS {
            data[variant * N_SAMPLES + sample_idx] = f64::NAN;
        }
    }

    fn set_variant_to_nan(data: &mut [f64], variant_idx: usize) {
        let start = variant_idx * N_SAMPLES;
        let end = start + N_SAMPLES;
        for value in &mut data[start..end] {
            *value = f64::NAN;
        }
    }
    #[test]
    fn renormalization_matches_baseline_without_missingness() {
        let model = fit_example_model();
        let data = sample_data();

        let mut baseline_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("baseline");
        let baseline = model
            .projector()
            .project(&mut baseline_source)
            .expect("baseline projection");

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let result = model
            .projector()
            .project_with_options(&mut renorm_source, &options)
            .expect("renormalized projection");
        assert_mats_close(&baseline, &result.scores, 1e-10);

        let alignment = result.alignment.expect("alignment");
        for row in 0..alignment.nrows() {
            for col in 0..alignment.ncols() {
                let norm = alignment[(row, col)];
                assert!(
                    (norm - 1.0).abs() <= 1e-12,
                    "alignment mismatch at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn ld_weighted_renormalization_matches_baseline_without_missingness() {
        let model = fit_example_model_with_ld();
        let data = sample_data();

        let mut raw_source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("raw");
        let raw_options = ProjectionOptions {
            missing_axis_renormalization: false,
            return_alignment: false,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let raw_scores = model
            .projector()
            .project_with_options(&mut raw_source, &raw_options)
            .expect("raw projection")
            .scores;

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let renorm_options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let renorm_result = model
            .projector()
            .project_with_options(&mut renorm_source, &renorm_options)
            .expect("renormalized projection");

        assert_mats_close(&raw_scores, &renorm_result.scores, TOLERANCE);

        let alignment = renorm_result.alignment.expect("alignment");
        for row in 0..alignment.nrows() {
            for col in 0..alignment.ncols() {
                let norm = alignment[(row, col)];
                assert!(
                    (norm - 1.0).abs() <= 1e-12,
                    "alignment mismatch at ({row}, {col})"
                );
            }
        }
    }
    #[test]
    fn zero_alignment_behavior_respected() {
        let model = fit_example_model();
        let mut data = sample_data();
        set_sample_to_nan(&mut data, 1);

        let mut raw_source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("raw");
        let raw_options = ProjectionOptions {
            missing_axis_renormalization: false,
            return_alignment: false,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let raw_scores = model
            .projector()
            .project_with_options(&mut raw_source, &raw_options)
            .expect("raw projection")
            .scores;

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::NaN,
        };
        let result = model
            .projector()
            .project_with_options(&mut renorm_source, &options)
            .expect("renormalized projection");
        let alignment = result.alignment.expect("alignment");

        for row in [0usize, 2usize] {
            for col in 0..result.scores.ncols() {
                let norm = alignment[(row, col)];
                if norm > 0.0 {
                    let expected = raw_scores[(row, col)] / norm;
                    let diff = (result.scores[(row, col)] - expected).abs();
                    assert!(
                        diff <= 1e-10,
                        "renormalized score mismatch at ({row}, {col})"
                    );
                    assert!(
                        (norm - 1.0).abs() <= 1e-12,
                        "alignment unexpectedly deviates from 1 at ({row}, {col})"
                    );
                }
            }
        }

        for col in 0..result.scores.ncols() {
            assert!(result.scores[(1, col)].is_nan());
            assert_eq!(alignment[(1, col)], 0.0);
        }
    }
    #[test]
    fn dropping_variant_matches_manual_renormalization() {
        let model = fit_example_model();
        let mut data = sample_data();
        let dropped_variant = 1;
        set_variant_to_nan(&mut data, dropped_variant);

        let mut raw_source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("raw");
        let raw_options = ProjectionOptions {
            missing_axis_renormalization: false,
            return_alignment: false,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let raw_scores = model
            .projector()
            .project_with_options(&mut raw_source, &raw_options)
            .expect("raw projection")
            .scores;

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let result = model
            .projector()
            .project_with_options(&mut renorm_source, &options)
            .expect("renormalized projection");
        let alignment = result.alignment.expect("alignment");

        let loadings = model.variant_loadings();
        let mut total_mass = Vec::with_capacity(result.scores.ncols());
        for col in 0..result.scores.ncols() {
            let mut sumsq = 0.0f64;
            for row in 0..loadings.nrows() {
                let value = loadings[(row, col)];
                sumsq += value * value;
            }
            total_mass.push(sumsq);
        }
        for col in 0..result.scores.ncols() {
            let missing = loadings[(dropped_variant, col)];
            let denom = total_mass[col];
            let retained_mass = (denom - missing * missing).max(0.0);
            let expected_norm = if denom > 0.0 {
                (retained_mass / denom).sqrt()
            } else {
                0.0
            };
            for row in 0..result.scores.nrows() {
                if expected_norm > 0.0 {
                    let expected = raw_scores[(row, col)] / expected_norm;
                    let actual = result.scores[(row, col)];
                    let diff = (actual - expected).abs();
                    assert!(
                        diff <= 1e-10,
                        "renormalized score mismatch at ({row}, {col})"
                    );
                    let norm = alignment[(row, col)];
                    let norm_diff = (norm - expected_norm).abs();
                    assert!(norm_diff <= 1e-12, "alignment mismatch at ({row}, {col})");
                } else {
                    assert_eq!(result.scores[(row, col)], 0.0);
                    assert_eq!(alignment[(row, col)], 0.0);
                }
            }
        }
    }

    #[test]
    fn ld_weighted_missingness_matches_manual_renormalization() {
        let model = fit_example_model_with_ld();
        let mut data = sample_data();
        data[1 * N_SAMPLES + 0] = f64::NAN;
        data[3 * N_SAMPLES + 2] = f64::NAN;

        let mut raw_source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("raw");
        let raw_options = ProjectionOptions {
            missing_axis_renormalization: false,
            return_alignment: false,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let raw_scores = model
            .projector()
            .project_with_options(&mut raw_source, &raw_options)
            .expect("raw projection")
            .scores;

        let mut renorm_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("renorm");
        let renorm_options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let renorm_result = model
            .projector()
            .project_with_options(&mut renorm_source, &renorm_options)
            .expect("renormalized projection");

        let alignment = renorm_result.alignment.expect("alignment");
        let renorm_scores = renorm_result.scores;
        let loadings = model.variant_loadings();
        let weights = model.ld().expect("ld weights").weights.clone();

        for row in 0..renorm_scores.nrows() {
            for col in 0..renorm_scores.ncols() {
                let mut observed_mass = 0.0f64;
                for variant in 0..loadings.nrows() {
                    let value = data[variant * N_SAMPLES + row];
                    if value.is_finite() {
                        let weight = if variant < weights.len() {
                            weights[variant]
                        } else {
                            1.0
                        };
                        let loading = loadings[(variant, col)];
                        let weighted = weight * loading;
                        observed_mass += weighted * weighted;
                    }
                }

                if observed_mass > 0.0 {
                    let expected_norm = observed_mass.sqrt();
                    let expected_score = raw_scores[(row, col)] / expected_norm;
                    let actual_score = renorm_scores[(row, col)];
                    let actual_norm = alignment[(row, col)];
                    assert!(
                        (actual_score - expected_score).abs() <= 1e-10,
                        "renormalized score mismatch at ({row}, {col})"
                    );
                    assert!(
                        (actual_norm - expected_norm).abs() <= 1e-12,
                        "alignment mismatch at ({row}, {col})"
                    );
                } else {
                    assert_eq!(renorm_scores[(row, col)], 0.0);
                    assert_eq!(alignment[(row, col)], 0.0);
                }
            }
        }
    }

    #[test]
    fn alignment_request_without_renormalization_fails() {
        let model = fit_example_model();
        let data = sample_data();

        let mut source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("source");
        let options = ProjectionOptions {
            missing_axis_renormalization: false,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };

        let err = model
            .projector()
            .project_with_options(&mut source, &options)
            .expect_err("alignment without renormalization should fail");

        assert!(matches!(err, HwePcaError::InvalidInput(_)));
    }
    struct ChunkedBlockSource<'a> {
        data: &'a [f64],
        dims: (usize, usize),
        cursor: usize,
        chunk: usize,
    }

    impl<'a> ChunkedBlockSource<'a> {
        fn new(data: &'a [f64], n_samples: usize, n_variants: usize, chunk: usize) -> Self {
            Self {
                data,
                dims: (n_samples, n_variants),
                cursor: 0,
                chunk: chunk.max(1),
            }
        }
    }

    impl<'a> VariantBlockSource for ChunkedBlockSource<'a> {
        type Error = Infallible;

        fn n_samples(&self) -> usize {
            self.dims.0
        }

        fn n_variants(&self) -> usize {
            self.dims.1
        }

        fn reset(&mut self) -> Result<(), Self::Error> {
            self.cursor = 0;
            Ok(())
        }

        fn next_block_into(
            &mut self,
            max_variants: usize,
            storage: &mut [f64],
        ) -> Result<usize, Self::Error> {
            if max_variants == 0 {
                return Ok(0);
            }
            let remaining = self.n_variants().saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(0);
            }
            let ncols = remaining.min(self.chunk).min(max_variants);
            let nrows = self.n_samples();
            let len = nrows * ncols;
            let start = self.cursor * nrows;
            let end = start + len;
            storage[..len].copy_from_slice(&self.data[start..end]);
            self.cursor += ncols;
            Ok(ncols)
        }
    }

    #[test]
    fn block_boundary_missingness_is_stable() {
        let model = fit_example_model();
        let mut data = sample_data();
        data[1 * N_SAMPLES + 0] = f64::NAN;
        data[2 * N_SAMPLES + 2] = f64::NAN;

        let mut dense_source = DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("dense");
        let options = ProjectionOptions {
            missing_axis_renormalization: true,
            return_alignment: true,
            on_zero_alignment: ZeroAlignmentAction::Zero,
        };
        let dense_result = model
            .projector()
            .project_with_options(&mut dense_source, &options)
            .expect("dense projection");

        let mut chunked_source = ChunkedBlockSource::new(&data, N_SAMPLES, N_VARIANTS, 2);
        let chunked_result = model
            .projector()
            .project_with_options(&mut chunked_source, &options)
            .expect("chunked projection");

        assert_mats_close(&dense_result.scores, &chunked_result.scores, TOLERANCE);

        let dense_alignment = dense_result.alignment.expect("dense alignment");
        let chunked_alignment = chunked_result.alignment.expect("chunked alignment");
        assert_mats_close(&dense_alignment, &chunked_alignment, TOLERANCE);
    }
}
