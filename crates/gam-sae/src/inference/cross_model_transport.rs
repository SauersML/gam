//! Cross-model concept transport on dimension-free chart coordinates.
//!
//! Inter-layer transport in [`layer_transport`](crate::inference::layer_transport)
//! assumes two charts live on matched observations of one model. Cross-model
//! transport keeps that contract but changes what is paired: each model first
//! extracts its own one-dimensional chart coordinate, then this module transports
//! those coordinates. For the top-2 activation-plane proxy, each model may have a
//! different ambient width (for example Qwen3-8B `d=4096` and Qwen3.6-35B
//! `d=2048`): the Procrustes/frame choice happens inside each model before this
//! API sees the data, and the transported object is only the circle angle. The
//! ambient dimension mismatch therefore never enters the transport fit.
//!
//! The map fit itself is not new machinery. [`fit_cross_model_transport`] calls
//! [`fit_transport_map`](crate::inference::layer_transport::fit_transport_map),
//! reads the empirical isometry defect from that fit, and classifies circle maps
//! with [`classify_circle_transport_fit`](crate::inference::transport_class::classify_circle_transport_fit).
//! The verdict follows the crate's measure-don't-latch convention: it reports
//! "consistent within noise/gauge" when the measured defects are at or below
//! their own uncertainty scale, and otherwise reports the measured obstruction
//! rather than fitting it away.

use crate::inference::layer_transport::{
    ChartTopology, DEFAULT_COMPOSITION_GRID, FittedTransport, fit_transport_map,
};
use crate::inference::transport_class::{
    CircleTransportClass, CircleTransportReport, classify_circle_transport_fit,
};
use ndarray::{Array1, ArrayView2};
use std::f64::consts::TAU;

/// A model-local one-dimensional coordinate ready for cross-model transport.
#[derive(Debug, Clone)]
pub struct ModelCoordinate {
    /// Human-readable model/checkpoint name for reports.
    pub model: String,
    /// Coordinate values on paired observations. Circle coordinates are radians.
    pub coordinate: Array1<f64>,
    /// Coordinate topology.
    pub topology: ChartTopology,
}

impl ModelCoordinate {
    pub fn new(model: impl Into<String>, coordinate: Array1<f64>, topology: ChartTopology) -> Self {
        Self {
            model: model.into(),
            coordinate,
            topology,
        }
    }

    /// Build a circle coordinate from model-local top-2 plane scores.
    ///
    /// `scores` has one row per observation and at least two columns. The angle
    /// is `atan2(col_b, col_a)` wrapped to `[0, 2π)`. The two columns are assumed
    /// to be the model's own aligned plane coordinates, such as PCA/SVD scores
    /// after any model-local sink/top-PC peeling. Because only the resulting
    /// angle is transported, the source activation widths may differ across
    /// models.
    pub fn from_plane_scores(
        model: impl Into<String>,
        scores: ArrayView2<'_, f64>,
        col_a: usize,
        col_b: usize,
    ) -> Result<Self, String> {
        let p = scores.ncols();
        if col_a >= p || col_b >= p || col_a == col_b {
            return Err(format!(
                "from_plane_scores: need two distinct columns inside {p}; got {col_a}, {col_b}"
            ));
        }
        let mut coord = Array1::<f64>::zeros(scores.nrows());
        for row in 0..scores.nrows() {
            let angle = scores[[row, col_b]].atan2(scores[[row, col_a]]);
            coord[row] = wrap_tau(angle);
        }
        Ok(Self::new(model, coord, ChartTopology::Circle))
    }
}

/// Measure-don't-latch verdict for a cross-model coordinate transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniversalityVerdict {
    /// The transport is a topology-preserving O(2) element, and the measured
    /// isometry/O(2) defects are no larger than their own noise/gauge scale.
    ConsistentWithSharedFeatureWithinNoise,
    /// A shared circle/interval is present, but the fitted map has a measured
    /// non-isometric reparameterization.
    SharedFeatureWithMeasuredReparameterization,
    /// The coordinate pairing does not support a shared feature chart.
    NotSharedFeature,
}

impl UniversalityVerdict {
    pub fn label(self) -> &'static str {
        match self {
            UniversalityVerdict::ConsistentWithSharedFeatureWithinNoise => {
                "consistent with a shared feature within noise/gauge"
            }
            UniversalityVerdict::SharedFeatureWithMeasuredReparameterization => {
                "shared feature with measured reparameterization"
            }
            UniversalityVerdict::NotSharedFeature => "not a shared feature by this coordinate",
        }
    }
}

/// Cross-model transport report, including the underlying fitted map.
#[derive(Debug, Clone)]
pub struct CrossModelTransportReport {
    pub model_from: String,
    pub model_to: String,
    pub n_obs: usize,
    pub fit: FittedTransport,
    pub circle: Option<CircleTransportReport>,
    pub verdict: UniversalityVerdict,
    /// Natural circular/O(2) gauge scale used for the verdict. For circle maps
    /// this is the classifier's `2/sqrt(n)` separation scale; for non-circle
    /// transports it is zero because no O(2) gauge is present.
    pub gauge_defect_scale: f64,
}

impl CrossModelTransportReport {
    pub fn isometry_defect(&self) -> f64 {
        self.fit.isometry_defect
    }

    pub fn winding(&self) -> Option<i8> {
        self.circle.as_ref().map(|r| r.winding)
    }

    pub fn phase(&self) -> Option<f64> {
        self.circle.as_ref().map(|r| r.phase)
    }

    pub fn o2_defect(&self) -> Option<f64> {
        self.circle.as_ref().map(|r| r.defect)
    }
}

/// Fit a cross-model transport between already-paired one-dimensional
/// coordinates.
pub fn fit_cross_model_transport(
    from: &ModelCoordinate,
    to: &ModelCoordinate,
) -> Result<CrossModelTransportReport, String> {
    validate_coordinate(from)?;
    validate_coordinate(to)?;
    if from.coordinate.len() != to.coordinate.len() {
        return Err(format!(
            "cross-model transport needs paired coordinates with equal length: {} vs {}",
            from.coordinate.len(),
            to.coordinate.len()
        ));
    }

    let fit = fit_transport_map(
        from.coordinate.view(),
        to.coordinate.view(),
        from.topology,
        to.topology,
    )
    .map_err(|e| {
        format!(
            "cross-model transport {}→{} failed: {e}",
            from.model, to.model
        )
    })?;

    let circle = classify_circle_transport_fit(
        &fit,
        from.topology,
        to.topology,
        0,
        1,
        DEFAULT_COMPOSITION_GRID,
    );
    let gauge_defect_scale = circle
        .as_ref()
        .map(|r| 2.0 / (r.n_samples as f64).sqrt())
        .unwrap_or(0.0);
    let verdict = universality_verdict(&fit, circle.as_ref(), gauge_defect_scale);

    Ok(CrossModelTransportReport {
        model_from: from.model.clone(),
        model_to: to.model.clone(),
        n_obs: from.coordinate.len(),
        fit,
        circle,
        verdict,
        gauge_defect_scale,
    })
}

fn validate_coordinate(coord: &ModelCoordinate) -> Result<(), String> {
    if coord.coordinate.is_empty() {
        return Err(format!("{} coordinate is empty", coord.model));
    }
    if coord.coordinate.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "{} coordinate contains non-finite values",
            coord.model
        ));
    }
    Ok(())
}

fn universality_verdict(
    fit: &FittedTransport,
    circle: Option<&CircleTransportReport>,
    gauge_defect_scale: f64,
) -> UniversalityVerdict {
    if !fit.topology_preserved {
        return UniversalityVerdict::NotSharedFeature;
    }

    if let Some(report) = circle {
        if report.class == CircleTransportClass::Mixing || !matches!(report.winding, -1 | 1) {
            return UniversalityVerdict::NotSharedFeature;
        }
    }

    let roundoff_scale = f64::EPSILON.sqrt();
    let isometry_noise_scale = fit.isometry_defect_se.max(roundoff_scale);
    let isometry_distinguished = fit.isometry_defect > isometry_noise_scale;
    let gauge_distinguished = circle
        .map(|report| report.defect > gauge_defect_scale)
        .unwrap_or(false);

    if isometry_distinguished || gauge_distinguished {
        UniversalityVerdict::SharedFeatureWithMeasuredReparameterization
    } else {
        UniversalityVerdict::ConsistentWithSharedFeatureWithinNoise
    }
}

fn wrap_tau(x: f64) -> f64 {
    x.rem_euclid(TAU)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn circle_coord(name: &str, values: Vec<f64>) -> ModelCoordinate {
        ModelCoordinate::new(name, Array1::from_vec(values), ChartTopology::Circle)
    }

    fn wrap_pi(x: f64) -> f64 {
        let w = (x + PI).rem_euclid(TAU) - PI;
        if w <= -PI { w + TAU } else { w }
    }

    #[test]
    fn recovers_cross_model_rotation_reflection() {
        let n = 256;
        let phi = 0.73_f64;
        let theta_a: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        let theta_b: Vec<f64> = theta_a.iter().map(|&t| wrap_tau(-t + phi)).collect();
        let a = circle_coord("model-a", theta_a);
        let b = circle_coord("model-b", theta_b);

        let report = fit_cross_model_transport(&a, &b).expect("cross-model fit");
        let circle = report.circle.as_ref().expect("circle classification");
        assert_eq!(circle.class, CircleTransportClass::Reflect);
        assert_eq!(circle.winding, -1);
        assert_eq!(report.o2_defect(), Some(circle.defect));
        assert!(wrap_pi(circle.phase - phi).abs() < 1e-6);
        assert!(report.fit.isometry_defect < 1e-10);
        assert!(report.fit.residual_rms < 1e-6);
        assert_eq!(
            report.verdict,
            UniversalityVerdict::ConsistentWithSharedFeatureWithinNoise
        );
    }

    #[test]
    fn different_manifold_reports_non_shared_feature() {
        let n = 256;
        let theta_a: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        let theta_b: Vec<f64> = theta_a.iter().map(|&t| wrap_tau(2.0 * t + 0.2)).collect();
        let a = circle_coord("model-a", theta_a);
        let b = circle_coord("model-b", theta_b);

        let report = fit_cross_model_transport(&a, &b).expect("cross-model fit");
        assert_eq!(report.fit.degree, Some(2));
        assert!(!report.fit.topology_preserved);
        assert_eq!(report.verdict, UniversalityVerdict::NotSharedFeature);
        assert!(report.fit.isometry_defect > 0.5);
    }

    #[test]
    fn plane_scores_become_dimension_free_angle_coordinate() {
        let n = 64;
        let mut scores = ndarray::Array2::<f64>::zeros((n, 3));
        for row in 0..n {
            let t = TAU * row as f64 / n as f64;
            scores[[row, 1]] = t.cos();
            scores[[row, 2]] = t.sin();
        }

        let coord = ModelCoordinate::from_plane_scores("wide-model", scores.view(), 1, 2)
            .expect("angle coordinate");
        assert_eq!(coord.topology, ChartTopology::Circle);
        assert_eq!(coord.coordinate.len(), n);
        for row in 0..n {
            let expected = TAU * row as f64 / n as f64;
            assert!(wrap_pi(coord.coordinate[row] - expected).abs() < 1e-12);
        }
    }
}
