//! Typed output-Fisher row-metric construction for SAE fits (issue #2236).
//!
//! Harvest shards expose factors as `(n, p, rank)` arrays. This module owns
//! their shape/rank validation, provenance policy, row-major packing, and the
//! choice of [`RowMetric`] constructor. Bindings only borrow the input array,
//! create [`SaeFisherRowMetricRequest`], and map the returned error.

use std::sync::Arc;

use gam_problem::RowMetric;
use ndarray::{Array2, ArrayView1, ArrayView3};

/// Scientific provenance and likelihood role of a supplied output-Fisher
/// factor stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeFisherMetricProvenance {
    /// Same-position output Fisher; drives geometry but not reconstruction GLS.
    OutputFisher,
    /// Future-position/KV aggregate; drives geometry but not reconstruction GLS.
    OutputFisherDownstream,
    /// Fixed output Fisher used as the reconstruction likelihood metric.
    BehavioralFisher,
}

impl SaeFisherMetricProvenance {
    /// Parse the public harvest tag. Absence selects same-position output Fisher.
    pub fn from_tag(tag: Option<&str>) -> Result<Self, String> {
        match tag.unwrap_or("output_fisher") {
            "output_fisher" => Ok(Self::OutputFisher),
            "output_fisher_downstream" => Ok(Self::OutputFisherDownstream),
            "behavioral_fisher" => Ok(Self::BehavioralFisher),
            other => Err(format!(
                "fisher_provenance must be 'output_fisher', 'output_fisher_downstream', or \
                 'behavioral_fisher'; got {other:?}"
            )),
        }
    }

    /// Stable public tag carried by reports and serialized payloads.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::OutputFisher => "output_fisher",
            Self::OutputFisherDownstream => "output_fisher_downstream",
            Self::BehavioralFisher => "behavioral_fisher",
        }
    }
}

/// Borrowed typed request for one SAE output-Fisher row metric.
#[derive(Debug, Clone, Copy)]
pub struct SaeFisherRowMetricRequest<'a> {
    factors: ArrayView3<'a, f64>,
    expected_rows: usize,
    expected_output_dim: usize,
    provenance: SaeFisherMetricProvenance,
    mass_residual: Option<ArrayView1<'a, f64>>,
}

impl<'a> SaeFisherRowMetricRequest<'a> {
    /// Build a typed request from the public provenance tag. The optional
    /// truncation-mass diagnostic is validated for row alignment but is not
    /// part of the metric itself.
    pub fn from_tag(
        factors: ArrayView3<'a, f64>,
        expected_rows: usize,
        expected_output_dim: usize,
        provenance: Option<&str>,
        mass_residual: Option<ArrayView1<'a, f64>>,
    ) -> Result<Self, String> {
        Ok(Self {
            factors,
            expected_rows,
            expected_output_dim,
            provenance: SaeFisherMetricProvenance::from_tag(provenance)?,
            mass_residual,
        })
    }

    /// Resolved scientific provenance after applying the default tag policy.
    pub const fn provenance(&self) -> SaeFisherMetricProvenance {
        self.provenance
    }
}

/// Validate and pack a natural `(n, p, rank)` factor stack into the canonical
/// `(n, p*rank)` [`RowMetric`] layout, then install its typed provenance.
pub fn build_sae_fisher_row_metric(
    request: SaeFisherRowMetricRequest<'_>,
) -> Result<RowMetric, String> {
    let shape = request.factors.shape();
    if shape[0] != request.expected_rows || shape[1] != request.expected_output_dim {
        return Err(format!(
            "fisher_factors U must be (n, p, rank)=({}, {}, rank); got shape {:?}",
            request.expected_rows, request.expected_output_dim, shape
        ));
    }
    let rank = shape[2];
    if rank == 0 {
        return Err("fisher_factors U rank (last axis) must be >= 1".to_string());
    }
    if rank > request.expected_output_dim {
        return Err(format!(
            "fisher_factors U rank {rank} exceeds output dim p={}",
            request.expected_output_dim
        ));
    }
    if let Some(mass_residual) = request.mass_residual {
        if mass_residual.len() != request.expected_rows {
            return Err(format!(
                "fisher_factors mass_residual must have {} rows; got {}",
                request.expected_rows,
                mass_residual.len()
            ));
        }
    }

    let packed_width = request
        .expected_output_dim
        .checked_mul(rank)
        .ok_or_else(|| {
            format!(
                "fisher_factors packed width overflows for p={} and rank={rank}",
                request.expected_output_dim
            )
        })?;
    let mut packed = Array2::<f64>::zeros((request.expected_rows, packed_width));
    for row in 0..request.expected_rows {
        for output in 0..request.expected_output_dim {
            for component in 0..rank {
                packed[[row, output * rank + component]] =
                    request.factors[[row, output, component]];
            }
        }
    }
    let packed = Arc::new(packed);
    match request.provenance {
        SaeFisherMetricProvenance::OutputFisher => {
            RowMetric::output_fisher(packed, request.expected_output_dim, rank)
        }
        SaeFisherMetricProvenance::OutputFisherDownstream => {
            RowMetric::output_fisher_downstream(packed, request.expected_output_dim, rank)
        }
        SaeFisherMetricProvenance::BehavioralFisher => {
            RowMetric::behavioral_fisher(packed, request.expected_output_dim, rank)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_problem::MetricProvenance;
    use ndarray::{Array1, Array3};

    #[test]
    fn default_tag_builds_same_position_output_fisher() {
        let mut factors = Array3::<f64>::zeros((2, 3, 2));
        factors[[0, 0, 0]] = 1.0;
        factors[[1, 2, 1]] = 2.0;
        let mass_residual = Array1::<f64>::zeros(2);
        let request = SaeFisherRowMetricRequest::from_tag(
            factors.view(),
            2,
            3,
            None,
            Some(mass_residual.view()),
        )
        .expect("typed request");
        assert_eq!(
            request.provenance(),
            SaeFisherMetricProvenance::OutputFisher
        );
        let metric = build_sae_fisher_row_metric(request).expect("row metric");
        assert_eq!(metric.n_rows(), 2);
        assert_eq!(metric.p_out(), 3);
        assert_eq!(
            metric.provenance(),
            MetricProvenance::OutputFisher { rank: 2 }
        );
        assert!(!metric.whitens_likelihood());
    }

    #[test]
    fn behavioral_tag_and_validation_are_core_owned() {
        let factors = Array3::<f64>::ones((2, 3, 1));
        let request = SaeFisherRowMetricRequest::from_tag(
            factors.view(),
            2,
            3,
            Some("behavioral_fisher"),
            None,
        )
        .expect("behavioral request");
        let metric = build_sae_fisher_row_metric(request).expect("behavioral metric");
        assert_eq!(
            metric.provenance(),
            MetricProvenance::BehavioralFisher { probes: 1 }
        );
        assert!(metric.whitens_likelihood());

        let bad_tag =
            SaeFisherRowMetricRequest::from_tag(factors.view(), 2, 3, Some("mystery"), None);
        assert!(bad_tag.is_err());
        let bad_rows = SaeFisherRowMetricRequest::from_tag(factors.view(), 3, 3, None, None)
            .expect("shape validation runs in entry");
        assert!(build_sae_fisher_row_metric(bad_rows).is_err());
    }
}
