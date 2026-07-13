//! Typed atom geometry authority shared by seed, fit, rebuild, and OOS paths.
//!
//! A basis width is a theorem of a topology plus its resolution; it is not an
//! independently writable property. Likewise, the smoothing operator belongs
//! to a declared reference metric. Persisted models store this tagged plan
//! directly. There is intentionally no loader that reconstructs it from the
//! former parallel `(kind, harmonic_order, basis_width)` scalars.

use super::*;

/// Basis-native resolution of one analytic atom family.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeBasisResolution {
    PeriodicHarmonics { order: usize },
    SphereChart,
    TorusHarmonics { per_axis_order: usize },
    ProjectivePlaneHarmonics { quotient_order: usize },
    KleinBottleHarmonics { per_axis_order: usize },
    /// Duchon centers are the resolution authority. The evaluator derives its
    /// width from these centers and the dimension-derived null-space order.
    DuchonCoordinates { centers: Array2<f64> },
    Polynomial { degree: usize },
    CylinderHarmonics {
        circle_order: usize,
        line_degree: usize,
    },
    MobiusHarmonics {
        circle_order: usize,
        width_degree: usize,
    },
    FiniteAnchors { anchors: usize },
    Precomputed { basis_size: usize },
}

/// Reference geometry whose function-space seminorm the atom stores.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeReferenceMetricPlan {
    UnitCircle,
    SphereChart,
    /// Flat rectangular torus with aspect `A = cosh(tau) >= 1`; `tau = 0`
    /// is square. This is the exact flat comparator for the donut at the same
    /// `tau` and therefore the same aspect.
    FlatRectangularTorus { tau: f64 },
    /// Embedded donut torus with aspect `A = cosh(tau) > 1`.
    EmbeddedDonutTorus { tau: f64 },
    RoundProjectivePlane,
    FlatKleinBottle,
    EuclideanDuchon,
    EuclideanPolynomial,
    UnitPoincareBall,
    CylinderProduct,
    MobiusQuotient,
    DiscreteCounting,
    CallerProvided,
}

/// Complete persisted analytic geometry plan for one atom.
#[derive(Debug, Clone, PartialEq)]
pub struct SaeAtomGeometryPlan {
    kind: SaeAtomBasisKind,
    latent_dim: usize,
    resolution: SaeBasisResolution,
    reference_metric: SaeReferenceMetricPlan,
}

impl SaeAtomGeometryPlan {
    /// Construct and validate an exact kind/resolution/reference-metric tuple.
    pub fn new(
        kind: SaeAtomBasisKind,
        latent_dim: usize,
        resolution: SaeBasisResolution,
        reference_metric: SaeReferenceMetricPlan,
    ) -> Result<Self, String> {
        if latent_dim == 0 {
            return Err("SaeAtomGeometryPlan requires latent_dim >= 1".to_string());
        }
        let semantic_match = match (&kind, latent_dim, &resolution, &reference_metric) {
            (
                SaeAtomBasisKind::Periodic,
                1,
                SaeBasisResolution::PeriodicHarmonics { order },
                SaeReferenceMetricPlan::UnitCircle,
            ) => *order >= 1,
            (
                SaeAtomBasisKind::Sphere,
                2,
                SaeBasisResolution::SphereChart,
                SaeReferenceMetricPlan::SphereChart,
            ) => true,
            (
                SaeAtomBasisKind::Torus,
                2,
                SaeBasisResolution::TorusHarmonics { per_axis_order },
                SaeReferenceMetricPlan::FlatRectangularTorus { tau },
            ) => *per_axis_order >= 1 && tau.is_finite() && *tau >= 0.0,
            (
                SaeAtomBasisKind::Torus,
                2,
                SaeBasisResolution::TorusHarmonics { per_axis_order },
                SaeReferenceMetricPlan::EmbeddedDonutTorus { tau },
            ) => *per_axis_order >= 1 && tau.is_finite() && *tau > 0.0,
            (
                SaeAtomBasisKind::ProjectivePlane,
                2,
                SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order },
                SaeReferenceMetricPlan::RoundProjectivePlane,
            ) => *quotient_order >= 1,
            (
                SaeAtomBasisKind::KleinBottle,
                2,
                SaeBasisResolution::KleinBottleHarmonics { per_axis_order },
                SaeReferenceMetricPlan::FlatKleinBottle,
            ) => *per_axis_order >= 2,
            (
                SaeAtomBasisKind::Duchon,
                _,
                SaeBasisResolution::DuchonCoordinates { centers },
                SaeReferenceMetricPlan::EuclideanDuchon,
            ) => {
                centers.nrows() > 0
                    && centers.ncols() == latent_dim
                    && centers.iter().all(|value| value.is_finite())
            }
            (
                SaeAtomBasisKind::Linear,
                _,
                SaeBasisResolution::Polynomial { degree },
                SaeReferenceMetricPlan::EuclideanPolynomial,
            ) => *degree == 1,
            (
                SaeAtomBasisKind::EuclideanPatch,
                _,
                SaeBasisResolution::Polynomial { degree },
                SaeReferenceMetricPlan::EuclideanPolynomial,
            ) => {
                (SAE_EUCLIDEAN_PATCH_MAX_DEGREE
                    ..=SAE_EUCLIDEAN_PATCH_RECOVERY_MAX_DEGREE)
                    .contains(degree)
            }
            (
                SaeAtomBasisKind::Poincare,
                _,
                SaeBasisResolution::Polynomial { degree },
                SaeReferenceMetricPlan::UnitPoincareBall,
            ) => *degree == SAE_EUCLIDEAN_PATCH_MAX_DEGREE,
            (
                SaeAtomBasisKind::Cylinder,
                2,
                SaeBasisResolution::CylinderHarmonics { circle_order, .. },
                SaeReferenceMetricPlan::CylinderProduct,
            ) => *circle_order >= 1,
            (
                SaeAtomBasisKind::Mobius,
                2,
                SaeBasisResolution::MobiusHarmonics { circle_order, .. },
                SaeReferenceMetricPlan::MobiusQuotient,
            ) => *circle_order >= 1,
            (
                SaeAtomBasisKind::FiniteSet,
                _,
                SaeBasisResolution::FiniteAnchors { anchors },
                SaeReferenceMetricPlan::DiscreteCounting,
            ) => *anchors >= 2,
            (
                SaeAtomBasisKind::Precomputed(_),
                _,
                SaeBasisResolution::Precomputed { basis_size },
                SaeReferenceMetricPlan::CallerProvided,
            ) => *basis_size >= 1,
            _ => false,
        };
        if !semantic_match {
            return Err(format!(
                "invalid atom geometry tuple: kind={kind:?}, latent_dim={latent_dim}, resolution={resolution:?}, reference_metric={reference_metric:?}"
            ));
        }
        let plan = Self {
            kind,
            latent_dim,
            resolution,
            reference_metric,
        };
        plan.basis_size()?;
        Ok(plan)
    }

    pub fn projective_plane(quotient_order: usize) -> Result<Self, String> {
        Self::new(
            SaeAtomBasisKind::ProjectivePlane,
            2,
            SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order },
            SaeReferenceMetricPlan::RoundProjectivePlane,
        )
    }

    pub fn klein_bottle(per_axis_order: usize) -> Result<Self, String> {
        Self::new(
            SaeAtomBasisKind::KleinBottle,
            2,
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order },
            SaeReferenceMetricPlan::FlatKleinBottle,
        )
    }

    pub fn kind(&self) -> &SaeAtomBasisKind {
        &self.kind
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn resolution(&self) -> &SaeBasisResolution {
        &self.resolution
    }

    pub fn reference_metric(&self) -> &SaeReferenceMetricPlan {
        &self.reference_metric
    }

    pub fn duchon_centers(&self) -> Option<&Array2<f64>> {
        match &self.resolution {
            SaeBasisResolution::DuchonCoordinates { centers } => Some(centers),
            _ => None,
        }
    }

    /// Width derived from the tagged resolution.
    pub fn basis_size(&self) -> Result<usize, String> {
        match &self.resolution {
            SaeBasisResolution::PeriodicHarmonics { order } => sae_periodic_basis_size(*order),
            SaeBasisResolution::SphereChart => Ok(SAE_SPHERE_BASIS_SIZE),
            SaeBasisResolution::TorusHarmonics { per_axis_order } => {
                TorusHarmonicEvaluator::new(self.latent_dim, *per_axis_order)
                    .map(|evaluator| evaluator.basis_size())
            }
            SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order } => {
                projective_plane_basis_size(*quotient_order)
            }
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order } => {
                klein_bottle_basis_size(*per_axis_order)
            }
            SaeBasisResolution::DuchonCoordinates { centers } => {
                let evaluator = DuchonCoordinateEvaluator::new(
                    centers.clone(),
                    sae_duchon_atom_m(self.latent_dim),
                )?;
                let probe = Array2::<f64>::zeros((1, self.latent_dim));
                evaluator
                    .evaluate(probe.view())
                    .map(|(phi, _)| phi.ncols())
            }
            SaeBasisResolution::Polynomial { degree } => Ok(
                gam_terms::basis::monomial_exponents(self.latent_dim, *degree).len(),
            ),
            SaeBasisResolution::CylinderHarmonics {
                circle_order,
                line_degree,
            } => CylinderHarmonicEvaluator::new(*circle_order, *line_degree)
                .map(|evaluator| evaluator.basis_size()),
            SaeBasisResolution::MobiusHarmonics {
                circle_order,
                width_degree,
            } => MobiusHarmonicEvaluator::new(*circle_order, *width_degree)
                .map(|evaluator| evaluator.basis_size()),
            SaeBasisResolution::FiniteAnchors { anchors } => Ok(*anchors),
            SaeBasisResolution::Precomputed { basis_size } => Ok(*basis_size),
        }
    }

    /// Build the one analytic evaluator declared by this plan.
    pub fn build_evaluator(&self) -> Result<Arc<dyn SaeBasisSecondJet>, String> {
        let evaluator: Arc<dyn SaeBasisSecondJet> = match &self.resolution {
            SaeBasisResolution::PeriodicHarmonics { order } => Arc::new(
                PeriodicHarmonicEvaluator::new(sae_periodic_basis_size(*order)?)?,
            ),
            SaeBasisResolution::SphereChart => Arc::new(SphereChartEvaluator),
            SaeBasisResolution::TorusHarmonics { per_axis_order } => Arc::new(
                TorusHarmonicEvaluator::new(self.latent_dim, *per_axis_order)?,
            ),
            SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order } => Arc::new(
                QuotientSpectralEvaluator::projective_plane(*quotient_order)?,
            ),
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order } => Arc::new(
                QuotientSpectralEvaluator::klein_bottle(*per_axis_order)?,
            ),
            SaeBasisResolution::DuchonCoordinates { centers } => {
                Arc::new(DuchonCoordinateEvaluator::new(
                    centers.clone(),
                    sae_duchon_atom_m(self.latent_dim),
                )?)
            }
            SaeBasisResolution::Polynomial { degree } => Arc::new(
                EuclideanPatchEvaluator::new(self.latent_dim, *degree)?,
            ),
            SaeBasisResolution::CylinderHarmonics {
                circle_order,
                line_degree,
            } => Arc::new(CylinderHarmonicEvaluator::new(
                *circle_order,
                *line_degree,
            )?),
            SaeBasisResolution::MobiusHarmonics {
                circle_order,
                width_degree,
            } => Arc::new(MobiusHarmonicEvaluator::new(
                *circle_order,
                *width_degree,
            )?),
            SaeBasisResolution::FiniteAnchors { .. } => {
                return Err("finite-set atoms have no continuous analytic evaluator".to_string());
            }
            SaeBasisResolution::Precomputed { .. } => {
                return Err("precomputed atoms have no analytic evaluator".to_string());
            }
        };
        Ok(evaluator)
    }

    /// Exact cover-restricted spectral penalty for quotient atoms.
    pub fn quotient_spectral_penalty(&self, power: u32) -> Result<Option<Array2<f64>>, String> {
        match &self.resolution {
            SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order } => Ok(Some(
                QuotientSpectralEvaluator::projective_plane(*quotient_order)?
                    .spectral_penalty(power)?,
            )),
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order } => Ok(Some(
                QuotientSpectralEvaluator::klein_bottle(*per_axis_order)?
                    .spectral_penalty(power)?,
            )),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotient_plan_is_the_only_width_and_metric_authority() {
        let rp2 = SaeAtomGeometryPlan::projective_plane(3).unwrap();
        assert_eq!(rp2.basis_size().unwrap(), 28);
        assert_eq!(
            rp2.reference_metric(),
            &SaeReferenceMetricPlan::RoundProjectivePlane
        );

        let klein = SaeAtomGeometryPlan::klein_bottle(3).unwrap();
        assert_eq!(klein.basis_size().unwrap(), 24);
        assert_eq!(
            klein.reference_metric(),
            &SaeReferenceMetricPlan::FlatKleinBottle
        );
        assert!(SaeAtomGeometryPlan::klein_bottle(1).is_err());
    }

    #[test]
    fn semantic_plan_mismatches_are_rejected() {
        assert!(
            SaeAtomGeometryPlan::new(
                SaeAtomBasisKind::Linear,
                2,
                SaeBasisResolution::Polynomial { degree: 2 },
                SaeReferenceMetricPlan::EuclideanPolynomial,
            )
            .is_err()
        );
        assert!(
            SaeAtomGeometryPlan::new(
                SaeAtomBasisKind::Torus,
                1,
                SaeBasisResolution::TorusHarmonics { per_axis_order: 2 },
                SaeReferenceMetricPlan::FlatRectangularTorus { tau: 0.0 },
            )
            .is_err()
        );
        assert!(
            SaeAtomGeometryPlan::new(
                SaeAtomBasisKind::FiniteSet,
                1,
                SaeBasisResolution::FiniteAnchors { anchors: 1 },
                SaeReferenceMetricPlan::DiscreteCounting,
            )
            .is_err()
        );
    }
}
