//! Typed atom geometry authority shared by seed, fit, rebuild, and OOS paths.
//!
//! A basis width is a theorem of a topology plus its resolution; it is not an
//! independently writable property.  Likewise, the smoothing operator belongs
//! to a declared reference metric.  This module keeps those two choices paired
//! so persisted scalar metadata is validated once at the boundary and every
//! internal consumer rebuilds the same evaluator and penalty.

use super::*;

/// Basis-native resolution of one analytic atom family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaeBasisResolution {
    PeriodicHarmonics { order: usize },
    SphereChart,
    TorusHarmonics { per_axis_order: usize },
    ProjectivePlaneHarmonics { quotient_order: usize },
    KleinBottleHarmonics { per_axis_order: usize },
    DuchonCoordinates { basis_size: usize },
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
///
/// These are geometric declarations, not coefficient-size fallbacks.  The two
/// quotient metrics route directly to the cover-restricted spectral operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeReferenceMetricPlan {
    UnitCircle,
    SphereChart,
    UnitFlatTorus,
    UnitRoundProjectivePlane,
    UnitFlatKleinBottle,
    EuclideanDuchon,
    EuclideanPolynomial,
    UnitPoincareBall,
    CylinderProduct,
    MobiusQuotient,
    DiscreteCounting,
    CallerProvided,
}

/// Complete analytic geometry plan for one atom.
#[derive(Debug, Clone, PartialEq, Eq)]
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
        let valid = matches!(
            (&kind, latent_dim, &resolution, reference_metric),
            (
                SaeAtomBasisKind::Periodic,
                1,
                SaeBasisResolution::PeriodicHarmonics { order: 1.. },
                SaeReferenceMetricPlan::UnitCircle,
            )
                | (
                    SaeAtomBasisKind::Sphere,
                    2,
                    SaeBasisResolution::SphereChart,
                    SaeReferenceMetricPlan::SphereChart,
                )
                | (
                    SaeAtomBasisKind::Torus,
                    _,
                    SaeBasisResolution::TorusHarmonics { per_axis_order: 1.. },
                    SaeReferenceMetricPlan::UnitFlatTorus,
                )
                | (
                    SaeAtomBasisKind::ProjectivePlane,
                    2,
                    SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order: 1.. },
                    SaeReferenceMetricPlan::UnitRoundProjectivePlane,
                )
                | (
                    SaeAtomBasisKind::KleinBottle,
                    2,
                    SaeBasisResolution::KleinBottleHarmonics { per_axis_order: 2.. },
                    SaeReferenceMetricPlan::UnitFlatKleinBottle,
                )
                | (
                    SaeAtomBasisKind::Duchon,
                    _,
                    SaeBasisResolution::DuchonCoordinates { basis_size: 1.. },
                    SaeReferenceMetricPlan::EuclideanDuchon,
                )
                | (
                    SaeAtomBasisKind::Linear | SaeAtomBasisKind::EuclideanPatch,
                    _,
                    SaeBasisResolution::Polynomial { .. },
                    SaeReferenceMetricPlan::EuclideanPolynomial,
                )
                | (
                    SaeAtomBasisKind::Poincare,
                    _,
                    SaeBasisResolution::Polynomial { .. },
                    SaeReferenceMetricPlan::UnitPoincareBall,
                )
                | (
                    SaeAtomBasisKind::Cylinder,
                    2,
                    SaeBasisResolution::CylinderHarmonics {
                        circle_order: 1..,
                        ..
                    },
                    SaeReferenceMetricPlan::CylinderProduct,
                )
                | (
                    SaeAtomBasisKind::Mobius,
                    2,
                    SaeBasisResolution::MobiusHarmonics {
                        circle_order: 1..,
                        ..
                    },
                    SaeReferenceMetricPlan::MobiusQuotient,
                )
                | (
                    SaeAtomBasisKind::FiniteSet,
                    _,
                    SaeBasisResolution::FiniteAnchors { anchors: 1.. },
                    SaeReferenceMetricPlan::DiscreteCounting,
                )
                | (
                    SaeAtomBasisKind::Precomputed(_),
                    _,
                    SaeBasisResolution::Precomputed { basis_size: 1.. },
                    SaeReferenceMetricPlan::CallerProvided,
                )
        );
        if !valid {
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
            SaeReferenceMetricPlan::UnitRoundProjectivePlane,
        )
    }

    pub fn klein_bottle(per_axis_order: usize) -> Result<Self, String> {
        Self::new(
            SaeAtomBasisKind::KleinBottle,
            2,
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order },
            SaeReferenceMetricPlan::UnitFlatKleinBottle,
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

    pub fn reference_metric(&self) -> SaeReferenceMetricPlan {
        self.reference_metric
    }

    /// Width derived from the typed resolution.  No caller supplies a second
    /// width for analytic families.
    pub fn basis_size(&self) -> Result<usize, String> {
        match self.resolution {
            SaeBasisResolution::PeriodicHarmonics { order } => sae_periodic_basis_size(order),
            SaeBasisResolution::SphereChart => Ok(SAE_SPHERE_BASIS_SIZE),
            SaeBasisResolution::TorusHarmonics { per_axis_order } => {
                TorusHarmonicEvaluator::new(self.latent_dim, per_axis_order)
                    .map(|evaluator| evaluator.basis_size())
            }
            SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order } => {
                projective_plane_basis_size(quotient_order)
            }
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order } => {
                klein_bottle_basis_size(per_axis_order)
            }
            SaeBasisResolution::DuchonCoordinates { basis_size }
            | SaeBasisResolution::Precomputed { basis_size } => Ok(basis_size),
            SaeBasisResolution::Polynomial { degree } => {
                Ok(gam_terms::basis::monomial_exponents(self.latent_dim, degree).len())
            }
            SaeBasisResolution::CylinderHarmonics {
                circle_order,
                line_degree,
            } => CylinderHarmonicEvaluator::new(circle_order, line_degree)
                .map(|evaluator| evaluator.basis_size()),
            SaeBasisResolution::MobiusHarmonics {
                circle_order,
                width_degree,
            } => MobiusHarmonicEvaluator::new(circle_order, width_degree)
                .map(|evaluator| evaluator.basis_size()),
            SaeBasisResolution::FiniteAnchors { anchors } => Ok(anchors),
        }
    }

    /// Scalar harmonic metadata emitted by the current artifact wire format.
    /// This is a derived projection of [`Self::resolution`], never an authority.
    pub fn stored_harmonic_order(&self) -> usize {
        match self.resolution {
            SaeBasisResolution::PeriodicHarmonics { order } => order,
            SaeBasisResolution::TorusHarmonics { per_axis_order }
            | SaeBasisResolution::KleinBottleHarmonics { per_axis_order } => per_axis_order,
            SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order } => quotient_order,
            SaeBasisResolution::CylinderHarmonics { circle_order, .. }
            | SaeBasisResolution::MobiusHarmonics { circle_order, .. } => circle_order,
            _ => 0,
        }
    }

    /// Reconstruct and validate a typed plan from persisted scalar metadata.
    /// `persisted_basis_size` is an integrity check for analytic families, not a
    /// second source of width.
    pub fn from_persisted(
        kind: SaeAtomBasisKind,
        latent_dim: usize,
        stored_harmonic_order: usize,
        persisted_basis_size: usize,
    ) -> Result<Self, String> {
        let (resolution, reference_metric) = match &kind {
            SaeAtomBasisKind::Periodic => (
                SaeBasisResolution::PeriodicHarmonics {
                    order: stored_harmonic_order,
                },
                SaeReferenceMetricPlan::UnitCircle,
            ),
            SaeAtomBasisKind::Sphere => {
                require_zero_harmonics(&kind, stored_harmonic_order)?;
                (
                    SaeBasisResolution::SphereChart,
                    SaeReferenceMetricPlan::SphereChart,
                )
            }
            SaeAtomBasisKind::Torus => (
                SaeBasisResolution::TorusHarmonics {
                    per_axis_order: stored_harmonic_order,
                },
                SaeReferenceMetricPlan::UnitFlatTorus,
            ),
            SaeAtomBasisKind::ProjectivePlane => (
                SaeBasisResolution::ProjectivePlaneHarmonics {
                    quotient_order: stored_harmonic_order,
                },
                SaeReferenceMetricPlan::UnitRoundProjectivePlane,
            ),
            SaeAtomBasisKind::KleinBottle => (
                SaeBasisResolution::KleinBottleHarmonics {
                    per_axis_order: stored_harmonic_order,
                },
                SaeReferenceMetricPlan::UnitFlatKleinBottle,
            ),
            SaeAtomBasisKind::Duchon => {
                require_zero_harmonics(&kind, stored_harmonic_order)?;
                (
                    SaeBasisResolution::DuchonCoordinates {
                        basis_size: persisted_basis_size,
                    },
                    SaeReferenceMetricPlan::EuclideanDuchon,
                )
            }
            SaeAtomBasisKind::Linear
            | SaeAtomBasisKind::EuclideanPatch
            | SaeAtomBasisKind::Poincare => {
                require_zero_harmonics(&kind, stored_harmonic_order)?;
                let degree = sae_euclidean_degree_for_basis_size(latent_dim, persisted_basis_size)?;
                let metric = if matches!(kind, SaeAtomBasisKind::Poincare) {
                    SaeReferenceMetricPlan::UnitPoincareBall
                } else {
                    SaeReferenceMetricPlan::EuclideanPolynomial
                };
                (SaeBasisResolution::Polynomial { degree }, metric)
            }
            SaeAtomBasisKind::Cylinder => {
                let circle_width = stored_harmonic_order
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| {
                        format!(
                            "cylinder harmonic width overflowed for order {stored_harmonic_order}"
                        )
                    })?;
                if stored_harmonic_order == 0
                    || persisted_basis_size == 0
                    || persisted_basis_size % circle_width != 0
                {
                    return Err(format!(
                        "cylinder width {persisted_basis_size} is incompatible with circle order {stored_harmonic_order}"
                    ));
                }
                (
                    SaeBasisResolution::CylinderHarmonics {
                        circle_order: stored_harmonic_order,
                        line_degree: persisted_basis_size / circle_width - 1,
                    },
                    SaeReferenceMetricPlan::CylinderProduct,
                )
            }
            SaeAtomBasisKind::Mobius => {
                if stored_harmonic_order == 0 {
                    return Err("Mobius harmonic order must be positive".to_string());
                }
                let width_degree = (0..=persisted_basis_size)
                    .find(|&degree| {
                        MobiusHarmonicEvaluator::new(stored_harmonic_order, degree)
                            .is_ok_and(|evaluator| evaluator.basis_size() == persisted_basis_size)
                    })
                    .ok_or_else(|| {
                        format!(
                            "Mobius width {persisted_basis_size} has no exact width degree for circle order {stored_harmonic_order}"
                        )
                    })?;
                (
                    SaeBasisResolution::MobiusHarmonics {
                        circle_order: stored_harmonic_order,
                        width_degree,
                    },
                    SaeReferenceMetricPlan::MobiusQuotient,
                )
            }
            SaeAtomBasisKind::FiniteSet => {
                require_zero_harmonics(&kind, stored_harmonic_order)?;
                (
                    SaeBasisResolution::FiniteAnchors {
                        anchors: persisted_basis_size,
                    },
                    SaeReferenceMetricPlan::DiscreteCounting,
                )
            }
            SaeAtomBasisKind::Precomputed(_) => {
                require_zero_harmonics(&kind, stored_harmonic_order)?;
                (
                    SaeBasisResolution::Precomputed {
                        basis_size: persisted_basis_size,
                    },
                    SaeReferenceMetricPlan::CallerProvided,
                )
            }
        };
        let plan = Self::new(kind, latent_dim, resolution, reference_metric)?;
        let derived = plan.basis_size()?;
        if derived != persisted_basis_size {
            return Err(format!(
                "persisted basis width {persisted_basis_size} disagrees with typed plan width {derived} for {:?}",
                plan.kind
            ));
        }
        Ok(plan)
    }

    /// Build the one analytic evaluator declared by this plan.
    pub fn build_evaluator(
        &self,
        centers: Option<&Array2<f64>>,
    ) -> Result<Arc<dyn SaeBasisSecondJet>, String> {
        let evaluator: Arc<dyn SaeBasisSecondJet> = match self.resolution {
            SaeBasisResolution::PeriodicHarmonics { order } => {
                Arc::new(PeriodicHarmonicEvaluator::new(sae_periodic_basis_size(order)?)?)
            }
            SaeBasisResolution::SphereChart => Arc::new(SphereChartEvaluator),
            SaeBasisResolution::TorusHarmonics { per_axis_order } => Arc::new(
                TorusHarmonicEvaluator::new(self.latent_dim, per_axis_order)?,
            ),
            SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order } => {
                Arc::new(QuotientSpectralEvaluator::projective_plane(quotient_order)?)
            }
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order } => Arc::new(
                QuotientSpectralEvaluator::klein_bottle(per_axis_order)?,
            ),
            SaeBasisResolution::DuchonCoordinates { .. } => {
                let centers = centers.ok_or_else(|| {
                    format!("{:?} requires persisted Duchon centers", self.kind)
                })?;
                Arc::new(DuchonCoordinateEvaluator::new(
                    centers.clone(),
                    sae_duchon_atom_m(centers.ncols()),
                )?)
            }
            SaeBasisResolution::Polynomial { degree } => {
                Arc::new(EuclideanPatchEvaluator::new(self.latent_dim, degree)?)
            }
            SaeBasisResolution::CylinderHarmonics {
                circle_order,
                line_degree,
            } => Arc::new(CylinderHarmonicEvaluator::new(circle_order, line_degree)?),
            SaeBasisResolution::MobiusHarmonics {
                circle_order,
                width_degree,
            } => Arc::new(MobiusHarmonicEvaluator::new(circle_order, width_degree)?),
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
        match self.resolution {
            SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order } => Ok(Some(
                QuotientSpectralEvaluator::projective_plane(quotient_order)?
                    .spectral_penalty(power)?,
            )),
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order } => Ok(Some(
                QuotientSpectralEvaluator::klein_bottle(per_axis_order)?
                    .spectral_penalty(power)?,
            )),
            _ => Ok(None),
        }
    }
}

fn require_zero_harmonics(kind: &SaeAtomBasisKind, order: usize) -> Result<(), String> {
    if order == 0 {
        Ok(())
    } else {
        Err(format!(
            "non-harmonic atom {kind:?} must store harmonic order zero; got {order}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotient_plan_is_the_only_width_and_metric_authority() {
        let rp2 = SaeAtomGeometryPlan::projective_plane(3).unwrap();
        assert_eq!(rp2.basis_size().unwrap(), 28);
        assert_eq!(rp2.stored_harmonic_order(), 3);
        assert_eq!(
            rp2.reference_metric(),
            SaeReferenceMetricPlan::UnitRoundProjectivePlane
        );

        let klein = SaeAtomGeometryPlan::klein_bottle(3).unwrap();
        assert_eq!(klein.basis_size().unwrap(), 24);
        assert_eq!(klein.stored_harmonic_order(), 3);
        assert_eq!(
            klein.reference_metric(),
            SaeReferenceMetricPlan::UnitFlatKleinBottle
        );
        assert!(SaeAtomGeometryPlan::klein_bottle(1).is_err());
    }

    #[test]
    fn persisted_quotient_width_is_validation_not_authority() {
        assert!(
            SaeAtomGeometryPlan::from_persisted(
                SaeAtomBasisKind::ProjectivePlane,
                2,
                3,
                27,
            )
            .is_err()
        );
        assert!(
            SaeAtomGeometryPlan::from_persisted(
                SaeAtomBasisKind::KleinBottle,
                2,
                3,
                25,
            )
            .is_err()
        );
    }
}
