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
    PeriodicHarmonics {
        order: usize,
    },
    SphereChart,
    TorusHarmonics {
        per_axis_order: usize,
    },
    ProjectivePlaneHarmonics {
        quotient_order: usize,
    },
    KleinBottleHarmonics {
        per_axis_order: usize,
    },
    /// Duchon centers are the resolution authority. The evaluator derives its
    /// width from these centers and the dimension-derived null-space order.
    DuchonCoordinates {
        centers: Array2<f64>,
    },
    Polynomial {
        degree: usize,
    },
    CylinderHarmonics {
        circle_order: usize,
        line_degree: usize,
    },
    MobiusHarmonics {
        circle_order: usize,
        width_degree: usize,
    },
    FiniteAnchors {
        anchors: usize,
    },
    Precomputed {
        basis_size: usize,
    },
}

/// Reference geometry whose function-space seminorm the atom stores.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeReferenceMetricPlan {
    UnitCircle,
    SphereChart,
    /// Flat rectangular torus with aspect `A = cosh(tau) >= 1`; `tau = 0`
    /// is square. This is the exact flat comparator for the donut at the same
    /// `tau` and therefore the same aspect.
    FlatRectangularTorus {
        tau: f64,
    },
    /// Embedded donut torus with aspect `A = cosh(tau) > 1`.
    EmbeddedDonutTorus {
        tau: f64,
    },
    RoundProjectivePlane,
    FlatKleinBottle,
    EuclideanDuchon,
    EuclideanPolynomial,
    /// Unit-curvature Poincare ball with the fixed reference rows that define
    /// the conformal Dirichlet function Gram. These rows are model data: OOS
    /// rebuild must replay them exactly, never replace them with query rows.
    UnitPoincareBall {
        reference_coords: Array2<f64>,
    },
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
            ) => (SAE_EUCLIDEAN_PATCH_MAX_DEGREE..=SAE_EUCLIDEAN_PATCH_RACE_MAX_DEGREE)
                .contains(degree),
            (
                SaeAtomBasisKind::Poincare,
                _,
                SaeBasisResolution::Polynomial { degree },
                SaeReferenceMetricPlan::UnitPoincareBall { reference_coords },
            ) => {
                *degree == SAE_EUCLIDEAN_PATCH_MAX_DEGREE
                    && reference_coords.nrows() > 0
                    && reference_coords.ncols() == latent_dim
                    && reference_coords.iter().all(|value| value.is_finite())
            }
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

    pub(crate) fn reference_roughness_kind(&self) -> SaeReferenceRoughnessKind {
        match &self.reference_metric {
            SaeReferenceMetricPlan::UnitPoincareBall { .. } => {
                SaeReferenceRoughnessKind::PoincareConformalDirichlet
            }
            _ => SaeReferenceRoughnessKind::ProvidedFunctionGram,
        }
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
                evaluator.evaluate(probe.view()).map(|(phi, _)| phi.ncols())
            }
            SaeBasisResolution::Polynomial { degree } => {
                Ok(gam_terms::basis::monomial_exponents(self.latent_dim, *degree).len())
            }
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
            SaeBasisResolution::KleinBottleHarmonics { per_axis_order } => {
                Arc::new(QuotientSpectralEvaluator::klein_bottle(*per_axis_order)?)
            }
            SaeBasisResolution::DuchonCoordinates { centers } => {
                Arc::new(DuchonCoordinateEvaluator::new(
                    centers.clone(),
                    sae_duchon_atom_m(self.latent_dim),
                )?)
            }
            SaeBasisResolution::Polynomial { degree } => {
                Arc::new(EuclideanPatchEvaluator::new(self.latent_dim, *degree)?)
            }
            SaeBasisResolution::CylinderHarmonics {
                circle_order,
                line_degree,
            } => Arc::new(CylinderHarmonicEvaluator::new(*circle_order, *line_degree)?),
            SaeBasisResolution::MobiusHarmonics {
                circle_order,
                width_degree,
            } => Arc::new(MobiusHarmonicEvaluator::new(*circle_order, *width_degree)?),
            SaeBasisResolution::FiniteAnchors { .. } => {
                return Err("finite-set atoms have no continuous analytic evaluator".to_string());
            }
            SaeBasisResolution::Precomputed { .. } => {
                return Err("precomputed atoms have no analytic evaluator".to_string());
            }
        };
        Ok(evaluator)
    }

    /// Materialize the declared reference-function Gram without evaluating a
    /// caller coordinate block. Used when a plan is attached to an atom so the
    /// persisted metric and the atom's already-installed Gram cannot disagree.
    pub(crate) fn build_reference_penalty(&self) -> Result<Array2<f64>, String> {
        let evaluator = self.build_evaluator()?;
        self.reference_penalty(evaluator.as_ref())
    }

    /// Evaluate the plan's analytic basis and materialize the one declared
    /// reference-function Gram. This is the sole seed/rebuild/OOS authority:
    /// callers only validate and copy these arrays, never reconstruct a
    /// topology-specific penalty from raw widths or kind tags.
    pub(crate) fn evaluate_bundle(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<SaeAtomEvaluationBundle, String> {
        if coords.ncols() != self.latent_dim {
            return Err(format!(
                "SaeAtomGeometryPlan::evaluate_bundle: coordinate width {} != plan latent_dim {}",
                coords.ncols(),
                self.latent_dim
            ));
        }
        if coords.iter().any(|value| !value.is_finite()) {
            return Err(
                "SaeAtomGeometryPlan::evaluate_bundle: coordinates must be finite".to_string(),
            );
        }
        let evaluator = self.build_evaluator()?;
        let (basis_values, basis_jacobian) = evaluator.evaluate(coords)?;
        let reference_penalty = self.reference_penalty(evaluator.as_ref())?;
        let expected_width = self.basis_size()?;
        let n_rows = coords.nrows();
        if basis_values.dim() != (n_rows, expected_width)
            || basis_jacobian.dim() != (n_rows, expected_width, self.latent_dim)
            || reference_penalty.dim() != (expected_width, expected_width)
        {
            return Err(format!(
                "SaeAtomGeometryPlan::evaluate_bundle: plan {:?} produced values={:?}, jacobian={:?}, reference_penalty={:?}; expected ({n_rows}, {expected_width}), ({n_rows}, {expected_width}, {}), ({expected_width}, {expected_width})",
                self.kind,
                basis_values.dim(),
                basis_jacobian.dim(),
                reference_penalty.dim(),
                self.latent_dim
            ));
        }
        if basis_values
            .iter()
            .chain(basis_jacobian.iter())
            .chain(reference_penalty.iter())
            .any(|value| !value.is_finite())
        {
            return Err(format!(
                "SaeAtomGeometryPlan::evaluate_bundle: plan {:?} produced non-finite basis or penalty data",
                self.kind
            ));
        }
        Ok(SaeAtomEvaluationBundle {
            basis_values,
            basis_jacobian,
            reference_penalty,
            evaluator,
        })
    }

    fn reference_penalty(&self, evaluator: &dyn SaeBasisSecondJet) -> Result<Array2<f64>, String> {
        match (&self.resolution, &self.reference_metric) {
            (
                SaeBasisResolution::PeriodicHarmonics { order },
                SaeReferenceMetricPlan::UnitCircle,
            ) => periodic_reference_penalty(*order),
            (SaeBasisResolution::SphereChart, SaeReferenceMetricPlan::SphereChart) => {
                Ok(sphere_chart_reference_penalty())
            }
            (
                SaeBasisResolution::TorusHarmonics { per_axis_order },
                SaeReferenceMetricPlan::FlatRectangularTorus { tau },
            ) => flat_rectangular_torus_reference_penalty(*per_axis_order, *tau),
            (
                SaeBasisResolution::TorusHarmonics { .. },
                SaeReferenceMetricPlan::EmbeddedDonutTorus { tau },
            ) => Err(format!(
                "embedded-donut reference penalty at tau={tau} requires the exact closed-form dense Fourier operator from Phase 4; refusing to substitute the flat rectangular penalty"
            )),
            (
                SaeBasisResolution::ProjectivePlaneHarmonics { quotient_order },
                SaeReferenceMetricPlan::RoundProjectivePlane,
            ) => QuotientSpectralEvaluator::projective_plane(*quotient_order)?.spectral_penalty(2),
            (
                SaeBasisResolution::KleinBottleHarmonics { per_axis_order },
                SaeReferenceMetricPlan::FlatKleinBottle,
            ) => QuotientSpectralEvaluator::klein_bottle(*per_axis_order)?.spectral_penalty(2),
            (
                SaeBasisResolution::DuchonCoordinates { centers },
                SaeReferenceMetricPlan::EuclideanDuchon,
            ) => gam_terms::basis::duchon_sae_atom_penalty(
                centers.view(),
                duchon_nullspace_from_m(sae_duchon_atom_m(self.latent_dim)),
            )
            .map_err(|error| error.to_string()),
            (
                SaeBasisResolution::Polynomial { degree },
                SaeReferenceMetricPlan::EuclideanPolynomial,
            ) => Ok(polynomial_reference_penalty(self.latent_dim, *degree)),
            (
                SaeBasisResolution::Polynomial { .. },
                SaeReferenceMetricPlan::UnitPoincareBall { reference_coords },
            ) => {
                let (_, reference_jacobian) = evaluator.evaluate(reference_coords.view())?;
                gam_geometry::manifolds::poincare::conformal_dirichlet_penalty(
                    reference_coords.view(),
                    reference_jacobian.view(),
                    -1.0,
                )
                .map_err(|error| {
                    format!(
                        "SaeAtomGeometryPlan::reference_penalty: Poincare conformal Dirichlet Gram failed: {error}"
                    )
                })
            }
            (
                SaeBasisResolution::CylinderHarmonics {
                    circle_order,
                    line_degree,
                },
                SaeReferenceMetricPlan::CylinderProduct,
            ) => Ok(CylinderHarmonicEvaluator::new(*circle_order, *line_degree)?.roughness_gram()),
            (
                SaeBasisResolution::MobiusHarmonics {
                    circle_order,
                    width_degree,
                },
                SaeReferenceMetricPlan::MobiusQuotient,
            ) => Ok(MobiusHarmonicEvaluator::new(*circle_order, *width_degree)?.roughness_gram()),
            (
                SaeBasisResolution::FiniteAnchors { .. },
                SaeReferenceMetricPlan::DiscreteCounting,
            ) => Err("finite-set atoms have no continuous analytic evaluation bundle".to_string()),
            (SaeBasisResolution::Precomputed { .. }, SaeReferenceMetricPlan::CallerProvided) => {
                Err("precomputed atoms require a caller-supplied evaluation bundle".to_string())
            }
            _ => Err(format!(
                "SaeAtomGeometryPlan::reference_penalty: internally inconsistent plan {:?}",
                self
            )),
        }
    }
}

pub(crate) struct SaeAtomEvaluationBundle {
    pub(crate) basis_values: Array2<f64>,
    pub(crate) basis_jacobian: Array3<f64>,
    pub(crate) reference_penalty: Array2<f64>,
    pub(crate) evaluator: Arc<dyn SaeBasisSecondJet>,
}

/// Coordinate velocities of the three `SO(3)` Killing fields on the spherical
/// cover used by `RP²`, ordered by rotations about the ambient `x`, `y`, and
/// `z` axes.
///
/// The cover chart is `(latitude, longitude)`. Its longitude coordinate is
/// singular at a pole, so an `SO(3)` orbit cannot be represented by a unique
/// two-component chart velocity there. Refusing that state is essential: a
/// fabricated finite longitude velocity would silently change the quotient
/// tangent space used by both Newton gauge deflation and the identifiability
/// certificate.
pub(crate) fn projective_plane_cover_killing_directions(
    latitude: f64,
    longitude: f64,
) -> Result<[[f64; 2]; 3], String> {
    if !(latitude.is_finite() && longitude.is_finite()) {
        return Err(
            "projective_plane_cover_killing_directions requires finite cover coordinates"
                .to_string(),
        );
    }
    let cos_latitude = latitude.cos();
    if cos_latitude.abs() <= f64::EPSILON.sqrt() {
        return Err(format!(
            "projective_plane_cover_killing_directions: latitude {latitude} is at the spherical-cover pole where longitude has a nontrivial stabilizer"
        ));
    }
    let tan_latitude = latitude.sin() / cos_latitude;
    let (sin_longitude, cos_longitude) = longitude.sin_cos();
    Ok([
        [sin_longitude, -tan_latitude * cos_longitude],
        [-cos_longitude, -tan_latitude * sin_longitude],
        [0.0, 1.0],
    ])
}

fn duchon_nullspace_from_m(m: usize) -> gam_terms::basis::DuchonNullspaceOrder {
    match m {
        1 => gam_terms::basis::DuchonNullspaceOrder::Zero,
        2 => gam_terms::basis::DuchonNullspaceOrder::Linear,
        other => gam_terms::basis::DuchonNullspaceOrder::Degree(other - 1),
    }
}

/// Squared normalized-Laplacian Gram on the unit circle under normalized Haar
/// measure. Every raw sine/cosine column has L2 weight one half.
fn periodic_reference_penalty(order: usize) -> Result<Array2<f64>, String> {
    let width = sae_periodic_basis_size(order)?;
    let mut penalty = Array2::<f64>::zeros((width, width));
    for harmonic in 1..=order {
        let weight = 0.5 * (harmonic as f64).powi(4);
        penalty[[2 * harmonic - 1, 2 * harmonic - 1]] = weight;
        penalty[[2 * harmonic, 2 * harmonic]] = weight;
    }
    Ok(penalty)
}

/// Squared round-sphere Laplacian Gram for `[1,x,y,z,xy,yz,xz]` under the
/// normalized area measure. Degree-one modes have eigenvalue 2 and L2 weight
/// 1/3; the degree-two cross modes have eigenvalue 6 and L2 weight 1/15.
fn sphere_chart_reference_penalty() -> Array2<f64> {
    let mut penalty = Array2::<f64>::zeros((SAE_SPHERE_BASIS_SIZE, SAE_SPHERE_BASIS_SIZE));
    for column in 1..=3 {
        penalty[[column, column]] = 4.0 / 3.0;
    }
    for column in 4..=6 {
        penalty[[column, column]] = 12.0 / 5.0;
    }
    penalty
}

/// Squared Laplace--Beltrami Gram for a flat rectangular torus of aspect
/// `A=cosh(tau)`. Axis 0 is the long cycle, so the normalized eigenvalue is
/// `k0^2/A^2 + k1^2`; tensor sine/cosine L2 weights come from normalized Haar
/// measure. The overall physical scale is intentionally left to smoothing.
fn flat_rectangular_torus_reference_penalty(
    per_axis_order: usize,
    tau: f64,
) -> Result<Array2<f64>, String> {
    let evaluator = TorusHarmonicEvaluator::new(2, per_axis_order)?;
    let aspect = tau.cosh();
    let inverse_aspect_squared = aspect.recip().powi(2);
    let modes = evaluator.spectral_modes();
    let mut penalty = Array2::<f64>::zeros((modes.len(), modes.len()));
    for (column, mode) in modes.iter().enumerate() {
        let long_frequency = mode.components[0].harmonic() as f64;
        let short_frequency = mode.components[1].harmonic() as f64;
        let eigenvalue = long_frequency.powi(2) * inverse_aspect_squared + short_frequency.powi(2);
        penalty[[column, column]] = eigenvalue.powi(2) * mode.l2_gram_weight;
    }
    Ok(penalty)
}

fn polynomial_reference_penalty(latent_dim: usize, degree: usize) -> Array2<f64> {
    let exponents = gam_terms::basis::monomial_exponents(latent_dim, degree);
    let mut penalty = Array2::<f64>::zeros((exponents.len(), exponents.len()));
    for (column, exponent) in exponents.iter().enumerate() {
        if exponent.iter().any(|power| *power != 0) {
            penalty[[column, column]] = 1.0;
        }
    }
    penalty
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

    #[test]
    fn projective_plane_killing_fields_descend_through_antipodal_deck_map() {
        let latitude = 0.37;
        let longitude = -0.81;
        let original = projective_plane_cover_killing_directions(latitude, longitude).unwrap();
        let deck =
            projective_plane_cover_killing_directions(-latitude, longitude + std::f64::consts::PI)
                .unwrap();

        // Dg = diag(-1, 1) for g(lat, lon) = (-lat, lon + pi).
        for generator in 0..3 {
            assert!((deck[generator][0] + original[generator][0]).abs() <= 1.0e-12);
            assert!((deck[generator][1] - original[generator][1]).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn projective_plane_killing_fields_refuse_cover_poles() {
        assert!(
            projective_plane_cover_killing_directions(std::f64::consts::FRAC_PI_2, 0.0).is_err()
        );
    }
}
