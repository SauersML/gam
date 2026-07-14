//! Typed atom geometry authority shared by seed, fit, rebuild, and OOS paths.
//!
//! A basis width is a theorem of a topology plus its resolution; it is not an
//! independently writable property. Likewise, the smoothing operator belongs
//! to a declared reference metric. Persisted models store this tagged plan
//! directly. There is intentionally no loader that reconstructs it from the
//! former parallel `(kind, harmonic_order, basis_width)` scalars.

use super::*;
use serde::{Deserialize, Serialize};

/// Basis-native resolution of one analytic atom family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SaeAtomGeometryPlanWire")]
pub struct SaeAtomGeometryPlan {
    kind: SaeAtomBasisKind,
    latent_dim: usize,
    resolution: SaeBasisResolution,
    reference_metric: SaeReferenceMetricPlan,
}

/// Deserialization proxy for [`SaeAtomGeometryPlan`]. The persisted wire carries
/// every semantic component, but it is never allowed to initialize the private
/// fields directly: `TryFrom` routes the tuple back through [`SaeAtomGeometryPlan::new`]
/// so a saved artifact cannot bypass topology/resolution/metric validation.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SaeAtomGeometryPlanWire {
    kind: SaeAtomBasisKind,
    latent_dim: usize,
    resolution: SaeBasisResolution,
    reference_metric: SaeReferenceMetricPlan,
}

impl TryFrom<SaeAtomGeometryPlanWire> for SaeAtomGeometryPlan {
    type Error = String;

    fn try_from(wire: SaeAtomGeometryPlanWire) -> Result<Self, Self::Error> {
        Self::new(
            wire.kind,
            wire.latent_dim,
            wire.resolution,
            wire.reference_metric,
        )
    }
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
                SaeBasisResolution::TorusHarmonics { per_axis_order },
                SaeReferenceMetricPlan::EmbeddedDonutTorus { tau },
            ) => embedded_donut_torus_reference_penalty(*per_axis_order, tau.cosh()),
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
/// `A=cosh(tau)`. This is the `tau`-parameterized entry point into the
/// anisotropic flat product-torus family: it forwards to
/// [`anisotropic_flat_product_torus_penalty`] with `A = cosh(tau)`, so the
/// donut's flat comparator and the standalone flat baseline model are one and
/// the same closed form evaluated at the same aspect.
fn flat_rectangular_torus_reference_penalty(
    per_axis_order: usize,
    tau: f64,
) -> Result<Array2<f64>, String> {
    if !(tau.is_finite() && tau >= 0.0) {
        return Err(format!(
            "flat_rectangular_torus_reference_penalty requires finite tau >= 0, got {tau}"
        ));
    }
    anisotropic_flat_product_torus_penalty(per_axis_order, tau.cosh())
}

/// Squared Laplace--Beltrami Gram for the **anisotropic flat product torus**
/// `S^1(R) x S^1(r)` with one relative aspect parameter `A = R/r >= 1`.
///
/// This is the identifiable flat *baseline model* of audit section 30: the
/// product metric is `ds^2 = A^2 dtheta^2 + dphi^2` (up to the overall `r^2`
/// scale absorbed by smoothing), so the Laplacian is diagonal in the tensor
/// Fourier basis with normalized eigenvalue `k^2/A^2 + l^2` (axis 0 is the long
/// cycle). Model selection contrasts this "is a flat anisotropic torus
/// sufficient?" family against [`embedded_donut_torus_reference_penalty`], the
/// `phi`-dependent embedded donut whose Laplacian couples Fourier modes; the
/// two agree only in the thin-tube limit `A -> infinity`. `A = 1` is the
/// isotropic square-torus reference `k^2 + l^2`.
pub fn anisotropic_flat_product_torus_penalty(
    per_axis_order: usize,
    aspect: f64,
) -> Result<Array2<f64>, String> {
    if !(aspect.is_finite() && aspect >= 1.0) {
        return Err(format!(
            "anisotropic_flat_product_torus_penalty requires a finite aspect A >= 1, got {aspect}"
        ));
    }
    let evaluator = TorusHarmonicEvaluator::new(2, per_axis_order)?;
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

// ─────────────────────────────────────────────────────────────────────────
// Embedded donut torus: exact closed-form Laplace--Beltrami penalty operator.
//
// For the standard embedding of `T^2` as a donut of aspect `A = R/r > 1`, the
// induced metric (r = 1) is
//     ds^2 = (A + cos phi)^2 dtheta^2 + dphi^2,
// with volume element `dvol = (A + cos phi) dtheta dphi`. The Laplace--Beltrami
// operator is
//     Delta_g f = (A+cos phi)^{-2} f_{theta theta}
//               + f_{phi phi} - (sin phi / (A+cos phi)) f_phi.
// The `phi`-dependent metric couples Fourier modes in `phi` *within* each fixed
// `theta`-frequency `k` (theta-modes stay uncoupled by orthogonality of the
// theta circle). We assemble the exact within-`k` blocks of the squared-
// Laplacian roughness Gram `P = integral (Delta_g f)^2 dvol` in the tensor real
// Fourier basis via the Galerkin identity `P_k = scale * B_k G^{-1} B_k`, where
//     G[U,V]  = integral U V (A+cos phi) dphi                    (donut L2 Gram)
//     B_k[U,V]= k^2 integral U V /(A+cos phi) dphi
//             +     integral U' V' (A+cos phi) dphi              (weak -Delta_g)
// so that `G^{-1} B_k` is the Galerkin `-Delta_g` and its eigenvalues converge
// to `k^2/A^2 + l^2` (anisotropic flat product torus), NOT `k^2 + l^2`, as
// `A -> infinity`. In the diagonal (flat) limit `P_k` reduces exactly to
// [`anisotropic_flat_product_torus_penalty`]. Every integral below is a closed
// form: the `1/(A+cos phi)` weight yields
//     integral_0^{2pi} cos(m phi)/(A+cos phi) dphi
//         = 2 pi (-1)^m beta^m / sqrt(A^2 - 1),   beta = A - sqrt(A^2 - 1),
// whose `beta^m` decay makes `B_k` numerically compressible (zeta-banded).

/// One real Fourier factor `amp * trig(freq * phi)`; `Constant` is the cosine of
/// frequency zero. This is the internal working representation used to expand
/// products and derivatives of donut basis functions into cosine harmonics.
#[derive(Debug, Clone, Copy)]
struct DonutTrigTerm {
    amp: f64,
    freq: usize,
    is_sine: bool,
}

impl DonutTrigTerm {
    fn from_component(component: RealHarmonicComponent) -> Self {
        match component {
            RealHarmonicComponent::Constant => Self {
                amp: 1.0,
                freq: 0,
                is_sine: false,
            },
            RealHarmonicComponent::Sine { harmonic } => Self {
                amp: 1.0,
                freq: harmonic,
                is_sine: true,
            },
            RealHarmonicComponent::Cosine { harmonic } => Self {
                amp: 1.0,
                freq: harmonic,
                is_sine: false,
            },
        }
    }

    /// `d/dphi` of this term: `d(cos)= -f sin`, `d(sin)= +f cos`.
    fn derivative(self) -> Self {
        Self {
            amp: if self.is_sine {
                self.amp * self.freq as f64
            } else {
                -self.amp * self.freq as f64
            },
            freq: self.freq,
            is_sine: !self.is_sine,
        }
    }
}

/// Cosine harmonics `m -> coeff` of the product of two trig terms. Products that
/// are pure sines (`sin*cos`) contribute nothing to any integral against the
/// even weights `A+cos phi` or `1/(A+cos phi)`, so only cosine output is kept.
fn donut_product_cos_coeffs(left: DonutTrigTerm, right: DonutTrigTerm) -> Vec<(usize, f64)> {
    let amp = left.amp * right.amp;
    if amp == 0.0 {
        return Vec::new();
    }
    let (a, b) = (left.freq, right.freq);
    let sum = a + b;
    let diff = a.abs_diff(b);
    // Each entry (m, coeff); `sum` and `diff` collide only when a == 0 or b == 0,
    // in which case the two half-weight cosines add to one full-weight cosine.
    let contributions: [(usize, f64); 2] = match (left.is_sine, right.is_sine) {
        // cos*cos = 1/2[cos(a+b) + cos(a-b)]
        (false, false) => [(sum, 0.5 * amp), (diff, 0.5 * amp)],
        // sin*sin = 1/2[cos(a-b) - cos(a+b)]
        (true, true) => [(diff, 0.5 * amp), (sum, -0.5 * amp)],
        // sin*cos or cos*sin = sines only -> no cosine content
        _ => return Vec::new(),
    };
    let mut out: Vec<(usize, f64)> = Vec::with_capacity(2);
    for (m, value) in contributions {
        if let Some(slot) = out.iter_mut().find(|(existing, _)| *existing == m) {
            slot.1 += value;
        } else {
            out.push((m, value));
        }
    }
    out
}

/// `integral_0^{2pi} cos(m phi) / (A + cos phi) dphi`, valid for `A > 1`.
fn donut_inverse_weight_integral(m: usize, aspect: f64) -> f64 {
    let root = (aspect * aspect - 1.0).sqrt();
    let beta = aspect - root;
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
    2.0 * std::f64::consts::PI * sign * beta.powi(m as i32) / root
}

/// `d/dA` of [`donut_inverse_weight_integral`].
fn donut_inverse_weight_integral_derivative(m: usize, aspect: f64) -> f64 {
    let root = (aspect * aspect - 1.0).sqrt();
    let beta = aspect - root;
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
    let two_pi = 2.0 * std::f64::consts::PI;
    let d_root = aspect / root;
    let d_beta = 1.0 - d_root;
    // C(m) = two_pi * sign * beta^m / root
    // dC/dA = two_pi * sign * [ m beta^{m-1} d_beta / root - beta^m d_root / root^2 ]
    let first = if m == 0 {
        0.0
    } else {
        (m as f64) * beta.powi(m as i32 - 1) * d_beta / root
    };
    let second = beta.powi(m as i32) * d_root / (root * root);
    two_pi * sign * (first - second)
}

/// The three `phi`-block integral matrices `(I_inv, I_stiff, G)` over the real
/// Fourier basis of order `H`, where
///   `I_inv[i,j]  = integral U_i U_j / (A+cos phi) dphi`,
///   `I_stiff[i,j]= integral U_i' U_j' (A+cos phi) dphi`,
///   `G[i,j]      = integral U_i U_j (A+cos phi) dphi`.
/// `I_inv` supplies the `k^2` (theta-curvature) contribution to the weak
/// Laplacian; `I_stiff` the `phi`-curvature contribution; `G` is the donut L2
/// Gram. Returned in the evaluator's per-axis column order.
fn donut_phi_block_integrals(per_axis_order: usize, aspect: f64) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let width = 2 * per_axis_order + 1;
    let mut i_inv = Array2::<f64>::zeros((width, width));
    let mut i_stiff = Array2::<f64>::zeros((width, width));
    let mut gram = Array2::<f64>::zeros((width, width));
    let two_pi = 2.0 * std::f64::consts::PI;
    let pi = std::f64::consts::PI;
    for i in 0..width {
        let term_i = DonutTrigTerm::from_component(TorusHarmonicEvaluator::axis_component(i));
        let dterm_i = term_i.derivative();
        for j in 0..width {
            let term_j = DonutTrigTerm::from_component(TorusHarmonicEvaluator::axis_component(j));
            let dterm_j = term_j.derivative();
            // I_inv: sum_m coeff * C(m).
            let mut inv = 0.0;
            for (m, coeff) in donut_product_cos_coeffs(term_i, term_j) {
                inv += coeff * donut_inverse_weight_integral(m, aspect);
            }
            i_inv[[i, j]] = inv;
            // G and I_stiff share the (A+cos phi) kernel: for a product with
            // cosine content `sum_m coeff cos(m phi)`, integral against
            // `(A+cos phi)` picks only m=0 (weight 2 pi A) and m=1 (weight pi).
            let mut g = 0.0;
            for (m, coeff) in donut_product_cos_coeffs(term_i, term_j) {
                if m == 0 {
                    g += coeff * two_pi * aspect;
                } else if m == 1 {
                    g += coeff * pi;
                }
            }
            gram[[i, j]] = g;
            let mut stiff = 0.0;
            for (m, coeff) in donut_product_cos_coeffs(dterm_i, dterm_j) {
                if m == 0 {
                    stiff += coeff * two_pi * aspect;
                } else if m == 1 {
                    stiff += coeff * pi;
                }
            }
            i_stiff[[i, j]] = stiff;
        }
    }
    (i_inv, i_stiff, gram)
}

/// `d/dA` of [`donut_phi_block_integrals`]. `G` and `I_stiff` are affine in `A`
/// (only the `m=0`, `2 pi A` term carries `A`); `I_inv` differentiates through
/// the closed-form inverse-weight integral.
fn donut_phi_block_integral_derivatives(
    per_axis_order: usize,
    aspect: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let width = 2 * per_axis_order + 1;
    let mut d_inv = Array2::<f64>::zeros((width, width));
    let mut d_stiff = Array2::<f64>::zeros((width, width));
    let mut d_gram = Array2::<f64>::zeros((width, width));
    let two_pi = 2.0 * std::f64::consts::PI;
    for i in 0..width {
        let term_i = DonutTrigTerm::from_component(TorusHarmonicEvaluator::axis_component(i));
        let dterm_i = term_i.derivative();
        for j in 0..width {
            let term_j = DonutTrigTerm::from_component(TorusHarmonicEvaluator::axis_component(j));
            let dterm_j = term_j.derivative();
            let mut inv = 0.0;
            for (m, coeff) in donut_product_cos_coeffs(term_i, term_j) {
                inv += coeff * donut_inverse_weight_integral_derivative(m, aspect);
            }
            d_inv[[i, j]] = inv;
            let mut g = 0.0;
            for (m, coeff) in donut_product_cos_coeffs(term_i, term_j) {
                if m == 0 {
                    g += coeff * two_pi;
                }
            }
            d_gram[[i, j]] = g;
            let mut stiff = 0.0;
            for (m, coeff) in donut_product_cos_coeffs(dterm_i, dterm_j) {
                if m == 0 {
                    stiff += coeff * two_pi;
                }
            }
            d_stiff[[i, j]] = stiff;
        }
    }
    (d_inv, d_stiff, d_gram)
}

/// Symmetrize `M` in place (kills roundoff asymmetry from the `G^{-1}` solve).
fn donut_symmetrize(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            let mean = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            matrix[[i, j]] = mean;
            matrix[[j, i]] = mean;
        }
    }
}

/// Per-`theta`-frequency normalization of the normalized-measure squared-
/// Laplacian penalty block: `1/(2 pi A)` for the constant `theta`-mode (L2
/// weight one) and `1/(4 pi A)` for each sine/cosine `theta`-mode (L2 weight
/// one half). These are the exact factors that make `P_k` reduce to
/// [`anisotropic_flat_product_torus_penalty`] in the diagonal limit.
fn donut_block_scale(k: usize, aspect: f64) -> f64 {
    let gamma = if k == 0 {
        1.0 / (2.0 * std::f64::consts::PI)
    } else {
        1.0 / (4.0 * std::f64::consts::PI)
    };
    gamma / aspect
}

/// Weak `-Delta_g` block `B_k = k^2 I_inv + I_stiff` for `theta`-frequency `k`.
fn donut_weak_laplacian_block(
    k: usize,
    i_inv: &Array2<f64>,
    i_stiff: &Array2<f64>,
) -> Array2<f64> {
    let mut block = i_stiff.clone();
    if k > 0 {
        let k2 = (k * k) as f64;
        block.scaled_add(k2, i_inv);
    }
    block
}

/// Generalized-eigenvalue spectrum of the exact embedded-donut Laplace--Beltrami
/// operator restricted to `theta`-frequency `k`: the eigenvalues of
/// `G^{-1} B_k` (a self-adjoint pencil `B_k u = lambda G u`). As `A -> infinity`
/// this multiset converges to `{ k^2/A^2 + l^2 : l = 0..H }` (with `l >= 1`
/// doubled for sine/cosine), i.e. the ANISOTROPIC flat product torus, not the
/// isotropic `k^2 + l^2`.
pub fn embedded_donut_laplacian_block_eigenvalues(
    per_axis_order: usize,
    aspect: f64,
    k: usize,
) -> Result<Vec<f64>, String> {
    if !(aspect.is_finite() && aspect > 1.0) {
        return Err(format!(
            "embedded_donut_laplacian_block_eigenvalues requires a finite aspect A > 1, got {aspect}"
        ));
    }
    if per_axis_order == 0 {
        return Err("embedded_donut_laplacian_block_eigenvalues requires per_axis_order >= 1".to_string());
    }
    let (i_inv, i_stiff, gram) = donut_phi_block_integrals(per_axis_order, aspect);
    let block = donut_weak_laplacian_block(k, &i_inv, &i_stiff);
    // Whiten by the SPD donut Gram: eig(G^{-1} B) = eig(L^{-1} B L^{-T}) with
    // G = L L^T. Two forward substitutions with L keep it self-adjoint.
    let factor = gram
        .cholesky(Side::Lower)
        .map_err(|error| format!("donut Gram Cholesky failed: {error}"))?;
    let lower = factor.lower_triangular();
    let whitened = {
        let step = donut_forward_substitution(&lower, &block);
        let step_t = step.t().to_owned();
        let second = donut_forward_substitution(&lower, &step_t);
        second.t().to_owned()
    };
    let mut symmetric = whitened;
    donut_symmetrize(&mut symmetric);
    let (values, _) = symmetric
        .eigh(Side::Lower)
        .map_err(|error| format!("donut whitened eigendecomposition failed: {error}"))?;
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite donut eigenvalues"));
    Ok(sorted)
}

/// Solve the lower-triangular system `L X = B` for `X` (column-by-column
/// forward substitution). `L` must be square lower-triangular with nonzero
/// diagonal; `B` has matching row count.
fn donut_forward_substitution(lower: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
    let n = lower.nrows();
    let cols = rhs.ncols();
    let mut x = Array2::<f64>::zeros((n, cols));
    for c in 0..cols {
        for i in 0..n {
            let mut acc = rhs[[i, c]];
            for j in 0..i {
                acc -= lower[[i, j]] * x[[j, c]];
            }
            x[[i, c]] = acc / lower[[i, i]];
        }
    }
    x
}

/// Exact closed-form squared Laplace--Beltrami roughness Gram for the embedded
/// donut torus of aspect `A = aspect > 1`, in the tensor real Fourier basis of
/// [`TorusHarmonicEvaluator`] (`latent_dim = 2`, order `per_axis_order`). The
/// operator is block-diagonal across `theta`-columns; each block is the dense
/// `phi`-coupled `P_k = scale_k * B_k G^{-1} B_k`.
pub fn embedded_donut_torus_reference_penalty(
    per_axis_order: usize,
    aspect: f64,
) -> Result<Array2<f64>, String> {
    let (penalty, _) = embedded_donut_penalty_and_optional_derivative(per_axis_order, aspect, false)?;
    Ok(penalty)
}

/// Analytic aspect (`A`) derivative `dP/dA` of
/// [`embedded_donut_torus_reference_penalty`]. Closed form throughout: no finite
/// differences or autodiff. Used to propagate the smoothing/aspect coupling into
/// the outer objective.
pub fn embedded_donut_torus_reference_penalty_aspect_derivative(
    per_axis_order: usize,
    aspect: f64,
) -> Result<Array2<f64>, String> {
    let (_, derivative) = embedded_donut_penalty_and_optional_derivative(per_axis_order, aspect, true)?;
    derivative.ok_or_else(|| "donut aspect derivative was not produced".to_string())
}

/// Shared assembler for the donut penalty and (optionally) its exact `A`
/// derivative. Building both together reuses the single Gram factorization.
fn embedded_donut_penalty_and_optional_derivative(
    per_axis_order: usize,
    aspect: f64,
    want_derivative: bool,
) -> Result<(Array2<f64>, Option<Array2<f64>>), String> {
    if !(aspect.is_finite() && aspect > 1.0) {
        return Err(format!(
            "embedded_donut_torus_reference_penalty requires a finite aspect A > 1, got {aspect}"
        ));
    }
    let evaluator = TorusHarmonicEvaluator::new(2, per_axis_order)?;
    let axis_m = evaluator.axis_basis_size();
    let total = evaluator.basis_size();
    let (i_inv, i_stiff, gram) = donut_phi_block_integrals(per_axis_order, aspect);
    let (d_inv, d_stiff, d_gram) = if want_derivative {
        let (a, b, c) = donut_phi_block_integral_derivatives(per_axis_order, aspect);
        (Some(a), Some(b), Some(c))
    } else {
        (None, None, None)
    };
    let factor = gram
        .cholesky(Side::Lower)
        .map_err(|error| format!("donut Gram Cholesky failed: {error}"))?;

    // Cache one dense (2H+1)^2 block per distinct theta-frequency k = 0..H, then
    // scatter each theta-column into the block-diagonal full penalty.
    let mut penalty_blocks: Vec<Option<Array2<f64>>> = vec![None; per_axis_order + 1];
    let mut derivative_blocks: Vec<Option<Array2<f64>>> = vec![None; per_axis_order + 1];

    let mut penalty = Array2::<f64>::zeros((total, total));
    let mut derivative = if want_derivative {
        Some(Array2::<f64>::zeros((total, total)))
    } else {
        None
    };

    let derivative_integrals = match (&d_inv, &d_stiff, &d_gram) {
        (Some(inv), Some(stiff), Some(g)) => Some((inv, stiff, g)),
        _ => None,
    };
    for theta_index in 0..axis_m {
        let k = TorusHarmonicEvaluator::axis_component(theta_index).harmonic();
        if penalty_blocks[k].is_none() {
            let (block_penalty, block_derivative) =
                donut_penalty_block(k, aspect, &i_inv, &i_stiff, &factor, derivative_integrals)?;
            penalty_blocks[k] = Some(block_penalty);
            derivative_blocks[k] = block_derivative;
        }
        let block_penalty = penalty_blocks[k]
            .as_ref()
            .expect("donut penalty block cached");
        let base = theta_index * axis_m;
        for a in 0..axis_m {
            for b in 0..axis_m {
                penalty[[base + a, base + b]] = block_penalty[[a, b]];
            }
        }
        if let Some(derivative) = derivative.as_mut() {
            let block_derivative = derivative_blocks[k]
                .as_ref()
                .expect("donut derivative block cached");
            for a in 0..axis_m {
                for b in 0..axis_m {
                    derivative[[base + a, base + b]] = block_derivative[[a, b]];
                }
            }
        }
    }
    Ok((penalty, derivative))
}

/// Derivative-integral triple `(dI_inv/dA, dI_stiff/dA, dG/dA)` for one aspect.
type DonutIntegralDerivatives<'a> = (&'a Array2<f64>, &'a Array2<f64>, &'a Array2<f64>);

/// One `theta`-frequency block `P_k = scale_k B_k G^{-1} B_k` and (optionally)
/// its exact `A` derivative. With `Y = G^{-1} B_k`,
///   `d/dA (B G^{-1} B) = dB Y + B G^{-1}(dB - dG Y)`,
/// and `P_k = (gamma_k / A) B G^{-1} B` adds the `-1/A^2` scale term.
fn donut_penalty_block(
    k: usize,
    aspect: f64,
    i_inv: &Array2<f64>,
    i_stiff: &Array2<f64>,
    factor: &FaerCholeskyFactor,
    derivative_integrals: Option<DonutIntegralDerivatives<'_>>,
) -> Result<(Array2<f64>, Option<Array2<f64>>), String> {
    let scale = donut_block_scale(k, aspect);
    let block = donut_weak_laplacian_block(k, i_inv, i_stiff);
    // Y = G^{-1} B_k.
    let response = factor.solve_mat(&block);
    let mut core = fast_ab(&block, &response); // B G^{-1} B
    donut_symmetrize(&mut core);
    let mut penalty = core.clone();
    penalty.mapv_inplace(|value| value * scale);

    let Some((d_inv, d_stiff, d_gram)) = derivative_integrals else {
        return Ok((penalty, None));
    };
    let d_block = donut_weak_laplacian_block(k, d_inv, d_stiff); // dB_k/dA
    // R = dB - dG Y ; Z = G^{-1} R ; dCore = dB Y + B Z.
    let dg_response = fast_ab(d_gram, &response);
    let rhs = &d_block - &dg_response;
    let z = factor.solve_mat(&rhs);
    let mut d_core = &fast_ab(&d_block, &response) + &fast_ab(&block, &z);
    donut_symmetrize(&mut d_core);
    // dP = gamma_k (-1/A^2 core + 1/A dCore) = -scale/A core + scale dCore.
    let mut derivative = d_core;
    derivative.mapv_inplace(|value| value * scale);
    derivative.scaled_add(-scale / aspect, &core);
    donut_symmetrize(&mut derivative);
    Ok((penalty, Some(derivative)))
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

    /// Expected block spectrum of the anisotropic flat product torus at
    /// `theta`-frequency `k`: `k^2/A^2 + l^2` with `l = 0` once and `l = 1..H`
    /// twice (sine and cosine), sorted ascending.
    fn expected_anisotropic_flat_block(per_axis_order: usize, aspect: f64, k: usize) -> Vec<f64> {
        let base = (k * k) as f64 / (aspect * aspect);
        let mut values = vec![base];
        for l in 1..=per_axis_order {
            let value = base + (l * l) as f64;
            values.push(value);
            values.push(value);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values
    }

    #[test]
    fn embedded_donut_eigenvalues_converge_to_anisotropic_flat_not_isotropic() {
        let per_axis_order = 4;
        // Thin-tube limit: A -> infinity. Convergence of the Laplace--Beltrami
        // spectrum to the flat product torus is O(1/A^2).
        let aspect = 2000.0;
        for k in 0..=per_axis_order {
            let eigenvalues =
                embedded_donut_laplacian_block_eigenvalues(per_axis_order, aspect, k).unwrap();
            let expected = expected_anisotropic_flat_block(per_axis_order, aspect, k);
            assert_eq!(eigenvalues.len(), expected.len());
            for (got, want) in eigenvalues.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() <= 1.0e-3,
                    "k={k}: donut eigenvalue {got} != anisotropic-flat target {want}"
                );
            }
        }

        // Decisive discriminant: at k=3, l=0 the anisotropic target is
        // k^2/A^2 = 9/A^2 (~2.25e-6), while the ISOTROPIC flat torus would put
        // this mode at k^2 = 9. The smallest block eigenvalue must land on the
        // former, nowhere near the latter.
        let smallest = embedded_donut_laplacian_block_eigenvalues(per_axis_order, aspect, 3)
            .unwrap()[0];
        let anisotropic_target = 9.0 / (aspect * aspect);
        assert!(
            (smallest - anisotropic_target).abs() <= 1.0e-3,
            "smallest k=3 eigenvalue {smallest} should track k^2/A^2 = {anisotropic_target}"
        );
        assert!(
            smallest < 0.5,
            "smallest k=3 eigenvalue {smallest} must not approach the isotropic k^2 = 9"
        );
    }

    #[test]
    fn embedded_donut_aspect_derivative_matches_central_difference() {
        let per_axis_order = 3;
        let aspect = 1.7;
        let analytic = embedded_donut_torus_reference_penalty_aspect_derivative(
            per_axis_order,
            aspect,
        )
        .unwrap();
        let h = 1.0e-6;
        let plus = embedded_donut_torus_reference_penalty(per_axis_order, aspect + h).unwrap();
        let minus = embedded_donut_torus_reference_penalty(per_axis_order, aspect - h).unwrap();
        let finite_difference = (&plus - &minus).mapv(|value| value / (2.0 * h));
        assert_eq!(analytic.dim(), finite_difference.dim());
        let mut max_gap = 0.0_f64;
        let mut max_scale = 1.0_f64;
        for (analytic_value, fd_value) in analytic.iter().zip(finite_difference.iter()) {
            max_gap = max_gap.max((analytic_value - fd_value).abs());
            max_scale = max_scale.max(analytic_value.abs());
        }
        assert!(
            max_gap <= 1.0e-5 * max_scale,
            "analytic donut aspect derivative deviates from central FD: gap {max_gap}, scale {max_scale}"
        );
    }

    #[test]
    fn anisotropic_flat_family_is_distinct_from_isotropic_reference() {
        let per_axis_order = 3;
        let isotropic = anisotropic_flat_product_torus_penalty(per_axis_order, 1.0).unwrap();
        let anisotropic = anisotropic_flat_product_torus_penalty(per_axis_order, 3.0).unwrap();
        assert_eq!(isotropic.dim(), anisotropic.dim());
        let max_gap = isotropic
            .iter()
            .zip(anisotropic.iter())
            .fold(0.0_f64, |acc, (iso, aniso)| acc.max((iso - aniso).abs()));
        assert!(
            max_gap > 1.0e-6,
            "anisotropic flat family must differ from the isotropic k^2+l^2 reference"
        );
        // The tau entry point (A = cosh(tau)) is exactly this family evaluated
        // at the same aspect.
        let via_tau = flat_rectangular_torus_reference_penalty(per_axis_order, (3.0_f64).acosh())
            .unwrap();
        let tau_gap = via_tau
            .iter()
            .zip(anisotropic.iter())
            .fold(0.0_f64, |acc, (lhs, rhs)| acc.max((lhs - rhs).abs()));
        assert!(tau_gap <= 1.0e-9, "tau entry point must match aspect entry point");
    }

    #[test]
    fn embedded_donut_plan_builds_finite_symmetric_penalty() {
        let plan = SaeAtomGeometryPlan::new(
            SaeAtomBasisKind::Torus,
            2,
            SaeBasisResolution::TorusHarmonics { per_axis_order: 3 },
            SaeReferenceMetricPlan::EmbeddedDonutTorus { tau: 0.9 },
        )
        .unwrap();
        let penalty = plan.build_reference_penalty().unwrap();
        let width = plan.basis_size().unwrap();
        assert_eq!(penalty.dim(), (width, width));
        assert!(penalty.iter().all(|value| value.is_finite()));
        let mut max_asymmetry = 0.0_f64;
        for i in 0..width {
            for j in 0..width {
                max_asymmetry = max_asymmetry.max((penalty[[i, j]] - penalty[[j, i]]).abs());
            }
        }
        assert!(max_asymmetry <= 1.0e-9, "donut penalty must be symmetric");
        // It must be a genuinely phi-coupled dense operator, not the diagonal
        // flat comparator: at least one within-k off-diagonal entry is nonzero.
        let has_offdiagonal = (0..width).any(|i| {
            (0..width).any(|j| i != j && penalty[[i, j]].abs() > 1.0e-9)
        });
        assert!(
            has_offdiagonal,
            "embedded donut penalty must couple Fourier modes (be non-diagonal)"
        );
    }
}
