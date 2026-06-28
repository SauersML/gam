use ndarray::{ArrayView1, ArrayViewMut1, s};

const TWO_PI: f64 = std::f64::consts::PI * 2.0;

pub trait Retraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>);
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EuclideanRetraction;

impl Retraction for EuclideanRetraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        assert_eq!(base.len(), tangent.len());
        let manifold = gam_geometry::EuclideanManifold::new(base.len());
        let next = gam_geometry::RiemannianManifold::exp_map(&manifold, base.view(), tangent)
            .expect("Euclidean retraction dimensions were prevalidated");
        for axis in 0..base.len() {
            base[axis] = next[axis];
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CircleRetraction;

impl Retraction for CircleRetraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        assert_eq!(base.len(), 1);
        assert_eq!(tangent.len(), 1);
        let manifold = gam_geometry::CircleManifold::new();
        let next = gam_geometry::RiemannianManifold::exp_map(&manifold, base.view(), tangent)
            .expect("Circle retraction dimensions were prevalidated");
        base[0] = next[0];
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereRetraction {
    pub dim: usize,
}

impl Retraction for SphereRetraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        assert_eq!(base.len(), self.dim);
        assert_eq!(tangent.len(), self.dim);
        assert!(
            self.dim >= 2,
            "SphereRetraction ambient dim must be at least 2"
        );
        let manifold = gam_geometry::SphereManifold::new(self.dim - 1);
        let next = gam_geometry::RiemannianManifold::exp_map(&manifold, base.view(), tangent)
            .expect("Sphere retraction dimensions were prevalidated");
        for axis in 0..self.dim {
            base[axis] = next[axis];
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProductRetraction {
    pub parts: Vec<RetractionKind>,
}

impl ProductRetraction {
    pub fn ambient_dim(&self) -> usize {
        self.parts.iter().map(RetractionKind::ambient_dim).sum()
    }

    pub fn is_all_euclidean(&self) -> bool {
        self.parts.iter().all(RetractionKind::is_euclidean)
    }
}

impl Retraction for ProductRetraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        assert_eq!(base.len(), tangent.len());
        let mut offset = 0_usize;
        for part in &self.parts {
            let dim = part.ambient_dim();
            let mut base_part = base.slice_mut(s![offset..offset + dim]);
            let tangent_part = tangent.slice(s![offset..offset + dim]);
            part.retract(&mut base_part, tangent_part);
            offset += dim;
        }
        assert_eq!(offset, base.len());
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RetractionKind {
    Euclidean { dim: usize },
    Circle,
    Sphere { dim: usize },
    Product(ProductRetraction),
}

impl RetractionKind {
    pub fn euclidean(dim: usize) -> Self {
        Self::Euclidean { dim }
    }

    pub fn ambient_dim(&self) -> usize {
        match self {
            Self::Euclidean { dim } | Self::Sphere { dim } => *dim,
            Self::Circle => 1,
            Self::Product(product) => product.ambient_dim(),
        }
    }

    pub fn is_euclidean(&self) -> bool {
        match self {
            Self::Euclidean { .. } => true,
            Self::Circle | Self::Sphere { .. } => false,
            Self::Product(product) => product.is_all_euclidean(),
        }
    }

    pub fn metric_weights(&self) -> Vec<f64> {
        match self {
            Self::Euclidean { dim } => vec![1.0; *dim],
            Self::Circle => vec![1.0 / (TWO_PI * TWO_PI)],
            Self::Sphere { dim } => {
                let weight = 1.0 / (std::f64::consts::PI * std::f64::consts::PI);
                vec![weight; *dim]
            }
            Self::Product(product) => {
                let mut out = Vec::with_capacity(product.ambient_dim());
                for part in &product.parts {
                    out.extend(part.metric_weights());
                }
                out
            }
        }
    }

    /// Per-ambient-axis periodicity, mirroring
    /// [`gam_terms::latent::LatentManifold::axis_periods`]. A `Circle`
    /// retraction wraps modulo `2π`; an embedded `Sphere` retraction is smooth
    /// with no cut and is reported non-periodic.
    pub fn axis_periods(&self) -> Vec<Option<f64>> {
        match self {
            Self::Euclidean { dim } => vec![None; *dim],
            Self::Circle => vec![Some(TWO_PI)],
            Self::Sphere { dim } => vec![None; *dim],
            Self::Product(product) => {
                let mut out = Vec::with_capacity(product.ambient_dim());
                for part in &product.parts {
                    out.extend(part.axis_periods());
                }
                out
            }
        }
    }
}

impl Retraction for RetractionKind {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        match self {
            Self::Euclidean { .. } => EuclideanRetraction.retract(base, tangent),
            Self::Circle => CircleRetraction.retract(base, tangent),
            Self::Sphere { dim } => SphereRetraction { dim: *dim }.retract(base, tangent),
            Self::Product(product) => product.retract(base, tangent),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LatentRetractionRegistry {
    block: Option<RetractionKind>,
}

impl LatentRetractionRegistry {
    pub fn all_euclidean() -> Self {
        Self { block: None }
    }

    pub fn new(block: RetractionKind) -> Self {
        if block.is_euclidean() {
            Self::all_euclidean()
        } else {
            Self { block: Some(block) }
        }
    }

    pub fn is_all_euclidean(&self) -> bool {
        self.block.is_none()
    }

    pub(crate) fn ambient_dim(&self, fallback_dim: usize) -> usize {
        self.block
            .as_ref()
            .map_or(fallback_dim, RetractionKind::ambient_dim)
    }

    pub fn metric_weights(&self, fallback_dim: usize) -> Vec<f64> {
        self.block
            .as_ref()
            .map_or_else(|| vec![1.0; fallback_dim], RetractionKind::metric_weights)
    }

    /// Per-ambient-axis periodicity for the override retraction, falling back
    /// to all-non-periodic (`None`) of length `fallback_dim` when no override
    /// is installed.
    pub fn axis_periods(&self, fallback_dim: usize) -> Vec<Option<f64>> {
        self.block
            .as_ref()
            .map_or_else(|| vec![None; fallback_dim], RetractionKind::axis_periods)
    }

    pub fn validate_dim(&self, latent_dim: usize, context: &str) -> Result<(), String> {
        let dim = self.ambient_dim(latent_dim);
        if dim != latent_dim {
            return Err(format!(
                "{context} retraction ambient dimension {dim} does not match latent d={latent_dim}"
            ));
        }
        Ok(())
    }

    pub fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        assert_eq!(base.len(), tangent.len());
        if let Some(block) = self.block.as_ref() {
            block.retract(base, tangent);
        } else {
            for (value, delta) in base.iter_mut().zip(tangent.iter()) {
                *value += *delta;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── RetractionKind::ambient_dim ───────────────────────────────────────────

    #[test]
    fn euclidean_ambient_dim() {
        assert_eq!(RetractionKind::euclidean(4).ambient_dim(), 4);
        assert_eq!(RetractionKind::euclidean(0).ambient_dim(), 0);
    }

    #[test]
    fn circle_ambient_dim_is_one() {
        assert_eq!(RetractionKind::Circle.ambient_dim(), 1);
    }

    #[test]
    fn sphere_ambient_dim() {
        assert_eq!(RetractionKind::Sphere { dim: 3 }.ambient_dim(), 3);
    }

    #[test]
    fn product_ambient_dim_is_sum() {
        let product = ProductRetraction {
            parts: vec![
                RetractionKind::euclidean(2),
                RetractionKind::Circle,
                RetractionKind::Sphere { dim: 3 },
            ],
        };
        assert_eq!(product.ambient_dim(), 6); // 2 + 1 + 3
    }

    // ── RetractionKind::is_euclidean ──────────────────────────────────────────

    #[test]
    fn euclidean_is_euclidean() {
        assert!(RetractionKind::euclidean(5).is_euclidean());
    }

    #[test]
    fn circle_is_not_euclidean() {
        assert!(!RetractionKind::Circle.is_euclidean());
    }

    #[test]
    fn sphere_is_not_euclidean() {
        assert!(!RetractionKind::Sphere { dim: 3 }.is_euclidean());
    }

    #[test]
    fn all_euclidean_product_is_euclidean() {
        let product = ProductRetraction {
            parts: vec![RetractionKind::euclidean(2), RetractionKind::euclidean(3)],
        };
        assert!(RetractionKind::Product(product).is_euclidean());
    }

    #[test]
    fn mixed_product_is_not_euclidean() {
        let product = ProductRetraction {
            parts: vec![RetractionKind::euclidean(2), RetractionKind::Circle],
        };
        assert!(!RetractionKind::Product(product).is_euclidean());
    }

    // ── RetractionKind::metric_weights ────────────────────────────────────────

    #[test]
    fn euclidean_metric_weights_are_all_one() {
        let w = RetractionKind::euclidean(3).metric_weights();
        assert_eq!(w, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn circle_metric_weight_is_inv_twopi_sq() {
        let w = RetractionKind::Circle.metric_weights();
        let expected = 1.0 / (TWO_PI * TWO_PI);
        assert_eq!(w.len(), 1);
        assert!((w[0] - expected).abs() < 1e-15);
    }

    // ── RetractionKind::axis_periods ─────────────────────────────────────────

    #[test]
    fn euclidean_axis_periods_all_none() {
        let p = RetractionKind::euclidean(3).axis_periods();
        assert_eq!(p, vec![None, None, None]);
    }

    #[test]
    fn circle_axis_period_is_two_pi() {
        let p = RetractionKind::Circle.axis_periods();
        assert_eq!(p.len(), 1);
        assert!((p[0].unwrap() - TWO_PI).abs() < 1e-15);
    }

    #[test]
    fn sphere_axis_periods_all_none() {
        let p = RetractionKind::Sphere { dim: 3 }.axis_periods();
        assert_eq!(p, vec![None, None, None]);
    }

    // ── LatentRetractionRegistry ──────────────────────────────────────────────

    #[test]
    fn all_euclidean_registry_reports_is_all_euclidean() {
        let r = LatentRetractionRegistry::all_euclidean();
        assert!(r.is_all_euclidean());
    }

    #[test]
    fn circle_registry_is_not_all_euclidean() {
        let r = LatentRetractionRegistry::new(RetractionKind::Circle);
        assert!(!r.is_all_euclidean());
    }

    #[test]
    fn euclidean_registry_collapses_to_all_euclidean() {
        // Constructing with a Euclidean kind must collapse to all-Euclidean.
        let r = LatentRetractionRegistry::new(RetractionKind::euclidean(3));
        assert!(r.is_all_euclidean());
    }

    #[test]
    fn registry_validate_dim_ok_when_matching() {
        let r = LatentRetractionRegistry::new(RetractionKind::Circle);
        assert!(r.validate_dim(1, "ctx").is_ok());
    }

    #[test]
    fn registry_validate_dim_err_when_mismatched() {
        let r = LatentRetractionRegistry::new(RetractionKind::Circle);
        let e = r.validate_dim(3, "ctx").unwrap_err();
        assert!(e.contains("ctx"), "error should mention context: {e}");
    }

    #[test]
    fn registry_euclidean_retract_adds_tangent() {
        let r = LatentRetractionRegistry::all_euclidean();
        let mut base = ndarray::array![1.0, 2.0, 3.0];
        let tangent = ndarray::array![0.1, -0.2, 0.5];
        r.retract(&mut base.view_mut(), tangent.view());
        assert!((base[0] - 1.1).abs() < 1e-15);
        assert!((base[1] - 1.8).abs() < 1e-15);
        assert!((base[2] - 3.5).abs() < 1e-15);
    }
}
