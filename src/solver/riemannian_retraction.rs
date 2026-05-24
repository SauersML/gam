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
        let manifold = crate::geometry::EuclideanManifold::new(base.len());
        let next = crate::geometry::RiemannianManifold::exp_map(&manifold, base.view(), tangent)
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
        let manifold = crate::geometry::CircleManifold::new();
        let next = crate::geometry::RiemannianManifold::exp_map(&manifold, base.view(), tangent)
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
        let manifold = crate::geometry::SphereManifold::new(self.dim - 1);
        let next = crate::geometry::RiemannianManifold::exp_map(&manifold, base.view(), tangent)
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
