use ndarray::{ArrayView1, ArrayViewMut1, s};

const TWO_PI: f64 = std::f64::consts::PI * 2.0;

pub trait Retraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>);
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EuclideanRetraction;

impl Retraction for EuclideanRetraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        debug_assert_eq!(base.len(), tangent.len());
        for (value, delta) in base.iter_mut().zip(tangent.iter()) {
            *value += *delta;
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CircleRetraction;

impl Retraction for CircleRetraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        debug_assert_eq!(base.len(), 1);
        debug_assert_eq!(tangent.len(), 1);
        base[0] = wrap_angle(base[0] + tangent[0]);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereRetraction {
    pub dim: usize,
}

impl Retraction for SphereRetraction {
    fn retract(&self, base: &mut ArrayViewMut1<f64>, tangent: ArrayView1<f64>) {
        debug_assert_eq!(base.len(), self.dim);
        debug_assert_eq!(tangent.len(), self.dim);
        let mut norm_sq = 0.0_f64;
        for axis in 0..self.dim {
            let next = base[axis] + tangent[axis];
            norm_sq += next * next;
        }
        let norm = norm_sq.sqrt();
        assert!(
            norm.is_finite() && norm > 0.0,
            "SphereRetraction cannot normalize a zero or non-finite update"
        );
        for axis in 0..self.dim {
            base[axis] = (base[axis] + tangent[axis]) / norm;
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
        debug_assert_eq!(base.len(), tangent.len());
        let mut offset = 0_usize;
        for part in &self.parts {
            let dim = part.ambient_dim();
            let mut base_part = base.slice_mut(s![offset..offset + dim]);
            let tangent_part = tangent.slice(s![offset..offset + dim]);
            part.retract(&mut base_part, tangent_part);
            offset += dim;
        }
        debug_assert_eq!(offset, base.len());
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

fn wrap_angle(theta: f64) -> f64 {
    (theta + std::f64::consts::PI).rem_euclid(TWO_PI) - std::f64::consts::PI
}
