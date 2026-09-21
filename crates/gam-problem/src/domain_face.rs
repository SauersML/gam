//! What each face of an outer coordinate's domain is (#2627).
//!
//! The outer optimizer certifies an optimum that rests on a face of its box by
//! box-KKT projection: at the face it removes the outward half of the gradient
//! and judges what is left. That is stationarity only where the face belongs to
//! the model's parameter space. A face that marks the edge of what the
//! computation can represent, or the limit of what a coordinate's gradient can
//! resolve, is not a constraint of the model, and an outward pull there beyond
//! the gradient's own band means the criterion wants the coordinate past the face.
//!
//! So every face carries a [`DomainFaceKind`], declared by the producer that
//! derives it. The certificate reads the declared kind; nothing classifies a
//! face by its value.

use ndarray::Array1;

/// The kind of one face of an outer coordinate's domain, declared by the
/// producer that derives the face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainFaceKind {
    /// A constraint of the model's parameter space, derived by the objective
    /// from its own geometry. An optimum resting on it with an outward pull is a
    /// box-KKT point, so projection certifies it.
    Constraint,
    /// The coordinate's own resolution limit (#2812): past it, every
    /// direction's share of the gradient is under its own round-off. A face
    /// priced correctly leaves at most an outward pull inside the gradient's
    /// resolution, so the certificate accepts the face only while the pull it
    /// would remove is within its stationarity band. A larger pull says the face
    /// was mispriced at that state.
    Resolution,
    /// The edge of what the computation represents
    /// ([`crate::log_strength`]). It is not part of the model, so an outward
    /// pull beyond the band means the criterion wants the coordinate past what
    /// can be represented, and the face never certifies.
    Representability,
}

impl DomainFaceKind {
    /// Whether box-KKT projection at a face of this kind is stationarity. Only
    /// a constraint of the model is.
    pub fn projection_certifies(self) -> bool {
        matches!(self, Self::Constraint)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Constraint => "constraint",
            Self::Resolution => "resolution",
            Self::Representability => "representability",
        }
    }

    /// Which kind a face keeps when two producers declare it at the same value.
    /// A constraint of the model that binds at that value is a constraint there
    /// whatever else also ends there, and a resolution limit is a statement
    /// about the model's coordinate that representability is not.
    fn outranks(self, other: Self) -> bool {
        let rank = |kind: Self| match kind {
            Self::Representability => 0,
            Self::Resolution => 1,
            Self::Constraint => 2,
        };
        rank(self) > rank(other)
    }
}

impl std::fmt::Display for DomainFaceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Which side of a coordinate's domain a face bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainFaceSide {
    Lower,
    Upper,
}

impl std::fmt::Display for DomainFaceSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Lower => "lower",
            Self::Upper => "upper",
        })
    }
}

/// A coordinate whose certificate cleared its bound only by projecting away an
/// outward pull at a face that is not a constraint of the model (#2627).
#[derive(Debug, Clone, PartialEq)]
pub struct RefusedDomainFace {
    /// The coordinate, in the caller's native order.
    pub coordinate: usize,
    /// The coordinate's value at the judged point.
    pub theta: f64,
    /// The face it rests on, which side, and what kind of face that is.
    pub face: f64,
    pub side: DomainFaceSide,
    pub kind: DomainFaceKind,
    /// The gradient component projection would have removed: the pull of the
    /// criterion past the face.
    pub outward_gradient: f64,
}

impl std::fmt::Display for RefusedDomainFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "coordinate {} at {:.6e} on its {} {} face {:.6e} with outward gradient {:.6e}",
            self.coordinate, self.theta, self.side, self.kind, self.face, self.outward_gradient,
        )
    }
}

/// One side of an outer domain box: a face value per coordinate and the kind
/// of face each is.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainFaces {
    values: Array1<f64>,
    kinds: Vec<DomainFaceKind>,
}

impl DomainFaces {
    /// Every coordinate's face declared with one kind.
    pub fn uniform(values: Array1<f64>, kind: DomainFaceKind) -> Self {
        let kinds = vec![kind; values.len()];
        Self { values, kinds }
    }

    /// Faces whose kinds were declared one per coordinate.
    pub fn from_parts(values: Array1<f64>, kinds: Vec<DomainFaceKind>) -> Result<Self, String> {
        if kinds.len() != values.len() {
            return Err(format!(
                "domain faces declare {} kind(s) for {} face value(s)",
                kinds.len(),
                values.len()
            ));
        }
        Ok(Self { values, kinds })
    }

    pub fn values(&self) -> &Array1<f64> {
        &self.values
    }

    pub fn kinds(&self) -> &[DomainFaceKind] {
        &self.kinds
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Lower coordinate `index`'s upper face to `value` of `kind` where that is
    /// tighter. At an equal value the outranking kind is kept.
    pub fn tighten_upper(&mut self, index: usize, value: f64, kind: DomainFaceKind) {
        let current = self.values[index];
        if value < current || (value == current && kind.outranks(self.kinds[index])) {
            self.values[index] = value;
            self.kinds[index] = kind;
        }
    }

    /// Raise coordinate `index`'s lower face to `value` of `kind` where that is
    /// tighter. At an equal value the outranking kind is kept.
    pub fn tighten_lower(&mut self, index: usize, value: f64, kind: DomainFaceKind) {
        let current = self.values[index];
        if value > current || (value == current && kind.outranks(self.kinds[index])) {
            self.values[index] = value;
            self.kinds[index] = kind;
        }
    }

    /// Replace coordinate `index`'s face outright, for a producer whose face
    /// supersedes every generic one on that coordinate rather than intersecting
    /// with it.
    pub fn replace(&mut self, index: usize, value: f64, kind: DomainFaceKind) {
        self.values[index] = value;
        self.kinds[index] = kind;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tightening_keeps_the_tighter_face_and_its_kind() {
        let mut upper = DomainFaces::uniform(array![5.0, 5.0], DomainFaceKind::Representability);
        upper.tighten_upper(0, 3.0, DomainFaceKind::Constraint);
        upper.tighten_upper(1, 7.0, DomainFaceKind::Constraint);
        assert_eq!(upper.values(), &array![3.0, 5.0]);
        assert_eq!(
            upper.kinds(),
            &[DomainFaceKind::Constraint, DomainFaceKind::Representability]
        );

        let mut lower = DomainFaces::uniform(array![-5.0, -5.0], DomainFaceKind::Representability);
        lower.tighten_lower(0, -3.0, DomainFaceKind::Resolution);
        lower.tighten_lower(1, -7.0, DomainFaceKind::Resolution);
        assert_eq!(lower.values(), &array![-3.0, -5.0]);
        assert_eq!(
            lower.kinds(),
            &[DomainFaceKind::Resolution, DomainFaceKind::Representability]
        );
    }

    #[test]
    fn a_tie_keeps_the_outranking_kind_in_either_order() {
        for (first, second) in [
            (DomainFaceKind::Representability, DomainFaceKind::Constraint),
            (DomainFaceKind::Constraint, DomainFaceKind::Representability),
        ] {
            let mut upper = DomainFaces::uniform(array![2.0], first);
            upper.tighten_upper(0, 2.0, second);
            assert_eq!(upper.kinds(), &[DomainFaceKind::Constraint]);
        }
        let mut lower = DomainFaces::uniform(array![-2.0], DomainFaceKind::Representability);
        lower.tighten_lower(0, -2.0, DomainFaceKind::Resolution);
        assert_eq!(lower.kinds(), &[DomainFaceKind::Resolution]);
    }

    #[test]
    fn only_a_constraint_face_certifies_by_projection() {
        assert!(DomainFaceKind::Constraint.projection_certifies());
        assert!(!DomainFaceKind::Resolution.projection_certifies());
        assert!(!DomainFaceKind::Representability.projection_certifies());
    }

    #[test]
    fn declared_kinds_must_cover_every_face() {
        assert!(
            DomainFaces::from_parts(array![1.0, 2.0], vec![DomainFaceKind::Constraint]).is_err()
        );
        let faces = DomainFaces::from_parts(
            array![1.0, 2.0],
            vec![DomainFaceKind::Constraint, DomainFaceKind::Resolution],
        )
        .expect("one kind per face");
        assert_eq!(
            faces.kinds(),
            &[DomainFaceKind::Constraint, DomainFaceKind::Resolution]
        );
    }
}
