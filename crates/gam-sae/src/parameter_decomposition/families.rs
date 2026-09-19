//! Parameter-family labels for fixed components (#2951).
//!
//! A decomposition's fixed components `P_c` of one `rows x cols` tensor class are
//! grouped into families. A family holds an affine matrix-valued field over labels
//! `z_c` in `R^d`,
//!
//! ```text
//! Gamma(z) = B_0 + sum_{a=1..d} z_a U_a,      P_c = Gamma(z_c) for every member c,
//! ```
//!
//! and the labels belong to the component, never to the input: they are
//! parameter-family labels, not computational-state coordinates. A family of one
//! component at `d = 0` is the literal component. Sharing merges families into one
//! field, splitting takes a component out into its own literal family, and refining
//! or reducing moves a family's label dimension ([`FamilyPartition`]). This file builds
//! those proposals and their labels, encodes a proposal as one message, and scores the
//! decoded artifact. Nothing here compares two scores: a proposal is decided on
//! decoded code at a declared fidelity by `fit`'s rule.
//!
//! # Labels are principal coordinates
//!
//! For a family with `n` members take the center `B_0 = (1/n) sum_c P_c`, the
//! centered members `X_c = P_c - B_0`, their Frobenius Gram `G_ck = <X_c, X_k>_F`
//! and its eigenpairs `(lambda_a, V_a)` in descending order. With
//!
//! ```text
//! U_a = sum_c V_ca X_c / sqrt(lambda_a),      z_ca = sqrt(lambda_a) V_ca,
//! ```
//!
//! `<U_a, U_b>_F = V_a^T G V_b / sqrt(lambda_a lambda_b) = delta_ab` and
//! `z_ca = (G V_a)_c / sqrt(lambda_a) = <X_c, U_a>_F`. So the labels are isometric: a
//! label error `dz` moves the instance by `||dz||_2` in Frobenius norm, the unit of a
//! coefficient error. `B_0 + sum_{a<=d} z_ca U_a` is the best rank-`d` affine
//! approximation of the members in `sum_c ||.||_F^2` (Eckart-Young on the centered
//! stack), and at `d` equal to the rank of `G` it reproduces every member. No
//! coefficient is regressed here. Fitting coefficients against network responses is
//! `fit`'s conditionally Gaussian block.
//!
//! The label gauge `z -> Q z`, `U -> U Q^T` with `Q` in `O(d)` leaves every instance
//! unchanged, and the eigensolver's representative is returned. The code sends that
//! representative and claims no minimum over the gauge.
//!
//! # Resolved label dimension
//!
//! Each Gram entry is an inner product over `m = rows cols` products of two rounded
//! differences, at most `m + 2` rounded operations on a path, so its formation error
//! is at most `gamma_{m+2} sum_i |X_ci X_ki| <= gamma_{m+2} ||X_c||_F ||X_k||_F`, and
//! the Frobenius norm of the Gram's formation error is at most `gamma_{m+2} tr(G)` to
//! first order. A label dimension `d` is admitted only when
//! `d <= resolved_eigenvalue_count(lambda, gamma_{m+2} tr(G))`. A direction built from
//! an eigenvalue inside that band is rounding divided by `sqrt(lambda)`, and it never
//! enters a proposal. The Gram route costs `O(n^2 m)` time and `O(n^2 + (d + 2) m)`
//! memory and never stacks the members. It squares their conditioning, so a mode whose
//! singular value lies below the band's square root is not proposed.
//!
//! # Code
//!
//! A [`FamilyArtifact`] is one message, and its length is the number of bits its
//! decoder reads back. The tensor shape is a declaration of the tensor registry and is
//! not written. The message holds:
//! * the component count `C` as `C + 1` in the prefix integer code;
//! * the families in canonical order, each led by the smallest component not yet
//!   assigned. Per family:
//!   * its other members as an enumerative subset of the other unassigned components,
//!     so every partition of the components has exactly one codeword;
//!   * its label dimension `d` as `d + 1`;
//!   * the coefficients `B_0, U_1, ..., U_d` as one `precision` lattice message;
//!   * for `d > 0`, the `n x d` labels as a second lattice message.
//!
//! Every real passes through `precision`'s declared lattice. The decoder refuses a
//! label dimension of at least the member count (such a family has no canonical
//! directions), a lattice message whose count or declared precision disagrees with
//! the family, trailing bits, and a component count whose index state exceeds the
//! host in-core budget, before it allocates that state. Decoding makes one pass over
//! the unassigned components per family, and every family costs at least three bits.
//!
//! # Scoring
//!
//! [`score_decoded_families`] decodes an artifact, executes the decoded artifact
//! through the caller's executor, and states the declared distortion against the native
//! reference as evidence through `precision::decode_then_evaluate`. The score reads
//! only an exact figure: the largest distortion over what the executor ran, with its
//! derived rounding. A bound, an estimate, a counterexample or a bracket is not that
//! figure, and it is refused. The artifact's code length with that fidelity is what
//! `codec`'s comparison at proven fidelity and `fit`'s rule read. It certifies nothing
//! beyond the executed inputs and masks (#2946 fr-census overclaim audit, comment
//! 5716123817).
//!
//! # Relation to `field`
//!
//! A family is `field`'s instance form `P_c = w_c Gamma'(z_c)` over the degree-one
//! patch basis `(1, z_1, ..., z_d)`, with `w_c = 1/n` on the scale gauge and
//! `Gamma' = n Gamma`. [`PrincipalField::parameter_family`] and
//! [`DecodedFamily::parameter_family`] build that [`ParameterFamily`], so the pullbacks
//! and the function-space penalty read a proposal through `field`. A literal family has
//! no label manifold. It is a native tensor, not a field, and it is refused.

use std::fmt;
use std::sync::Arc;

use super::codec::{
    BitReader, BitString, CodecError, decode_prefix_integer, decode_subset, encode_prefix_integer,
    encode_subset,
};
use super::field::{FieldCoefficient, MatrixParameterField, ParameterFamily};
use super::precision::{
    DecodableArtifact, DecodedFidelity, DeclaredPrecision, LatticeCode, decode_then_evaluate,
};
use super::supports::EvidenceStatus;
use crate::basis::{EuclideanPatchEvaluator, SaeBasisEvaluator};
use faer::Side;
use gam_linalg::faer_ndarray::{FaerLinalgError, strict_symmetric_eigh};
use gam_linalg::roundoff::{accumulation_growth, resolved_eigenvalue_count};
use ndarray::{Array2, ArrayView2};

/// Why a family proposal was not built, encoded or decoded.
#[derive(Debug)]
pub enum FamilyError {
    /// The families are not a canonical partition of the components.
    InvalidPartition(String),
    /// The components are not one finite, nonempty tensor class.
    InvalidComponents(String),
    /// A family asks for more label dimensions than its centered Gram resolves.
    UnresolvedDimension {
        first_member: usize,
        dimension: usize,
        resolved: usize,
    },
    /// The dense state exceeds the host in-core budget. `required_bytes` is `None`
    /// when the count overflows `usize`.
    AdmissionRefused {
        required_bytes: Option<usize>,
        budget_bytes: usize,
    },
    /// The eigensolver refused the Gram.
    Eigen(FaerLinalgError),
    /// A message could not be written or read.
    Code(String),
    /// `field` refused the family's instance form, or the family has no label manifold.
    Field(String),
}

impl fmt::Display for FamilyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPartition(reason) => write!(f, "family partition refused: {reason}"),
            Self::InvalidComponents(reason) => write!(f, "family components refused: {reason}"),
            Self::UnresolvedDimension {
                first_member,
                dimension,
                resolved,
            } => write!(
                f,
                "the family led by component {first_member} asks for {dimension} label \
                 dimensions, while its centered Gram resolves {resolved} above its rounding band"
            ),
            Self::AdmissionRefused {
                required_bytes: Some(required),
                budget_bytes,
            } => write!(
                f,
                "family refused: its dense state needs at least {required} bytes, above the host \
                 in-core budget of {budget_bytes} bytes"
            ),
            Self::AdmissionRefused {
                required_bytes: None,
                budget_bytes,
            } => write!(
                f,
                "family refused: its byte count overflows usize (host in-core budget \
                 {budget_bytes} bytes)"
            ),
            Self::Eigen(error) => write!(f, "family spectrum refused: {error}"),
            Self::Code(reason) => write!(f, "family message refused: {reason}"),
            Self::Field(reason) => write!(f, "family field refused: {reason}"),
        }
    }
}

impl std::error::Error for FamilyError {}

fn code_error(error: CodecError) -> FamilyError {
    FamilyError::Code(error.to_string())
}

/// Refuses a dense state whose byte count overflows or exceeds the host in-core
/// budget.
fn admit_bytes(required_bytes: Option<usize>) -> Result<(), FamilyError> {
    let budget_bytes = crate::manifold::sae_host_in_core_budget_bytes().0;
    match required_bytes {
        Some(required) if required <= budget_bytes => Ok(()),
        required_bytes => Err(FamilyError::AdmissionRefused {
            required_bytes,
            budget_bytes,
        }),
    }
}

/// One family: its members, ascending, and its label dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilySpec {
    pub members: Vec<usize>,
    pub dimension: usize,
}

/// A partition of `components` fixed components into families, held in canonical
/// order (by smallest member).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyPartition {
    components: usize,
    families: Vec<FamilySpec>,
}

impl FamilyPartition {
    /// Refuses a partition with no components, an empty family, members that are not
    /// strictly ascending or not below `components`, a component in two families or
    /// in none, and a label dimension of at least the family's member count.
    pub fn new(components: usize, mut families: Vec<FamilySpec>) -> Result<Self, FamilyError> {
        if components == 0 {
            return Err(FamilyError::InvalidPartition(
                "a partition needs at least one component".to_string(),
            ));
        }
        let mut assigned = vec![false; components];
        for (index, family) in families.iter().enumerate() {
            check_members(&family.members, components)
                .map_err(|reason| FamilyError::InvalidPartition(format!("family {index}: {reason}")))?;
            if family.dimension >= family.members.len() {
                return Err(FamilyError::InvalidPartition(format!(
                    "family {index} has {} members, which span at most {} label dimensions, not {}",
                    family.members.len(),
                    family.members.len() - 1,
                    family.dimension
                )));
            }
            for &member in &family.members {
                if assigned[member] {
                    return Err(FamilyError::InvalidPartition(format!(
                        "component {member} is in two families"
                    )));
                }
                assigned[member] = true;
            }
        }
        if let Some(member) = assigned.iter().position(|&taken| !taken) {
            return Err(FamilyError::InvalidPartition(format!(
                "component {member} is in no family"
            )));
        }
        families.sort_by_key(|family| family.members[0]);
        Ok(Self {
            components,
            families,
        })
    }

    /// Every component in its own family at label dimension zero.
    pub fn literal(components: usize) -> Result<Self, FamilyError> {
        Self::new(
            components,
            (0..components)
                .map(|component| FamilySpec {
                    members: vec![component],
                    dimension: 0,
                })
                .collect(),
        )
    }

    pub fn components(&self) -> usize {
        self.components
    }

    /// The families in canonical order.
    pub fn families(&self) -> &[FamilySpec] {
        &self.families
    }

    fn family(&self, index: usize) -> Result<&FamilySpec, FamilyError> {
        self.families.get(index).ok_or_else(|| {
            FamilyError::InvalidPartition(format!(
                "family {index} does not exist; the partition has {}",
                self.families.len()
            ))
        })
    }

    /// Share: families `first` and `second` become one family at label dimension
    /// `dimension`.
    pub fn share(&self, first: usize, second: usize, dimension: usize) -> Result<Self, FamilyError> {
        if first == second {
            return Err(FamilyError::InvalidPartition(format!(
                "family {first} cannot be shared with itself"
            )));
        }
        let mut members = self.family(first)?.members.clone();
        members.extend_from_slice(&self.family(second)?.members);
        members.sort_unstable();
        let mut families = Vec::with_capacity(self.families.len() - 1);
        for (index, family) in self.families.iter().enumerate() {
            if index != first && index != second {
                families.push(family.clone());
            }
        }
        families.push(FamilySpec { members, dimension });
        Self::new(self.components, families)
    }

    /// Split: `member` leaves family `family` as a literal, and the rest of the family
    /// keeps label dimension `remaining_dimension`.
    pub fn split(
        &self,
        family: usize,
        member: usize,
        remaining_dimension: usize,
    ) -> Result<Self, FamilyError> {
        let source = self.family(family)?;
        if source.members.len() < 2 {
            return Err(FamilyError::InvalidPartition(format!(
                "family {family} has one member and nothing to split off"
            )));
        }
        let position = source.members.binary_search(&member).map_err(|insertion| {
            FamilyError::InvalidPartition(format!(
                "component {member} is not in family {family} (it would sit at position {insertion})"
            ))
        })?;
        let mut families = self.families.clone();
        families[family].members.remove(position);
        families[family].dimension = remaining_dimension;
        families.push(FamilySpec {
            members: vec![member],
            dimension: 0,
        });
        Self::new(self.components, families)
    }

    /// Refine or reduce: family `family` moves to label dimension `dimension`.
    pub fn with_dimension(&self, family: usize, dimension: usize) -> Result<Self, FamilyError> {
        self.family(family)?;
        let mut families = self.families.clone();
        families[family].dimension = dimension;
        Self::new(self.components, families)
    }
}

/// Refuses an empty member list, members that are not strictly ascending, and a
/// member at or above `components`.
fn check_members(members: &[usize], components: usize) -> Result<(), String> {
    let Some(&largest) = members.last() else {
        return Err("a family needs at least one member".to_string());
    };
    if let Some(pair) = members.windows(2).find(|pair| pair[0] >= pair[1]) {
        return Err(format!(
            "members must be strictly ascending, got {} then {}",
            pair[0], pair[1]
        ));
    }
    if largest >= components {
        return Err(format!(
            "member {largest} is outside the {components} components"
        ));
    }
    Ok(())
}

/// The shape of a finite, nonempty tensor class shared by every component.
fn component_class(components: &[ArrayView2<'_, f64>]) -> Result<(usize, usize), FamilyError> {
    let first = components
        .first()
        .ok_or_else(|| FamilyError::InvalidComponents("no components".to_string()))?;
    let (rows, cols) = first.dim();
    if rows == 0 || cols == 0 {
        return Err(FamilyError::InvalidComponents(format!(
            "components must be nonempty tensors, got {rows} x {cols}"
        )));
    }
    for (index, component) in components.iter().enumerate() {
        if component.dim() != (rows, cols) {
            return Err(FamilyError::InvalidComponents(format!(
                "component {index} is {:?}, component 0 is {rows} x {cols}",
                component.dim()
            )));
        }
        if component.iter().any(|value| !value.is_finite()) {
            return Err(FamilyError::InvalidComponents(format!(
                "component {index} has a non-finite entry"
            )));
        }
    }
    Ok((rows, cols))
}

/// The principal affine field of one family's members.
#[derive(Clone, Debug)]
pub struct PrincipalField {
    /// `B_0`, the members' mean.
    pub center: Array2<f64>,
    /// `U_1, ..., U_d`, orthonormal in the Frobenius product.
    pub directions: Vec<Array2<f64>>,
    /// `n x d`: row `i` is the label of the `i`-th member in ascending order.
    pub labels: Array2<f64>,
    /// Every eigenvalue of the centered Gram, descending.
    pub eigenvalues: Vec<f64>,
    /// `gamma_{m+2} tr(G)`, the Gram's formation band.
    pub assembly_band: f64,
    /// The number of label dimensions the spectrum resolves.
    pub resolved_dimension: usize,
}

/// The principal affine field of `members` at label dimension `dimension` (see the
/// module docs).
///
/// Refuses components that are not one finite tensor class, malformed members, a
/// dimension of at least the member count or above the resolved dimension, and a
/// dense state beyond the host in-core budget.
pub fn principal_field(
    components: &[ArrayView2<'_, f64>],
    members: &[usize],
    dimension: usize,
) -> Result<PrincipalField, FamilyError> {
    let (rows, cols) = component_class(components)?;
    principal_field_of_class(components, rows, cols, members, dimension)
}

/// [`principal_field`] for components already checked to be one `rows x cols` class.
fn principal_field_of_class(
    components: &[ArrayView2<'_, f64>],
    rows: usize,
    cols: usize,
    members: &[usize],
    dimension: usize,
) -> Result<PrincipalField, FamilyError> {
    check_members(members, components.len()).map_err(FamilyError::InvalidPartition)?;
    let n = members.len();
    if dimension >= n {
        return Err(FamilyError::InvalidPartition(format!(
            "{n} members span at most {} label dimensions, not {dimension}",
            n - 1
        )));
    }
    let entries = rows * cols;
    // The Gram and its eigenvectors, the center and the directions.
    admit_bytes(
        n.checked_mul(n)
            .and_then(|square| square.checked_mul(2))
            .and_then(|squares| {
                dimension
                    .checked_add(1)
                    .and_then(|tensors| tensors.checked_mul(entries))
                    .and_then(|doubles| doubles.checked_add(squares))
            })
            .and_then(|doubles| doubles.checked_mul(std::mem::size_of::<f64>())),
    )?;
    let mut center = Array2::<f64>::zeros((rows, cols));
    for &member in members {
        center += &components[member];
    }
    let count = n as f64;
    center.mapv_inplace(|value| value / count);
    let mut gram = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for k in i..n {
            let value: f64 = components[members[i]]
                .iter()
                .zip(components[members[k]].iter())
                .zip(center.iter())
                .map(|((&left, &right), &mean)| (left - mean) * (right - mean))
                .sum();
            gram[[i, k]] = value;
            gram[[k, i]] = value;
        }
    }
    let trace: f64 = (0..n).map(|i| gram[[i, i]]).sum();
    let assembly_band = accumulation_growth(entries + 2) * trace;
    let (values, vectors) = strict_symmetric_eigh(&gram, Side::Lower).map_err(FamilyError::Eigen)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&left, &right| values[right].total_cmp(&values[left]));
    let eigenvalues: Vec<f64> = order.iter().map(|&index| values[index]).collect();
    let resolved_dimension = resolved_eigenvalue_count(&eigenvalues, assembly_band);
    if dimension > resolved_dimension {
        return Err(FamilyError::UnresolvedDimension {
            first_member: members[0],
            dimension,
            resolved: resolved_dimension,
        });
    }
    let roots: Vec<f64> = eigenvalues[..dimension]
        .iter()
        .map(|value| value.sqrt())
        .collect();
    let directions = (0..dimension)
        .map(|a| {
            let mut direction = Array2::<f64>::zeros((rows, cols));
            for (i, &member) in members.iter().enumerate() {
                let weight = vectors[[i, order[a]]] / roots[a];
                for ((slot, &value), &mean) in direction
                    .iter_mut()
                    .zip(components[member].iter())
                    .zip(center.iter())
                {
                    *slot += weight * (value - mean);
                }
            }
            direction
        })
        .collect();
    let labels = Array2::from_shape_fn((n, dimension), |(i, a)| roots[a] * vectors[[i, order[a]]]);
    Ok(PrincipalField {
        center,
        directions,
        labels,
        eigenvalues,
        assembly_band,
        resolved_dimension,
    })
}

/// A family partition of components, as one message (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyArtifact {
    rows: usize,
    cols: usize,
    message: BitString,
}

impl FamilyArtifact {
    /// A transmitted message for components of the declared `rows x cols` class.
    /// Refuses an empty or overflowing shape; the message itself is checked when it is
    /// decoded.
    pub fn from_message(rows: usize, cols: usize, message: BitString) -> Result<Self, FamilyError> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols).is_none() {
            return Err(FamilyError::InvalidComponents(format!(
                "a family message needs a nonempty tensor class, got {rows} x {cols}"
            )));
        }
        Ok(Self {
            rows,
            cols,
            message,
        })
    }

    /// The exact artifact length in bits.
    pub fn code_bits(&self) -> u64 {
        self.message.len_bits()
    }

    pub fn message(&self) -> &BitString {
        &self.message
    }
}

/// Encodes `partition` of `components` at the declared `precision`.
///
/// Refuses components that are not one finite tensor class, a partition of a
/// different component count, a family whose label dimension is above its resolved
/// dimension, and a lattice index the precision cannot decode exactly.
pub fn encode_families(
    components: &[ArrayView2<'_, f64>],
    partition: &FamilyPartition,
    precision: DeclaredPrecision,
) -> Result<FamilyArtifact, FamilyError> {
    let (rows, cols) = component_class(components)?;
    if components.len() != partition.components() {
        return Err(FamilyError::InvalidPartition(format!(
            "the partition covers {} components, {} were given",
            partition.components(),
            components.len()
        )));
    }
    let mut message = BitString::new();
    encode_prefix_integer(&mut message, components.len() as u64 + 1).map_err(code_error)?;
    let mut remaining: Vec<usize> = (0..components.len()).collect();
    for family in partition.families() {
        // Canonical order puts the smallest unassigned component first in this family.
        if remaining.first() != Some(&family.members[0]) {
            return Err(FamilyError::InvalidPartition(format!(
                "the family led by component {} is not in canonical order",
                family.members[0]
            )));
        }
        let mut positions: Vec<usize> = Vec::with_capacity(family.members.len() - 1);
        for member in &family.members[1..] {
            positions.push(remaining[1..].binary_search(member).map_err(|insertion| {
                FamilyError::InvalidPartition(format!(
                    "component {member} was already assigned (unassigned position {insertion})"
                ))
            })?);
        }
        encode_subset(&mut message, remaining.len() - 1, &positions).map_err(code_error)?;
        encode_prefix_integer(&mut message, family.dimension as u64 + 1).map_err(code_error)?;
        let field = principal_field_of_class(components, rows, cols, &family.members, family.dimension)?;
        let mut coefficients: Vec<f64> = Vec::with_capacity((family.dimension + 1) * rows * cols);
        coefficients.extend(field.center.iter());
        for direction in &field.directions {
            coefficients.extend(direction.iter());
        }
        LatticeCode::encode(&coefficients, precision)
            .and_then(|code| code.write(&mut message))
            .map_err(FamilyError::Code)?;
        if family.dimension > 0 {
            let labels: Vec<f64> = field.labels.iter().copied().collect();
            LatticeCode::encode(&labels, precision)
                .and_then(|code| code.write(&mut message))
                .map_err(FamilyError::Code)?;
        }
        remaining.retain(|component| family.members.binary_search(component).is_err());
    }
    Ok(FamilyArtifact {
        rows,
        cols,
        message,
    })
}

/// One decoded family.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedFamily {
    /// The members, ascending.
    pub members: Vec<usize>,
    /// `B_0, U_1, ..., U_d` as the decoder rebuilt them.
    pub coefficients: Vec<Array2<f64>>,
    /// `n x d`: row `i` is the decoded label of the `i`-th member.
    pub labels: Array2<f64>,
}

/// What a decoder rebuilds from a [`FamilyArtifact`].
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedFamilies {
    rows: usize,
    cols: usize,
    precision: DeclaredPrecision,
    families: Vec<DecodedFamily>,
    /// `(family, position)` of each component.
    placement: Vec<(usize, usize)>,
}

impl DecodedFamilies {
    /// `(rows, cols)` of every instance.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The precision every real of the message declared.
    pub fn precision(&self) -> DeclaredPrecision {
        self.precision
    }

    pub fn components(&self) -> usize {
        self.placement.len()
    }

    pub fn families(&self) -> &[DecodedFamily] {
        &self.families
    }

    /// The decoded family structure.
    pub fn partition(&self) -> Result<FamilyPartition, FamilyError> {
        FamilyPartition::new(
            self.placement.len(),
            self.families
                .iter()
                .map(|family| FamilySpec {
                    members: family.members.clone(),
                    dimension: family.labels.ncols(),
                })
                .collect(),
        )
    }

    /// The decoded instance `B_0 + sum_a z_ca U_a` of `component`.
    pub fn instance(&self, component: usize) -> Result<Array2<f64>, String> {
        let &(family, position) = self.placement.get(component).ok_or_else(|| {
            format!(
                "component {component} is outside the {} decoded components",
                self.placement.len()
            )
        })?;
        let family = &self.families[family];
        let mut value = family.coefficients[0].clone();
        for (a, direction) in family.coefficients[1..].iter().enumerate() {
            value.scaled_add(family.labels[[position, a]], direction);
        }
        Ok(value)
    }
}

/// `field`'s instance form of an affine family over the degree-one patch basis.
///
/// `EuclideanPatchEvaluator::new(d, 1)` evaluates the raw monomials in the column order
/// of `monomial_exponents(d, 1)`, which is `(1, z_1, ..., z_d)`. So the coefficients are
/// `[B_0, U_1, ..., U_d]` as given. The raw weights are one per member.
/// `ParameterFamily::new` moves their total `n` into the coefficients, so `w_c = 1/n`,
/// `Gamma' = n Gamma`, and every instance `w_c Gamma'(z_c)` equals `B_0 + sum_a z_ca U_a`
/// up to rounding.
fn affine_parameter_family(
    coefficients: Vec<Array2<f64>>,
    labels: Array2<f64>,
) -> Result<ParameterFamily, FamilyError> {
    let dimension = labels.ncols();
    if dimension == 0 {
        return Err(FamilyError::Field(
            "a literal family (d = 0) has no label manifold, so it is a native tensor, not a field"
                .to_string(),
        ));
    }
    if coefficients.len() != dimension + 1 {
        return Err(FamilyError::Field(format!(
            "{} coefficients for label dimension {dimension}; the degree-one basis has {}",
            coefficients.len(),
            dimension + 1
        )));
    }
    let basis: Arc<dyn SaeBasisEvaluator> =
        Arc::new(EuclideanPatchEvaluator::new(dimension, 1).map_err(FamilyError::Field)?);
    let field = MatrixParameterField::new(
        basis,
        coefficients.into_iter().map(FieldCoefficient::Dense).collect(),
    )
    .map_err(FamilyError::Field)?;
    let weights = ndarray::Array1::from_elem(labels.nrows(), 1.0);
    ParameterFamily::new(field, labels, weights).map_err(FamilyError::Field)
}

impl PrincipalField {
    /// The family as `field` instances over the degree-one patch basis (see the module
    /// docs). A literal family (`d = 0`) is refused.
    pub fn parameter_family(&self) -> Result<ParameterFamily, FamilyError> {
        let mut coefficients = Vec::with_capacity(self.directions.len() + 1);
        coefficients.push(self.center.clone());
        coefficients.extend(self.directions.iter().cloned());
        affine_parameter_family(coefficients, self.labels.clone())
    }
}

impl DecodedFamily {
    /// The decoded family as `field` instances: the reals the artifact carries, as the
    /// decoder rebuilt them. A literal family (`d = 0`) is refused.
    pub fn parameter_family(&self) -> Result<ParameterFamily, FamilyError> {
        affine_parameter_family(self.coefficients.clone(), self.labels.clone())
    }
}

impl DecodableArtifact for FamilyArtifact {
    type Decoded = DecodedFamilies;

    fn decode(&self) -> Result<DecodedFamilies, String> {
        decode_families(self).map_err(|error| error.to_string())
    }
}

/// Reads one lattice message of exactly `count` reals, all at one declared precision.
fn read_reals(
    reader: &mut BitReader<'_>,
    count: usize,
    precision: &mut Option<DeclaredPrecision>,
) -> Result<Vec<f64>, FamilyError> {
    let code = LatticeCode::read(reader).map_err(FamilyError::Code)?;
    if code.indices().len() != count {
        return Err(FamilyError::Code(format!(
            "a lattice message of {} reals where the family needs {count}",
            code.indices().len()
        )));
    }
    let declared = *precision.get_or_insert(code.precision());
    if declared != code.precision() {
        return Err(FamilyError::Code(format!(
            "a lattice message at 2^-{} where the artifact declared 2^-{}",
            code.precision().fraction_bits(),
            declared.fraction_bits()
        )));
    }
    code.decode().map_err(FamilyError::Code)
}

fn decode_families(artifact: &FamilyArtifact) -> Result<DecodedFamilies, FamilyError> {
    let (rows, cols) = (artifact.rows, artifact.cols);
    let entries = rows * cols;
    let mut reader = artifact.message.reader();
    let announced = decode_prefix_integer(&mut reader).map_err(code_error)? - 1;
    if announced == 0 {
        return Err(FamilyError::InvalidPartition(
            "the message announces no components".to_string(),
        ));
    }
    let components = usize::try_from(announced).map_err(|error| {
        FamilyError::InvalidPartition(format!("component count {announced}: {error}"))
    })?;
    // The unassigned list, the placements and every member list: 4C indices.
    admit_bytes(components.checked_mul(4 * std::mem::size_of::<usize>()))?;
    let mut remaining: Vec<usize> = (0..components).collect();
    let mut placement = vec![(0_usize, 0_usize); components];
    let mut families: Vec<DecodedFamily> = Vec::new();
    let mut precision = None;
    while let Some(&first) = remaining.first() {
        let others = decode_subset(&mut reader, remaining.len() - 1).map_err(code_error)?;
        let mut members = Vec::with_capacity(others.len() + 1);
        members.push(first);
        members.extend(others.iter().map(|&position| remaining[position + 1]));
        let announced_dimension = decode_prefix_integer(&mut reader).map_err(code_error)? - 1;
        if announced_dimension >= members.len() as u64 {
            return Err(FamilyError::InvalidPartition(format!(
                "the family led by component {first} has {} members and announces label dimension \
                 {announced_dimension}",
                members.len()
            )));
        }
        let dimension = announced_dimension as usize;
        let overflow = || FamilyError::Code(format!("the family led by component {first} overflows usize"));
        let coefficient_count = (dimension + 1).checked_mul(entries).ok_or_else(overflow)?;
        let label_count = members.len().checked_mul(dimension).ok_or_else(overflow)?;
        let values = read_reals(&mut reader, coefficient_count, &mut precision)?;
        let coefficients = values
            .chunks(entries)
            .map(|chunk| {
                Array2::from_shape_vec((rows, cols), chunk.to_vec())
                    .map_err(|error| FamilyError::Code(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let labels = if dimension == 0 {
            Array2::<f64>::zeros((members.len(), 0))
        } else {
            let values = read_reals(&mut reader, label_count, &mut precision)?;
            Array2::from_shape_vec((members.len(), dimension), values)
                .map_err(|error| FamilyError::Code(error.to_string()))?
        };
        for (position, &member) in members.iter().enumerate() {
            placement[member] = (families.len(), position);
        }
        remaining.retain(|component| members.binary_search(component).is_err());
        families.push(DecodedFamily {
            members,
            coefficients,
            labels,
        });
    }
    reader.finish().map_err(code_error)?;
    let precision = precision.ok_or_else(|| {
        FamilyError::Code("the message carries no real, so it declares no precision".to_string())
    })?;
    Ok(DecodedFamilies {
        rows,
        cols,
        precision,
        families,
        placement,
    })
}

/// Decodes `artifact`, executes the decoded artifact with `evaluate`, and scores it
/// against `native_reference` under the declared `tolerance`.
///
/// `measure` states the largest declared distortion over what `evaluate` executed as
/// evidence. Returns the artifact's exact message length with that fidelity evidence and
/// its proven verdict, the pair `codec`'s comparison at proven fidelity reads. The
/// measurement covers only the executed inputs and masks (#2946 fr-census overclaim
/// audit, comment 5716123817). Refuses what `decode_then_evaluate` refuses, and a
/// status that is not exact.
pub fn score_decoded_families<O, E, M, W, D>(
    artifact: &FamilyArtifact,
    evaluate: E,
    native_reference: &O,
    measure: M,
    tolerance: f64,
) -> Result<(u64, DecodedFidelity<W, D>), String>
where
    E: FnOnce(&DecodedFamilies) -> Result<O, String>,
    M: FnOnce(&O, &O) -> Result<EvidenceStatus<W, D>, String>,
{
    let fidelity = decode_then_evaluate(artifact, evaluate, native_reference, measure, tolerance)?;
    match fidelity.status() {
        EvidenceStatus::Exact { .. } => {}
        EvidenceStatus::UniformBound { .. } => {
            return Err(not_the_largest_distortion("a uniform bound"));
        }
        EvidenceStatus::StatisticalEstimate { .. } => {
            return Err(not_the_largest_distortion("a statistical estimate"));
        }
        EvidenceStatus::Counterexample { .. } => {
            return Err(not_the_largest_distortion("a counterexample"));
        }
        EvidenceStatus::Unresolved { .. } => {
            return Err(not_the_largest_distortion("an unresolved bracket"));
        }
    }
    Ok((artifact.code_bits(), fidelity))
}

fn not_the_largest_distortion(kind: &str) -> String {
    format!(
        "score_decoded_families: the distortion evidence is {kind}, not the exact largest \
         distortion over the executed family that a code comparison reads"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::codec::{
        code_saving_at_proven_fidelity, prefix_integer_len_bits, subset_code_len_bits,
    };
    use crate::parameter_decomposition::precision::FidelityVerdict;
    use crate::parameter_decomposition::fit::{
        ProposalAcceptance, ProposalKind, ProposalRejection, decide_proposal,
    };
    use crate::parameter_decomposition::supports::ExactBasis;
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, symmetric_spectrum_rounding_band};
    use ndarray::Array1;

    const ROWS: usize = 4;
    const COLS: usize = 5;
    const MEMBERS: usize = 6;
    /// The declared fidelity tolerance of the scoring fixtures.
    const TOLERANCE: f64 = 1.0e-3;
    /// The declared precision of the scoring fixtures: 24 binary digits after the point.
    const FRACTION_BITS: i32 = 24;

    /// Deterministic fixture entries in `[-1, 1)`.
    fn entries(seed: usize, count: usize) -> Vec<f64> {
        (0..count)
            .map(|k| {
                let x = ((seed * 7919 + k * 104_729 + 1) as f64).sin() * 43_758.545_3;
                (x - x.floor()) * 2.0 - 1.0
            })
            .collect()
    }

    fn matrix(seed: usize, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), entries(seed, rows * cols)).expect("fixture shape")
    }

    fn views(components: &[Array2<f64>]) -> Vec<ArrayView2<'_, f64>> {
        components.iter().map(|component| component.view()).collect()
    }

    /// Six components on a planted two-dimensional affine field `B_0 + z_1 U_1 + z_2 U_2`.
    fn planted_components() -> Vec<Array2<f64>> {
        let center = matrix(1, ROWS, COLS);
        let first = matrix(2, ROWS, COLS);
        let second = matrix(3, ROWS, COLS);
        let labels = entries(4, 2 * MEMBERS);
        (0..MEMBERS)
            .map(|c| &center + &(&first * labels[2 * c]) + &(&second * labels[2 * c + 1]))
            .collect()
    }

    /// Six components drawn as a random initialization: no shared field.
    fn random_components() -> Vec<Array2<f64>> {
        (0..MEMBERS).map(|c| matrix(10 + c, ROWS, COLS)).collect()
    }

    /// `P_c = (2/C) v(t_c) v(t_c)^T` at `t_c = c pi / C`, which sums to `I_2` (P11).
    fn projector_components(instances: usize) -> Vec<Array2<f64>> {
        (0..instances)
            .map(|c| {
                let angle = std::f64::consts::PI * c as f64 / instances as f64;
                let direction = [angle.cos(), angle.sin()];
                Array2::from_shape_fn((2, 2), |(r, k)| 2.0 / instances as f64 * direction[r] * direction[k])
            })
            .collect()
    }

    /// The quantities the tests' bands are built from.
    ///
    /// The premise is the owner band's (`symmetric_spectrum_rounding_band`): the
    /// eigensolver returns an orthonormal `V` and a `Lambda` that exactly diagonalize
    /// `G_x + E`, where `G_x` is the exact Gram of the computed centered members and
    /// `||E||_2 <= beta = p eps ||G||_2 + gamma_{m+2} tr(G)`. Every bound below is first
    /// order in `eps`.
    struct Bands {
        /// The computed centered members `X_c = P_c - B_0`, in member order.
        centered: Vec<Array2<f64>>,
        /// `sqrt(sum_k X_ki^2)` per entry. For a unit `V_a`,
        /// `sum_k |V_ka X_ki| <= reach_i`, so forming `U_a` rounds each entry by at most
        /// `gamma reach_i / sqrt(lambda_a)`.
        reach: Array2<f64>,
        beta: f64,
    }

    fn bands(components: &[Array2<f64>], members: &[usize], field: &PrincipalField) -> Bands {
        let centered: Vec<Array2<f64>> = members
            .iter()
            .map(|&member| &components[member] - &field.center)
            .collect();
        let mut reach = Array2::<f64>::zeros(field.center.dim());
        for x in &centered {
            reach += &x.mapv(|value| value * value);
        }
        reach.mapv_inplace(f64::sqrt);
        Bands {
            centered,
            reach,
            beta: symmetric_spectrum_rounding_band(&field.eigenvalues) + field.assembly_band,
        }
    }

    /// `|U_a| + reach / sqrt(lambda_a)`: the magnitude of the terms that formed `U_a`.
    fn direction_magnitude(field: &PrincipalField, bands: &Bands, a: usize) -> Array2<f64> {
        field.directions[a].mapv(f64::abs) + &(&bands.reach / field.eigenvalues[a].sqrt())
    }

    fn frobenius(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter().zip(right.iter()).map(|(x, y)| x * y).sum()
    }

    /// `sqrt(sum_c ||X_c - sum_{a<=d} z_ca U_a||_F^2)` and the bound it must meet.
    ///
    /// `sum_a z_ca U_a = sum_k (V_d V_d^T)_ck X_k`, so the squared residual is
    /// `tr((I - P_d) G_x)`. Under the premise that is
    /// `sum_{a>d} lambda_a - tr((I - P_d) E) <= sum_{a>d} lambda_a + (n - d) beta`,
    /// and every unresolved eigenvalue is at most `beta`, so the residual is at most
    /// `sqrt(2 (n - d) beta)`. The reconstruction adds its own formation rounding: a
    /// path forms `U_a` (at most `n + 2` operations), the label (2), the product and the
    /// `d + 1` accumulations, at most `n + 2d + 8` operations against the magnitude
    /// `|X_c| + sum_a |z_ca| (|U_a| + reach / sqrt(lambda_a))`.
    fn reproduction(field: &PrincipalField, bands: &Bands) -> (f64, f64) {
        let n = bands.centered.len();
        let dimension = field.directions.len();
        let growth = accumulation_growth(n + 2 * dimension + 8);
        let mut squared = 0.0;
        let mut formation = 0.0;
        for (i, x) in bands.centered.iter().enumerate() {
            let mut residual = x.clone();
            let mut magnitude = x.mapv(f64::abs);
            for a in 0..dimension {
                residual.scaled_add(-field.labels[[i, a]], &field.directions[a]);
                magnitude.scaled_add(field.labels[[i, a]].abs(), &direction_magnitude(field, bands, a));
            }
            squared += residual.iter().map(|value| value * value).sum::<f64>();
            formation += magnitude.iter().map(|value| (growth * value).powi(2)).sum::<f64>();
        }
        let sum_growth = 1.0 + accumulation_growth(n * field.center.len());
        let bound = sum_growth * ((2.0 * (n - dimension) as f64 * bands.beta).sqrt() + formation.sqrt());
        (squared.sqrt(), bound)
    }

    /// Asserts `<U_a, U_b>_F = delta_ab` and `z_ca = <X_c, U_a>_F` within their bands,
    /// and returns whether the uncentered `<P_c, U_a>_F` is refuted for some label.
    /// That negative control applies only where the center has a component along the
    /// directions: the projector family's center `I/C` is orthogonal to its traceless
    /// directions.
    ///
    /// Under the premise `V_a^T G_x V_b = lambda_a delta_ab - V_a^T E V_b`, so
    /// `|<U_a, U_b> - delta_ab| <= beta / sqrt(lambda_a lambda_b)` and
    /// `|<X_c, U_a> - z_ca| <= beta / sqrt(lambda_a)`. Each inner product adds the
    /// rounding of forming its factors and summing its `m` products, at most
    /// `m + n + 6` operations against the magnitudes of the terms.
    fn assert_isometric_labels(components: &[Array2<f64>], members: &[usize], field: &PrincipalField) -> bool {
        let bands = bands(components, members, field);
        let dimension = field.directions.len();
        let growth = accumulation_growth(field.center.len() + members.len() + 6);
        let magnitudes: Vec<Array2<f64>> = (0..dimension)
            .map(|a| direction_magnitude(field, &bands, a))
            .collect();
        for a in 0..dimension {
            for b in 0..dimension {
                let inner = frobenius(&field.directions[a], &field.directions[b]);
                let target = if a == b { 1.0 } else { 0.0 };
                let bound = bands.beta / (field.eigenvalues[a] * field.eigenvalues[b]).sqrt()
                    + growth * frobenius(&magnitudes[a], &magnitudes[b]);
                assert!(
                    (inner - target).abs() <= bound,
                    "<U_{a}, U_{b}> = {inner}, not {target} within {bound:e}"
                );
            }
        }
        let mut uncentered_refuted = false;
        for (i, (&member, x)) in members.iter().zip(&bands.centered).enumerate() {
            for a in 0..dimension {
                let label = field.labels[[i, a]];
                let centered_inner = frobenius(x, &field.directions[a]);
                let bound = bands.beta / field.eigenvalues[a].sqrt()
                    + growth * frobenius(&x.mapv(f64::abs), &magnitudes[a])
                    + accumulation_growth(2) * label.abs();
                assert!(
                    (label - centered_inner).abs() <= bound,
                    "member {member} label {a}: z = {label} against <X_c, U_a> = {centered_inner} \
                     (bound {bound:e})"
                );
                let uncentered_inner = frobenius(&components[member], &field.directions[a]);
                let uncentered_band = bound + growth * frobenius(&components[member].mapv(f64::abs), &magnitudes[a]);
                uncentered_refuted |= (label - uncentered_inner).abs() > uncentered_band;
            }
        }
        uncentered_refuted
    }

    #[test]
    fn planted_field_labels_are_isometric_and_reproduce_the_members_at_the_resolved_dimension() {
        let components = planted_components();
        let view = views(&components);
        let everyone: Vec<usize> = (0..MEMBERS).collect();
        let field = principal_field(&view, &everyone, 2).expect("the planted field is resolved");
        assert_eq!(
            field.resolved_dimension, 2,
            "eigenvalues {:?} against the formation band {:e}",
            field.eigenvalues, field.assembly_band
        );
        // Guard: a label dimension above the resolved spectrum is refused; d = 2 above is
        // the positive control.
        let refused = principal_field(&view, &everyone, 3);
        assert!(
            matches!(refused, Err(FamilyError::UnresolvedDimension { first_member: 0, dimension: 3, resolved: 2 })),
            "got {refused:?}"
        );
        assert!(
            assert_isometric_labels(&components, &everyone, &field),
            "negative control: the uncentered inner product must disagree with some planted label"
        );
        let (residual, bound) = reproduction(&field, &bands(&components, &everyone, &field));
        assert!(
            residual <= bound,
            "the d = 2 field misses the planted members by {residual:e}, above {bound:e}"
        );
        // Negative control: one label dimension fewer discards a direction the members use.
        let reduced = principal_field(&view, &everyone, 1).expect("d = 1 is resolved");
        let (reduced_residual, reduced_bound) = reproduction(&reduced, &bands(&components, &everyone, &reduced));
        assert!(
            reduced_residual > reduced_bound,
            "the d = 1 field must miss the members: residual {reduced_residual:e}, bound {reduced_bound:e}"
        );
    }

    #[test]
    fn random_init_components_need_every_centered_direction() {
        let components = random_components();
        let view = views(&components);
        let everyone: Vec<usize> = (0..MEMBERS).collect();
        let full = principal_field(&view, &everyone, MEMBERS - 1).expect("every centered direction is resolved");
        assert_eq!(
            full.resolved_dimension,
            MEMBERS - 1,
            "random components resolve every centered direction: eigenvalues {:?}, band {:e}",
            full.eigenvalues,
            full.assembly_band
        );
        assert!(
            assert_isometric_labels(&components, &everyone, &full),
            "negative control: the uncentered inner product must disagree with some random label"
        );
        let (residual, bound) = reproduction(&full, &bands(&components, &everyone, &full));
        assert!(
            residual <= bound,
            "the full field misses the members by {residual:e}, above {bound:e}"
        );
        // A shared field through a random initialization needs n - 1 directions: one
        // fewer misses the members.
        let truncated = principal_field(&view, &everyone, MEMBERS - 2).expect("n - 2 directions are resolved");
        let (truncated_residual, truncated_bound) =
            reproduction(&truncated, &bands(&components, &everyone, &truncated));
        assert!(
            truncated_residual > truncated_bound,
            "n - 2 directions must miss random members: residual {truncated_residual:e}, bound \
             {truncated_bound:e}"
        );
        // n members span n - 1 directions: d = n is refused.
        assert!(matches!(
            principal_field(&view, &everyone, MEMBERS),
            Err(FamilyError::InvalidPartition(..))
        ));
    }

    #[test]
    fn the_projector_family_is_a_two_dimensional_affine_field_p11() {
        let instances = 16;
        let components = projector_components(instances);
        let view = views(&components);
        let everyone: Vec<usize> = (0..instances).collect();
        let field = principal_field(&view, &everyone, 2).expect("the projector family is an affine field");
        assert_eq!(
            field.resolved_dimension, 2,
            "the centered projectors span the two traceless symmetric directions: eigenvalues \
             {:?}, band {:e}",
            field.eigenvalues, field.assembly_band
        );
        assert!(matches!(
            principal_field(&view, &everyone, 3),
            Err(FamilyError::UnresolvedDimension { resolved: 2, .. })
        ));
        assert_isometric_labels(&components, &everyone, &field);
        let (residual, bound) = reproduction(&field, &bands(&components, &everyone, &field));
        assert!(
            residual <= bound,
            "the d = 2 field misses the projectors by {residual:e}, above {bound:e}"
        );
        let line = principal_field(&view, &everyone, 1).expect("d = 1 is resolved");
        let (line_residual, line_bound) = reproduction(&line, &bands(&components, &everyone, &line));
        assert!(
            line_residual > line_bound,
            "one label dimension must miss the projectors: residual {line_residual:e}, bound {line_bound:e}"
        );
    }

    #[test]
    fn principal_field_refuses_malformed_components_and_members() {
        let components = planted_components();
        let view = views(&components);
        // Positive control: a valid family of three members.
        assert!(principal_field(&view, &[0, 2, 4], 1).is_ok());
        for members in [&[][..], &[2, 0][..], &[0, 6][..], &[1, 1][..]] {
            assert!(
                matches!(principal_field(&view, members, 0), Err(FamilyError::InvalidPartition(..))),
                "members {members:?} were admitted"
            );
        }
        let mut misshapen = components.clone();
        misshapen[3] = matrix(40, ROWS, COLS + 1);
        let mut non_finite = components.clone();
        non_finite[1][[0, 0]] = f64::NAN;
        for refused in [views(&misshapen), views(&non_finite), Vec::new()] {
            assert!(
                matches!(principal_field(&refused, &[0, 1], 0), Err(FamilyError::InvalidComponents(..))),
                "malformed components were admitted"
            );
        }
    }

    #[test]
    fn partition_moves_keep_a_canonical_cover_and_refuse_malformed_families() {
        let family = |members: &[usize], dimension: usize| FamilySpec {
            members: members.to_vec(),
            dimension,
        };
        let literal = FamilyPartition::literal(4).expect("literal partition");
        assert_eq!(literal.components(), 4);
        let shared = literal.share(0, 2, 1).expect("share components 0 and 2");
        assert_eq!(
            shared.families(),
            &[family(&[0, 2], 1), family(&[1], 0), family(&[3], 0)]
        );
        assert_eq!(shared.split(0, 2, 0).expect("split component 2 back out"), literal);
        assert!(shared.with_dimension(0, 0).is_ok());
        assert!(shared.with_dimension(0, 2).is_err(), "two members span one label dimension");
        assert!(literal.share(1, 1, 0).is_err(), "a family is not shared with itself");
        assert!(literal.share(1, 4, 0).is_err(), "there is no fifth family");
        assert!(shared.split(1, 1, 0).is_err(), "a literal family has nothing to split off");
        assert!(shared.split(0, 3, 0).is_err(), "component 3 is not in family 0");

        // Positive control: families given out of order are accepted in canonical order.
        let reordered = FamilyPartition::new(3, vec![family(&[2], 0), family(&[0, 1], 1)]).expect("a cover");
        assert_eq!(reordered.families()[0].members, vec![0, 1]);
        for (components, families) in [
            (3, vec![family(&[0, 1], 0), family(&[1, 2], 0)]),
            (3, vec![family(&[0], 0), family(&[1], 0)]),
            (3, vec![family(&[1, 0], 0), family(&[2], 0)]),
            (3, vec![family(&[], 0), family(&[0, 1, 2], 0)]),
            (3, vec![family(&[0, 1, 3], 0)]),
            (3, vec![family(&[0, 1, 2], 3)]),
            (0, vec![]),
        ] {
            assert!(
                FamilyPartition::new(components, families.clone()).is_err(),
                "{components} components {families:?} were admitted"
            );
        }
    }

    fn precision() -> DeclaredPrecision {
        DeclaredPrecision::new(FRACTION_BITS).expect("precision in range")
    }

    fn declared_inputs(cols: usize) -> Vec<Array1<f64>> {
        (0..3).map(|seed| Array1::from(entries(90 + seed, cols))).collect()
    }

    /// Which masks a readout executes, as a function of the component count.
    #[derive(Clone, Copy)]
    enum MaskFamily {
        /// All-on, then every single deletion.
        AllOnAndSingleDeletions,
        /// All-on only: the one setting a program without components can execute.
        AllOn,
    }

    impl MaskFamily {
        fn masks(self, components: usize) -> Vec<Vec<f64>> {
            let all_on = vec![1.0; components];
            match self {
                Self::AllOn => vec![all_on],
                Self::AllOnAndSingleDeletions => {
                    let mut masks = vec![all_on.clone()];
                    masks.extend((0..components).map(|deleted| {
                        let mut mask = all_on.clone();
                        mask[deleted] = 0.0;
                        mask
                    }));
                    masks
                }
            }
        }
    }

    /// Output coordinates over the declared masks and inputs, each with a bound on its
    /// rounding.
    struct Executed {
        values: Vec<f64>,
        rounding: Vec<f64>,
    }

    /// `sum_c m_c P_c x` for every declared mask and input. Each instance comes with the
    /// entrywise absolute sum of the terms that formed it, and `formation` is the most
    /// rounded operations forming one tensor entry. An output coordinate sums `C cols`
    /// products `m_c P_c[o, j] x_j`, so a path has at most `formation + 2 + C cols`
    /// rounded operations, and Higham's gamma of that count times the absolute sum of the
    /// monomials bounds its rounding.
    fn execute(
        instances: &[(Array2<f64>, Array2<f64>)],
        formation: usize,
        family: MaskFamily,
        inputs: &[Array1<f64>],
    ) -> Executed {
        let (rows, cols) = instances[0].0.dim();
        let growth = accumulation_growth(formation + 2 + instances.len() * cols);
        let mut values = Vec::new();
        let mut rounding = Vec::new();
        for mask in family.masks(instances.len()) {
            for input in inputs {
                for o in 0..rows {
                    let mut value = 0.0;
                    let mut absolute = 0.0;
                    for (c, (tensor, magnitude)) in instances.iter().enumerate() {
                        for j in 0..cols {
                            value += mask[c] * tensor[[o, j]] * input[j];
                            absolute += mask[c].abs() * magnitude[[o, j]] * input[j].abs();
                        }
                    }
                    values.push(value);
                    rounding.push(growth * absolute);
                }
            }
        }
        Executed { values, rounding }
    }

    /// Native components are formed exactly: each is its own magnitude.
    fn native_instances(components: &[Array2<f64>]) -> Vec<(Array2<f64>, Array2<f64>)> {
        components
            .iter()
            .map(|component| (component.clone(), component.mapv(f64::abs)))
            .collect()
    }

    /// Executes a decoded artifact. Instance `B_0 + sum_a z_a U_a` is formed with `2d`
    /// rounded operations, and its terms have magnitude `|B_0| + sum_a |z_a| |U_a|`.
    fn execute_decoded(
        decoded: &DecodedFamilies,
        family: MaskFamily,
        inputs: &[Array1<f64>],
    ) -> Result<Executed, String> {
        let mut instances: Vec<Option<(Array2<f64>, Array2<f64>)>> = vec![None; decoded.components()];
        let mut formation = 0;
        for decoded_family in decoded.families() {
            let dimension = decoded_family.labels.ncols();
            formation = formation.max(2 * dimension);
            for (position, &member) in decoded_family.members.iter().enumerate() {
                let mut magnitude = decoded_family.coefficients[0].mapv(f64::abs);
                for a in 0..dimension {
                    magnitude.scaled_add(
                        decoded_family.labels[[position, a]].abs(),
                        &decoded_family.coefficients[a + 1].mapv(f64::abs),
                    );
                }
                instances[member] = Some((decoded.instance(member)?, magnitude));
            }
        }
        let instances = instances
            .into_iter()
            .enumerate()
            .map(|(component, slot)| slot.ok_or_else(|| format!("component {component} was not decoded")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(execute(&instances, formation, family, inputs))
    }

    type Status = EvidenceStatus<usize, &'static str>;

    /// The largest absolute output difference, exhaustive over the declared output
    /// coordinates, with a bound on its rounding (both outputs' bands and the one
    /// subtraction). The witness is the maximizing coordinate.
    fn measure(outputs: &Executed, native: &Executed) -> Result<Status, String> {
        if outputs.values.len() != native.values.len() {
            return Err(format!(
                "{} outputs against {} native outputs",
                outputs.values.len(),
                native.values.len()
            ));
        }
        let mut distortion = 0.0_f64;
        let mut roundoff = 0.0_f64;
        let mut witness = 0;
        for (index, (((&value, &band), &reference), &reference_band)) in outputs
            .values
            .iter()
            .zip(&outputs.rounding)
            .zip(&native.values)
            .zip(&native.rounding)
            .enumerate()
        {
            let gap = (value - reference).abs();
            if gap > distortion {
                distortion = gap;
                witness = index;
            }
            roundoff = roundoff.max(band + reference_band + UNIT_ROUNDOFF * (gap + band + reference_band));
        }
        EvidenceStatus::exact(
            distortion,
            roundoff,
            ExactBasis::Exhaustive {
                cardinality: outputs.values.len() as u64,
            },
            Some(witness),
            "declared masks x inputs x output coordinates",
        )
        .map_err(|error| error.to_string())
    }

    /// An artifact's code length with its decoded fidelity.
    type Score = (u64, DecodedFidelity<usize, &'static str>);

    /// The score of `artifact` over the declared masks and inputs.
    fn score(
        artifact: &FamilyArtifact,
        family: MaskFamily,
        inputs: &[Array1<f64>],
        native: &Executed,
    ) -> Score {
        score_decoded_families(
            artifact,
            |decoded: &DecodedFamilies| execute_decoded(decoded, family, inputs),
            native,
            measure,
            TOLERANCE,
        )
        .expect("the decoded artifact executes")
    }

    /// A score as the pair `code_saving_at_proven_fidelity` reads.
    fn pair(score: &Score) -> (u64, &DecodedFidelity<usize, &'static str>) {
        (score.0, &score.1)
    }

    fn one_family(components: usize, dimension: usize) -> FamilyPartition {
        FamilyPartition::new(
            components,
            vec![FamilySpec {
                members: (0..components).collect(),
                dimension,
            }],
        )
        .expect("one family of every component")
    }

    /// Encodes the literals and one family at `dimension`, and scores both over all-on
    /// plus every single deletion.
    fn literal_and_family_scores(
        components: &[Array2<f64>],
        dimension: usize,
        inputs: &[Array1<f64>],
    ) -> (FamilyArtifact, Score, FamilyArtifact, Score) {
        let view = views(components);
        let native = execute(
            &native_instances(components),
            0,
            MaskFamily::AllOnAndSingleDeletions,
            inputs,
        );
        let literal = encode_families(
            &view,
            &FamilyPartition::literal(components.len()).expect("literal partition"),
            precision(),
        )
        .expect("the literals encode");
        let family = encode_families(&view, &one_family(components.len(), dimension), precision())
            .expect("the family encodes");
        let literal_score = score(&literal, MaskFamily::AllOnAndSingleDeletions, inputs, &native);
        let family_score = score(&family, MaskFamily::AllOnAndSingleDeletions, inputs, &native);
        (literal, literal_score, family, family_score)
    }

    #[test]
    fn a_planted_field_saves_code_and_random_init_components_lose_on_code_at_equal_fidelity() {
        let inputs = declared_inputs(COLS);
        // Positive control: the planted field is shorter than its literals at the declared
        // fidelity.
        let planted = planted_components();
        let (literal, literal_score, shared, shared_score) = literal_and_family_scores(&planted, 2, &inputs);
        let saving = code_saving_at_proven_fidelity(pair(&literal_score), pair(&shared_score));
        assert!(
            matches!(saving, Ok(bits) if bits > 0 && bits == i128::from(literal.code_bits()) - i128::from(shared.code_bits())),
            "the planted field ({} bits) must save code against its literals ({} bits), got {saving:?}",
            shared.code_bits(),
            literal.code_bits()
        );
        // One label dimension fewer is shorter still but misses the tolerance, so the
        // comparison is refused.
        let (reduced_literal, reduced_literal_score, reduced, reduced_score) =
            literal_and_family_scores(&planted, 1, &inputs);
        assert!(reduced.code_bits() < shared.code_bits());
        assert_eq!(reduced_literal, literal, "the literals do not depend on the family");
        assert_eq!(
            reduced_score.1.verdict(),
            FidelityVerdict::Violates,
            "a dropped label dimension must be proven to miss the tolerance: {reduced_score:?}"
        );
        // A re-execution of the native reference scores the reduced family the same.
        let planted_native = execute(
            &native_instances(&planted),
            0,
            MaskFamily::AllOnAndSingleDeletions,
            &inputs,
        );
        let rescored = score(&reduced, MaskFamily::AllOnAndSingleDeletions, &inputs, &planted_native);
        assert_eq!(rescored, reduced_score);
        assert!(code_saving_at_proven_fidelity(pair(&reduced_literal_score), pair(&reduced_score)).is_err());

        // Random-init control: the full-dimension family meets the tolerance, so both are
        // at equal fidelity, and it loses on code.
        let random = random_components();
        let (random_literal, random_literal_score, full, full_score) =
            literal_and_family_scores(&random, MEMBERS - 1, &inputs);
        let lost = code_saving_at_proven_fidelity(pair(&random_literal_score), pair(&full_score));
        assert!(
            matches!(lost, Ok(bits) if bits < 0),
            "a field through random components ({} bits) must lose to the literals ({} bits), got {lost:?}",
            full.code_bits(),
            random_literal.code_bits()
        );
        // Every shorter family through random components misses the tolerance.
        let (truncated_literal, truncated_literal_score, truncated, truncated_score) =
            literal_and_family_scores(&random, MEMBERS - 2, &inputs);
        assert_eq!(truncated_literal, random_literal);
        assert!(truncated.code_bits() < full.code_bits());
        assert!(
            code_saving_at_proven_fidelity(pair(&truncated_literal_score), pair(&truncated_score)).is_err(),
            "a truncated field through random components must miss the tolerance: {truncated_score:?}"
        );
    }

    #[test]
    fn the_projector_family_saves_code_over_its_literals_but_identity_is_shorter_p11() {
        let instances = 16;
        let inputs = declared_inputs(2);
        let components = projector_components(instances);
        let (literal, literal_score, shared, shared_score) = literal_and_family_scores(&components, 2, &inputs);
        let saving = code_saving_at_proven_fidelity(pair(&literal_score), pair(&shared_score));
        assert!(
            matches!(saving, Ok(bits) if bits > 0),
            "the smooth family ({} bits) must save code against its literals ({} bits), got {saving:?}",
            shared.code_bits(),
            literal.code_bits()
        );
        // The identity program: the native tensor sum_c P_c = I_2 as one literal
        // component, compared with the family at the one setting both execute.
        let identity = vec![Array2::<f64>::eye(2)];
        let identity_artifact = encode_families(
            &views(&identity),
            &FamilyPartition::literal(1).expect("one component"),
            precision(),
        )
        .expect("the identity encodes");
        let native_all_on = execute(&native_instances(&identity), 0, MaskFamily::AllOn, &inputs);
        let shared_all_on = score(&shared, MaskFamily::AllOn, &inputs, &native_all_on);
        let identity_score = score(&identity_artifact, MaskFamily::AllOn, &inputs, &native_all_on);
        assert_eq!(
            (shared_all_on.1.verdict(), identity_score.1.verdict()),
            (FidelityVerdict::Meets, FidelityVerdict::Meets),
            "both programs are proven within the tolerance at all-on"
        );
        // Guard: the score reads only the exact largest distortion. The same figure
        // restated as a uniform bound is refused; the exact status above is the positive
        // control.
        let restated = score_decoded_families(
            &identity_artifact,
            |decoded: &DecodedFamilies| execute_decoded(decoded, MaskFamily::AllOn, &inputs),
            &native_all_on,
            |outputs: &Executed, native: &Executed| -> Result<Status, String> {
                let exact = measure(outputs, native)?;
                let upper = exact
                    .upper_bound()
                    .ok_or_else(|| "an exact figure has an upper bound".to_string())?;
                EvidenceStatus::uniform_bound(upper, 0.0, "declared masks x inputs x output coordinates")
                    .map_err(|error| error.to_string())
            },
            TOLERANCE,
        );
        assert!(restated.is_err(), "a uniform bound must not stand in for the largest distortion");
        let identity_saving = code_saving_at_proven_fidelity(pair(&shared_all_on), pair(&identity_score));
        assert!(
            matches!(identity_saving, Ok(bits) if bits > 0),
            "identity ({} bits) must be the short program against the family ({} bits), got {identity_saving:?}",
            identity_artifact.code_bits(),
            shared.code_bits()
        );
        // Negative control: from the identity start, the family is longer at equal fidelity.
        assert!(matches!(
            code_saving_at_proven_fidelity(pair(&identity_score), pair(&shared_all_on)),
            Ok(bits) if bits < 0
        ));
    }

    #[test]
    fn a_family_message_reads_back_at_its_exact_length_and_refuses_what_it_cannot_decode() {
        // Components 0, 2 and 4 lie on a one-dimensional affine field; 1 and 3 are literals.
        let (rows, cols) = (3, 4);
        let center = matrix(20, rows, cols);
        let axis = matrix(21, rows, cols);
        let components: Vec<Array2<f64>> = (0..5)
            .map(|c| {
                if c % 2 == 0 {
                    &center + &(&axis * (0.3 * c as f64 - 0.5))
                } else {
                    matrix(22 + c, rows, cols)
                }
            })
            .collect();
        let view = views(&components);
        let family = |members: &[usize], dimension: usize| FamilySpec {
            members: members.to_vec(),
            dimension,
        };
        let partition = FamilyPartition::new(5, vec![family(&[3], 0), family(&[0, 2, 4], 1), family(&[1], 0)])
            .expect("a partition");
        let artifact = encode_families(&view, &partition, precision()).expect("the partition encodes");
        let decoded = artifact.decode().expect("the message decodes");
        assert_eq!(decoded.partition().expect("a valid decoded partition"), partition);
        assert_eq!(decoded.shape(), (rows, cols));
        assert_eq!(decoded.precision(), precision());

        // The length is the sum of the codewords the module docs name, and every instance
        // is its lattice-decoded field value.
        let mut expected_bits = prefix_integer_len_bits(6).expect("count codeword");
        let mut unassigned = 5_usize;
        for spec in partition.families() {
            let field = principal_field(&view, &spec.members, spec.dimension).expect("the family's field");
            expected_bits += subset_code_len_bits(unassigned - 1, spec.members.len() - 1).expect("subset codeword")
                + prefix_integer_len_bits(spec.dimension as u64 + 1).expect("dimension codeword");
            let mut coefficients: Vec<f64> = field.center.iter().copied().collect();
            for direction in &field.directions {
                coefficients.extend(direction.iter());
            }
            let coefficient_code = LatticeCode::encode(&coefficients, precision()).expect("coefficients encode");
            let mut coefficient_message = BitString::new();
            coefficient_code.write(&mut coefficient_message).expect("coefficients write");
            expected_bits += coefficient_message.len_bits();
            let blocks: Vec<Array2<f64>> = coefficient_code
                .decode()
                .expect("coefficients decode")
                .chunks(rows * cols)
                .map(|chunk| Array2::from_shape_vec((rows, cols), chunk.to_vec()).expect("block shape"))
                .collect();
            let labels = if spec.dimension == 0 {
                Array2::<f64>::zeros((spec.members.len(), 0))
            } else {
                let values: Vec<f64> = field.labels.iter().copied().collect();
                let label_code = LatticeCode::encode(&values, precision()).expect("labels encode");
                let mut label_message = BitString::new();
                label_code.write(&mut label_message).expect("labels write");
                expected_bits += label_message.len_bits();
                Array2::from_shape_vec(
                    (spec.members.len(), spec.dimension),
                    label_code.decode().expect("labels decode"),
                )
                .expect("label shape")
            };
            for (position, &member) in spec.members.iter().enumerate() {
                let mut rebuilt = blocks[0].clone();
                for a in 0..spec.dimension {
                    rebuilt.scaled_add(labels[[position, a]], &blocks[a + 1]);
                }
                assert_eq!(
                    decoded.instance(member).expect("a decoded component"),
                    rebuilt,
                    "component {member} decodes to its lattice-decoded field value"
                );
            }
            unassigned -= spec.members.len();
        }
        assert_eq!(artifact.code_bits(), expected_bits);
        assert!(decoded.instance(5).is_err(), "there is no sixth component");

        // A trailing bit and a missing bit are refused.
        let mut trailing = artifact.message().clone();
        trailing.push_bit(false);
        assert!(FamilyArtifact::from_message(rows, cols, trailing).expect("shape").decode().is_err());
        let mut short = BitString::new();
        let mut reader = artifact.message().reader();
        for _ in 1..artifact.code_bits() {
            short.push_bit(reader.read_bit().expect("inside the message"));
        }
        assert!(FamilyArtifact::from_message(rows, cols, short).expect("shape").decode().is_err());

        // One component spans no label direction: dimension 1 is refused, while dimension
        // 0 with its twelve reals reads back.
        let lone = |dimension: u64| {
            let mut message = BitString::new();
            encode_prefix_integer(&mut message, 2).expect("count codeword");
            encode_subset(&mut message, 0, &[]).expect("empty subset codeword");
            encode_prefix_integer(&mut message, dimension + 1).expect("dimension codeword");
            LatticeCode::encode(&[0.25; 12], precision())
                .expect("reals encode")
                .write(&mut message)
                .expect("reals write");
            FamilyArtifact::from_message(rows, cols, message).expect("shape").decode()
        };
        assert!(lone(0).is_ok(), "positive control: a literal component reads back");
        assert!(lone(1).is_err(), "a lone component announcing a label dimension must be refused");

        // A header announcing 2^62 components is refused before anything is allocated.
        let mut huge = BitString::new();
        encode_prefix_integer(&mut huge, (1_u64 << 62) + 1).expect("count codeword");
        let refusal = FamilyArtifact::from_message(rows, cols, huge)
            .expect("shape")
            .decode()
            .expect_err("2^62 components exceed the host");
        assert!(refusal.contains("budget"), "{refusal}");
    }

    /// `field`'s instance `sum_j s_j B'_j` of `member`, read through the anchor at the mask
    /// selecting that member with the residual removed:
    /// `s = sum_k (m_k - m_Delta) w_k phi(z_k)`.
    fn field_instance(family: &ParameterFamily, member: usize, shape: (usize, usize)) -> Array2<f64> {
        let mut mask = Array1::<f64>::zeros(family.labels().nrows());
        mask[member] = 1.0;
        let weights = family
            .anchor_basis_weights(mask.view(), 0.0)
            .expect("anchor weights at a member mask");
        let mut value = Array2::<f64>::zeros(shape);
        for (j, coefficient) in family.field().coefficients().iter().enumerate() {
            assert!(
                matches!(coefficient, FieldCoefficient::Dense(..)),
                "coefficient {j} of an affine family is dense"
            );
            if let FieldCoefficient::Dense(b) = coefficient {
                value.scaled_add(weights[j], b);
            }
        }
        value
    }

    #[test]
    fn principal_and_decoded_families_are_field_instances_over_the_degree_one_patch_basis() {
        let components = planted_components();
        let view = views(&components);
        let everyone: Vec<usize> = (0..MEMBERS).collect();
        let principal = principal_field(&view, &everyone, 2).expect("the planted field is resolved");
        let dimension = principal.directions.len();
        let family = principal.parameter_family().expect("the planted field converts");
        assert_eq!(family.field().coefficients().len(), dimension + 1);
        // Both sides form `B_0 + sum_a z_a U_a`. The field side scales by n and 1/n, forms
        // `phi_j w`, the product with `B'_j` and the d + 1 accumulations, at most d + 5
        // operations. The reference takes at most 2d + 1. Each rounds against
        // `|B_0| + sum_a |z_a| |U_a|` to first order.
        let growth = accumulation_growth(2 * dimension + 6);
        let mut exchanged = principal.clone();
        for a in 0..dimension {
            exchanged.labels[[0, a]] = principal.labels[[1, a]];
            exchanged.labels[[1, a]] = principal.labels[[0, a]];
        }
        let exchanged_family = exchanged.parameter_family().expect("exchanged labels convert");
        let mut exchanged_refuted = false;
        for member in 0..MEMBERS {
            let mut reference = principal.center.clone();
            let mut magnitude = principal.center.mapv(f64::abs);
            for a in 0..dimension {
                reference.scaled_add(principal.labels[[member, a]], &principal.directions[a]);
                magnitude.scaled_add(
                    principal.labels[[member, a]].abs(),
                    &principal.directions[a].mapv(f64::abs),
                );
            }
            let value = field_instance(&family, member, (ROWS, COLS));
            let moved = field_instance(&exchanged_family, member, (ROWS, COLS));
            for ((index, &field_entry), &reference_entry) in value.indexed_iter().zip(reference.iter()) {
                let band = 2.0 * growth * magnitude[index];
                assert!(
                    (field_entry - reference_entry).abs() <= band,
                    "member {member} entry {index:?}: field instance {field_entry} against \
                     {reference_entry} (band {band:e})"
                );
                exchanged_refuted |= (moved[index] - reference_entry).abs() > band;
            }
        }
        assert!(exchanged_refuted, "negative control: exchanged labels must move some instance");

        // The decoded family is field instances too, at the reals the message carries.
        let partition = FamilyPartition::new(MEMBERS, vec![FamilySpec { members: everyone.clone(), dimension }])
            .expect("one family");
        let decoded = encode_families(&view, &partition, precision())
            .expect("the family encodes")
            .decode()
            .expect("the message decodes");
        let decoded_family = decoded.families()[0].parameter_family().expect("the decoded family converts");
        for member in 0..MEMBERS {
            let rebuilt = decoded.instance(member).expect("a decoded member");
            let mut magnitude = decoded.families()[0].coefficients[0].mapv(f64::abs);
            for a in 0..dimension {
                magnitude.scaled_add(
                    decoded.families()[0].labels[[member, a]].abs(),
                    &decoded.families()[0].coefficients[a + 1].mapv(f64::abs),
                );
            }
            let value = field_instance(&decoded_family, member, (ROWS, COLS));
            for ((index, &field_entry), &rebuilt_entry) in value.indexed_iter().zip(rebuilt.iter()) {
                assert!(
                    (field_entry - rebuilt_entry).abs() <= 2.0 * growth * magnitude[index],
                    "decoded member {member} entry {index:?}: field instance {field_entry} against \
                     {rebuilt_entry}"
                );
            }
        }

        // Guard: a literal family is a native tensor, not a field (positive control above).
        let literal = principal_field(&view, &[3], 0).expect("a literal family");
        assert!(matches!(literal.parameter_family(), Err(FamilyError::Field(..))));
        let literal_decoded = encode_families(
            &view,
            &FamilyPartition::literal(MEMBERS).expect("literal partition"),
            precision(),
        )
        .expect("the literals encode")
        .decode()
        .expect("the literals decode");
        assert!(matches!(
            literal_decoded.families()[0].parameter_family(),
            Err(FamilyError::Field(..))
        ));
    }

    /// The score and the fidelity status its measure proved over the declared family: the
    /// status `fit`'s proposal rule reads.
    fn score_and_status(
        artifact: &FamilyArtifact,
        family: MaskFamily,
        inputs: &[Array1<f64>],
        native: &Executed,
    ) -> (Score, Status) {
        let scored = score(artifact, family, inputs, native);
        let status = scored.1.status().clone();
        (scored, status)
    }

    fn encode_one_family(components: &[Array2<f64>], dimension: usize) -> FamilyArtifact {
        encode_families(&views(components), &one_family(components.len(), dimension), precision())
            .expect("the family encodes")
    }

    fn encode_literals(components: &[Array2<f64>]) -> FamilyArtifact {
        encode_families(
            &views(components),
            &FamilyPartition::literal(components.len()).expect("literal partition"),
            precision(),
        )
        .expect("the literals encode")
    }

    #[test]
    fn family_proposals_are_decided_by_fits_rule_on_decoded_code_at_declared_fidelity() {
        let inputs = declared_inputs(COLS);
        let deletions = MaskFamily::AllOnAndSingleDeletions;

        // Planted field: Share at d = 2 is accepted and certified, with the saving equal
        // to the code-length difference.
        let planted = planted_components();
        let planted_native = execute(&native_instances(&planted), 0, deletions, &inputs);
        let literal = encode_literals(&planted);
        let shared = encode_one_family(&planted, 2);
        let (literal_score, literal_status) = score_and_status(&literal, deletions, &inputs, &planted_native);
        let (shared_score, shared_status) = score_and_status(&shared, deletions, &inputs, &planted_native);
        let expected_saving = i128::from(literal.code_bits()) - i128::from(shared.code_bits());
        let accepted = decide_proposal(ProposalKind::Share, pair(&literal_score), pair(&shared_score), shared_status);
        assert!(
            matches!(
                &accepted,
                Ok(ProposalAcceptance { kind: ProposalKind::Share, saving_bits, fidelity_certified: true, .. })
                    if *saving_bits == expected_saving && *saving_bits > 0
            ),
            "sharing the planted field must be accepted, got {accepted:?}"
        );
        // Reduce d from 2 to 1: shorter, but its exhaustive status refutes the tolerance.
        let reduced = encode_one_family(&planted, 1);
        let (reduced_score, reduced_status) = score_and_status(&reduced, deletions, &inputs, &planted_native);
        assert!(reduced.code_bits() < shared.code_bits());
        assert_eq!(reduced_score.1.verdict(), FidelityVerdict::Violates);
        let refuted = decide_proposal(ProposalKind::Reduce, pair(&shared_score), pair(&reduced_score), reduced_status);
        assert!(
            matches!(refuted, Err(ProposalRejection::FidelityRefuted(..))),
            "a dropped label dimension must be refuted, got {refuted:?}"
        );
        // Split back into the literals: they are proven within the tolerance and longer, so
        // the verdict is on code.
        let split = decide_proposal(ProposalKind::Split, pair(&shared_score), pair(&literal_score), literal_status);
        assert!(
            matches!(split, Err(ProposalRejection::NoShorterCode { saving_bits }) if saving_bits < 0),
            "splitting the planted field into literals must lose on code, got {split:?}"
        );

        // Random-init control: at equal (proven) fidelity the full-dimension share loses on code.
        let random = random_components();
        let random_native = execute(&native_instances(&random), 0, deletions, &inputs);
        let random_literal = encode_literals(&random);
        let full = encode_one_family(&random, MEMBERS - 1);
        let (random_literal_score, random_literal_status) =
            score_and_status(&random_literal, deletions, &inputs, &random_native);
        let (full_score, full_status) = score_and_status(&full, deletions, &inputs, &random_native);
        assert!(
            random_literal_status.certifies_at_most(TOLERANCE) && full_status.certifies_at_most(TOLERANCE),
            "both artifacts are proven within the tolerance: equal fidelity"
        );
        let lost = decide_proposal(ProposalKind::Share, pair(&random_literal_score), pair(&full_score), full_status);
        assert!(
            matches!(lost, Err(ProposalRejection::NoShorterCode { saving_bits }) if saving_bits < 0),
            "a field through random components must lose on code at equal fidelity, got {lost:?}"
        );

        // P11: the smooth family shares, Reduce to identity at all-on is accepted, and Expose
        // from identity loses on code.
        let instances = 16;
        let projectors = projector_components(instances);
        let projector_inputs = declared_inputs(2);
        let projector_native = execute(&native_instances(&projectors), 0, deletions, &projector_inputs);
        let projector_literal = encode_literals(&projectors);
        let projector_family = encode_one_family(&projectors, 2);
        let projector_literal_score =
            score_and_status(&projector_literal, deletions, &projector_inputs, &projector_native).0;
        let (projector_family_score, projector_family_status) =
            score_and_status(&projector_family, deletions, &projector_inputs, &projector_native);
        let projector_share = decide_proposal(
            ProposalKind::Share,
            pair(&projector_literal_score),
            pair(&projector_family_score),
            projector_family_status,
        );
        assert!(
            matches!(&projector_share, Ok(acceptance) if acceptance.saving_bits > 0),
            "the smooth family must beat its literals, got {projector_share:?}"
        );
        let identity = vec![Array2::<f64>::eye(2)];
        let identity_artifact = encode_literals(&identity);
        let native_all_on = execute(&native_instances(&identity), 0, MaskFamily::AllOn, &projector_inputs);
        let (family_all_on, family_all_on_status) =
            score_and_status(&projector_family, MaskFamily::AllOn, &projector_inputs, &native_all_on);
        let (identity_score, identity_status) =
            score_and_status(&identity_artifact, MaskFamily::AllOn, &projector_inputs, &native_all_on);
        let reduce_to_identity =
            decide_proposal(ProposalKind::Reduce, pair(&family_all_on), pair(&identity_score), identity_status);
        assert!(
            matches!(
                &reduce_to_identity,
                Ok(acceptance) if acceptance.saving_bits > 0 && acceptance.fidelity_certified
            ),
            "identity must be the short program, got {reduce_to_identity:?}"
        );
        let expose =
            decide_proposal(ProposalKind::Expose, pair(&identity_score), pair(&family_all_on), family_all_on_status);
        assert!(
            matches!(expose, Err(ProposalRejection::NoShorterCode { saving_bits }) if saving_bits < 0),
            "exposing the smooth family from identity must lose on code, got {expose:?}"
        );
    }
}
