//! Spectral recovery of rotation planes from a frozen matrix, and of the isolating
//! parameter blocks of commuting response projectors (#2951 P3).
//!
//! # The result
//!
//! A rotation acting in orthogonal planes is
//!
//! ```text
//! W = I + sum_k U_k (R_{alpha_k} - I) U_k^T,     U_k^T U_l = delta_kl I_2.
//! ```
//!
//! For any orthogonal `W` the symmetric part `S = (W + W^T)/2` has the plane
//! projectors `P_k = U_k U_k^T` as eigenspaces with eigenvalue `cos alpha_k`, and
//! the skew part `K = (W - W^T)/2` satisfies `P_k K P_k = sin alpha_k J_k` with
//! `J_k^2 = -P_k`. So
//!
//! ```text
//! W = I + sum_k [(cos alpha_k - 1) P_k + sin alpha_k J_k]
//! ```
//!
//! is read off one symmetric eigendecomposition and one compression of `K` per
//! eigenspace. Eigenvalue `+1` is the fixed space and `-1` the half-turn space;
//! every other eigenvalue has even multiplicity.
//!
//! # What the matrix does not determine (reported, never guessed)
//!
//! * **Repeated cosines.** An eigenspace of dimension `2p`, `p >= 2`, carries its
//!   projector and its complex structure `J`, but every `J`-invariant split into
//!   planes is equally valid, so the individual planes are not identified.
//! * **Half-turn.** At `cos alpha = -1`, `sin alpha = 0` and `J` is undefined: a
//!   2-dimensional `-1` space is a plane with no orientation. An odd-dimensional
//!   `-1` space contains a reflection.
//! * **Identity.** No plane at all. Only rotations below a derived angle can hide.
//! * **Winding.** `alpha` and `alpha + 2 pi k` give the same matrix, as do
//!   `(alpha, J)` and `(-alpha, -J)`. The reported angle is the representative in
//!   `(0, pi)` with the orientation carried by `J`; choosing the shortest path is
//!   a convention.
//!
//! # Orthogonality is declared and checked, never approximated
//!
//! The caller declares `delta`, a bound on `||W - O||_2` for the orthogonal matrix `O`
//! that `W` represents (the precision of its construction or export); there is no
//! default. The distance from `W` to the orthogonal group is
//! `rho = max_i |sigma_i(W) - 1|`, so a computed `rho` beyond `delta` plus the
//! singular-value band proves that no such `O` exists, and the recovery refuses. A
//! non-orthogonal operator such as `I + U (R - I) U^+` with non-orthonormal `U` keeps
//! its planes in its real Schur form, not in `(W + W^T)/2`, and is not approximated.
//!
//! # Grouping is derived, not a constant
//!
//! Every claim holds for every orthogonal `O` within `delta` of `W`. `S(W)` differs from
//! `S(O)` by at most `delta`; adding the rounding of forming `S` and the eigensolver's
//! backward error gives `beta >= ||S_computed - S(O)||_2`, and by Weyl each computed
//! eigenvalue is within `beta` of its true one. Neighbouring computed eigenvalues
//! further apart than `2 beta` belong to provably distinct true eigenvalues; nearer
//! ones are not resolved and stay in one cluster. A cluster's projector carries the
//! Davis–Kahan bar `beta / (gap - beta)` of [`projector_error_bar`], `gap` being its
//! measured separation from the rest of the spectrum.
//!
//! The route is dense: `O(d^3)` time and `d x d` workspace for a `d x d` matrix.
//!
//! # Commuting response projectors
//!
//! A response `e` reads the parameter through `A_e` (`r_e x p`). When every
//! `H_e = A_e^T A_e` is an orthogonal projector and the `H_e` commute, they generate
//! a Boolean algebra whose atoms
//!
//! ```text
//! Pi_T = prod_{e in T} H_e prod_{f not in T} (I - H_f)
//! ```
//!
//! are orthogonal projectors onto the directions seen by exactly the responses in
//! `T`, summing to `I` over all subsets. A decomposition `theta* = sum_T v_T` is
//! *isolating* when `H_e v_T = v_T` for `e in T` and `H_f v_T = 0` for `f not in T`.
//! Then `v_T` lies in `range(Pi_T)`, the atoms are mutually orthogonal, and
//! `Pi_T theta* = v_T`: the isolating decomposition is unique, and response `e`'s
//! component is `P_e = H_e theta* = sum_{T containing e} v_T`.
//!
//! The atoms are built by refinement without forming a `p x p` matrix. Each
//! response splits every existing block by the singular values of its compression
//! (`0`: unseen, `1`: seen) and adds the part of its row space outside every block
//! as a new block. The caller declares `delta`, a bound on each `A_e`'s distance to
//! the partial isometry it represents; a computed distance
//! `max_i min(sigma_i, |1 - sigma_i|)` beyond `delta` plus the band refuses. A
//! compressed singular value within the derived band of neither `0` nor `1` is a
//! counterexample to commutation. A successful refinement is certified a posteriori:
//! the polar factor of the block bases defines exactly commuting projectors `P'_e`,
//! and `||A_e^T A_e - P'_e||_2` is bounded from measured quantities.

use faer::Side;
use gam_linalg::decision::projector_error_bar;
use gam_linalg::faer_ndarray::{FaerLinalgError, FaerSvd, strict_symmetric_eigh};
use gam_linalg::roundoff::{
    accumulation_growth, factor_singular_band, symmetric_spectrum_rounding_band,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use std::f64::consts::SQRT_2;

/// Why a matrix admits no plane-rotation recovery.
#[derive(Debug)]
pub enum PlaneRotationError {
    /// The matrix is empty or not square.
    NotSquare { rows: usize, cols: usize },
    /// An entry is not finite.
    NonFinite { row: usize, col: usize },
    /// The declared error is negative or not finite.
    InvalidDeclaredError { declared_error: f64 },
    /// Counterexample: the matrix is `distance` from the orthogonal group, beyond the
    /// declared error plus the singular-value band, so no orthogonal matrix lies within
    /// the declared error of it.
    NotOrthogonal { distance: f64, declared_error: f64 },
    /// The singular value or symmetric eigendecomposition failed.
    Linalg(FaerLinalgError),
    /// A cluster whose cosine interval excludes both `+1` and `-1` has odd
    /// dimension. The eigenvalues of `S(O)` inside `(-1, 1)` come in pairs, and a
    /// valid `beta` keeps every cluster a union of whole true eigenvalue groups, so
    /// an odd count means the derived bound was violated and no claim stands.
    OddInteriorCluster { cluster: usize, dimension: usize },
}

impl std::fmt::Display for PlaneRotationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSquare { rows, cols } => write!(
                formatter,
                "plane-rotation recovery needs a non-empty square matrix, got {rows}x{cols}"
            ),
            Self::NonFinite { row, col } => write!(
                formatter,
                "plane-rotation recovery needs finite entries; entry ({row}, {col}) is not finite"
            ),
            Self::InvalidDeclaredError { declared_error } => write!(
                formatter,
                "plane-rotation recovery needs a finite non-negative declared error, got \
                 {declared_error}"
            ),
            Self::NotOrthogonal {
                distance,
                declared_error,
            } => write!(
                formatter,
                "plane-rotation recovery refused: the matrix is {distance:.3e} from the \
                 orthogonal group, beyond the declared error {declared_error:.3e}"
            ),
            Self::Linalg(error) => {
                write!(formatter, "plane-rotation recovery decomposition failed: {error}")
            }
            Self::OddInteriorCluster { cluster, dimension } => write!(
                formatter,
                "plane-rotation recovery: interior cluster {cluster} has odd dimension \
                 {dimension}, so the derived perturbation bound was violated"
            ),
        }
    }
}

impl std::error::Error for PlaneRotationError {}

/// What one cluster of the symmetric part's spectrum is.
#[derive(Clone, Debug)]
pub enum RotationClusterKind {
    /// The cosine interval admits `+1`: directions the rotation fixes. A rotation
    /// by at most `max_hidden_angle` would be indistinguishable from fixing them.
    Fixed { max_hidden_angle: f64 },
    /// The cosine interval excludes `+1` and `-1`: `planes` planes whose cosines
    /// are not resolved from one another. `planes > 1` is the repeated-cosine case.
    Rotation {
        planes: usize,
        /// `atan2(s, c)`, with `c` the cluster's mean computed cosine and `s` the
        /// root-mean-square singular value of the compressed skew part.
        angle: f64,
        /// `J` in the cluster basis (`m x m` with `m = 2 planes`, skew, `J^2 = -I`),
        /// or `None` when the orientation is not certified (see
        /// [`recover_plane_rotations`]).
        complex_structure: Option<Array2<f64>>,
    },
    /// The cosine interval admits `-1`: half-turn planes without orientation, and a
    /// reflection when the dimension is odd. A rotation by at least
    /// `min_hidden_angle` would be indistinguishable from a half-turn.
    HalfTurn { min_hidden_angle: f64 },
    /// The cosine interval admits both `+1` and `-1`: no structure is resolved.
    Unresolved,
}

/// One cluster of the symmetric part's spectrum, provably separated from the rest.
#[derive(Clone, Debug)]
pub struct RotationCluster {
    /// Orthonormal basis of the cluster's computed eigenspace (`d x m`).
    pub basis: Array2<f64>,
    /// Interval containing every true cosine of the cluster:
    /// `[lowest computed - beta, highest computed + beta]`.
    pub cosine_interval: (f64, f64),
    /// Measured distance from the cluster's computed eigenvalues to the nearest
    /// other computed eigenvalue; infinite when the cluster is the whole spectrum.
    pub separation: f64,
    /// Davis–Kahan bound on the 2-norm distance between the projector onto `basis`
    /// and the true spectral projector of `S(O)`. Zero for the whole spectrum, whose
    /// projector is the identity.
    pub projector_bar: f64,
    pub kind: RotationClusterKind,
}

/// A structural fact about the recovered rotation that the matrix does not decide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RotationAmbiguity {
    /// Every cluster is fixed: no plane at all.
    Identity,
    /// Cluster `cluster` holds `planes >= 2` planes with unresolved cosines.
    RepeatedCosine { cluster: usize, planes: usize },
    /// Cluster `cluster` admits cosine `-1`: no orientation.
    HalfTurn { cluster: usize },
    /// Cluster `cluster` admits both `+1` and `-1`.
    Unresolved { cluster: usize },
    /// Angles are fixed only modulo `2 pi`; the reported representative is a
    /// convention.
    Winding,
}

/// Plane-rotation structure of a frozen square matrix, valid for every orthogonal `O`
/// within the declared error of it.
#[derive(Clone, Debug)]
pub struct PlaneRotationRecovery {
    /// Certified bound on the 2-norm distance from the matrix to the orthogonal
    /// group, `rho`; at most the declared error plus the singular-value band.
    pub orthogonality_defect: f64,
    /// `beta`, the bound on the distance between the decomposed symmetric part and
    /// `S(O)`.
    pub perturbation_bound: f64,
    /// Clusters in increasing cosine order.
    pub clusters: Vec<RotationCluster>,
}

impl PlaneRotationRecovery {
    /// The ambiguities the clusters carry, in cluster order, with
    /// [`RotationAmbiguity::Winding`] last whenever any rotation or half-turn exists.
    pub fn ambiguities(&self) -> Vec<RotationAmbiguity> {
        let mut ambiguities = Vec::new();
        if self
            .clusters
            .iter()
            .all(|cluster| matches!(cluster.kind, RotationClusterKind::Fixed { .. }))
        {
            ambiguities.push(RotationAmbiguity::Identity);
        }
        let mut winding = false;
        for (index, cluster) in self.clusters.iter().enumerate() {
            match &cluster.kind {
                RotationClusterKind::Rotation { planes, .. } => {
                    winding = true;
                    if *planes > 1 {
                        ambiguities.push(RotationAmbiguity::RepeatedCosine {
                            cluster: index,
                            planes: *planes,
                        });
                    }
                }
                RotationClusterKind::HalfTurn { .. } => {
                    winding = true;
                    ambiguities.push(RotationAmbiguity::HalfTurn { cluster: index });
                }
                RotationClusterKind::Unresolved => {
                    ambiguities.push(RotationAmbiguity::Unresolved { cluster: index });
                }
                RotationClusterKind::Fixed { .. } => {}
            }
        }
        if winding {
            ambiguities.push(RotationAmbiguity::Winding);
        }
        ambiguities
    }
}

/// Recover the rotation planes of a frozen square matrix that the caller declares to be
/// within `declared_error` (2-norm) of an orthogonal matrix.
///
/// Refuses when the matrix is provably further from the orthogonal group (module
/// documentation). Otherwise clusters the spectrum of `S = (W + W^T)/2` at the derived
/// resolution `2 beta` and classifies each cluster by whether its cosine interval admits
/// `+1` or `-1`. An interior cluster of dimension `m` compresses the skew part to
/// `G = V^T K V`.
///
/// # Orientation certificate
///
/// With `V Q` the true eigenbasis for the Procrustes-optimal `Q`,
/// `||V - V Q||_2 <= sqrt(2) bar` and `||K(O)||_2 <= 1`, so
///
/// ```text
/// ||G - Q^T V^T K(O) V Q||_2 <= ||K_computed - K(O)||_2 + 2 sqrt(2) bar + rounding.
/// ```
///
/// The true compression is `sin alpha` times a complex structure, with
/// `sin alpha >= sqrt(1 - max c^2)` over the cosine interval. The opposite
/// orientation lies `2 sin alpha` away, so when that sine floor exceeds the error
/// above, `J = G / s` carries the true orientation and is reported; otherwise it is
/// `None`.
pub fn recover_plane_rotations(
    matrix: ArrayView2<'_, f64>,
    declared_error: f64,
) -> Result<PlaneRotationRecovery, PlaneRotationError> {
    let (rows, cols) = matrix.dim();
    if rows == 0 || rows != cols {
        return Err(PlaneRotationError::NotSquare { rows, cols });
    }
    if !(declared_error.is_finite() && declared_error >= 0.0) {
        return Err(PlaneRotationError::InvalidDeclaredError { declared_error });
    }
    if let Some(((row, col), _)) = matrix.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(PlaneRotationError::NonFinite { row, col });
    }
    let dimension = rows;
    let (_, singular_values, _) = matrix
        .svd(false, false)
        .map_err(PlaneRotationError::Linalg)?;
    let sigma_max = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value));
    let singular_band = factor_singular_band(dimension, dimension, sigma_max);
    let distance = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max((value - 1.0).abs()));
    // Each true singular value is within the band of its computed one, so a computed
    // distance beyond `declared_error + band` proves the true distance exceeds it.
    if distance > declared_error + singular_band {
        return Err(PlaneRotationError::NotOrthogonal {
            distance,
            declared_error,
        });
    }
    let orthogonality_defect = distance + singular_band;

    // `a + b` and `b + a` round identically, so `symmetric` is exactly symmetric.
    let mut symmetric = Array2::<f64>::zeros((dimension, dimension));
    let mut skew = Array2::<f64>::zeros((dimension, dimension));
    for row in 0..dimension {
        for col in 0..dimension {
            symmetric[[row, col]] = 0.5 * (matrix[[row, col]] + matrix[[col, row]]);
            skew[[row, col]] = 0.5 * (matrix[[row, col]] - matrix[[col, row]]);
        }
    }
    // One rounded addition per entry, halved exactly: `u (|w_ij| + |w_ji|) / 2`,
    // whose Frobenius norm is at most `u ||W||_F`. The same bound holds for `K`.
    let formation_band = accumulation_growth(1) * frobenius_norm(matrix);
    let (values, vectors) =
        strict_symmetric_eigh(&symmetric, Side::Lower).map_err(PlaneRotationError::Linalg)?;
    let mut order: Vec<usize> = (0..dimension).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let cosines: Vec<f64> = order.iter().map(|&index| values[index]).collect();
    let perturbation_bound =
        declared_error + formation_band + symmetric_spectrum_rounding_band(&cosines);

    let resolution = 2.0 * perturbation_bound;
    let mut starts = vec![0_usize];
    for index in 1..dimension {
        if cosines[index] - cosines[index - 1] > resolution {
            starts.push(index);
        }
    }
    let mut clusters = Vec::with_capacity(starts.len());
    for (cluster_index, &start) in starts.iter().enumerate() {
        let end = starts
            .get(cluster_index + 1)
            .copied()
            .unwrap_or(dimension);
        let width = end - start;
        let lowest = cosines[start];
        let highest = cosines[end - 1];
        let below = if start > 0 {
            lowest - cosines[start - 1]
        } else {
            f64::INFINITY
        };
        let above = if end < dimension {
            cosines[end] - highest
        } else {
            f64::INFINITY
        };
        let separation = below.min(above);
        let projector_bar = if separation.is_finite() {
            projector_error_bar(separation, perturbation_bound)
        } else {
            0.0
        };
        let mut basis = Array2::<f64>::zeros((dimension, width));
        for (column, &index) in order[start..end].iter().enumerate() {
            basis.column_mut(column).assign(&vectors.column(index));
        }
        let cosine_interval = (lowest - perturbation_bound, highest + perturbation_bound);
        let admits_fixed = cosine_interval.1 >= 1.0;
        let admits_half_turn = cosine_interval.0 <= -1.0;
        // A true cosine lies in the interval and in `[-1, 1]`.
        let kind = match (admits_fixed, admits_half_turn) {
            (true, true) => RotationClusterKind::Unresolved,
            (true, false) => RotationClusterKind::Fixed {
                max_hidden_angle: cosine_interval.0.min(1.0).acos(),
            },
            (false, true) => RotationClusterKind::HalfTurn {
                min_hidden_angle: cosine_interval.1.max(-1.0).acos(),
            },
            (false, false) => {
                if width % 2 != 0 {
                    return Err(PlaneRotationError::OddInteriorCluster {
                        cluster: cluster_index,
                        dimension: width,
                    });
                }
                interior_rotation(
                    &skew,
                    &basis,
                    &cosines[start..end],
                    cosine_interval,
                    declared_error + formation_band + 2.0 * SQRT_2 * projector_bar,
                )
            }
        };
        clusters.push(RotationCluster {
            basis,
            cosine_interval,
            separation,
            projector_bar,
            kind,
        });
    }
    Ok(PlaneRotationRecovery {
        orthogonality_defect,
        perturbation_bound,
        clusters,
    })
}

/// Angle and certified orientation of an interior cluster. `skew_error` bounds
/// `||K_computed - K(O)||_2 + 2 sqrt(2) bar`; the compression's own rounding is
/// added here.
fn interior_rotation(
    skew: &Array2<f64>,
    basis: &Array2<f64>,
    cosines: &[f64],
    cosine_interval: (f64, f64),
    skew_error: f64,
) -> RotationClusterKind {
    let (dimension, width) = basis.dim();
    let compressed = basis.t().dot(&skew.dot(basis));
    // Two nested length-`d` accumulations per entry: at most `gamma_{2d}` times the
    // absolute sum of the terms, `(|V|^T |K| |V|)_ab`.
    let absolute_basis = basis.mapv(f64::abs);
    let absolute_compressed = absolute_basis
        .t()
        .dot(&skew.mapv(f64::abs).dot(&absolute_basis));
    let compression_band =
        accumulation_growth(2 * dimension) * frobenius_norm(absolute_compressed.view());
    let sine = frobenius_norm(compressed.view()) / (width as f64).sqrt();
    let mean_cosine = cosines.iter().sum::<f64>() / width as f64;
    let magnitude = cosine_interval.0.abs().max(cosine_interval.1.abs());
    let sine_floor = (1.0 - magnitude * magnitude).sqrt();
    let complex_structure = if sine_floor > skew_error + compression_band {
        Some(compressed.mapv(|value| value / sine))
    } else {
        None
    };
    RotationClusterKind::Rotation {
        planes: width / 2,
        angle: sine.atan2(mean_cosine),
        complex_structure,
    }
}

fn frobenius_norm(matrix: ArrayView2<'_, f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn vector_norm(vector: ArrayView1<'_, f64>) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Why a family of responses admits no isolating block decomposition.
#[derive(Debug)]
pub enum ResponseBlockError {
    /// No responses were supplied.
    NoResponses,
    /// Response `response` has `columns` columns, not one per parameter.
    ColumnMismatch {
        response: usize,
        columns: usize,
        parameters: usize,
    },
    /// An entry of response `response`, or of the parameter when `None`, is not finite.
    NonFinite { response: Option<usize> },
    /// The declared error is negative or not finite.
    InvalidDeclaredError { declared_error: f64 },
    /// Counterexample: response `response` is `distance` from the nearest partial
    /// isometry, beyond the declared error plus the singular-value band, so its Gram is
    /// not within the declared error of a projector.
    NotProjector {
        response: usize,
        distance: f64,
        declared_error: f64,
    },
    /// A singular value decomposition failed.
    Linalg(FaerLinalgError),
    /// Counterexample: compressed to the block seen by exactly `seen_by` (the
    /// complement of every earlier block when empty), response `response` has a
    /// singular value further than `band` from both `0` and `1`, so no commuting
    /// partial isometries lie within the declared error of the responses.
    NotIsolating {
        response: usize,
        seen_by: Vec<usize>,
        singular_value: f64,
        band: f64,
    },
    /// The classification band reached `1/2`, so `0` and `1` are not separated.
    Unresolved {
        response: usize,
        seen_by: Vec<usize>,
        band: f64,
    },
}

impl std::fmt::Display for ResponseBlockError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoResponses => write!(formatter, "response block recovery needs a response"),
            Self::ColumnMismatch {
                response,
                columns,
                parameters,
            } => write!(
                formatter,
                "response {response} has {columns} columns for {parameters} parameters"
            ),
            Self::NonFinite {
                response: Some(response),
            } => write!(formatter, "response {response} has a non-finite entry"),
            Self::NonFinite { response: None } => {
                write!(formatter, "the parameter has a non-finite entry")
            }
            Self::InvalidDeclaredError { declared_error } => write!(
                formatter,
                "response block recovery needs a finite non-negative declared error, got \
                 {declared_error}"
            ),
            Self::NotProjector {
                response,
                distance,
                declared_error,
            } => write!(
                formatter,
                "response {response} is {distance:.3e} from a partial isometry, beyond the \
                 declared error {declared_error:.3e}"
            ),
            Self::Linalg(error) => {
                write!(formatter, "response block recovery decomposition failed: {error}")
            }
            Self::NotIsolating {
                response,
                seen_by,
                singular_value,
                band,
            } => write!(
                formatter,
                "response {response} compressed to the block seen by {seen_by:?} has singular \
                 value {singular_value}, further than {band:.3e} from 0 and 1"
            ),
            Self::Unresolved {
                response,
                seen_by,
                band,
            } => write!(
                formatter,
                "response {response} on the block seen by {seen_by:?}: band {band:.3e} does \
                 not separate 0 from 1"
            ),
        }
    }
}

impl std::error::Error for ResponseBlockError {}

/// Parameter directions seen by exactly one set of responses.
#[derive(Clone, Debug)]
pub struct ResponseBlock {
    /// The responses that see the block, increasing. Every other response sees none
    /// of it.
    pub responses: Vec<usize>,
    /// Orthonormal basis of the block (`p x m`).
    pub basis: Array2<f64>,
    /// Bound on the 2-norm distance between the projector onto `basis` and the atom
    /// `Pi'_T` of the certified family (see [`ResponseBlockRecovery`]).
    pub projector_bar: f64,
    /// `Pi'_T theta*`.
    pub component: Array1<f64>,
    /// Bound on `||component - Pi'_T theta*||_2`.
    pub component_bar: f64,
}

/// The unique isolating decomposition of a parameter under a certified family of
/// commuting projectors.
///
/// With `W` the polar factor of every block basis stacked, the family
/// `P'_e = sum_{T containing e} W_T W_T^T` is exactly commuting projectors whose atoms
/// are `Pi'_T = W_T W_T^T`, and `||A_e^T A_e - P'_e||_2 <= response_defects[e]`. When the
/// supplied Grams are themselves commuting projectors, their atoms lie within
/// `sum_e response_defects[e]` of `Pi'_T`, since an atom is a product of norm-one
/// factors.
#[derive(Clone, Debug)]
pub struct ResponseBlockRecovery {
    /// Blocks ordered by their response sets.
    pub blocks: Vec<ResponseBlock>,
    /// `theta* - sum_T component_T`, the part no response sees.
    pub invisible: Array1<f64>,
    /// Bound on `||invisible - (I - sum_T Pi'_T) theta*||_2`.
    pub invisible_bar: f64,
    /// Bound on `||A_e^T A_e - P'_e||_2` per response.
    pub response_defects: Vec<f64>,
    /// Bound on `||V - W||_2` for the stacked block bases `V`.
    pub orthonormality_defect: f64,
    /// `||theta*||_2`.
    pub parameter_norm: f64,
}

impl ResponseBlockRecovery {
    /// `P_e = H_e theta*`, the sum of the components of every block `response` sees,
    /// with a bound on its distance to `A_e^T A_e theta*`. The sum is `P'_e theta*` up to
    /// the component bars, and `||(A_e^T A_e - P'_e) theta*|| <= response_defects[e]
    /// ||theta*||` holds whether or not the supplied Grams commute.
    pub fn response_component(&self, response: usize) -> (Array1<f64>, f64) {
        let mut component = Array1::<f64>::zeros(self.invisible.len());
        let mut bar = 0.0_f64;
        let mut magnitude = 0.0_f64;
        let mut terms = 0_usize;
        for block in self
            .blocks
            .iter()
            .filter(|block| block.responses.binary_search(&response).is_ok())
        {
            component += &block.component;
            bar += block.component_bar;
            magnitude += vector_norm(block.component.view());
            terms += 1;
        }
        (
            component,
            bar + accumulation_growth(terms.saturating_sub(1)) * magnitude
                + self.response_defects[response] * self.parameter_norm,
        )
    }
}

/// A block before its component is formed.
struct RefinementBlock {
    responses: Vec<usize>,
    basis: Array2<f64>,
    bar: f64,
}

/// Right singular directions (rows of `right`) classified as seen (singular value
/// within `band` of `1`) or unseen (within `band` of `0`), with the Davis–Kahan bar
/// on the split. A wide matrix's omitted singular values are zero, and their
/// directions are not returned.
struct VisibilitySplit {
    right: Array2<f64>,
    seen: Vec<usize>,
    unseen: Vec<usize>,
    bar: f64,
}

fn split_by_visibility(
    matrix: &Array2<f64>,
    perturbation: f64,
    response: usize,
    seen_by: &[usize],
) -> Result<VisibilitySplit, ResponseBlockError> {
    let (rows, cols) = matrix.dim();
    let (_, singular_values, right) = matrix
        .svd(false, true)
        .map_err(ResponseBlockError::Linalg)?;
    let right = right.ok_or(ResponseBlockError::Linalg(
        FaerLinalgError::SvdNoConvergence {
            context: "response block split: right singular vectors",
        },
    ))?;
    let sigma_max = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value));
    let band = perturbation + factor_singular_band(rows, cols, sigma_max);
    if !(band < 0.5) {
        return Err(ResponseBlockError::Unresolved {
            response,
            seen_by: seen_by.to_vec(),
            band,
        });
    }
    let mut seen = Vec::new();
    let mut unseen = Vec::new();
    for (index, &value) in singular_values.iter().enumerate() {
        if value <= band {
            unseen.push(index);
        } else if (value - 1.0).abs() <= band {
            seen.push(index);
        } else {
            return Err(ResponseBlockError::NotIsolating {
                response,
                seen_by: seen_by.to_vec(),
                singular_value: value,
                band,
            });
        }
    }
    let omitted_zero = singular_values.len() < cols;
    let seen_floor = seen
        .iter()
        .map(|&index| singular_values[index])
        .fold(f64::INFINITY, f64::min);
    let unseen_ceiling = unseen
        .iter()
        .map(|&index| singular_values[index])
        .fold(
            if omitted_zero {
                0.0
            } else {
                f64::NEG_INFINITY
            },
            f64::max,
        );
    // Nothing seen, or everything seen: the split projector is 0 or I.
    let bar = if seen.is_empty() || unseen_ceiling == f64::NEG_INFINITY {
        0.0
    } else {
        projector_error_bar(seen_floor - unseen_ceiling, band)
    };
    Ok(VisibilitySplit {
        right,
        seen,
        unseen,
        bar,
    })
}

/// Recover the unique isolating decomposition of `parameter` under the commuting
/// response projectors `A_e^T A_e`, for responses the caller declares to be within
/// `declared_error` (2-norm) of commuting partial isometries (module documentation).
///
/// # Classification bands
///
/// With `A~_e` those partial isometries and `V Q` a block's true basis,
/// `||A_e V^ - A~_e V Q||_2 <= delta + sqrt(2) bar + rounding` (`||A~_e||_2 <= 1`,
/// `||V^ - V Q||_2 <= sqrt(2) bar`), where a child block's bar is `2 sqrt(2)` times its
/// parent's plus the split's Davis–Kahan bar plus the rounding of the rotated basis.
/// The complement compression `A_e (I - sum_b V_b V_b^T)` differs from the true one
/// by at most `delta + sum_b bar_b + rounding`, summing the bars of the blocks before
/// the split: a split rotates a block within its own span. A singular value outside
/// that band refutes the declaration, so a refusal is a counterexample.
///
/// # Reported bounds
///
/// A classification that succeeds does not prove the declaration, so the reported bars
/// are a posteriori: the blocks define the certified family of
/// [`ResponseBlockRecovery`], and each `response_defects[e]` is measured against it.
pub fn recover_response_projector_blocks(
    responses: &[ArrayView2<'_, f64>],
    parameter: ArrayView1<'_, f64>,
    declared_error: f64,
) -> Result<ResponseBlockRecovery, ResponseBlockError> {
    if responses.is_empty() {
        return Err(ResponseBlockError::NoResponses);
    }
    if !(declared_error.is_finite() && declared_error >= 0.0) {
        return Err(ResponseBlockError::InvalidDeclaredError { declared_error });
    }
    let parameters = parameter.len();
    if parameter.iter().any(|value| !value.is_finite()) {
        return Err(ResponseBlockError::NonFinite { response: None });
    }
    for (response, matrix) in responses.iter().enumerate() {
        if matrix.ncols() != parameters {
            return Err(ResponseBlockError::ColumnMismatch {
                response,
                columns: matrix.ncols(),
                parameters,
            });
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(ResponseBlockError::NonFinite {
                response: Some(response),
            });
        }
    }
    let mut blocks: Vec<RefinementBlock> = Vec::new();
    for (response, matrix) in responses.iter().enumerate() {
        let rows = matrix.nrows();
        if rows == 0 {
            continue;
        }
        let (_, singular_values, _) = matrix
            .svd(false, false)
            .map_err(ResponseBlockError::Linalg)?;
        let sigma_max = singular_values
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value));
        // The nearest partial isometry rounds every singular value to 0 or 1.
        let distance = singular_values
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max(value.min((1.0 - value).abs())));
        if distance > declared_error + factor_singular_band(rows, parameters, sigma_max) {
            return Err(ResponseBlockError::NotProjector {
                response,
                distance,
                declared_error,
            });
        }
        let absolute_matrix = matrix.mapv(f64::abs);

        let mut refined = Vec::with_capacity(blocks.len() + 1);
        // Splitting a block leaves its span unchanged, so the union's projector error is
        // bounded by the parents' bars plus the rounding of the rotated bases, not by
        // the children's larger bars.
        let mut union_bar = 0.0_f64;
        for block in blocks {
            let width = block.basis.ncols();
            union_bar += block.bar;
            // Zero rows complete a wide compression, so every block direction is
            // classified.
            let mut compressed = Array2::<f64>::zeros((rows.max(width), width));
            compressed
                .slice_mut(s![..rows, ..])
                .assign(&matrix.dot(&block.basis));
            let absolute_basis = block.basis.mapv(f64::abs);
            let perturbation = declared_error
                + SQRT_2 * block.bar
                + accumulation_growth(parameters)
                    * frobenius_norm(absolute_matrix.dot(&absolute_basis).view());
            let split =
                split_by_visibility(&compressed, perturbation, response, &block.responses)?;
            if split.seen.is_empty() || split.unseen.is_empty() {
                let mut seeing = block.responses;
                if split.unseen.is_empty() {
                    seeing.push(response);
                }
                refined.push(RefinementBlock {
                    responses: seeing,
                    basis: block.basis,
                    bar: block.bar,
                });
                continue;
            }
            for (indices, sees) in [(&split.seen, true), (&split.unseen, false)] {
                let rotation = split.right.select(Axis(0), indices).t().to_owned();
                let basis = block.basis.dot(&rotation);
                let rounding = accumulation_growth(width)
                    * frobenius_norm(absolute_basis.dot(&rotation.mapv(f64::abs)).view());
                union_bar += 2.0 * rounding;
                let mut seeing = block.responses.clone();
                if sees {
                    seeing.push(response);
                }
                refined.push(RefinementBlock {
                    responses: seeing,
                    basis,
                    bar: 2.0 * SQRT_2 * block.bar + split.bar + 2.0 * rounding,
                });
            }
        }

        let union = stack_bases(parameters, refined.iter());
        let union_width = union.ncols();
        let (residual, rounding) = if union_width == 0 {
            (matrix.to_owned(), 0.0)
        } else {
            let absolute_union = union.mapv(f64::abs);
            // Two nested products and one subtraction per entry.
            let magnitude = absolute_matrix.dot(&absolute_union).dot(&absolute_union.t())
                + &absolute_matrix;
            (
                matrix.to_owned() - matrix.dot(&union).dot(&union.t()),
                accumulation_growth(parameters + union_width + 1)
                    * frobenius_norm(magnitude.view()),
            )
        };
        let split = split_by_visibility(
            &residual,
            declared_error + union_bar + rounding,
            response,
            &[],
        )?;
        if !split.seen.is_empty() {
            refined.push(RefinementBlock {
                responses: vec![response],
                basis: split.right.select(Axis(0), &split.seen).t().to_owned(),
                bar: split.bar,
            });
        }
        blocks = refined;
    }

    // `||V - W||_2 <= ||V^T V - I||_F` for the polar factor `W`, plus the Gram's rounding.
    let stacked = stack_bases(parameters, blocks.iter());
    let union_width = stacked.ncols();
    let orthonormality_defect = if union_width == 0 {
        0.0
    } else {
        let absolute_stacked = stacked.mapv(f64::abs);
        frobenius_norm((stacked.t().dot(&stacked) - Array2::<f64>::eye(union_width)).view())
            + accumulation_growth(parameters)
                * frobenius_norm(absolute_stacked.t().dot(&absolute_stacked).view())
    };
    let mut response_defects = Vec::with_capacity(responses.len());
    for (response, matrix) in responses.iter().enumerate() {
        let seen = stack_bases(
            parameters,
            blocks
                .iter()
                .filter(|block| block.responses.binary_search(&response).is_ok()),
        );
        response_defects.push(response_defect(*matrix, &seen, orthonormality_defect)?);
    }

    // `||V_T V_T^T - W_T W_T^T||_2 <= ||V_T - W_T|| (||V_T|| + ||W_T||)`.
    let basis_shift = orthonormality_defect * (2.0 + orthonormality_defect);
    let parameter_norm = vector_norm(parameter);
    let absolute_parameter = parameter.mapv(f64::abs);
    let mut invisible = parameter.to_owned();
    let mut invisible_bar = 0.0_f64;
    let mut component_magnitude = parameter_norm;
    let mut recovered = Vec::with_capacity(blocks.len());
    for block in blocks {
        let component = block.basis.dot(&block.basis.t().dot(&parameter));
        let absolute_basis = block.basis.mapv(f64::abs);
        let magnitude = absolute_basis.dot(&absolute_basis.t().dot(&absolute_parameter));
        let component_bar = basis_shift * parameter_norm
            + accumulation_growth(parameters + block.basis.ncols())
                * vector_norm(magnitude.view());
        invisible -= &component;
        invisible_bar += component_bar;
        component_magnitude += vector_norm(component.view());
        recovered.push(ResponseBlock {
            responses: block.responses,
            basis: block.basis,
            projector_bar: basis_shift,
            component,
            component_bar,
        });
    }
    invisible_bar += accumulation_growth(recovered.len()) * component_magnitude;
    recovered.sort_by(|left, right| left.responses.cmp(&right.responses));
    Ok(ResponseBlockRecovery {
        blocks: recovered,
        invisible,
        invisible_bar,
        response_defects,
        orthonormality_defect,
        parameter_norm,
    })
}

fn stack_bases<'a>(
    parameters: usize,
    blocks: impl Iterator<Item = &'a RefinementBlock> + Clone,
) -> Array2<f64> {
    let width: usize = blocks.clone().map(|block| block.basis.ncols()).sum();
    let mut stacked = Array2::<f64>::zeros((parameters, width));
    let mut offset = 0;
    for block in blocks {
        let block_width = block.basis.ncols();
        stacked
            .slice_mut(s![.., offset..offset + block_width])
            .assign(&block.basis);
        offset += block_width;
    }
    stacked
}

/// Bound on `||A^T A - W W^T||_2`, where `W` is the polar factor of the stacked bases
/// `seen` of every block the response sees (orthonormality defect `eta`).
///
/// With `Y = A W` and `X = A (I - W W^T)`, `A = Y W^T + X` and `X W = 0`, so
/// `A^T A - W W^T = W (Y^T Y - I) W^T + W Y^T X + X^T Y W^T + X^T X`, bounded by
/// `max_i |sigma_i(Y)^2 - 1| + 2 ||X|| ||Y|| + ||X||^2`. The computed `Y` and `X` use
/// `seen` instead of `W`: `||seen - W|| <= eta` moves each `sigma_i(Y)` by at most
/// `||A|| eta` and `X` by at most `||A|| eta (2 + eta)`, plus the rounding of both.
fn response_defect(
    matrix: ArrayView2<'_, f64>,
    seen: &Array2<f64>,
    orthonormality_defect: f64,
) -> Result<f64, ResponseBlockError> {
    let (rows, parameters) = matrix.dim();
    let width = seen.ncols();
    let matrix_norm = frobenius_norm(matrix);
    if width == 0 {
        return Ok(matrix_norm * matrix_norm);
    }
    let absolute_matrix = matrix.mapv(f64::abs);
    let absolute_seen = seen.mapv(f64::abs);
    let projected = matrix.dot(seen);
    // Zero rows complete a wide projection so every seen direction has a singular value.
    let mut completed = Array2::<f64>::zeros((rows.max(width), width));
    completed.slice_mut(s![..rows, ..]).assign(&projected);
    let (_, singular_values, _) = completed
        .svd(false, false)
        .map_err(ResponseBlockError::Linalg)?;
    let sigma_max = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value));
    let shift = factor_singular_band(rows.max(width), width, sigma_max)
        + accumulation_growth(parameters)
            * frobenius_norm(absolute_matrix.dot(&absolute_seen).view())
        + matrix_norm * orthonormality_defect;
    let isometry = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max((value * value - 1.0).abs()))
        + accumulation_growth(2) * sigma_max * sigma_max
        + shift * (2.0 * sigma_max + shift);
    let magnitude =
        absolute_matrix.dot(&absolute_seen).dot(&absolute_seen.t()) + &absolute_matrix;
    let residual = matrix.to_owned() - projected.dot(&seen.t());
    let residual_norm = frobenius_norm(residual.view())
        + accumulation_growth(parameters + width + 1) * frobenius_norm(magnitude.view())
        + matrix_norm * orthonormality_defect * (2.0 + orthonormality_defect);
    let projected_norm = sigma_max + shift;
    Ok(isometry + 2.0 * residual_norm * projected_norm + residual_norm * residual_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::test_support::{
        Planted, hidden_basis, orthogonality_bound, plant, projector_distance,
    };
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};
    use std::f64::consts::PI;
    use std::ops::Range;

    const DIMENSION: usize = 8;

    fn recover_planted(planted: &Planted) -> PlaneRotationRecovery {
        recover_plane_rotations(planted.matrix.view(), planted.matrix_defect).expect("recovery")
    }

    /// The cluster spans the planted columns: within its Davis–Kahan bar about
    /// `Q_o B_o Q_o^T` (the planted matrix declares its own defect), plus
    /// `||Q_k Q_k^T - Q_o,k Q_o,k^T|| <= eta_Q (2 + eta_Q)` and the measurement band.
    fn assert_spans_planted(cluster: &RotationCluster, planted: &Planted, columns: Range<usize>) {
        let column_defect = planted.basis_defect * (2.0 + planted.basis_defect);
        let (distance, band) = projector_distance(
            cluster.basis.view(),
            planted.basis.slice(s![.., columns.clone()]),
        );
        assert!(
            distance <= cluster.projector_bar + column_defect + band,
            "planted columns {columns:?}: projector distance {distance:.3e} exceeds bar \
             {:.3e} + column defect {column_defect:.3e} + band {band:.3e}",
            cluster.projector_bar
        );
    }

    /// The planted cosine lies in the cluster's interval widened by `|c - c/r| <= eta_B`.
    fn assert_cosine_admitted(cluster: &RotationCluster, planted: &Planted, angle: f64) {
        let widen = planted.form_defect;
        let cosine = angle.sin_cos().1;
        assert!(
            cluster.cosine_interval.0 - widen <= cosine
                && cosine <= cluster.cosine_interval.1 + widen,
            "planted cosine {cosine} outside {:?} widened by {widen:.3e}",
            cluster.cosine_interval
        );
    }

    /// `<J_recovered, J_planted>_F / 2`, which is `+1` for the planted orientation and
    /// `-1` for the reverse. `J_planted = q_2 q_1^T - q_1 q_2^T` from `[[0, -1], [1, 0]]`.
    fn orientation_alignment(cluster: &RotationCluster, planted: &Planted, first: usize) -> f64 {
        let structure = match &cluster.kind {
            RotationClusterKind::Rotation {
                complex_structure: Some(structure),
                ..
            } => structure,
            other => panic!("expected a certified rotation, got {other:?}"),
        };
        let recovered = cluster.basis.dot(structure).dot(&cluster.basis.t());
        let (u, v) = (planted.basis.column(first), planted.basis.column(first + 1));
        let dimension = planted.basis.nrows();
        let planted_structure =
            Array2::<f64>::from_shape_fn((dimension, dimension), |(row, col)| {
                v[row] * u[col] - u[row] * v[col]
            });
        (&recovered * &planted_structure).sum() / 2.0
    }

    fn rotation_planes(cluster: &RotationCluster) -> usize {
        match &cluster.kind {
            RotationClusterKind::Rotation { planes, angle, .. } => {
                assert!(*angle > 0.0 && *angle < PI, "angle {angle} outside (0, pi)");
                *planes
            }
            other => panic!("expected a rotation cluster, got {other:?}"),
        }
    }

    #[test]
    fn planted_planes_hidden_by_a_random_basis_are_recovered_2951() {
        let angles = [0.35, 1.2, 2.6];
        let planted = plant(DIMENSION, &angles, 0, 0x2951_0001);
        let recovery = recover_planted(&planted);
        assert_eq!(recovery.clusters.len(), 4, "{recovery:?}");
        // Increasing cosine: 2.6, 1.2, 0.35, then the fixed space.
        for (cluster, (angle, first)) in recovery
            .clusters
            .iter()
            .zip([(2.6, 4_usize), (1.2, 2), (0.35, 0)])
        {
            assert_eq!(rotation_planes(cluster), 1, "{cluster:?}");
            assert_cosine_admitted(cluster, &planted, angle);
            assert_spans_planted(cluster, &planted, first..first + 2);
            let alignment = orientation_alignment(cluster, &planted, first);
            assert!(alignment > 0.0, "angle {angle}: alignment {alignment}");
        }
        let fixed = &recovery.clusters[3];
        assert!(
            matches!(fixed.kind, RotationClusterKind::Fixed { .. }),
            "{fixed:?}"
        );
        assert_eq!(fixed.basis.ncols(), 2);
        assert_spans_planted(fixed, &planted, 6..8);
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
    }

    #[test]
    fn repeated_cosines_report_the_invariant_subspace_not_the_planes_2951() {
        let planted = plant(DIMENSION, &[0.9, 0.9, 2.0], 0, 0x2951_0002);
        let recovery = recover_planted(&planted);
        assert_eq!(recovery.clusters.len(), 3, "{recovery:?}");
        let repeated = &recovery.clusters[1];
        assert_eq!(rotation_planes(repeated), 2, "{repeated:?}");
        assert_cosine_admitted(repeated, &planted, 0.9);
        assert_spans_planted(repeated, &planted, 0..4);
        assert_eq!(
            recovery.ambiguities(),
            vec![
                RotationAmbiguity::RepeatedCosine {
                    cluster: 1,
                    planes: 2
                },
                RotationAmbiguity::Winding
            ]
        );

        // Positive control: distinct cosines are separated into identified planes.
        let distinct = plant(DIMENSION, &[0.9, 1.0, 2.0], 0, 0x2951_0002);
        let recovery = recover_planted(&distinct);
        assert_eq!(recovery.clusters.len(), 4, "{recovery:?}");
        for (cluster, (angle, first)) in recovery
            .clusters
            .iter()
            .zip([(2.0, 4_usize), (1.0, 2), (0.9, 0)])
        {
            assert_eq!(rotation_planes(cluster), 1, "{cluster:?}");
            assert_spans_planted(cluster, &distinct, first..first + 2);
            assert_cosine_admitted(cluster, &distinct, angle);
        }
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
    }

    #[test]
    fn a_coarser_declared_error_merges_cosines_it_cannot_separate_2951() {
        let planted = plant(DIMENSION, &[0.9, 1.0], 0, 0x2951_0003);
        let fine = recover_planted(&planted);
        assert_eq!(fine.clusters.len(), 3, "{fine:?}");
        assert_eq!(fine.ambiguities(), vec![RotationAmbiguity::Winding]);

        // Declaring the cosine gap itself makes the resolution 2 beta exceed that gap:
        // no pair of distinct true cosines is provable, so the planes merge.
        let gap = 0.9_f64.cos() - 1.0_f64.cos();
        let coarse = recover_plane_rotations(planted.matrix.view(), gap).expect("recovery");
        assert!(
            2.0 * coarse.perturbation_bound > gap,
            "resolution {} does not exceed the gap {gap}",
            2.0 * coarse.perturbation_bound
        );
        assert_eq!(coarse.clusters.len(), 2, "{coarse:?}");
        assert_eq!(rotation_planes(&coarse.clusters[0]), 2);
        assert!(matches!(
            coarse.clusters[1].kind,
            RotationClusterKind::Fixed { .. }
        ));
        assert_eq!(
            coarse.ambiguities(),
            vec![
                RotationAmbiguity::RepeatedCosine {
                    cluster: 0,
                    planes: 2
                },
                RotationAmbiguity::Winding
            ]
        );
    }

    #[test]
    fn a_non_orthogonal_operator_is_refused_not_approximated_2951() {
        let planted = plant(DIMENSION, &[1.1], 0, 0x2951_0008);
        assert!(recover_plane_rotations(planted.matrix.view(), planted.matrix_defect).is_ok());

        // `I + U (R - I) U^+` with `U = Q_{0..2} diag(1, 2)` rotates the same plane but is
        // not orthogonal: `D (R - I) D^-1` has off-diagonals `-s/2` and `2s`.
        let (sine, cosine) = 1.1_f64.sin_cos();
        let generator = ndarray::arr2(&[[cosine - 1.0, -sine / 2.0], [2.0 * sine, cosine - 1.0]]);
        let columns = planted.basis.slice(s![.., 0..2]);
        let conjugated = Array2::<f64>::eye(DIMENSION) + columns.dot(&generator).dot(&columns.t());
        match recover_plane_rotations(conjugated.view(), planted.matrix_defect) {
            Err(PlaneRotationError::NotOrthogonal {
                distance,
                declared_error,
            }) => assert!(distance > declared_error, "{distance} vs {declared_error}"),
            other => panic!("expected a non-orthogonal refusal, got {other:?}"),
        }

        // Scaling moves every singular value to 1.05, beyond the planted declaration.
        let scaled = planted.matrix.mapv(|value| 1.05 * value);
        match recover_plane_rotations(scaled.view(), planted.matrix_defect) {
            Err(PlaneRotationError::NotOrthogonal {
                distance,
                declared_error,
            }) => assert!(distance > declared_error, "{distance} vs {declared_error}"),
            other => panic!("expected a non-orthogonal refusal, got {other:?}"),
        }
        match recover_plane_rotations(planted.matrix.view(), -1.0) {
            Err(PlaneRotationError::InvalidDeclaredError { declared_error }) => {
                assert_eq!(declared_error, -1.0)
            }
            other => panic!("expected an invalid declaration refusal, got {other:?}"),
        }
    }

    #[test]
    fn half_turns_and_reflections_report_no_orientation_2951() {
        let planted = plant(DIMENSION, &[PI, 1.1], 0, 0x2951_0004);
        let recovery = recover_planted(&planted);
        assert_eq!(recovery.clusters.len(), 3, "{recovery:?}");
        let half_turn = &recovery.clusters[0];
        assert!(
            matches!(half_turn.kind, RotationClusterKind::HalfTurn { .. }),
            "{half_turn:?}"
        );
        assert_eq!(half_turn.basis.ncols(), 2);
        assert_spans_planted(half_turn, &planted, 0..2);
        assert_eq!(rotation_planes(&recovery.clusters[1]), 1);
        assert_eq!(
            recovery.ambiguities(),
            vec![
                RotationAmbiguity::HalfTurn { cluster: 0 },
                RotationAmbiguity::Winding
            ]
        );

        // A single -1 axis is a reflection: an odd-dimensional half-turn space.
        let reflection = plant(DIMENSION, &[1.1], 1, 0x2951_0004);
        let recovery = recover_planted(&reflection);
        assert_eq!(recovery.clusters.len(), 3, "{recovery:?}");
        let axis = &recovery.clusters[0];
        assert!(
            matches!(axis.kind, RotationClusterKind::HalfTurn { .. }),
            "{axis:?}"
        );
        assert_eq!(axis.basis.ncols(), 1);
        assert_spans_planted(axis, &reflection, 2..3);
    }

    #[test]
    fn identity_reports_no_plane_and_the_largest_angle_it_cannot_exclude_2951() {
        let identity = Array2::<f64>::eye(DIMENSION);
        let recovery = recover_plane_rotations(identity.view(), 0.0).expect("recovery");
        assert_eq!(recovery.clusters.len(), 1, "{recovery:?}");
        let cluster = &recovery.clusters[0];
        assert!(cluster.separation.is_infinite());
        assert_eq!(cluster.projector_bar, 0.0);
        let max_hidden_angle = match cluster.kind {
            RotationClusterKind::Fixed { max_hidden_angle } => max_hidden_angle,
            ref other => panic!("expected a fixed cluster, got {other:?}"),
        };
        // `1 - cos(hidden) = beta + (1 - lowest computed eigenvalue) <= 2 beta`.
        assert!(max_hidden_angle > 0.0);
        assert!(1.0 - max_hidden_angle.cos() <= 2.0 * recovery.perturbation_bound);
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Identity]);

        // Positive control: a rotation well below that angle is reported as identity,
        // while a resolvable one is a plane.
        let hidden = plant(DIMENSION, &[max_hidden_angle / 4.0], 0, 0x2951_0005);
        assert_eq!(
            recover_planted(&hidden).ambiguities(),
            vec![RotationAmbiguity::Identity]
        );
        let visible = plant(DIMENSION, &[0.3], 0, 0x2951_0005);
        assert_eq!(
            recover_planted(&visible).ambiguities(),
            vec![RotationAmbiguity::Winding]
        );
    }

    #[test]
    fn winding_and_orientation_resolve_to_the_principal_representative_2951() {
        for (angle, sign) in [(1.1, 1.0), (1.1 + 2.0 * PI, 1.0), (-1.1, -1.0)] {
            let planted = plant(DIMENSION, &[angle], 0, 0x2951_0006);
            let recovery = recover_planted(&planted);
            assert_eq!(recovery.clusters.len(), 2, "{recovery:?}");
            let plane = &recovery.clusters[0];
            assert_eq!(rotation_planes(plane), 1);
            assert_cosine_admitted(plane, &planted, angle);
            assert_spans_planted(plane, &planted, 0..2);
            let alignment = orientation_alignment(plane, &planted, 0);
            assert!(
                sign * alignment > 0.0,
                "angle {angle}: alignment {alignment} has the wrong sign"
            );
            assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
        }
    }

    #[test]
    fn orientation_is_withheld_when_the_declared_error_hides_the_skew_part_2951() {
        let dimension = 4;
        let angles = [0.896_f64.acos(), (-0.896_f64).acos()];
        let planted = plant(dimension, &angles, 0, 0x2951_0007);
        let exact = recover_planted(&planted);
        assert_eq!(exact.clusters.len(), 2, "{exact:?}");
        for cluster in &exact.clusters {
            assert!(
                matches!(
                    cluster.kind,
                    RotationClusterKind::Rotation {
                        complex_structure: Some(_),
                        ..
                    }
                ),
                "{cluster:?}"
            );
        }

        // Declared error 0.1 keeps the cosine intervals [0.796, 0.996] and their mirror
        // clear of +-1, but their sine floor sqrt(1 - 0.996^2) ~ 0.089 is below the
        // skew error 0.1 + 2 sqrt(2) bar, so no orientation is certified.
        let withheld = recover_plane_rotations(planted.matrix.view(), 0.1).expect("recovery");
        assert_eq!(withheld.clusters.len(), 2, "{withheld:?}");
        for cluster in &withheld.clusters {
            assert!(
                matches!(
                    cluster.kind,
                    RotationClusterKind::Rotation {
                        complex_structure: None,
                        ..
                    }
                ),
                "{cluster:?}"
            );
        }
    }

    /// An exactly orthogonal, exactly representable basis of `R^16` that mixes every
    /// coordinate: a seeded signed row permutation of the Kronecker square of the
    /// normalized 4x4 Hadamard matrix (entries `+-1/4`).
    fn exact_hidden_basis(seed: u64) -> Array2<f64> {
        let hadamard = ndarray::arr2(&[
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ])
        .mapv(|value| value / 2.0);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut order: Vec<usize> = (0..16).collect();
        order.shuffle(&mut rng);
        let signs: Vec<f64> = (0..16)
            .map(|_| if rng.random_range(0..2) == 0 { 1.0 } else { -1.0 })
            .collect();
        Array2::<f64>::from_shape_fn((16, 16), |(row, col)| {
            let source = order[row];
            signs[row] * hadamard[[source / 4, col / 4]] * hadamard[[source % 4, col % 4]]
        })
    }

    /// A response `A = fl(G C^T)` reading orthonormal columns `C` through a random
    /// rotation `G` of its rows.
    struct ResponseFixture {
        matrix: Array2<f64>,
        /// Bound on `||A - G_o C_o^T||_2`, the distance to the planted partial isometry:
        /// `eta_G (1 + eta_C) + eta_C + F`, with `F` the product's rounding.
        declared_error: f64,
        /// Bound on `||A^T A - C C^T||_2`:
        /// `(1 + eta_C)^2 eta_G (2 + eta_G) + 2 (1 + eta_G)(1 + eta_C) F + F^2`.
        gram_shift: f64,
    }

    fn response_fixture(columns: ArrayView2<'_, f64>, seed: u64) -> ResponseFixture {
        let rotation = hidden_basis(columns.ncols(), seed);
        let matrix = rotation.dot(&columns.t());
        let rotation_defect = orthogonality_bound(&rotation);
        let column_defect = orthogonality_bound(&columns.to_owned());
        let formation = accumulation_growth(columns.ncols())
            * frobenius_norm(
                rotation
                    .mapv(f64::abs)
                    .dot(&columns.t().mapv(f64::abs))
                    .view(),
            );
        ResponseFixture {
            matrix,
            declared_error: rotation_defect * (1.0 + column_defect) + column_defect + formation,
            gram_shift: (1.0 + column_defect).powi(2) * rotation_defect * (2.0 + rotation_defect)
                + 2.0 * (1.0 + rotation_defect) * (1.0 + column_defect) * formation
                + formation * formation,
        }
    }

    /// `P P^T theta` for planted columns `P`, with its rounding band.
    fn planted_component(
        planted: ArrayView2<'_, f64>,
        parameter: &Array1<f64>,
    ) -> (Array1<f64>, f64) {
        let component = planted.dot(&planted.t().dot(parameter));
        let absolute = planted.mapv(f64::abs);
        let magnitude = absolute.dot(&absolute.t().dot(&parameter.mapv(f64::abs)));
        (
            component,
            accumulation_growth(planted.nrows() + planted.ncols())
                * vector_norm(magnitude.view()),
        )
    }

    /// `||left - right||_2 <= bar`, counting the rounding of the subtraction.
    fn assert_within(label: &str, left: &Array1<f64>, right: &Array1<f64>, bar: f64) {
        let error = vector_norm((left - right).view());
        let band =
            accumulation_growth(1) * (vector_norm(left.view()) + vector_norm(right.view()));
        assert!(
            error <= bar + band,
            "{label}: error {error:.3e} exceeds bar {bar:.3e} + band {band:.3e}"
        );
    }

    fn uniform_parameter(seed: u64) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::<f64>::from_shape_fn(16, |_| rng.random_range(-1.0..1.0))
    }

    #[test]
    fn commuting_response_projectors_recover_the_unique_isolating_blocks_2951() {
        let basis = exact_hidden_basis(0x2951_0010);
        let planted_responses = [
            (0_usize, 0..6, 0x2951_0011_u64),
            (1, 3..9, 0x2951_0012),
            (2, 11..13, 0x2951_0013),
        ];
        let fixtures: Vec<ResponseFixture> = planted_responses
            .iter()
            .map(|(_, columns, seed)| response_fixture(basis.slice(s![.., columns.clone()]), *seed))
            .collect();
        let views: Vec<ArrayView2<'_, f64>> =
            fixtures.iter().map(|fixture| fixture.matrix.view()).collect();
        let declared_error = fixtures
            .iter()
            .fold(0.0_f64, |acc, fixture| acc.max(fixture.declared_error));
        let parameter = uniform_parameter(0x2951_0014);
        let recovery = recover_response_projector_blocks(&views, parameter.view(), declared_error)
            .expect("recovery");
        // The certified family lies within `d_e + shift_e` of each planted projector, and
        // every planted atom within the sum over responses.
        let truth_total: f64 = recovery
            .response_defects
            .iter()
            .zip(&fixtures)
            .map(|(defect, fixture)| defect + fixture.gram_shift)
            .sum();
        let parameter_norm = vector_norm(parameter.view());
        let expected: [(&[usize], Range<usize>); 4] =
            [(&[0], 0..3), (&[0, 1], 3..6), (&[1], 6..9), (&[2], 11..13)];
        assert_eq!(recovery.blocks.len(), expected.len(), "{recovery:?}");
        for (block, (seeing, columns)) in recovery.blocks.iter().zip(expected.iter()) {
            assert_eq!(block.responses.as_slice(), *seeing);
            let planted = basis.slice(s![.., columns.clone()]);
            let (distance, band) = projector_distance(block.basis.view(), planted);
            assert!(
                distance <= block.projector_bar + truth_total + band,
                "block {seeing:?}: projector distance {distance:.3e} exceeds bar {:.3e} + \
                 planted shift {truth_total:.3e} + band {band:.3e}",
                block.projector_bar
            );
            let (truth, truth_band) = planted_component(planted, &parameter);
            assert_within(
                "block component",
                &block.component,
                &truth,
                block.component_bar + truth_total * parameter_norm + truth_band,
            );
        }
        for ((response, columns, _), fixture) in planted_responses.iter().zip(&fixtures) {
            let (component, bar) = recovery.response_component(*response);
            let (truth, truth_band) =
                planted_component(basis.slice(s![.., columns.clone()]), &parameter);
            assert_within(
                "response component",
                &component,
                &truth,
                bar + fixture.gram_shift * parameter_norm + truth_band,
            );
        }
        let invisible_basis = basis.select(Axis(1), &[9, 10, 13, 14, 15]);
        let (truth, truth_band) = planted_component(invisible_basis.view(), &parameter);
        assert_within(
            "invisible",
            &recovery.invisible,
            &truth,
            recovery.invisible_bar + truth_total * parameter_norm + truth_band,
        );

        // Uniqueness: moving the {0, 1} component into the {0} block keeps the sum, but
        // response 1 then sees the {0} block's vector. For `x = Pi'_{0} theta*`,
        // `P'_1 x = 0`, so `||A_1 x||^2 = x^T (A_1^T A_1 - P'_1) x <= d_1 ||x||^2`; the
        // computed component is within its bar of `x`.
        let only_first = &recovery.blocks[0];
        let moved = &only_first.component + &recovery.blocks[1].component;
        let response = &fixtures[1].matrix;
        let response_norm = frobenius_norm(response.view());
        let absolute_response = response.mapv(f64::abs);
        let isolation_band = |vector: &Array1<f64>| {
            recovery.response_defects[1].sqrt()
                * (vector_norm(vector.view()) + only_first.component_bar)
                + response_norm * only_first.component_bar
                + accumulation_growth(16)
                    * vector_norm(absolute_response.dot(&vector.mapv(f64::abs)).view())
        };
        let recovered_defect = vector_norm(response.dot(&only_first.component).view());
        assert!(
            recovered_defect <= isolation_band(&only_first.component),
            "the recovered {{0}} block is seen by response 1: {recovered_defect:.3e}"
        );
        let moved_defect = vector_norm(response.dot(&moved).view());
        assert!(
            moved_defect > isolation_band(&moved),
            "the non-isolating decomposition passed the isolation check: {moved_defect:.3e}"
        );
    }

    #[test]
    fn a_response_that_does_not_commute_with_a_block_is_a_counterexample_2951() {
        let basis = exact_hidden_basis(0x2951_0010);
        let parameter = uniform_parameter(0x2951_0014);
        let first = response_fixture(basis.slice(s![.., 0..6]), 0x2951_0011);
        let commuting = response_fixture(basis.slice(s![.., 3..9]), 0x2951_0012);
        let control = recover_response_projector_blocks(
            &[first.matrix.view(), commuting.matrix.view()],
            parameter.view(),
            first.declared_error.max(commuting.declared_error),
        );
        assert!(control.is_ok(), "{control:?}");

        // Tilting q5 toward q9 keeps the second response a projector but mixes a
        // direction the first sees with one it does not: the compression onto the
        // first block has singular value cos 0.3.
        let (sine, cosine) = 0.3_f64.sin_cos();
        let mut tilted_columns = basis.slice(s![.., 3..9]).to_owned();
        let tilted = &basis.column(5) * cosine + &basis.column(9) * sine;
        tilted_columns.column_mut(2).assign(&tilted);
        let tilted = response_fixture(tilted_columns.view(), 0x2951_0012);
        match recover_response_projector_blocks(
            &[first.matrix.view(), tilted.matrix.view()],
            parameter.view(),
            first.declared_error.max(tilted.declared_error),
        ) {
            Err(ResponseBlockError::NotIsolating {
                response,
                seen_by,
                singular_value,
                band,
            }) => {
                assert_eq!(response, 1);
                assert_eq!(seen_by, vec![0]);
                assert!(singular_value > band && 1.0 - singular_value > band);
            }
            other => panic!("expected a commutation counterexample, got {other:?}"),
        }
    }

    #[test]
    fn scaled_responses_are_refused_or_widen_the_certified_defect_2951() {
        let basis = exact_hidden_basis(0x2951_0010);
        let parameter = uniform_parameter(0x2951_0014);
        let first = response_fixture(basis.slice(s![.., 0..6]), 0x2951_0011);
        let second = response_fixture(basis.slice(s![.., 3..9]), 0x2951_0012);
        let fine = first.declared_error.max(second.declared_error);
        let clean = recover_response_projector_blocks(
            &[first.matrix.view(), second.matrix.view()],
            parameter.view(),
            fine,
        )
        .expect("recovery");

        // 0.9 A is 0.1 from a partial isometry: refused at the fine declaration.
        let widened_first = first.matrix.mapv(|value| 0.9 * value);
        match recover_response_projector_blocks(
            &[widened_first.view(), second.matrix.view()],
            parameter.view(),
            fine,
        ) {
            Err(ResponseBlockError::NotProjector {
                response,
                distance,
                declared_error,
            }) => {
                assert_eq!(response, 0);
                assert!(distance > declared_error, "{distance} vs {declared_error}");
            }
            other => panic!("expected a non-projector refusal, got {other:?}"),
        }

        // `||0.9 A - G_o C^T|| <= 0.1 + 0.9 fine` plus the scaling's rounding: declaring
        // that accepts the same blocks with a larger certified Gram defect.
        let scaling_rounding = accumulation_growth(1) * frobenius_norm(first.matrix.view());
        let coarse = fine + 0.1 + scaling_rounding;
        let widened = recover_response_projector_blocks(
            &[widened_first.view(), second.matrix.view()],
            parameter.view(),
            coarse,
        )
        .expect("recovery");
        assert_eq!(widened.blocks.len(), clean.blocks.len(), "{widened:?}");
        for (wide, narrow) in widened.blocks.iter().zip(&clean.blocks) {
            assert_eq!(wide.responses, narrow.responses);
        }
        assert!(
            widened.response_defects[0] > clean.response_defects[0],
            "widened defect {} does not exceed clean defect {}",
            widened.response_defects[0],
            clean.response_defects[0]
        );

        // 0.5 A declared at 1/2 is equally far from 0 and 1.
        let halved_first = first.matrix.mapv(|value| 0.5 * value);
        match recover_response_projector_blocks(
            &[halved_first.view(), second.matrix.view()],
            parameter.view(),
            fine + 0.5 + scaling_rounding,
        ) {
            Err(ResponseBlockError::Unresolved {
                response,
                seen_by,
                band,
            }) => {
                assert_eq!(response, 0);
                assert!(seen_by.is_empty());
                assert!(band >= 0.5);
            }
            other => panic!("expected an unresolved band, got {other:?}"),
        }
    }
}
