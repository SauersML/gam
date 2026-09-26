//! Spectral recovery of rotation planes from a frozen matrix (#2951 P3).
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
//! # Orthogonality is measured, not declared
//!
//! A square `W = U Sigma V^T` represents its orthogonal polar factor `O = U V^T`, the
//! orthogonal matrix nearest to it in the 2-norm (Fan–Hoffman): `W - O = U (Sigma - I)
//! V^T`, so `rho = ||W - O||_2 = max_i |sigma_i - 1|`, the distance from `W` to the
//! orthogonal group. A backward-stable SVD returns each singular value within the band
//! [`factor_singular_band`] of the true one (Weyl), so
//!
//! ```text
//! rho <= rho_bar = max_i |sigma_i,computed - 1| + band.
//! ```
//!
//! `rho_bar` is the perturbation every claim below is made against, so there is no
//! caller-supplied tolerance. A smaller declaration would contradict the matrix, and a
//! larger one could only coarsen the resolution. The Gram bound
//! `rho <= ||W^T W - I||_F` (from `|sigma - 1| = |sigma^2 - 1| / (sigma + 1) <=
//! |sigma^2 - 1|`) is valid too, but it is about twice as loose near `sigma = 1`,
//! because `|sigma^2 - 1| ~ 2 |sigma - 1|`, and the Frobenius norm can add up to
//! `sqrt(d)`. The SVD reads `rho` directly for the same `O(d^3)` cost.
//!
//! `O` is unique exactly when `W` is invertible. A smallest computed singular value
//! within the band of zero does not rule out a singular `W`, whose polar factor is not
//! determined, and the recovery refuses it. Any other matrix is recovered as its polar
//! factor with `rho_bar` reported. A far-from-orthogonal operator such as
//! `I + U (R - I) U^+` with non-orthonormal `U` keeps its planes in its real Schur form,
//! not in `(W + W^T)/2`. It is described through its polar factor at its large `rho_bar`,
//! not approximated.
//!
//! # Grouping is derived, not a constant
//!
//! Every claim holds for every orthogonal `O` within `rho_bar` of `W`, the polar factor
//! among them. `S(W)` differs from `S(O)` by at most `rho_bar`. Adding the rounding of
//! forming `S` and the eigensolver's backward error gives
//! `beta >= ||S_computed - S(O)||_2`, and by Weyl each computed eigenvalue is within
//! `beta` of its true one. Neighbouring computed eigenvalues further apart than
//! `2 beta` belong to provably distinct true eigenvalues. Nearer ones are not resolved
//! and stay in one cluster. A cluster's projector carries the Davis–Kahan bar
//! `beta / (gap - beta)` of [`projector_error_bar`], `gap` being its measured
//! separation from the rest of the spectrum.
//!
//! The route is dense: `O(d^3)` time and `d x d` workspace for a `d x d` matrix.

use faer::Side;
use gam_linalg::decision::projector_error_bar;
use gam_linalg::faer_ndarray::{FaerLinalgError, FaerSvd, strict_symmetric_eigh};
use gam_linalg::roundoff::{
    SymmetricAssembly, accumulation_growth, factor_singular_band, symmetric_spectrum_rounding_band,
};
use gam_runtime::resource::{MemoryGovernor, MemoryReservationError};
use ndarray::{Array2, ArrayView2};
use std::f64::consts::SQRT_2;

/// Why a matrix admits no plane-rotation recovery.
#[derive(Debug)]
pub enum PlaneRotationError {
    /// The matrix is empty or not square.
    NotSquare { rows: usize, cols: usize },
    /// An entry is not finite.
    NonFinite { row: usize, col: usize },
    /// The smallest computed singular value is within the singular-value band of zero,
    /// so the matrix is not resolved from a singular one, whose orthogonal polar factor
    /// is not unique: there is no single orthogonal matrix it represents.
    NotInvertible {
        smallest_singular_value: f64,
        band: f64,
    },
    /// The singular value or symmetric eigendecomposition failed.
    Linalg(FaerLinalgError),
    /// The dense working set of the recovery does not fit the memory budget.
    Memory(MemoryReservationError),
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
            Self::NotInvertible {
                smallest_singular_value,
                band,
            } => write!(
                formatter,
                "plane-rotation recovery refused: the smallest singular value \
                 {smallest_singular_value:.3e} is within the band {band:.3e} of zero, so \
                 the matrix has no unique nearest orthogonal matrix"
            ),
            Self::Linalg(error) => {
                write!(formatter, "plane-rotation recovery decomposition failed: {error}")
            }
            Self::Memory(error) => write!(formatter, "plane-rotation recovery: {error}"),
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
/// within `orthogonality_defect` of it, its polar factor among them.
#[derive(Clone, Debug)]
pub struct PlaneRotationRecovery {
    /// `rho_bar`, the certified bound on the 2-norm distance `rho` from the matrix to
    /// the orthogonal group: the measured distance plus the singular-value band.
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

/// Recover the rotation planes of the orthogonal polar factor of a frozen square matrix.
///
/// Measures the matrix's distance from the orthogonal group and refuses a matrix not
/// resolved from a singular one (module documentation). Otherwise clusters the spectrum of `S = (W + W^T)/2` at the derived
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
    governor: &MemoryGovernor,
    matrix: ArrayView2<'_, f64>,
) -> Result<PlaneRotationRecovery, PlaneRotationError> {
    let (rows, cols) = matrix.dim();
    if rows == 0 || rows != cols {
        return Err(PlaneRotationError::NotSquare { rows, cols });
    }
    if let Some(((row, col), _)) = matrix.indexed_iter().find(|(_, value)| !value.is_finite()) {
        return Err(PlaneRotationError::NonFinite { row, col });
    }
    let dimension = rows;
    // `d × d` arrays live at the peak, while the eigendecomposition runs: `S`, `K`, the
    // decomposition's working copy and its eigenvectors, and the cluster bases (whose
    // widths sum to `d`). The singular value decomposition's one working copy is freed
    // before `S` and `K` are formed.
    let working = governor
        .try_reserve_dense_f64_copies(dimension, dimension, 5, "plane-rotation recovery")
        .map_err(PlaneRotationError::Memory)?;
    let (_, singular_values, _) = matrix
        .svd(false, false)
        .map_err(PlaneRotationError::Linalg)?;
    let sigma_max = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value));
    let sigma_min = singular_values
        .iter()
        .fold(f64::INFINITY, |acc, &value| acc.min(value));
    let singular_band = factor_singular_band(dimension, dimension, sigma_max);
    // Each true singular value is within the band of its computed one: a smallest one
    // inside the band may be zero, and a computed distance plus the band covers `rho`.
    if sigma_min <= singular_band {
        return Err(PlaneRotationError::NotInvertible {
            smallest_singular_value: sigma_min,
            band: singular_band,
        });
    }
    let orthogonality_defect = singular_values
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max((value - 1.0).abs()))
        + singular_band;

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
        strict_symmetric_eigh(&symmetric, SymmetricAssembly::Mirrored, Side::Lower)
            .map_err(PlaneRotationError::Linalg)?;
    let mut order: Vec<usize> = (0..dimension).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let cosines: Vec<f64> = order.iter().map(|&index| values[index]).collect();
    let perturbation_bound =
        orthogonality_defect + formation_band + symmetric_spectrum_rounding_band(&cosines);

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
                    orthogonality_defect + formation_band + 2.0 * SQRT_2 * projector_bar,
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
    drop(working);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::test_support::test_governor;
    use crate::parameter_decomposition::test_support::{Planted, plant, projector_distance};
    use gam_linalg::decision::projector_error_bar;
    use ndarray::s;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};
    use std::f64::consts::PI;
    use std::ops::Range;

    const DIMENSION: usize = 8;

    fn recover_planted(planted: &Planted) -> PlaneRotationRecovery {
        recover_plane_rotations(test_governor(), planted.matrix.view()).expect("recovery")
    }

    /// Bound on `||O - Q_o B_o Q_o^T||_2` for the polar factor `O` the recovery certifies:
    /// `||O - W|| + ||W - Q_o B_o Q_o^T|| <= rho_bar + ` the planted matrix's own defect.
    fn planted_shift(recovery: &PlaneRotationRecovery, planted: &Planted) -> f64 {
        recovery.orthogonality_defect + planted.matrix_defect
    }

    /// The cluster spans the planted columns: within the Davis–Kahan bar about
    /// `Q_o B_o Q_o^T`, whose symmetric part is within `beta + ` [`planted_shift`] of the
    /// decomposed one, plus `||Q_k Q_k^T - Q_o,k Q_o,k^T|| <= eta_Q (2 + eta_Q)` and the
    /// measurement band.
    fn assert_spans_planted(
        recovery: &PlaneRotationRecovery,
        cluster: &RotationCluster,
        planted: &Planted,
        columns: Range<usize>,
    ) {
        let bar = if cluster.separation.is_finite() {
            projector_error_bar(
                cluster.separation,
                recovery.perturbation_bound + planted_shift(recovery, planted),
            )
        } else {
            0.0
        };
        let column_defect = planted.basis_defect * (2.0 + planted.basis_defect);
        let (distance, band) = projector_distance(
            cluster.basis.view(),
            planted.basis.slice(s![.., columns.clone()]),
        );
        assert!(
            distance <= bar + column_defect + band,
            "planted columns {columns:?}: projector distance {distance:.3e} exceeds bar \
             {bar:.3e} + column defect {column_defect:.3e} + band {band:.3e}"
        );
    }

    /// The planted cosine lies in the cluster's interval widened by [`planted_shift`]
    /// (Weyl) and by `|c - c/r| <= eta_B`.
    fn assert_cosine_admitted(
        recovery: &PlaneRotationRecovery,
        cluster: &RotationCluster,
        planted: &Planted,
        angle: f64,
    ) {
        let widen = planted_shift(recovery, planted) + planted.form_defect;
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
            assert_cosine_admitted(&recovery, cluster, &planted, angle);
            assert_spans_planted(&recovery, cluster, &planted, first..first + 2);
            let alignment = orientation_alignment(cluster, &planted, first);
            assert!(alignment > 0.0, "angle {angle}: alignment {alignment}");
        }
        let fixed = &recovery.clusters[3];
        assert!(
            matches!(fixed.kind, RotationClusterKind::Fixed { .. }),
            "{fixed:?}"
        );
        assert_eq!(fixed.basis.ncols(), 2);
        assert_spans_planted(&recovery, fixed, &planted, 6..8);
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
    }

    #[test]
    fn repeated_cosines_report_the_invariant_subspace_not_the_planes_2951() {
        let planted = plant(DIMENSION, &[0.9, 0.9, 2.0], 0, 0x2951_0002);
        let recovery = recover_planted(&planted);
        assert_eq!(recovery.clusters.len(), 3, "{recovery:?}");
        let repeated = &recovery.clusters[1];
        assert_eq!(rotation_planes(repeated), 2, "{repeated:?}");
        assert_cosine_admitted(&recovery, repeated, &planted, 0.9);
        assert_spans_planted(&recovery, repeated, &planted, 0..4);
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
            assert_spans_planted(&recovery, cluster, &distinct, first..first + 2);
            assert_cosine_admitted(&recovery, cluster, &distinct, angle);
        }
        assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
    }

    #[test]
    fn a_matrix_further_from_the_orthogonal_group_merges_cosines_it_cannot_separate_2951() {
        let planted = plant(DIMENSION, &[0.9, 1.0], 0, 0x2951_0003);
        let fine = recover_planted(&planted);
        assert_eq!(fine.clusters.len(), 3, "{fine:?}");
        assert_eq!(fine.ambiguities(), vec![RotationAmbiguity::Winding]);

        // `t W` has the polar factor of `W` and singular values `t` (up to the planted
        // defect), so `rho >= 1 - t`. At `t = 1 - gap`, the resolution `2 beta >= 2 gap`
        // exceeds the scaled cosine gap `t gap`: no pair of distinct true cosines is
        // provable, so the planes merge.
        let gap = 0.9_f64.cos() - 1.0_f64.cos();
        let scale = 1.0 - gap;
        let scaled = planted.matrix.mapv(|value| scale * value);
        let coarse = recover_plane_rotations(test_governor(), scaled.view()).expect("recovery");
        assert!(
            coarse.orthogonality_defect >= gap - scale * planted.matrix_defect,
            "rho_bar {} does not cover the scaling's distance {gap}",
            coarse.orthogonality_defect
        );
        assert!(
            2.0 * coarse.perturbation_bound > scale * gap,
            "resolution {} does not exceed the scaled gap {}",
            2.0 * coarse.perturbation_bound,
            scale * gap
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
    fn the_orthogonality_defect_covers_the_true_distance_and_a_singular_matrix_is_refused_2951() {
        // Exactly orthogonal (entries `+-1/4`): `rho = 0`, so `rho_bar` is at most the
        // singular-value band twice, the measurement's and the certificate's.
        let basis = exact_hidden_basis(0x2951_0010);
        let dimension = basis.nrows();
        let exact = recover_plane_rotations(test_governor(), basis.view()).expect("recovery");
        assert!(
            exact.orthogonality_defect
                <= 2.0
                    * factor_singular_band(dimension, dimension, 1.0 + exact.orthogonality_defect),
            "rho_bar {} of an exactly orthogonal matrix exceeds twice its band",
            exact.orthogonality_defect
        );
        let planted = plant(DIMENSION, &[1.1], 0, 0x2951_0008);
        let recovery = recover_planted(&planted);
        assert!(
            recovery.orthogonality_defect
                <= planted.matrix_defect
                    + 2.0
                        * factor_singular_band(
                            DIMENSION,
                            DIMENSION,
                            1.0 + recovery.orthogonality_defect
                        ),
            "rho_bar {} of the planted rotation exceeds its defect {}",
            recovery.orthogonality_defect,
            planted.matrix_defect
        );

        // `H diag(s)` with dyadic `s_j` is formed exactly and has singular values exactly
        // `s_j`, so `rho = max |s_j - 1| = 1/32`.
        let steps: Vec<f64> = (0..dimension)
            .map(|index| 1.0 + (index as f64 - 8.0) / 256.0)
            .collect();
        let rho = steps
            .iter()
            .fold(0.0_f64, |acc, &value| acc.max((value - 1.0).abs()));
        let perturbed = Array2::from_shape_fn((dimension, dimension), |(row, col)| {
            basis[[row, col]] * steps[col]
        });
        let measured = recover_plane_rotations(test_governor(), perturbed.view()).expect("recovery");
        let sigma_max = steps.iter().fold(0.0_f64, |acc, &value| acc.max(value));
        let band = factor_singular_band(
            dimension,
            dimension,
            sigma_max + measured.orthogonality_defect,
        );
        assert!(
            rho <= measured.orthogonality_defect
                && measured.orthogonality_defect <= rho + 2.0 * band,
            "rho_bar {} does not bracket rho {rho} within twice the band {band:.3e}",
            measured.orthogonality_defect
        );

        // `I + C (M - I) C^T` with `M = D R D^-1`, `D = diag(1, 2)` rotates the plane of
        // `C` but is not orthogonal. It is recovered through its polar factor, and
        // `rho_bar` covers its distance `sigma_max(M) - 1`, read from `||M||_F^2` and
        // `det M = 1` in closed form, up to the rounding of forming it.
        let (sine, cosine) = 1.1_f64.sin_cos();
        let generator = ndarray::arr2(&[[cosine - 1.0, -sine / 2.0], [2.0 * sine, cosine - 1.0]]);
        let columns = basis.slice(s![.., 0..2]);
        let conjugated = Array2::<f64>::eye(dimension) + columns.dot(&generator).dot(&columns.t());
        let absolute_columns = columns.mapv(f64::abs);
        let formation = accumulation_growth(5)
            * frobenius_norm(
                (Array2::<f64>::eye(dimension)
                    + absolute_columns
                        .dot(&generator.mapv(f64::abs))
                        .dot(&absolute_columns.t()))
                .view(),
            );
        let frobenius_squared = 2.0 * cosine * cosine + sine * sine / 4.0 + 4.0 * sine * sine;
        let largest = ((frobenius_squared + (frobenius_squared * frobenius_squared - 4.0).sqrt())
            / 2.0)
            .sqrt();
        let operator = recover_plane_rotations(test_governor(), conjugated.view()).expect("recovery");
        assert!(
            operator.orthogonality_defect >= largest - 1.0 - formation,
            "rho_bar {} does not cover the operator's distance {}",
            operator.orthogonality_defect,
            largest - 1.0
        );

        // A zero column leaves a singular value inside the band of zero: the polar
        // factor is not unique and the recovery refuses. Positive control: the same
        // column shrunk to a resolved singular value is recovered.
        let mut singular = basis.clone();
        singular.column_mut(3).fill(0.0);
        match recover_plane_rotations(test_governor(), singular.view()) {
            Err(PlaneRotationError::NotInvertible {
                smallest_singular_value,
                band,
            }) => assert!(
                smallest_singular_value <= band,
                "{smallest_singular_value} vs {band}"
            ),
            other => panic!("expected a singular refusal, got {other:?}"),
        }
        let mut shrunk = basis.clone();
        shrunk.column_mut(3).mapv_inplace(|value| value / 1024.0);
        let resolved = recover_plane_rotations(test_governor(), shrunk.view()).expect("recovery");
        assert!(resolved.orthogonality_defect >= 1.0 - 1.0 / 1024.0);
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
        assert_spans_planted(&recovery, half_turn, &planted, 0..2);
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
        assert_spans_planted(&recovery, axis, &reflection, 2..3);
    }

    #[test]
    fn identity_reports_no_plane_and_the_largest_angle_it_cannot_exclude_2951() {
        let identity = Array2::<f64>::eye(DIMENSION);
        let recovery = recover_plane_rotations(test_governor(), identity.view()).expect("recovery");
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
            assert_cosine_admitted(&recovery, plane, &planted, angle);
            assert_spans_planted(&recovery, plane, &planted, 0..2);
            let alignment = orientation_alignment(plane, &planted, 0);
            assert!(
                sign * alignment > 0.0,
                "angle {angle}: alignment {alignment} has the wrong sign"
            );
            assert_eq!(recovery.ambiguities(), vec![RotationAmbiguity::Winding]);
        }
    }

    #[test]
    fn orientation_is_withheld_when_the_distance_to_the_orthogonal_group_hides_the_skew_part_2951()
    {
        let dimension = 4;
        let angles = [0.995_f64.acos(), (-0.995_f64).acos()];
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

        // `0.9 W` has the same polar factor at `rho ~ 0.1`. Its cosine intervals
        // `0.9 * 0.995 +- 0.1 = [0.796, 0.996]` and their mirror stay clear of +-1, but
        // their sine floor `sqrt(1 - 0.996^2) ~ 0.095` is below the skew error
        // `rho_bar + 2 sqrt(2) bar`, so no orientation is certified.
        let scaled = planted.matrix.mapv(|value| 0.9 * value);
        let withheld = recover_plane_rotations(test_governor(), scaled.view()).expect("recovery");
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
}
