//! Deterministic quadrature primitives shared across model and solver crates.
//!
//! Gaussian quadrature needs only the eigenvalues of a symmetric tridiagonal
//! Jacobi matrix and the first component of each eigenvector.  Carrying a full
//! dense eigenvector matrix makes an `n`-node rule consume `O(n²)` storage,
//! while embedding that tridiagonal in a dense matrix adds an avoidable
//! `O(n³)` eigendecomposition.  The implicit-shift QL routine here carries only
//! the first row of the accumulated eigenvector matrix: `O(n²)` arithmetic and
//! `O(n)` storage.

use std::fmt;

/// A failure to construct a deterministic quadrature rule.
#[derive(Clone, Debug, PartialEq)]
pub enum QuadratureError {
    /// A symmetric tridiagonal has `n` diagonal and exactly `n - 1`
    /// off-diagonal entries.
    InvalidTridiagonalShape {
        diagonal: usize,
        off_diagonal: usize,
    },
    /// Eigenvalue iteration requires finite matrix entries.
    NonFiniteEntry {
        diagonal: bool,
        index: usize,
        value: f64,
    },
    /// The implicit-shift QL iteration did not deflate one eigenvalue within
    /// its dimension-derived work bound.
    EigenIterationDidNotConverge {
        eigenvalue: usize,
        iterations: usize,
    },
    /// A Gauss-Hermite rule must contain at least one node.
    EmptyGaussHermiteRule,
    /// Golub-Welsch produced weights that cannot represent the Hermite mass.
    InvalidGaussHermiteMass { mass: f64 },
}

impl fmt::Display for QuadratureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTridiagonalShape {
                diagonal,
                off_diagonal,
            } => write!(
                formatter,
                "symmetric tridiagonal shape mismatch: {diagonal} diagonal entries require {}, got {off_diagonal}",
                diagonal.saturating_sub(1)
            ),
            Self::NonFiniteEntry {
                diagonal,
                index,
                value,
            } => write!(
                formatter,
                "symmetric tridiagonal {}[{index}] is non-finite: {value}",
                if *diagonal {
                    "diagonal"
                } else {
                    "off-diagonal"
                }
            ),
            Self::EigenIterationDidNotConverge {
                eigenvalue,
                iterations,
            } => write!(
                formatter,
                "symmetric tridiagonal QL failed to converge for eigenvalue {eigenvalue} after {iterations} iterations"
            ),
            Self::EmptyGaussHermiteRule => {
                formatter.write_str("Gauss-Hermite quadrature needs at least one node")
            }
            Self::InvalidGaussHermiteMass { mass } => write!(
                formatter,
                "Gauss-Hermite Golub-Welsch weights have invalid mass {mass}"
            ),
        }
    }
}

impl std::error::Error for QuadratureError {}

impl From<QuadratureError> for String {
    fn from(error: QuadratureError) -> Self {
        error.to_string()
    }
}

/// Eigenvalues and first eigenvector components of a symmetric tridiagonal.
///
/// For `T = Q diag(lambda) Q'`, the returned vectors contain `lambda_i` and
/// `Q[0, i]` in matching (not necessarily sorted) order.  This is precisely the
/// spectral information used by Golub-Welsch and Lanczos quadrature.
pub(crate) fn symmetric_tridiagonal_eigen_first_components(
    diagonal: &[f64],
    off_diagonal: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), QuadratureError> {
    let dimension = diagonal.len();
    let expected_off_diagonal = dimension.saturating_sub(1);
    if off_diagonal.len() != expected_off_diagonal {
        return Err(QuadratureError::InvalidTridiagonalShape {
            diagonal: dimension,
            off_diagonal: off_diagonal.len(),
        });
    }
    if let Some((index, &value)) = diagonal
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(QuadratureError::NonFiniteEntry {
            diagonal: true,
            index,
            value,
        });
    }
    if let Some((index, &value)) = off_diagonal
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(QuadratureError::NonFiniteEntry {
            diagonal: false,
            index,
            value,
        });
    }
    if dimension == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut eigenvalues = diagonal.to_vec();
    // The implicit QL recurrence reads one sentinel beyond the physical
    // off-diagonal.  Keeping that zero explicitly removes boundary branches
    // from the rotations.
    let mut off = vec![0.0; dimension];
    off[..expected_off_diagonal].copy_from_slice(off_diagonal);
    let mut first_components = vec![0.0; dimension];
    first_components[0] = 1.0;

    // LAPACK's symmetric-tridiagonal eigensolvers bound total iteration by a
    // constant multiple of matrix dimension.  Applying that same 30*n work
    // envelope to each active block makes the guard scale with the problem
    // rather than introducing a quadrature-order ceiling.
    let maximum_iterations = 30usize.saturating_mul(dimension.max(1));
    for left in 0..dimension {
        let mut iterations = 0usize;
        loop {
            let mut split = dimension - 1;
            for index in left..dimension - 1 {
                let local_scale = eigenvalues[index].abs() + eigenvalues[index + 1].abs();
                if off[index].abs() <= f64::EPSILON * local_scale {
                    split = index;
                    break;
                }
            }
            if split == left {
                break;
            }
            iterations += 1;
            if iterations > maximum_iterations {
                return Err(QuadratureError::EigenIterationDidNotConverge {
                    eigenvalue: left,
                    iterations,
                });
            }

            let mut shift_coordinate =
                (eigenvalues[left + 1] - eigenvalues[left]) / (2.0 * off[left]);
            let mut radius = shift_coordinate.hypot(1.0);
            shift_coordinate = eigenvalues[split] - eigenvalues[left]
                + off[left]
                    / (shift_coordinate + radius.copysign(shift_coordinate));
            let (mut sine, mut cosine) = (1.0, 1.0);
            let mut diagonal_correction = 0.0;
            let mut deflated_inside_sweep = false;

            for index in (left..split).rev() {
                let mut rotated_off = sine * off[index];
                let preserved_off = cosine * off[index];
                radius = rotated_off.hypot(shift_coordinate);
                off[index + 1] = radius;
                if radius == 0.0 {
                    eigenvalues[index + 1] -= diagonal_correction;
                    off[split] = 0.0;
                    deflated_inside_sweep = true;
                    break;
                }
                sine = rotated_off / radius;
                cosine = shift_coordinate / radius;
                shift_coordinate = eigenvalues[index + 1] - diagonal_correction;
                radius = (eigenvalues[index] - shift_coordinate) * sine
                    + 2.0 * cosine * preserved_off;
                diagonal_correction = sine * radius;
                eigenvalues[index + 1] = shift_coordinate + diagonal_correction;
                shift_coordinate = cosine * radius - preserved_off;

                // Carry only row zero of Q through the rotations.  Quadrature
                // weights never inspect any other eigenvector component.
                rotated_off = first_components[index + 1];
                first_components[index + 1] =
                    sine * first_components[index] + cosine * rotated_off;
                first_components[index] =
                    cosine * first_components[index] - sine * rotated_off;
            }
            if deflated_inside_sweep {
                continue;
            }
            eigenvalues[left] -= diagonal_correction;
            off[left] = shift_coordinate;
            off[split] = 0.0;
        }
    }

    Ok((eigenvalues, first_components))
}

/// Physicists' Gauss-Hermite rule for `integral exp(-x²) f(x) dx`.
#[derive(Clone, Debug)]
pub struct GaussHermiteRule {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

/// Construct an `n`-node physicists' Gauss-Hermite rule by Golub-Welsch.
pub fn gauss_hermite_rule(node_count: usize) -> Result<GaussHermiteRule, QuadratureError> {
    if node_count == 0 {
        return Err(QuadratureError::EmptyGaussHermiteRule);
    }
    let diagonal = vec![0.0; node_count];
    let off_diagonal = (1..node_count)
        .map(|index| ((index as f64) * 0.5).sqrt())
        .collect::<Vec<_>>();
    let (nodes, first_components) =
        symmetric_tridiagonal_eigen_first_components(&diagonal, &off_diagonal)?;
    let hermite_mass = std::f64::consts::PI.sqrt();
    let mut pairs = nodes
        .into_iter()
        .zip(first_components)
        .map(|(node, first)| (node, hermite_mass * first * first))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mass = pairs.iter().map(|(_, weight)| weight).sum::<f64>();
    if !(mass.is_finite() && mass > 0.0) {
        return Err(QuadratureError::InvalidGaussHermiteMass { mass });
    }
    let normalization = hermite_mass / mass;
    Ok(GaussHermiteRule {
        nodes: pairs.iter().map(|(node, _)| *node).collect(),
        weights: pairs
            .into_iter()
            .map(|(_, weight)| weight * normalization)
            .collect(),
    })
}

/// Gauss-Hermite rule for the standard normal weight, as `(node, weight)` pairs. The nodes
/// are `√2·x_i` and the weights `w_i/√π` of the physicists' rule, so the weights sum to one
/// and the rule integrates polynomials of degree `2n−1` exactly against `N(0,1)`.
pub fn standard_normal_gauss_hermite_rule(
    node_count: usize,
) -> Result<Vec<(f64, f64)>, QuadratureError> {
    let rule = gauss_hermite_rule(node_count)?;
    let sqrt_pi = std::f64::consts::PI.sqrt();
    Ok(rule
        .nodes
        .iter()
        .zip(rule.weights.iter())
        .map(|(&node, &weight)| (std::f64::consts::SQRT_2 * node, weight / sqrt_pi))
        .collect())
}

/// The largest order whose [`standard_normal_gauss_hermite_rule`] builds with every weight
/// positive, where every lower order does too (#784). A block quadrature refuses the next
/// order, so an order search that raises one order at a time stops exactly here.
///
/// Measured once per process from this arithmetic and this rule builder, so no order
/// ceiling is chosen. The scan ends because the extreme weight of an `n`-node rule decays
/// like `e^{−2n}` and underflows at a few hundred nodes.
pub fn max_representable_standard_normal_gauss_hermite_order() -> usize {
    static MAX_REPRESENTABLE_ORDER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX_REPRESENTABLE_ORDER.get_or_init(|| {
        let representable = |order: usize| {
            standard_normal_gauss_hermite_rule(order)
                .is_ok_and(|rule| rule.iter().all(|&(_, weight)| weight > 0.0))
        };
        let mut order = 1usize;
        while order
            .checked_add(1)
            .is_some_and(|next| representable(next))
        {
            order += 1;
        }
        order
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_normal_rule_integrates_normal_moments_through_degree_nine_784() {
        // E[z^k] against N(0,1): 1, 0, 1, 0, 3, 0, 15, 0, 105, 0. A five-node rule is exact
        // through degree nine, so every one of these moments must come back to roundoff.
        let rule = standard_normal_gauss_hermite_rule(5).expect("five-node standard-normal rule");
        let expected = [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0, 105.0, 0.0];
        for (degree, &moment) in expected.iter().enumerate() {
            let integrated: f64 = rule
                .iter()
                .map(|&(node, weight)| weight * node.powi(degree as i32))
                .sum();
            assert!(
                (integrated - moment).abs() <= 1e-11 * (1.0 + moment),
                "E[z^{degree}] by the five-node rule is {integrated}, expected {moment}"
            );
        }
    }

    #[test]
    fn the_representable_order_ceiling_is_where_a_weight_first_stops_being_positive_784() {
        let ceiling = max_representable_standard_normal_gauss_hermite_order();
        assert!(
            ceiling > 4,
            "the ceiling {ceiling} must sit above the order search's starting order four"
        );
        let at_ceiling = standard_normal_gauss_hermite_rule(ceiling).expect("rule at the ceiling");
        assert!(
            at_ceiling.iter().all(|&(_, weight)| weight > 0.0),
            "every weight of the order-{ceiling} rule must be positive"
        );
        let past_ceiling = standard_normal_gauss_hermite_rule(ceiling + 1);
        assert!(
            past_ceiling
                .as_ref()
                .map_or(true, |rule| rule.iter().any(|&(_, weight)| !(weight > 0.0))),
            "the order-{} rule must carry a non-positive weight or fail to build",
            ceiling + 1
        );
    }

    #[test]
    fn seven_node_gauss_hermite_matches_reference_rule() {
        let expected_nodes = [
            -2.651_961_356_835_233_4,
            -1.673_551_628_767_471_4,
            -0.816_287_882_858_964_7,
            0.0,
            0.816_287_882_858_964_7,
            1.673_551_628_767_471_4,
            2.651_961_356_835_233_4,
        ];
        let expected_weights = [
            0.000_971_781_245_099_519_1,
            0.054_515_582_819_127_03,
            0.425_607_252_610_127_8,
            0.810_264_617_556_807_3,
            0.425_607_252_610_127_8,
            0.054_515_582_819_127_03,
            0.000_971_781_245_099_519_1,
        ];
        let rule = gauss_hermite_rule(7).expect("seven-node rule");
        for index in 0..7 {
            assert!((rule.nodes[index] - expected_nodes[index]).abs() <= 1.0e-12);
            assert!((rule.weights[index] - expected_weights[index]).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn tridiagonal_shape_and_finiteness_are_validated() {
        assert!(matches!(
            symmetric_tridiagonal_eigen_first_components(&[0.0, 0.0], &[]),
            Err(QuadratureError::InvalidTridiagonalShape { .. })
        ));
        assert!(matches!(
            symmetric_tridiagonal_eigen_first_components(&[f64::NAN], &[]),
            Err(QuadratureError::NonFiniteEntry { diagonal: true, .. })
        ));
    }
}
