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
    /// A Gauss-Kronrod pair must extend a Gauss rule of at least one node.
    EmptyGaussKronrodRule,
    /// Laurie's recurrence produced a Jacobi matrix whose Kronrod extension is
    /// not a real positive rule interleaving the Gauss nodes on `(-1, 1)`.
    InvalidGaussKronrodRule { gauss_nodes: usize, node: usize },
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
            Self::EmptyGaussKronrodRule => {
                formatter.write_str("a Gauss-Kronrod pair needs a Gauss rule of at least one node")
            }
            Self::InvalidGaussKronrodRule { gauss_nodes, node } => write!(
                formatter,
                "the {}-node Kronrod extension of the {gauss_nodes}-node Gauss-Legendre rule is not a real positive interleaving rule at node {node}",
                2 * gauss_nodes + 1
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

/// Whether [`standard_normal_gauss_hermite_rule`] of `order` builds with every weight
/// positive (#784). A block quadrature refuses an order this rejects, so an order search
/// that raises one order at a time asks this of the next order before it raises.
///
/// One rule build, the same one the block quadrature performs at that order, so asking it
/// costs no more than the step it guards.
pub fn standard_normal_gauss_hermite_order_is_representable(order: usize) -> bool {
    standard_normal_gauss_hermite_rule(order)
        .is_ok_and(|rule| rule.iter().all(|&(_, weight)| weight > 0.0))
}

/// The largest order whose [`standard_normal_gauss_hermite_rule`] builds with every weight
/// positive, where every lower order does too (#784): the order at which a search that
/// raises one order at a time and asks
/// [`standard_normal_gauss_hermite_order_is_representable`] of the next order stops.
///
/// Measured by scanning every order from one, so no order ceiling is chosen. The scan ends
/// because the extreme weight of an `n`-node rule decays like `e^{−2n}` and underflows at a
/// few hundred nodes. The scan builds every rule up to that order, which costs most of a
/// second, so no fit calls it: it is the reference an order search's stop is checked
/// against.
pub fn max_representable_standard_normal_gauss_hermite_order() -> usize {
    static MAX_REPRESENTABLE_ORDER: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX_REPRESENTABLE_ORDER.get_or_init(|| {
        let mut order = 1usize;
        while order
            .checked_add(1)
            .is_some_and(standard_normal_gauss_hermite_order_is_representable)
        {
            order += 1;
        }
        order
    })
}

/// A Gauss-Legendre rule and its Kronrod extension on `[-1, 1]` sharing nodes (#784).
///
/// `nodes` holds the `2n+1` Kronrod nodes ascending. `kronrod_weights` integrates
/// polynomials through degree `3n+1` exactly; `gauss_weights` is aligned with `nodes`, zero
/// at the `n+1` Kronrod-only nodes and the `n`-node Gauss-Legendre weight at the rest, so
/// both estimates come from one set of integrand values and their difference measures
/// the Gauss estimate's error.
#[derive(Clone, Debug)]
pub struct GaussKronrodRule {
    pub nodes: Vec<f64>,
    pub kronrod_weights: Vec<f64>,
    pub gauss_weights: Vec<f64>,
}

/// The `(2n+1)`-node Gauss-Kronrod-Legendre rule extending the `n`-node Gauss rule.
///
/// Laurie's algorithm (Math. Comp. 66, 1997, in the form of Gautschi's `r_kronrod`)
/// turns the Legendre recurrence into the Jacobi matrix of the Kronrod rule; Golub-Welsch
/// then reads its nodes and weights from [`symmetric_tridiagonal_eigen_first_components`],
/// the same `O(n²)` path as the Gauss-Hermite rules here. The Gauss weights come from
/// [`crate::special::gauss_legendre`]; the Gauss nodes are the Kronrod nodes at the odd
/// sorted positions, which the construction checks.
pub fn gauss_kronrod_legendre_rule(
    gauss_nodes: usize,
) -> Result<GaussKronrodRule, QuadratureError> {
    let n = gauss_nodes;
    if n == 0 {
        return Err(QuadratureError::EmptyGaussKronrodRule);
    }
    let size = 2 * n + 1;
    // Legendre recurrence on [-1, 1]: alpha_k = 0, beta_0 = mass 2, beta_k = k²/(4k² − 1).
    // Laurie needs the first floor(3n/2)+1 alphas and ceil(3n/2)+1 betas.
    let mut alpha = vec![0.0_f64; size];
    let mut beta = vec![0.0_f64; size];
    for (k, value) in beta.iter_mut().enumerate().take((3 * n).div_ceil(2) + 1) {
        *value = if k == 0 {
            2.0
        } else {
            let k = k as f64;
            k * k / (4.0 * k * k - 1.0)
        };
    }
    let mut s = vec![0.0_f64; n / 2 + 2];
    let mut t = vec![0.0_f64; n / 2 + 2];
    t[1] = beta[n + 1];
    let mut terms = Vec::with_capacity(n);
    for m in 0..n.saturating_sub(1) {
        terms.clear();
        for k in (0..=(m + 1) / 2).rev() {
            let l = m - k;
            terms.push((
                k,
                (alpha[k + n + 1] - alpha[l]) * t[k + 1] + beta[k + n + 1] * s[k]
                    - beta[l] * s[k + 1],
            ));
        }
        let mut running = 0.0;
        for &(k, term) in &terms {
            running += term;
            s[k + 1] = running;
        }
        std::mem::swap(&mut s, &mut t);
    }
    for j in (0..=n / 2).rev() {
        s[j + 1] = s[j];
    }
    for m in (n - 1)..(2 * n).saturating_sub(2) {
        terms.clear();
        for k in (m + 1 - n)..=((m - 1) / 2) {
            let l = m - k;
            let j = n - 1 - l;
            terms.push((
                j,
                -(alpha[k + n + 1] - alpha[l]) * t[j + 1] - beta[k + n + 1] * s[j + 1]
                    + beta[l] * s[j + 2],
            ));
        }
        let mut running = 0.0;
        for &(j, term) in &terms {
            running += term;
            s[j + 1] = running;
        }
        let last = terms.last().map(|&(j, _)| j).expect("non-empty Laurie sweep");
        if m % 2 == 0 {
            let k = m / 2;
            alpha[k + n + 1] = alpha[k] + (s[last + 1] - beta[k + n + 1] * s[last + 2]) / t[last + 2];
        } else {
            let k = (m + 1) / 2;
            beta[k + n + 1] = s[last + 1] / s[last + 2];
        }
        std::mem::swap(&mut s, &mut t);
    }
    alpha[2 * n] = alpha[n - 1] - beta[2 * n] * s[1] / t[1];
    let invalid = |node: usize| QuadratureError::InvalidGaussKronrodRule { gauss_nodes: n, node };
    let mut off_diagonal = Vec::with_capacity(size - 1);
    for (index, &value) in beta[1..].iter().enumerate() {
        if !(value > 0.0) {
            return Err(invalid(index + 1));
        }
        off_diagonal.push(value.sqrt());
    }
    let (nodes, first_components) =
        symmetric_tridiagonal_eigen_first_components(&alpha, &off_diagonal)?;
    let mut pairs = nodes
        .into_iter()
        .zip(first_components)
        .map(|(node, first)| (node, beta[0] * first * first))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let (gauss_abscissae, gauss_legendre_weights) = crate::special::gauss_legendre(n);
    let mut gauss_weights = vec![0.0; size];
    for (index, (&abscissa, &weight)) in gauss_abscissae
        .iter()
        .zip(gauss_legendre_weights.iter())
        .enumerate()
    {
        let shared = 2 * index + 1;
        // Both constructions resolve a shared node to a few ulps of the unit interval.
        if (pairs[shared].0 - abscissa).abs() > (size as f64) * f64::EPSILON {
            return Err(invalid(shared));
        }
        gauss_weights[shared] = weight;
    }
    for (index, &(node, weight)) in pairs.iter().enumerate() {
        let inside = node > -1.0 && node < 1.0;
        let ascending = index == 0 || node > pairs[index - 1].0;
        if !(inside && ascending && weight > 0.0) {
            return Err(invalid(index));
        }
    }
    Ok(GaussKronrodRule {
        nodes: pairs.iter().map(|&(node, _)| node).collect(),
        kronrod_weights: pairs.into_iter().map(|(_, weight)| weight).collect(),
        gauss_weights,
    })
}

/// The standard normal restricted to `(lower, upper)`, `lower < 0 < upper`, and the
/// map that transports a standard-normal rule onto it.
///
/// With `Z = Φ(upper) − Φ(lower)`, the map `u ↦ ψ(u)` solves
/// `Φ(ψ) = Φ(lower) + Z·Φ(u)`. It pushes `N(0,1)` forward onto the normal truncated to
/// the interval, so for any `f`
///
/// ```text
/// ∫_lower^upper φ(x) f(x) dx = Z · E_{u∼N(0,1)}[ f(ψ(u)) ],
/// ```
///
/// and a Gauss–Hermite rule on the right is a rule for the left. `ψ` is analytic on
/// the whole line and reaches the ends only as `u → ±∞`, so an integrand with a root
/// or a jump at an end of the interval becomes smooth in `u`, where Gauss–Hermite
/// converges geometrically in its order instead of algebraically.
///
/// A node with `u ≤ 0` is inverted through the lower tail `Φ(lower) + Z·Φ(u)` and a
/// node with `u > 0` through the upper tail `(1 − Φ(upper)) + Z·(1 − Φ(u))`, each in
/// log space. Both probabilities are below `3/4`, so neither inversion reads a
/// probability that has rounded toward one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TruncatedNormalTransport {
    lower: f64,
    upper: f64,
    log_mass: f64,
    log_cdf_lower: f64,
    log_sf_upper: f64,
}

impl TruncatedNormalTransport {
    /// The transport onto `(lower, upper)`. Either end may be infinite; the interval
    /// must contain zero in its interior.
    pub fn new(lower: f64, upper: f64) -> Result<Self, String> {
        if !(lower < 0.0 && upper > 0.0) {
            return Err(format!(
                "a truncated-normal transport needs lower < 0 < upper, got ({lower}, {upper})"
            ));
        }
        // Z = ½(erf(upper/√2) + erf(−lower/√2)): both terms are positive, so the
        // mass is formed without cancellation however close either end is to zero.
        let mass = 0.5
            * (libm::erf(upper / std::f64::consts::SQRT_2)
                + libm::erf(-lower / std::f64::consts::SQRT_2));
        Ok(Self {
            lower,
            upper,
            log_mass: mass.ln(),
            log_cdf_lower: crate::probability::normal_logcdf(lower),
            log_sf_upper: crate::probability::normal_logsf(upper),
        })
    }

    pub fn lower(&self) -> f64 {
        self.lower
    }

    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// `ln Z`, the log of the standard-normal mass of the interval.
    pub fn log_mass(&self) -> f64 {
        self.log_mass
    }

    /// `ψ(u)`, the image of the standard-normal point `u` in the interval.
    pub fn transport(&self, u: f64) -> Result<f64, String> {
        use crate::probability::{
            normal_logcdf, normal_logsf, standard_normal_quantile_from_log_cdf,
        };
        use crate::special::logaddexp;
        if u <= 0.0 {
            let log_cdf = logaddexp(self.log_cdf_lower, self.log_mass + normal_logcdf(u));
            standard_normal_quantile_from_log_cdf(log_cdf)
        } else {
            let log_sf = logaddexp(self.log_sf_upper, self.log_mass + normal_logsf(u));
            standard_normal_quantile_from_log_cdf(log_sf).map(|x| -x)
        }
    }

    /// `(∂ψ/∂lower, ∂ψ/∂upper)` at the node `u` with image `psi = ψ(u)`, from
    /// differentiating `Φ(ψ) = Φ(lower) + Z·Φ(u)`:
    ///
    /// ```text
    /// ∂ψ/∂lower = φ(lower)·(1 − Φ(u)) / φ(ψ),   ∂ψ/∂upper = φ(upper)·Φ(u) / φ(ψ).
    /// ```
    ///
    /// An infinite end does not move, and its sensitivity is zero.
    pub fn endpoint_sensitivities(&self, u: f64, psi: f64) -> (f64, f64) {
        use crate::probability::{normal_logcdf, normal_logsf};
        let half_psi_sq = 0.5 * psi * psi;
        let lower = if self.lower.is_finite() {
            (half_psi_sq - 0.5 * self.lower * self.lower + normal_logsf(u)).exp()
        } else {
            0.0
        };
        let upper = if self.upper.is_finite() {
            (half_psi_sq - 0.5 * self.upper * self.upper + normal_logcdf(u)).exp()
        } else {
            0.0
        };
        (lower, upper)
    }

    /// `(∂ ln Z/∂lower, ∂ ln Z/∂upper) = (−φ(lower)/Z, φ(upper)/Z)`.
    pub fn log_mass_endpoint_gradient(&self) -> (f64, f64) {
        let density_over_mass = |end: f64| {
            if end.is_finite() {
                (crate::probability::normal_pdf(end).ln() - self.log_mass).exp()
            } else {
                0.0
            }
        };
        (
            -density_over_mass(self.lower),
            density_over_mass(self.upper),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `E[f | lower < X < upper]·Z` by a transported order-`order` rule.
    fn transported_integral(
        transport: &TruncatedNormalTransport,
        order: usize,
        f: impl Fn(f64) -> f64,
    ) -> f64 {
        let rule = standard_normal_gauss_hermite_rule(order).expect("standard-normal rule");
        let mean: f64 = rule
            .iter()
            .map(|&(u, w)| w * f(transport.transport(u).expect("transported node")))
            .sum();
        mean * transport.log_mass().exp()
    }

    #[test]
    fn transported_rule_integrates_a_root_at_the_cut_geometrically() {
        // ∫_a^∞ φ(x)·√(x − a) dx has a square-root kink at the cut. Over the whole
        // line with the indicator the rule converges algebraically; transported onto
        // (a, ∞) the integrand is analytic in u. Reference: the transported rule at
        // twice the order, which the geometric rate makes exact to roundoff.
        let a = -0.7;
        let transport = TruncatedNormalTransport::new(a, f64::INFINITY).expect("transport");
        let f = |x: f64| (x - a).sqrt();
        let reference = transported_integral(&transport, 160, f);
        let transported = transported_integral(&transport, 40, f);
        assert!(
            (transported - reference).abs() <= 1e-9,
            "order-40 transported rule {transported} vs reference {reference}"
        );
        let rule = standard_normal_gauss_hermite_rule(160).expect("rule");
        let indicator: f64 = rule
            .iter()
            .filter(|&&(u, _)| u > a)
            .map(|&(u, w)| w * f(u))
            .sum();
        assert!(
            (indicator - reference).abs() > 1e-5,
            "the untransported order-160 rule {indicator} should still miss {reference}"
        );
    }

    #[test]
    fn transported_rule_recovers_truncated_normal_moments() {
        // E[X·1(a<X<b)] = φ(a) − φ(b) and the mass Φ(b) − Φ(a).
        let (a, b) = (-1.3, 0.4);
        let transport = TruncatedNormalTransport::new(a, b).expect("transport");
        let pdf = crate::probability::normal_pdf;
        let cdf = crate::probability::normal_cdf;
        // Both ends cut: the image of the Gaussian tails flattens onto the ends, so
        // the rate is geometric but slower than one-sided (≈1e-9 at order 30,
        // ≈1e-12 at 60, roundoff by 120).
        let mass = transported_integral(&transport, 120, |_| 1.0);
        assert!((mass - (cdf(b) - cdf(a))).abs() <= 1e-15);
        let first = transported_integral(&transport, 120, |x| x);
        assert!(
            (first - (pdf(a) - pdf(b))).abs() <= 1e-13,
            "first moment {first} vs {}",
            pdf(a) - pdf(b)
        );
        // The map is monotone and stays inside the interval. The outermost nodes of a
        // high order rule sit within roundoff of an end, so their images may tie or
        // land an ulp past it; the block integrand gives such a node zero weight,
        // which is its value at the end.
        let rule = standard_normal_gauss_hermite_rule(60).expect("rule");
        let images: Vec<f64> = rule
            .iter()
            .map(|&(u, _)| transport.transport(u).expect("node"))
            .collect();
        assert!(images.windows(2).all(|pair| pair[0] <= pair[1]));
        let ulp = |end: f64| 2.0 * f64::EPSILON * end.abs();
        assert!(
            images
                .iter()
                .all(|&x| a - ulp(a) <= x && x <= b + ulp(b)),
            "images {:?}",
            (images[0] - a, images[images.len() - 1] - b)
        );
    }

    #[test]
    fn endpoint_sensitivities_match_the_moved_map() {
        let (a, b) = (-0.9, 1.6);
        let transport = TruncatedNormalTransport::new(a, b).expect("transport");
        let h = 1e-6;
        let moved_lower = TruncatedNormalTransport::new(a + h, b).expect("moved lower");
        let moved_lower_back = TruncatedNormalTransport::new(a - h, b).expect("moved lower");
        let moved_upper = TruncatedNormalTransport::new(a, b + h).expect("moved upper");
        let moved_upper_back = TruncatedNormalTransport::new(a, b - h).expect("moved upper");
        for u in [-4.0, -1.2, -0.1, 0.0, 0.3, 2.5, 5.0] {
            let psi = transport.transport(u).expect("node");
            let (d_lower, d_upper) = transport.endpoint_sensitivities(u, psi);
            let fd_lower = (moved_lower.transport(u).unwrap()
                - moved_lower_back.transport(u).unwrap())
                / (2.0 * h);
            let fd_upper = (moved_upper.transport(u).unwrap()
                - moved_upper_back.transport(u).unwrap())
                / (2.0 * h);
            assert!((d_lower - fd_lower).abs() <= 1e-7, "u={u}: {d_lower} vs {fd_lower}");
            assert!((d_upper - fd_upper).abs() <= 1e-7, "u={u}: {d_upper} vs {fd_upper}");
        }
        let (g_lower, g_upper) = transport.log_mass_endpoint_gradient();
        let fd_lower = (moved_lower.log_mass() - moved_lower_back.log_mass()) / (2.0 * h);
        let fd_upper = (moved_upper.log_mass() - moved_upper_back.log_mass()) / (2.0 * h);
        assert!((g_lower - fd_lower).abs() <= 1e-8);
        assert!((g_upper - fd_upper).abs() <= 1e-8);
    }

    #[test]
    fn seven_point_gauss_kronrod_pair_matches_the_quadpack_rule_784() {
        // QUADPACK's qk15 abscissae and weights (the non-negative half; the rule is symmetric).
        let nodes = [
            0.991_455_371_120_812_639,
            0.949_107_912_342_758_525,
            0.864_864_423_359_769_073,
            0.741_531_185_599_394_440,
            0.586_087_235_467_691_130,
            0.405_845_151_377_397_167,
            0.207_784_955_007_898_468,
            0.0,
        ];
        let kronrod = [
            0.022_935_322_010_529_225,
            0.063_092_092_629_978_553,
            0.104_790_010_322_250_184,
            0.140_653_259_715_525_919,
            0.169_004_726_639_267_903,
            0.190_350_578_064_785_410,
            0.204_432_940_075_298_892,
            0.209_482_141_084_727_828,
        ];
        let gauss = [
            0.0,
            0.129_484_966_168_869_693,
            0.0,
            0.279_705_391_489_276_668,
            0.0,
            0.381_830_050_505_118_945,
            0.0,
            0.417_959_183_673_469_388,
        ];
        let rule = gauss_kronrod_legendre_rule(7).expect("G7K15");
        assert_eq!(rule.nodes.len(), 15);
        for (offset, index) in (7..15).rev().enumerate() {
            assert!((rule.nodes[index] - nodes[offset]).abs() <= 1e-14, "node {index}");
            assert!((rule.nodes[14 - index] + nodes[offset]).abs() <= 1e-14, "node {}", 14 - index);
            assert!((rule.kronrod_weights[index] - kronrod[offset]).abs() <= 1e-14);
            assert!((rule.gauss_weights[index] - gauss[offset]).abs() <= 1e-14);
            assert_eq!(rule.gauss_weights[index], rule.gauss_weights[14 - index]);
        }
    }

    #[test]
    fn gauss_kronrod_pairs_integrate_their_exact_degrees_784() {
        // The Kronrod rule is exact through degree 3n+1, the embedded Gauss rule through 2n-1,
        // and at degree 2n the Gauss rule misses by exactly the squared norm of the monic
        // Legendre polynomial, 2·prod_k k²/(2k−1)² / (2n+1): the error the pair measures.
        for n in 1..=30 {
            let rule = gauss_kronrod_legendre_rule(n).expect("Gauss-Kronrod pair");
            let moment = |degree: usize| {
                if degree % 2 == 1 { 0.0 } else { 2.0 / (degree as f64 + 1.0) }
            };
            let integrate = |weights: &[f64], degree: usize| {
                rule.nodes
                    .iter()
                    .zip(weights)
                    .map(|(&x, &w)| w * x.powi(degree as i32))
                    .sum::<f64>()
            };
            for degree in 0..=3 * n + 1 {
                let k = integrate(&rule.kronrod_weights, degree);
                assert!((k - moment(degree)).abs() <= 1e-13, "K{} degree {degree}: {k}", 2 * n + 1);
            }
            for degree in 0..2 * n {
                let g = integrate(&rule.gauss_weights, degree);
                assert!((g - moment(degree)).abs() <= 1e-13, "G{n} degree {degree}: {g}");
            }
            let monic_norm = 2.0
                * (1..=n)
                    .map(|k| (k * k) as f64 / ((2 * k - 1) * (2 * k - 1)) as f64)
                    .product::<f64>()
                / (2 * n + 1) as f64;
            let g = integrate(&rule.gauss_weights, 2 * n);
            let miss = moment(2 * n) - g;
            assert!((miss - monic_norm).abs() <= 1e-13, "G{n} misses degree {} by {miss}, expected {monic_norm}", 2 * n);
        }
        assert!(matches!(
            gauss_kronrod_legendre_rule(0),
            Err(QuadratureError::EmptyGaussKronrodRule)
        ));
    }

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
        assert!(
            (1..=ceiling).all(standard_normal_gauss_hermite_order_is_representable),
            "every order through the ceiling {ceiling} must be representable"
        );
        assert!(
            !standard_normal_gauss_hermite_order_is_representable(ceiling + 1),
            "the order past the ceiling {ceiling} must not be representable"
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
