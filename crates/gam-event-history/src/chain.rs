//! Gauss-Hermite–Lagrange representation of one subject's latent chain.
//!
//! The latent state at every node is a product of independent unit-variance
//! atoms. Every density or likelihood-weighted function on the state is
//! carried as its values on an adaptive product Gauss-Hermite grid whose
//! per-axis centre and scale follow the predicted moments. The two operators
//! the chain needs — predict forward across a gap and condition backward
//! across a gap — are Gaussian convolutions. The product of the transition
//! kernel and a Gaussian is again a Gaussian, so each convolution is an
//! expectation under a Gaussian of an interpolated residual, evaluated by
//! Gauss-Hermite quadrature at inner points; this holds for gaps of any size,
//! including gaps far shorter than the grid spacing where direct kernel
//! quadrature fails.
//!
//! Both operators interpolate a logarithm. Forward ([`ForwardKernel`]), the
//! operand is a density, carried by its logarithm: `ln(density / envelope)`
//! is interpolated by the tensor Lagrange polynomial on the nodes (continued
//! beyond the hull by the quadratic through the end nodes) and exponentiated
//! at the inner points, so every predicted density is a sum of positive terms
//! and exact for any Gaussian density. Backward, the operand
//! is the logarithm of a smoother residual, bounded and smooth, carried by
//! the not-a-knot cubic spline through the nodes, continued linearly beyond
//! the hull with the end slope. No density is ever a signed interpolant, so
//! neither pass can lose positivity at any order.
//!
//! A conditional expectation of a carried function (not a density) given the
//! next state is a Lagrange interpolant of its values averaged under the
//! forward inner weights. The Lebesgue constant of that interpolant on
//! Hermite nodes grows exponentially with the order, so the rule records it
//! and the fit refuses an order whose roundoff amplification would exceed the
//! certificate's tolerance.
//!
//! Everything is generic over a [`JetField`] scalar so the same code yields
//! the value, one directional derivative, or a mixed second directional
//! derivative of every quantity downstream.

use super::cohort::EventHistoryError;
use super::scalar::{add_real, div, exp, ln, recip, sqrt, square};
use gam_math::nested_dual::JetField;

/// Physicists' Gauss-Hermite rule with the derived constants the chain uses.
#[derive(Clone, Debug)]
pub(crate) struct GaussHermite {
    pub order: usize,
    /// Nodes `x_l` of `∫ e^{-x²} f(x) dx ≈ Σ w_l f(x_l)`.
    pub nodes: Vec<f64>,
    /// `w_l / √π`: weights of the standard-normal expectation
    /// `E[f(Z)] = Σ (w_l/√π) f(√2 x_l)`.
    pub normal_weights: Vec<f64>,
    /// `w_l e^{x_l²}`: weights of plain integration `∫ g(x) dx ≈ Σ (w_l e^{x_l²}) g(x_l)`.
    pub plain_weights: Vec<f64>,
    /// Lagrange weights `1 / Π_{m≠i} (x_i − x_m)`, so that
    /// `L_i(x) = lagrange_weights[i] · Π_{m≠i} (x − x_m)`.
    pub lagrange_weights: Vec<f64>,
    /// Second derivatives at the nodes of the cardinal not-a-knot cubic
    /// splines: `spline_second[j * order + m]` is `S_j''(x_m)` for the spline
    /// that is one at node `j` and zero at every other node (all zero below
    /// order four, where the spline is the broken line).
    pub spline_second: Vec<f64>,
    /// The Lebesgue constant `max_x Σ_i |L_i(x)|` of Lagrange interpolation
    /// on the nodes over their hull: the factor by which nodal roundoff is
    /// amplified by the forward operator's interpolant.
    pub lebesgue_constant: f64,
}

impl GaussHermite {
    pub fn new(order: usize) -> Result<Self, EventHistoryError> {
        if order == 0 {
            return Err(EventHistoryError::InvalidInput {
                reason: "Gauss-Hermite order must be positive".to_string(),
            });
        }
        let rule = gam_math::quadrature::gauss_hermite_rule(order).map_err(|error| {
            EventHistoryError::NumericalFailure {
                reason: format!("Gauss-Hermite rule of order {order} failed: {error}"),
            }
        })?;
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let nodes = rule.nodes.clone();
        let normal_weights: Vec<f64> = rule.weights.iter().map(|w| w / sqrt_pi).collect();
        let plain_weights: Vec<f64> = rule
            .weights
            .iter()
            .zip(nodes.iter())
            .map(|(w, x)| (w.ln() + x * x).exp())
            .collect();
        let mut lagrange_weights = vec![0.0; order];
        for i in 0..order {
            let mut log_weight = 0.0;
            let mut sign = 1.0;
            for m in 0..order {
                if m != i {
                    let d = nodes[i] - nodes[m];
                    log_weight -= d.abs().ln();
                    if d < 0.0 {
                        sign = -sign;
                    }
                }
            }
            lagrange_weights[i] = sign * log_weight.exp();
        }
        if lagrange_weights.iter().any(|w| !w.is_finite() || *w == 0.0) {
            return Err(EventHistoryError::NumericalFailure {
                reason: format!(
                    "Gauss-Hermite order {order} is too large for the Lagrange product form"
                ),
            });
        }
        // Cardinal not-a-knot cubic splines: one tridiagonal solve per node.
        // The third derivative is continuous at the second and second-to-last
        // nodes, which keeps fourth-order accuracy up to the boundary (a
        // natural spline's zero end curvature is only second-order there).
        let mut spline_second = vec![0.0; order * order];
        if order >= 4 {
            let h: Vec<f64> = nodes.windows(2).map(|w| w[1] - w[0]).collect();
            let last = order - 1;
            for j in 0..order {
                let unit = |m: usize| if m == j { 1.0 } else { 0.0 };
                // Unknowns M_1 ..= M_{order-2}; M_0 and M_last are eliminated
                // through the not-a-knot conditions
                //   M_0 = M_1 (1 + h_0/h_1) − M_2 h_0/h_1,
                //   M_last = M_{last-1} (1 + h_{last-1}/h_{last-2}) − M_{last-2} h_{last-1}/h_{last-2}.
                let interior = order - 2;
                let mut lower = vec![0.0; interior];
                let mut diagonal = vec![0.0; interior];
                let mut upper = vec![0.0; interior];
                let mut rhs = vec![0.0; interior];
                for r in 0..interior {
                    let m = r + 1;
                    rhs[r] = 6.0 * ((unit(m + 1) - unit(m)) / h[m] - (unit(m) - unit(m - 1)) / h[m - 1]);
                    lower[r] = h[m - 1];
                    diagonal[r] = 2.0 * (h[m - 1] + h[m]);
                    upper[r] = h[m];
                }
                let ratio0 = h[0] / h[1];
                diagonal[0] += h[0] * (1.0 + ratio0);
                upper[0] -= h[0] * ratio0;
                let ratio_last = h[last - 1] / h[last - 2];
                diagonal[interior - 1] += h[last - 1] * (1.0 + ratio_last);
                lower[interior - 1] -= h[last - 1] * ratio_last;
                for r in 1..interior {
                    let factor = lower[r] / diagonal[r - 1];
                    diagonal[r] -= factor * upper[r - 1];
                    rhs[r] -= factor * rhs[r - 1];
                }
                let mut second = vec![0.0; interior];
                second[interior - 1] = rhs[interior - 1] / diagonal[interior - 1];
                for r in (0..interior - 1).rev() {
                    second[r] = (rhs[r] - upper[r] * second[r + 1]) / diagonal[r];
                }
                let first = second[0] * (1.0 + ratio0) - second[1] * ratio0;
                let end = second[interior - 1] * (1.0 + ratio_last) - second[interior - 2] * ratio_last;
                spline_second[j * order] = first;
                spline_second[j * order + last] = end;
                for (r, value) in second.into_iter().enumerate() {
                    spline_second[j * order + r + 1] = value;
                }
            }
        }
        let mut rule = Self {
            order,
            nodes,
            normal_weights,
            plain_weights,
            lagrange_weights,
            spline_second,
            lebesgue_constant: 1.0,
        };
        rule.lebesgue_constant = rule.lebesgue_function_maximum();
        Ok(rule)
    }

    /// `max_x Σ_i |L_i(x)|` over the node hull, sampled densely between
    /// consecutive nodes (the Lebesgue function of a Lagrange basis has one
    /// local maximum per interior interval, so a fine sample per interval
    /// resolves the maximum to the sampling resolution, which is all the
    /// certificate needs: it reads the order of magnitude).
    fn lebesgue_function_maximum(&self) -> f64 {
        const SAMPLES_PER_INTERVAL: usize = 32;
        let mut maximum = 1.0_f64;
        for pair in self.nodes.windows(2) {
            for s in 1..SAMPLES_PER_INTERVAL {
                let x = pair[0] + (pair[1] - pair[0]) * s as f64 / SAMPLES_PER_INTERVAL as f64;
                let total: f64 = self.lagrange_basis(&x).iter().map(|l| l.abs()).sum();
                maximum = maximum.max(total);
            }
        }
        maximum
    }

    /// Cardinal not-a-knot cubic spline basis at `xi`: `Σ_j basis[j] f_j` is
    /// the not-a-knot cubic spline through `(x_j, f_j)`, continued linearly
    /// beyond the hull with the spline's end slope.
    ///
    /// This serves the interpolation of a smoother residual (`log β`), a
    /// smooth but non-polynomial function whose degree-`G−1` Lagrange
    /// interpolant on Hermite nodes overshoots by the Lebesgue constant
    /// (thousands, at the orders the certificate reaches) and whose
    /// immediate exponential factors are evaluated exactly elsewhere. A
    /// cubic spline can overshoot the nodal range too, but only by a
    /// bounded factor: the operator norm of cubic spline interpolation is a
    /// small constant independent of the node count (the tests measure it
    /// below three on these nodes), where the Lagrange operator's grows
    /// exponentially. It converges like `h⁴` and is a fixed linear map of
    /// the nodal values, piecewise cubic and `C²` in `xi`, so every
    /// derivative channel of `xi` passes through it as through a polynomial
    /// (the piece selection below reads the value only; the pieces agree in
    /// value, slope and curvature at the nodes).
    pub(crate) fn spline_basis<S: JetField>(&self, xi: &S) -> Vec<S> {
        let g = self.order;
        let value = xi.value();
        let zero = xi.constant_like(0.0);
        if g == 1 {
            return vec![xi.constant_like(1.0)];
        }
        let second = |j: usize, m: usize| self.spline_second[j * g + m];
        // Linear continuation beyond the hull with the spline's end slope:
        // S'(x_0) = (f_1 − f_0)/h − h (2 M_0 + M_1) / 6, and the mirror image
        // at the top.
        if value <= self.nodes[0] || value >= self.nodes[g - 1] {
            let (edge, inner, sign) = if value <= self.nodes[0] {
                (0, 1, 1.0)
            } else {
                (g - 1, g - 2, -1.0)
            };
            let h = (self.nodes[inner] - self.nodes[edge]).abs();
            let offset = add_real(xi, -self.nodes[edge]);
            return (0..g)
                .map(|j| {
                    let at_edge = if j == edge { 1.0 } else { 0.0 };
                    let at_inner = if j == inner { 1.0 } else { 0.0 };
                    let slope = sign
                        * ((at_inner - at_edge) / h
                            - h * (2.0 * second(j, edge) + second(j, inner)) / 6.0);
                    offset.scale(slope).add(&xi.constant_like(at_edge))
                })
                .collect();
        }
        let m = self
            .nodes
            .windows(2)
            .position(|w| value >= w[0] && value <= w[1])
            .unwrap_or(g - 2);
        let h = self.nodes[m + 1] - self.nodes[m];
        let a = add_real(&xi.neg(), self.nodes[m + 1]).scale(1.0 / h);
        let b = add_real(xi, -self.nodes[m]).scale(1.0 / h);
        let c = a.mul(&a).mul(&a).sub(&a).scale(h * h / 6.0);
        let d = b.mul(&b).mul(&b).sub(&b).scale(h * h / 6.0);
        (0..g)
            .map(|j| {
                let mut basis = zero.clone();
                if j == m {
                    basis = basis.add(&a);
                }
                if j == m + 1 {
                    basis = basis.add(&b);
                }
                basis
                    .add(&c.scale(second(j, m)))
                    .add(&d.scale(second(j, m + 1)))
            })
            .collect()
    }

    /// Lagrange basis on the rule's nodes evaluated at `xi`.
    ///
    /// Product form `L_i(x) = w_i Π_{m≠i} (x − x_m)` through prefix and
    /// suffix products. Every channel of the result is a sum of products of
    /// the factors `x − x_m`, so nothing cancels catastrophically when `x`
    /// lies within roundoff of a node — the barycentric quotient is stable in
    /// value there but its derivative channels are differences of terms of
    /// size `1/(x − x_m)²`, and adaptive grids put points exactly on nodes.
    pub(crate) fn lagrange_basis<S: JetField>(&self, xi: &S) -> Vec<S> {
        let g = self.order;
        let factors: Vec<S> = self.nodes.iter().map(|&x| add_real(xi, -x)).collect();
        let one = xi.constant_like(1.0);
        let mut prefix = vec![one.clone(); g + 1];
        for i in 0..g {
            prefix[i + 1] = prefix[i].mul(&factors[i]);
        }
        let mut suffix = vec![one; g + 1];
        for i in (0..g).rev() {
            suffix[i] = suffix[i + 1].mul(&factors[i]);
        }
        (0..g)
            .map(|i| prefix[i].mul(&suffix[i + 1]).scale(self.lagrange_weights[i]))
            .collect()
    }

    /// The forward kernel's basis at `xi`: the Lagrange basis inside the
    /// hull `[x_0, x_{G−1}]` and, beyond it, the cardinal basis of the
    /// quadratic through the three end nodes on that side.
    ///
    /// The forward inner points reach up to `√2·x_{G−1}`, past the last node,
    /// where the degree-`G−1` interpolant is an extrapolation whose overshoot
    /// grows like the leading coefficient times the distance to the power
    /// `G−1`. The end quadratic reads three nodal values with bounded
    /// coefficients, so it is exact for every quadratic (every Gaussian
    /// `ln r`) and amplifies nodal errors by no more than the in-hull Lebesgue
    /// constant across the whole reach. The two pieces agree in value at the
    /// hull edge (both interpolate the end node), so the basis is continuous
    /// in `xi`. The side is selected by the value alone, and every derivative
    /// channel passes through each piece as through a polynomial.
    pub(crate) fn forward_basis<S: JetField>(&self, xi: &S) -> Vec<S> {
        let g = self.order;
        let value = xi.value();
        if g < 3 || (value >= self.nodes[0] && value <= self.nodes[g - 1]) {
            return self.lagrange_basis(xi);
        }
        let end: [usize; 3] = if value < self.nodes[0] { [0, 1, 2] } else { [g - 1, g - 2, g - 3] };
        let mut basis = vec![xi.constant_like(0.0); g];
        for (a, &i) in end.iter().enumerate() {
            let mut term = xi.constant_like(1.0);
            let mut denominator = 1.0;
            for (b, &m) in end.iter().enumerate() {
                if a != b {
                    term = term.mul(&add_real(xi, -self.nodes[m]));
                    denominator *= self.nodes[i] - self.nodes[m];
                }
            }
            basis[i] = term.scale(1.0 / denominator);
        }
        basis
    }
}

/// One axis of a product grid: centre, scale, node values and plain weights.
#[derive(Clone, Debug)]
pub(crate) struct Axis<S> {
    pub mu: S,
    pub sigma: S,
    /// `z_i = mu + √2 sigma x_i`.
    pub points: Vec<S>,
    /// `√2 sigma · w_i e^{x_i²}`: plain integration weights in `z`.
    pub weights: Vec<S>,
}

/// Product Gauss-Hermite grid over `k` independent axes, axis 0 fastest.
#[derive(Clone, Debug)]
pub(crate) struct Grid<S> {
    pub axes: Vec<Axis<S>>,
    pub order: usize,
    /// `order^k` for axis `k`: the flat stride of each axis.
    pub strides: Vec<usize>,
    /// Product integration weight of every flat point.
    pub weights: Vec<S>,
}

/// `order^axes` with overflow reported instead of wrapped.
pub(crate) fn product_grid_size(order: usize, axes: usize) -> Result<usize, EventHistoryError> {
    let mut size = 1usize;
    for _ in 0..axes {
        size = size.checked_mul(order).ok_or_else(|| EventHistoryError::InvalidInput {
            reason: format!("a product grid of order {order} over {axes} atoms is not representable"),
        })?;
    }
    Ok(size)
}

impl<S: JetField> Grid<S> {
    pub fn new(gh: &GaussHermite, centres: &[S], scales: &[S], like: &S) -> Self {
        let order = gh.order;
        let axes: Vec<Axis<S>> = centres
            .iter()
            .zip(scales.iter())
            .map(|(mu, sigma)| {
                let points = gh
                    .nodes
                    .iter()
                    .map(|&x| mu.add(&sigma.scale(std::f64::consts::SQRT_2 * x)))
                    .collect();
                let weights = gh
                    .plain_weights
                    .iter()
                    .map(|&w| sigma.scale(std::f64::consts::SQRT_2 * w))
                    .collect();
                Axis {
                    mu: mu.clone(),
                    sigma: sigma.clone(),
                    points,
                    weights,
                }
            })
            .collect();
        let mut strides = Vec::with_capacity(axes.len());
        let mut size = 1usize;
        for _ in 0..axes.len() {
            strides.push(size);
            size *= order;
        }
        let mut weights = Vec::with_capacity(size);
        for flat in 0..size {
            let mut w = like.constant_like(1.0);
            let mut rest = flat;
            for axis in &axes {
                let i = rest % order;
                rest /= order;
                w = w.mul(&axis.weights[i]);
            }
            weights.push(w);
        }
        Grid {
            axes,
            order,
            strides,
            weights,
        }
    }

    pub fn dimension(&self) -> usize {
        self.axes.len()
    }

    pub fn size(&self) -> usize {
        self.weights.len()
    }

    /// Coordinate index of flat point `flat` along `axis`.
    #[inline]
    pub fn index(&self, flat: usize, axis: usize) -> usize {
        (flat / self.strides[axis]) % self.order
    }

    /// The `axis` coordinate of flat point `flat`.
    #[inline]
    pub fn coordinate(&self, flat: usize, axis: usize) -> &S {
        &self.axes[axis].points[self.index(flat, axis)]
    }
}

/// Transition of one unit-variance Ornstein–Uhlenbeck atom across a gap of
/// dimensionless length `kappa = rate · gap`: `z' | z ~ N(φ z, 1 − φ²)`.
#[derive(Clone, Debug)]
pub(crate) struct AtomTransition<S> {
    /// `φ = e^{-κ}`.
    pub phi: S,
    /// `q = 1 − φ²`, the innovation variance.
    pub innovation: S,
    /// `dφ/dρ = −κ φ` with `ρ = ln rate`.
    pub dphi: S,
    /// `d²φ/dρ² = κ φ (κ − 1)`.
    pub d2phi: S,
}

impl<S: JetField> AtomTransition<S> {
    pub fn new(kappa: &S) -> Self {
        let k = kappa.value();
        if k == f64::INFINITY {
            return Self {
                phi: kappa.constant_like(0.0),
                innovation: kappa.constant_like(1.0),
                dphi: kappa.constant_like(0.0),
                d2phi: kappa.constant_like(0.0),
            };
        }
        let e = (-k).exp();
        let one_minus_phi = kappa.compose_unary([-(-k).exp_m1(), e, -e, e, -e]);
        // Reconstructing phi as 1-(1-exp(-k)) rounds it to zero near k=38,
        // even though its value and derivatives remain representable.
        let phi = kappa.compose_unary([e, -e, e, -e, e]);
        let innovation = one_minus_phi.mul(&add_real(&phi, 1.0));
        // Keep k*exp(-k) and k²*exp(-k) representable even after exp(-k)
        // underflows. Their derivative stacks follow from Leibniz's rule.
        let (p1, p2) = if k <= 1.0 {
            (k * e, k * k * e)
        } else {
            ((k.ln() - k).exp(), (2.0 * k.ln() - k).exp())
        };
        let dphi = kappa.compose_unary([
            -p1,
            p1 - e,
            2.0 * e - p1,
            p1 - 3.0 * e,
            4.0 * e - p1,
        ]);
        let d2phi = kappa.compose_unary([
            p2 - p1,
            -p2 + 3.0 * p1 - e,
            p2 - 5.0 * p1 + 4.0 * e,
            -p2 + 7.0 * p1 - 9.0 * e,
            p2 - 9.0 * p1 + 16.0 * e,
        ]);
        Self {
            phi,
            innovation,
            dphi,
            d2phi,
        }
    }
}

/// `ln √(2π)`.
pub(crate) const LOG_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_8;

/// `ln N(x; mean, variance)` as a scalar.
pub(crate) fn log_normal_density<S: JetField>(x: &S, mean: &S, variance: &S) -> S {
    let d = x.sub(mean);
    let quad = div(&square(&d), variance).scale(-0.5);
    add_real(&quad.add(&ln(variance).scale(-0.5)), -LOG_SQRT_TWO_PI)
}

/// `ln` of the standard normal density of every atom at every point of
/// `grid`: the stationary prior every chain starts from.
pub(crate) fn log_standard_prior<S: JetField>(grid: &Grid<S>, like: &S) -> Vec<S> {
    (0..grid.size())
        .map(|i| {
            (0..grid.dimension()).fold(like.constant_like(0.0), |acc, k| {
                let z = grid.coordinate(i, k);
                add_real(&acc.sub(&square(z).scale(0.5)), -LOG_SQRT_TWO_PI)
            })
        })
        .collect()
}

/// The Gaussian transition across one gap, from a filtered density on `from`
/// to the grid `to`, evaluated one target point at a time.
///
/// Per axis `z' | z ~ N(φz, q)`. Write the filtered density against the
/// envelope `e = N(μ, σ²)` of `from` as `α(z) = e(z) r(z)`. The product of the
/// kernel and the envelope is `N(z'; φμ, τ²) N(z; m(z'), σ² q/τ²)` with
/// `τ² = φ²σ² + q`, so
///
/// ```text
/// p̂(z') = ∫ N(z'; φz, q) α(z) dz = N(z'; φμ, τ²) · E[r(z) | z'],
/// ```
///
/// an expectation under a Gaussian whose standardised coordinate on `from`
/// is `centre + √(q/τ²) x` at Gauss-Hermite node `x`, with
/// `centre = φσ(z' − φμ)/(τ²√2)`.
///
/// The source density arrives split ([`SplitDensity`]) as `ln α = s + f`,
/// with `f` the node log-likelihood factors conditioned on since the last
/// gap. `f` is not polynomial in `z`: past an event with a large loading its
/// compensator `−Δ e^{a z}` is a wall that falls by orders of magnitude
/// across the hull, and the polynomial through the nodes oscillates between
/// them by as much (on a unit grid at order 21 with `a = 2`, `Δ e^{b} = 0.3`,
/// one event and `φ = e^{−0.4}`, interpolating `ln α` whole puts the largest
/// `ln p̂` at `+164`, where no predicted density can exceed
/// `sup α / φ = e^{0.18}`). `f` is an explicit formula, so it is evaluated
/// exactly at every inner point; only `ln r = s − ln e` is interpolated.
///
/// The ratio enters through its logarithm: `ln r` is carried on the nodes of
/// `from` by the tensor product of [`GaussHermite::forward_basis`], `f` is
/// added at the inner points, and the sum is exponentiated there. Every inner
/// weight is then positive,
/// the predicted density is a log-sum-exp of them, and the representation
/// cannot lose positivity at any order. Inside the hull the basis is the
/// Lagrange interpolant, exact whenever `ln r` is a polynomial of degree
/// below the order. The inner points reach past the last node (up to
/// `√2·x_{G−1}`), and there each axis continues by the quadratic through its
/// three end nodes instead of extrapolating the degree-`G−1` polynomial,
/// whose overshoot beyond the hull `exp` turns into spurious mass (a
/// predicted density far above `sup α / Π φ`). The prediction is exact for
/// every Gaussian `α` (`ln r` is then quadratic) whatever its centre and
/// spread relative to the grid, and the continuation amplifies nodal errors
/// by no more than the in-hull Lebesgue constant.
///
/// Normalising the inner weights of target point `z'_j` gives `ŵ_{jl}`, the
/// quadrature rule of the law of `z` given `z'_j` and the data so far. A
/// conditional expectation `E[f(z) | z'_j]` of a function carried on `from`
/// is `Σ_l ŵ_{jl} f̃(z_{jl})` with `f̃` its Lagrange interpolant
/// ([`Self::transfer`]); a polynomial in the inner coordinates and the
/// innovation is evaluated at them exactly ([`Self::innovation_moment`]).
/// Nothing of size `|to| × |from|` is formed: each target row is built,
/// used and dropped.
pub(crate) struct ForwardKernel<S> {
    order: usize,
    /// `ln r` on the points of `from`.
    log_ratio: Vec<S>,
    /// `bases[k][(j_k G + l) G + i] = B_i(raw_{k, j_k, l})`: the forward
    /// basis of `from` ([`GaussHermite::forward_basis`]) at the inner points
    /// of target coordinate `j_k`.
    bases: Vec<Vec<S>>,
    /// `ln N(z'_{j_k}; φμ, τ²)` per axis, `[k][j_k]`.
    log_gauss: Vec<Vec<S>>,
    /// The standardised innovation `u = (z' − φz)/√q` at every inner point,
    /// `[k][j_k G + l]`.
    innovations: Vec<Vec<S>>,
    /// The source coordinate `z` of every inner point, `[k][j_k G + l]`.
    coordinates: Vec<Vec<S>>,
    /// `Σ_k ln(w_{l_k}/√π)` per flat inner point, axis 0 fastest.
    log_inner_weights: Vec<f64>,
    /// The explicit factor `f` of the source density, evaluated at the inner
    /// points of each row.
    factor: LogFactor<S>,
}

/// One target point of a [`ForwardKernel`]: the log predicted density there
/// and the normalised inner weights `ŵ_{j·}`, axis 0 fastest.
pub(crate) struct KernelRow<S> {
    pub log_predicted: S,
    pub weights: Vec<S>,
}

/// One mark's Poisson node term `y η − Δ e^{η}` with `η = b + Σ_k a_k z_k`,
/// as a function of the latent state.
#[derive(Clone, Debug)]
pub(crate) struct FactorMark<S> {
    /// The event count `y`; its term is absent when zero.
    pub count: f64,
    /// `ln Δ`, or `None` where the mark has no compensator at this node.
    pub log_exposure: Option<f64>,
    /// The centred baseline `b`.
    pub base: S,
    /// The loadings `a_k`, one per atom.
    pub loadings: Vec<S>,
}

/// A sum of node log-likelihood terms plus a constant, as an explicit
/// function of the latent state `z`: `Σ_marks (y η − Δ e^{η}) + c`.
#[derive(Clone, Debug)]
pub(crate) struct LogFactor<S> {
    pub marks: Vec<FactorMark<S>>,
    pub constant: S,
}

impl<S: JetField> LogFactor<S> {
    /// The factor that is identically zero.
    pub fn zero(like: &S) -> Self {
        Self { marks: Vec::new(), constant: like.constant_like(0.0) }
    }

    /// This factor plus the constant `c`.
    pub fn shifted(&self, c: &S) -> Self {
        Self { marks: self.marks.clone(), constant: self.constant.add(c) }
    }

    /// The sum of this factor and `other`.
    pub fn joined(&self, other: &Self) -> Self {
        let mut marks = self.marks.clone();
        marks.extend(other.marks.iter().cloned());
        Self { marks, constant: self.constant.add(&other.constant) }
    }

    /// The factor at every point of the tensor product of `axes`, axis 0
    /// fastest. The axis contributions `a_k z_k` are tabulated once per mark,
    /// and the intensity is formed from their sum in log space.
    pub fn on_tensor(&self, axes: &[&[S]]) -> Vec<S> {
        let size: usize = axes.iter().map(|axis| axis.len()).product();
        let mut out = vec![self.constant.clone(); size];
        for mark in &self.marks {
            let tables: Vec<Vec<S>> = axes
                .iter()
                .zip(mark.loadings.iter())
                .map(|(axis, a)| axis.iter().map(|z| a.mul(z)).collect())
                .collect();
            for (p, value) in out.iter_mut().enumerate() {
                let mut eta = mark.base.clone();
                let mut rest = p;
                for (k, table) in tables.iter().enumerate() {
                    eta = eta.add(&table[rest % axes[k].len()]);
                    rest /= axes[k].len();
                }
                if mark.count != 0.0 {
                    *value = value.add(&eta.scale(mark.count));
                }
                if let Some(log_exposure) = mark.log_exposure {
                    *value = value.sub(&exp(&add_real(&eta, log_exposure)));
                }
            }
        }
        out
    }
}

/// A filtered log density split as `ln α(z) = s(z) + f(z)`: a smooth part
/// `s` carried by its values on the density's grid, and the node
/// log-likelihood factors `f` conditioned on since the last gap, kept as the
/// explicit function they are. The forward kernel interpolates only `s`;
/// `f` is evaluated exactly wherever the kernel needs it.
#[derive(Clone, Debug)]
pub(crate) struct SplitDensity<S> {
    /// `s` at the points of the density's grid.
    pub smooth: Vec<S>,
    pub factor: LogFactor<S>,
}

impl<S: JetField> SplitDensity<S> {
    /// `ln α` carried whole by its grid values, with no explicit factor.
    pub fn whole(log_alpha: &[S]) -> Self {
        Self { smooth: log_alpha.to_vec(), factor: LogFactor::zero(&log_alpha[0]) }
    }
}

impl<S: JetField> ForwardKernel<S> {
    pub fn new(
        gh: &GaussHermite,
        from: &Grid<S>,
        density: &SplitDensity<S>,
        to: &Grid<S>,
        transitions: &[AtomTransition<S>],
    ) -> Self {
        let g = gh.order;
        let atoms = from.dimension();
        // ln r_i = s_i − ln e(z_i), with ln e(z_i) = Σ_k (−x_{i_k}² − ln σ_k − ln √(2π)).
        let log_sigma: Vec<S> = from.axes.iter().map(|axis| ln(&axis.sigma)).collect();
        let log_ratio: Vec<S> = density
            .smooth
            .iter()
            .enumerate()
            .map(|(i, log_a)| {
                (0..atoms).fold(log_a.clone(), |acc, k| {
                    let x = gh.nodes[from.index(i, k)];
                    add_real(&acc.add(&log_sigma[k]), x * x + LOG_SQRT_TWO_PI)
                })
            })
            .collect();
        let mut bases = Vec::with_capacity(atoms);
        let mut log_gauss = Vec::with_capacity(atoms);
        let mut innovations = Vec::with_capacity(atoms);
        let mut coordinates = Vec::with_capacity(atoms);
        for (axis, transition) in transitions.iter().enumerate() {
            let old = &from.axes[axis];
            let new = &to.axes[axis];
            let phi = &transition.phi;
            let q = &transition.innovation;
            let root_q = if q.value() == 0.0 { q.constant_like(0.0) } else { sqrt(q) };
            let tau2 = square(phi).mul(&square(&old.sigma)).add(q);
            let inv_tau2 = recip(&tau2);
            let ratio = if q.value() == 0.0 { q.constant_like(0.0) } else { sqrt(&q.mul(&inv_tau2)) };
            let phi_mu = phi.mul(&old.mu);
            // The standardised innovation at inner node `x` of target point `z'`:
            // with `z = μ + √2 σ (centre + ratio·x)` and `d = z' − φμ`,
            //   z' − φz = d q/τ² − φ √2 σ √(q/τ²) x,
            // so `u = (z' − φz)/√q = d √q/τ² − φ √2 σ x/τ`. Forming `z' − φz` as
            // a difference would cancel to roundoff at small `q` (a short gap or
            // a slow atom) and dividing by `√q` would amplify that roundoff
            // without bound; the closed form is exact in the limit.
            let u_slope = phi
                .mul(&old.sigma)
                .mul(&sqrt(&inv_tau2))
                .scale(-std::f64::consts::SQRT_2);
            let u_offset_factor = root_q.mul(&inv_tau2);
            let spread = old.sigma.scale(std::f64::consts::SQRT_2);
            let mut axis_bases = Vec::with_capacity(g * g * g);
            let mut axis_gauss = Vec::with_capacity(g);
            let mut axis_innovations = Vec::with_capacity(g * g);
            let mut axis_coordinates = Vec::with_capacity(g * g);
            for j in 0..g {
                let d = new.points[j].sub(&phi_mu);
                axis_gauss.push(log_normal_density(&new.points[j], &phi_mu, &tau2));
                let centre = phi
                    .mul(&old.sigma)
                    .mul(&d)
                    .mul(&inv_tau2)
                    .scale(1.0 / std::f64::consts::SQRT_2);
                let u_offset = d.mul(&u_offset_factor);
                for &x in &gh.nodes {
                    let raw = centre.add(&ratio.scale(x));
                    axis_bases.extend(gh.forward_basis(&raw));
                    axis_innovations.push(u_offset.add(&u_slope.scale(x)));
                    axis_coordinates.push(old.mu.add(&spread.mul(&raw)));
                }
            }
            bases.push(axis_bases);
            log_gauss.push(axis_gauss);
            innovations.push(axis_innovations);
            coordinates.push(axis_coordinates);
        }
        let log_inner_weights = (0..from.size())
            .map(|l| {
                let mut rest = l;
                let mut acc = 0.0;
                for _ in 0..atoms {
                    acc += gh.normal_weights[rest % g].ln();
                    rest /= g;
                }
                acc
            })
            .collect();
        Self {
            order: g,
            log_ratio,
            bases,
            log_gauss,
            innovations,
            coordinates,
            log_inner_weights,
            factor: density.factor.clone(),
        }
    }

    /// Axis-`k` coordinate index of flat target point `j`.
    fn target_index(&self, j: usize, k: usize) -> usize {
        (j / self.order.pow(k as u32)) % self.order
    }

    /// The log predicted density at target point `j` and its normalised
    /// inner weights.
    pub fn row(&self, j: usize) -> KernelRow<S> {
        let g = self.order;
        let at_inner = interpolate_at_inner_points(g, &self.bases, &self.log_ratio, j);
        let inner_axes: Vec<&[S]> = self
            .coordinates
            .iter()
            .enumerate()
            .map(|(k, axis)| {
                let base = self.target_index(j, k) * g;
                &axis[base..base + g]
            })
            .collect();
        let factor = self.factor.on_tensor(&inner_axes);
        let terms: Vec<S> = at_inner
            .iter()
            .zip(factor.iter())
            .zip(self.log_inner_weights.iter())
            .map(|((t, f), &w)| add_real(&t.add(f), w))
            .collect();
        let log_mass = log_sum_exp(&terms);
        let weights = terms.iter().map(|t| exp(&t.sub(&log_mass))).collect();
        let log_predicted = self
            .log_gauss
            .iter()
            .enumerate()
            .fold(log_mass, |acc, (k, gauss)| acc.add(&gauss[self.target_index(j, k)]));
        KernelRow {
            log_predicted,
            weights,
        }
    }

    /// The log predicted density at every point of the target grid.
    pub fn log_predicted(&self, target_size: usize) -> Vec<S> {
        (0..target_size).map(|j| self.row(j).log_predicted).collect()
    }

    /// Row `j` of the conditional-expectation operator: `Σ_i M_{ji} f(z_i)` is
    /// `E[f(z) | z'_j]` for `f` carried by its values on `from`, with
    /// `M_{ji} = Σ_l ŵ_{jl} Π_k B_{i_k}(raw_{k, j_k, l_k})`. The inner weights
    /// are contracted one axis at a time against the transposed per-axis
    /// bases, using O(G^K) transient storage.
    pub fn transfer(&self, j: usize, weights: &[S]) -> Vec<S> {
        let g = self.order;
        let zero = weights[0].constant_like(0.0);
        let mut cur = weights.to_vec();
        let mut stride = 1;
        for (k, basis) in self.bases.iter().enumerate() {
            let target = self.target_index(j, k);
            let blocks = cur.len() / (stride * g);
            let mut next = vec![zero.clone(); cur.len()];
            for block in 0..blocks {
                for i in 0..g {
                    for inner in 0..stride {
                        let mut acc = zero.clone();
                        for l in 0..g {
                            acc = acc.add(&basis[(target * g + l) * g + i]
                                .mul(&cur[inner + stride * (l + g * block)]));
                        }
                        next[inner + stride * (i + g * block)] = acc;
                    }
                }
            }
            cur = next;
            stride *= g;
        }
        cur
    }

    /// The per-axis marginals of a row's inner weights, `[k][l_k]`.
    pub fn axis_marginals(&self, weights: &[S]) -> Vec<Vec<S>> {
        let g = self.order;
        let zero = weights[0].constant_like(0.0);
        (0..self.bases.len())
            .map(|k| {
                let stride = g.pow(k as u32);
                let mut marginal = vec![zero.clone(); g];
                for (l, w) in weights.iter().enumerate() {
                    let lk = (l / stride) % g;
                    marginal[lk] = marginal[lk].add(w);
                }
                marginal
            })
            .collect()
    }

    /// `E[u_k^b z_k^a | z'_j]` from the axis-`k` marginal of row `j`, with the
    /// innovation and the source coordinate evaluated at the inner points
    /// exactly.
    pub fn innovation_moment(&self, j: usize, marginal: &[S], k: usize, b: usize, a: usize) -> S {
        let g = self.order;
        let base = self.target_index(j, k) * g;
        let mut acc = marginal[0].constant_like(0.0);
        for (l, w) in marginal.iter().enumerate() {
            let mut term = w.clone();
            for _ in 0..b {
                term = term.mul(&self.innovations[k][base + l]);
            }
            for _ in 0..a {
                term = term.mul(&self.coordinates[k][base + l]);
            }
            acc = acc.add(&term);
        }
        acc
    }
}

/// Per-axis spline basis of `to` at the backward inner points
/// `ζ = φ_k z_{i_k} + √(2 q_k) x_l` of every source point of `from`, as the
/// cardinal natural cubic spline basis of `to` (linear beyond its hull):
/// `bases[k][(i * G + l) * G + j]`
/// for source index `i`, inner node `l` and target basis `j` along axis `k`.
pub(crate) fn backward_axis_bases<S: JetField>(
    gh: &GaussHermite,
    from: &Grid<S>,
    to: &Grid<S>,
    transitions: &[AtomTransition<S>],
) -> Vec<Vec<S>> {
    let g = gh.order;
    transitions
        .iter()
        .enumerate()
        .map(|(axis, transition)| {
            let old = &from.axes[axis];
            let new = &to.axes[axis];
            let spread = if transition.innovation.value() == 0.0 {
                transition.innovation.constant_like(0.0)
            } else { sqrt(&transition.innovation.scale(2.0)) };
            let inverse_scale = recip(&new.sigma.scale(std::f64::consts::SQRT_2));
            let mut bases = vec![old.mu.constant_like(0.0); g * g * g];
            for i in 0..g {
                let phi_z = transition.phi.mul(&old.points[i]);
                for (l, &x) in gh.nodes.iter().enumerate() {
                    let zeta = phi_z.add(&spread.scale(x));
                    let basis = gh.spline_basis(&zeta.sub(&new.mu).mul(&inverse_scale));
                    for (j, value) in basis.into_iter().enumerate() {
                        bases[(i * g + l) * g + j] = value;
                    }
                }
            }
            bases
        })
        .collect()
}

/// The tensor-product Lagrange interpolant of `values` (on the target grid,
/// axis 0 fastest) at the backward inner points of one source point.
/// The output indexes the inner point, axis 0 fastest. One axis is
/// contracted at a time, using O(G^K) transient storage.
pub(crate) fn interpolate_at_inner_points<S: JetField>(
    order: usize, bases: &[Vec<S>], values: &[S], source: usize,
) -> Vec<S> {
    let g = order;
    let zero = values[0].constant_like(0.0);
    let mut cur = values.to_vec();
    let mut stride = 1;
    // Contract one source row at a time. Every intermediate has G^K
    // entries; no all-source G^(2K) kernel is ever materialised.
    for basis in bases {
        let source_axis = (source / stride) % g;
        let blocks = values.len() / (stride * g);
        let mut next = vec![zero.clone(); values.len()];
        for block in 0..blocks {
            for l in 0..g {
                for inner in 0..stride {
                    let mut acc = zero.clone();
                    for j in 0..g {
                        acc = acc.add(&basis[(source_axis * g + l) * g + j]
                            .mul(&cur[inner + stride * (j + g * block)]));
                    }
                    next[inner + stride * (l + g * block)] = acc;
                }
            }
        }
        cur = next;
        stride *= g;
    }
    cur
}

/// `ln Σ exp(terms)`, stabilised by the largest value.
///
/// A sum of no mass is `-inf`, and it is returned as one. Stabilising by the
/// largest value divides through by `exp(shift)`, which is the identity only
/// while `shift` is finite: when every term is `-inf` the shift is `-inf` too,
/// `t - shift` is `inf - inf`, and the stabilised sum is NaN. That NaN then
/// travels — a filter's log normaliser, a predicted log density, a smoothed
/// marginal's mass — and every consumer downstream reports it instead of the
/// condition that produced it, which is a node whose whole grid holds no mass.
/// Returning the value the sum has removes the NaN and leaves that condition
/// to be reported by whoever cares that it is not finite.
pub(crate) fn log_sum_exp<S: JetField>(terms: &[S]) -> S {
    let shift = terms
        .iter()
        .map(|t| t.value())
        .fold(f64::NEG_INFINITY, f64::max);
    if !shift.is_finite() {
        // Every term is `-inf` (no mass), or one is `+inf` or NaN and the sum
        // is that. Either way the stabilised form cannot be formed and the
        // answer is the shift itself.
        return terms[0].constant_like(shift);
    }
    let sum = terms
        .iter()
        .fold(terms[0].constant_like(0.0), |acc, t| acc.add(&exp(&add_real(t, -shift))));
    add_real(&ln(&sum), shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::roundoff::accumulation_growth;

    /// A sum of no mass is `-inf`, and a sum of some mass is unchanged by the
    /// empty case being handled.
    ///
    /// The stabilised form divides through by `exp(max)`, which is the
    /// identity only while the max is finite. With every term `-inf` the max
    /// is `-inf`, `t - max` is `inf - inf`, and the old form returned NaN for
    /// a quantity whose value is `-inf`. Five event-history tests reported
    /// that NaN through a filter's log normaliser rather than the condition
    /// that produced it.
    #[test]
    fn a_log_sum_exp_of_no_mass_is_minus_infinity_not_a_nan() {
        let empty = log_sum_exp(&[f64::NEG_INFINITY; 4]);
        assert!(
            empty == f64::NEG_INFINITY,
            "a sum of four terms of no mass reads {empty}, and the sum of four zeros is zero"
        );
        // One term carrying mass is the whole sum, however many do not.
        let one = log_sum_exp(&[f64::NEG_INFINITY, -3.25, f64::NEG_INFINITY]);
        assert!(
            (one - -3.25).abs() <= f64::EPSILON * 3.25,
            "a sum whose only mass is exp(-3.25) reads {one}"
        );
        // The ordinary case is untouched: `ln(e^a + e^b)` to the rounding of
        // both routes to it. A logarithm turns its argument's RELATIVE rounding
        // into ABSOLUTE error (`d ln s = ds/s`), so each route is off by the
        // relative rounding of the sum it forms (an `exp`, an add and the `exp`
        // of the other term, or the shift the stabilised form makes) plus the
        // log's own `ε|ln s|`. The bar is both routes' share, in absolute units:
        // here `s ≈ 1.0019` and `ln s ≈ 0.0019`, so a bar relative to `ln s`
        // alone asks for 500 times the digits the sum carries.
        let (a, b) = (-1.5_f64, -0.25_f64);
        let both = log_sum_exp(&[a, b]);
        let exact = (a.exp() + b.exp()).ln();
        assert!(
            (both - exact).abs() <= 2.0 * accumulation_growth(4) * (1.0 + exact.abs()),
            "ln(e^{a} + e^{b}) reads {both}, against {exact}"
        );
    }
}
