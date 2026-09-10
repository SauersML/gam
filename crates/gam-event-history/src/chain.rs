//! Gauss-Hermite–Lagrange representation of one subject's latent chain.
//!
//! The latent state at every node is a product of independent unit-variance
//! atoms. Every density or likelihood-weighted function on the state is
//! carried as its values on an adaptive product Gauss-Hermite grid whose
//! per-axis centre and scale follow the predicted moments. The two operators
//! the chain needs — predict forward across a gap and condition backward
//! across a gap — are Gaussian convolutions, and a Gaussian convolution of
//! `envelope × polynomial` is exact: the product of the transition kernel and
//! the grid envelope is again a Gaussian, so the convolution is an expectation
//! of the Lagrange interpolant under that Gaussian, evaluated by Gauss-Hermite
//! quadrature to which it is exact. Both operators are therefore `G × G`
//! matrices per axis, valid for gaps of any size, including gaps far shorter
//! than the grid spacing where direct kernel quadrature fails.
//!
//! The two operators interpolate different classes. Forward, the operand is
//! a density, Gaussian-enveloped, so the interpolant is of `density /
//! envelope` and the convolution is exact for `envelope × polynomial`; the
//! Lagrange interpolant is used as the polynomial it is, inside and beyond
//! the hull, because the envelope's Gaussian decay beats its growth there
//! and a clamp would put a kink into the objective. Backward, the operand is
//! the logarithm of a smoother residual, bounded and smooth but not
//! enveloped, so it is carried by the not-a-knot cubic spline through the
//! nodes, continued linearly beyond the hull with the end slope.
//!
//! Both interpolants are signed linear maps of the nodal values: neither is
//! a positivity-preserving operator, and neither claims to be. A density's
//! interpolant can dip below zero far in the tails, at the level of the
//! interpolation error; the filter treats such values as the numerical noise
//! they are (they carry no smoothed mass) and the Gauss-Hermite certificate
//! of the fit is what bounds them. The Lebesgue constant of the Lagrange
//! interpolant on Hermite nodes grows exponentially with the order, so the
//! rule records it and the fit refuses an order whose roundoff amplification
//! would exceed the certificate's tolerance.
//!
//! Everything is generic over a [`JetField`] scalar so the same code yields
//! the value, one directional derivative, or a mixed second directional
//! derivative of every quantity downstream.

use super::cohort::EventHistoryError;
use super::scalar::{add_real, div, exp, recip, sqrt, square};
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
    pub fn spline_basis<S: JetField>(&self, xi: &S) -> Vec<S> {
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
    pub fn lagrange_basis<S: JetField>(&self, xi: &S) -> Vec<S> {
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

/// Apply a per-axis `G × G` matrix (row-major `[j * G + i]`, output index
/// `j`, input index `i`) along `axis` of a flat tensor.
pub(crate) fn apply_axis<S: JetField>(values: &[S], matrix: &[S], order: usize, axis: usize) -> Vec<S> {
    let stride = order.pow(axis as u32);
    let block = stride * order;
    let total = values.len();
    let zero = values[0].constant_like(0.0);
    let mut out = vec![zero.clone(); total];
    let mut base = 0;
    while base < total {
        for inner in 0..stride {
            for j in 0..order {
                let mut acc = zero.clone();
                for i in 0..order {
                    acc = acc.add(&matrix[j * order + i].mul(&values[base + inner + i * stride]));
                }
                out[base + inner + j * stride] = acc;
            }
        }
        base += block;
    }
    out
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

/// Beyond this `κ`, `e^{-κ}` is below the smallest subnormal `f64`, so `φ`
/// is exactly zero and every quantity derived from the transition is exactly
/// constant; holding `κ` there keeps `κ φ` from becoming `∞ · 0`.
const KAPPA_SATURATION: f64 = 745.0;

impl<S: JetField> AtomTransition<S> {
    pub fn new(kappa: &S) -> Self {
        let saturated;
        let kappa = if kappa.value() >= KAPPA_SATURATION {
            saturated = kappa.constant_like(KAPPA_SATURATION);
            &saturated
        } else {
            kappa
        };
        let k = kappa.value();
        let e = (-k).exp();
        let one_minus_phi = kappa.compose_unary([-(-k).exp_m1(), e, -e, e, -e]);
        let phi = one_minus_phi.neg().add(&kappa.constant_like(1.0));
        let innovation = one_minus_phi.mul(&add_real(&phi, 1.0));
        let dphi = kappa.mul(&phi).neg();
        let d2phi = kappa.mul(&phi).mul(&add_real(kappa, -1.0));
        Self {
            phi,
            innovation,
            dphi,
            d2phi,
        }
    }
}

const LOG_SQRT_TWO_PI: f64 = 0.918_938_533_204_672_8;

/// `N(x; mean, variance)` as a scalar.
pub(crate) fn normal_density<S: JetField>(x: &S, mean: &S, variance: &S) -> S {
    let d = x.sub(mean);
    let quad = div(&square(&d), variance).scale(-0.5);
    let log_norm = super::scalar::ln(variance).scale(-0.5);
    exp(&quad.add(&log_norm).add(&x.constant_like(-LOG_SQRT_TWO_PI)))
}

/// The innovation-weighted operators of one gap, every axis and every
/// innovation power at once: `per_axis[k][b]` is the `G × G` matrix of axis
/// `k` with weight `u_k^b`. One Lagrange-basis pass per axis serves all
/// powers, and the plain operator is power zero.
#[derive(Clone, Debug)]
pub(crate) struct OperatorFamily<S> {
    pub per_axis: Vec<Vec<Vec<S>>>,
    pub order: usize,
}

impl<S: JetField> OperatorFamily<S> {
    /// Apply the separable operator whose axis-`k` factor carries power
    /// `powers[k]`.
    pub fn apply(&self, powers: &[u8], values: &[S]) -> Vec<S> {
        let mut current = values.to_vec();
        for (axis, family) in self.per_axis.iter().enumerate() {
            let power = usize::from(powers.get(axis).copied().unwrap_or(0));
            current = apply_axis(&current, &family[power], self.order, axis);
        }
        current
    }

    /// The plain (power-zero) operator.
    pub fn plain(&self, values: &[S]) -> Vec<S> {
        let zeros = vec![0u8; self.per_axis.len()];
        self.apply(&zeros, values)
    }
}

/// Build the forward (predict) operators across one gap for innovation
/// powers `0..=max_power`: input values on `from`, output values on `to`,
/// `F_b[f](z') = ∫ N(z'; φz, q) u^b f(z) dz` with `u = (z' − φz)/√q` the
/// standardised innovation.
pub(crate) fn forward_operators<S: JetField>(
    gh: &GaussHermite,
    from: &Grid<S>,
    to: &Grid<S>,
    transitions: &[AtomTransition<S>],
    max_power: u8,
) -> OperatorFamily<S> {
    let g = gh.order;
    let powers = usize::from(max_power) + 1;
    let mut per_axis = Vec::with_capacity(from.dimension());
    for (axis, transition) in transitions.iter().enumerate() {
        let old = &from.axes[axis];
        let new = &to.axes[axis];
        let phi = &transition.phi;
        let q = &transition.innovation;
        let root_q = sqrt(q);
        let sigma2 = square(&old.sigma);
        let tau2 = square(phi).mul(&sigma2).add(q);
        let inv_tau2 = recip(&tau2);
        let ratio = sqrt(&q.mul(&inv_tau2));
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
        // 1 / e(z_i) with e = N(z_i; mu, sigma²): √(2π) σ e^{x_i²}.
        let inverse_envelope: Vec<S> = gh
            .nodes
            .iter()
            .map(|&x| {
                old.sigma
                    .scale((2.0 * std::f64::consts::PI).sqrt() * (x * x).exp())
            })
            .collect();
        let mut matrices = vec![vec![old.mu.constant_like(0.0); g * g]; powers];
        for j in 0..g {
            let d = new.points[j].sub(&phi_mu);
            let gauss = normal_density(&new.points[j], &phi_mu, &tau2);
            let centre = phi
                .mul(&old.sigma)
                .mul(&d)
                .mul(&inv_tau2)
                .scale(1.0 / std::f64::consts::SQRT_2);
            let mut accumulated = vec![vec![old.mu.constant_like(0.0); g]; powers];
            for (l, &x) in gh.nodes.iter().enumerate() {
                // The interpolant is used as the polynomial it is, inside
                // and beyond the hull: the source envelope's Gaussian decay
                // beats the polynomial's growth there, and a clamp would put
                // a kink into an otherwise smooth objective.
                let raw = centre.add(&ratio.scale(x));
                let basis = gh.lagrange_basis(&raw);
                let u = d.mul(&u_offset_factor).add(&u_slope.scale(x));
                let mut weight = old.mu.constant_like(gh.normal_weights[l]);
                for power in 0..powers {
                    if power > 0 {
                        weight = weight.mul(&u);
                    }
                    for i in 0..g {
                        accumulated[power][i] = accumulated[power][i].add(&basis[i].mul(&weight));
                    }
                }
            }
            for power in 0..powers {
                for i in 0..g {
                    matrices[power][j * g + i] = gauss
                        .mul(&accumulated[power][i])
                        .mul(&inverse_envelope[i]);
                }
            }
        }
        per_axis.push(matrices);
    }
    OperatorFamily { per_axis, order: g }
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
            let spread = sqrt(&transition.innovation.scale(2.0));
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
pub(crate) fn log_sum_exp<S: JetField>(terms: &[S]) -> S {
    let shift = terms
        .iter()
        .map(|t| t.value())
        .fold(f64::NEG_INFINITY, f64::max);
    let sum = terms
        .iter()
        .fold(terms[0].constant_like(0.0), |acc, t| acc.add(&exp(&add_real(t, -shift))));
    add_real(&super::scalar::ln(&sum), shift)
}
