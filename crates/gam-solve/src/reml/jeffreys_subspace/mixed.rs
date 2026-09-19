//! Mixed derivatives in a fixed spectral frame. Higher divided differences
//! carry eigenvector motion, including repeated interior eigenvalues, without
//! differentiating a particular choice of eigenvectors.
use super::*;

/// Divided differences of the capped inverse and its first two floor partials.
/// Same-piece rational identities avoid subtraction at repeated/nearby nodes.
pub(super) fn inverse_difference(nodes: &[f64], floor: f64, floor_order: usize) -> f64 {
    if nodes.iter().all(|&x| x < floor) && nodes.iter().any(|&x| x < 0.0) {
        return below_floor_difference(nodes, floor, floor_order);
    }
    let cap = floor.max(CONDITIONING_GATE_ABSOLUTE_CLEAR);
    let piece = |x: f64| inverse_kernel_piece(x, floor, cap);
    let branch = piece(nodes[0]);
    if nodes.iter().all(|&x| piece(x) == branch) {
        let sign = if nodes.len() % 2 == 1 { 1.0 } else { -1.0 };
        return match branch {
            3 => {
                let coefficient = match floor_order {
                    0 => cap,
                    1 if floor > CONDITIONING_GATE_ABSOLUTE_CLEAR => 1.0,
                    _ => 0.0,
                };
                sign * coefficient
                    * nodes.iter().map(|x| x.recip()).product::<f64>()
                    * nodes.iter().map(|x| x.recip()).sum::<f64>()
            }
            2 => {
                if floor_order == 0 {
                    sign * nodes.iter().map(|x| x.recip()).product::<f64>()
                } else {
                    0.0
                }
            }
            // The floor plateau; a node set holding a negative node left above.
            _ => {
                if nodes.len() == 1 {
                    match floor_order {
                        0 => floor.recip(),
                        1 => -floor.recip().powi(2),
                        _ => 2.0 * floor.recip().powi(3),
                    }
                } else {
                    0.0
                }
            }
        };
    }
    let mut storage = [0.0; 4];
    let sorted = &mut storage[..nodes.len()];
    sorted.copy_from_slice(nodes);
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len() - 1;
    (inverse_difference(&sorted[1..], floor, floor_order)
        - inverse_difference(&sorted[..last], floor, floor_order))
        / (sorted[last] - sorted[0])
}

/// [`inverse_difference`] on nodes below the floor, at least one of them on the
/// bottom saturation `d = w(λ/floor)/floor`, `w(t) = 1/(1 + t⁴)` (gam#2982).
///
/// Below the floor `∂ʲ_floor d` is its plateau value `(−1)ʲ j!/floor^{j+1}` plus a
/// remainder `ρ_j` that vanishes on the plateau and is `O(t⁴)` below zero. A
/// divided difference of two or more nodes does not see the constant, so it is
/// the divided difference of `ρ_j`, formed without subtracting two plateau-sized
/// values: [`bottom_saturation_difference`] when every node is negative, and the
/// recursion on `ρ_j` otherwise.
fn below_floor_difference(nodes: &[f64], floor: f64, floor_order: usize) -> f64 {
    if let [lam] = nodes {
        return match floor_order {
            0 => floored_inverse(*lam, floor),
            1 => floored_inverse_floor_sensitivity(*lam, floor),
            _ => floored_inverse_floor_second_sensitivity(*lam, floor),
        };
    }
    bottom_remainder_difference(nodes, floor, floor_order)
}

/// The divided difference of the remainder `ρ_j` of [`below_floor_difference`].
fn bottom_remainder_difference(nodes: &[f64], floor: f64, floor_order: usize) -> f64 {
    if let [lam] = nodes {
        if *lam >= 0.0 {
            return 0.0;
        }
        // With τ = t⁴w = 1 − w: ρ₀ = −τ/f, ρ₁ = (τ − t·w')/f², ρ₂ = (−2τ + 4t·w' + t²·w'')/f³.
        let profile = BottomProfile::new(*lam, floor);
        let tau = profile.monomial(4, 0);
        return match floor_order {
            0 => -tau / floor,
            1 => (tau - profile.monomial(1, 1)) / (floor * floor),
            _ => {
                (-2.0 * tau + 4.0 * profile.monomial(1, 1) + profile.monomial(2, 2))
                    / (floor * floor * floor)
            }
        };
    }
    if nodes.iter().all(|&x| x < 0.0) {
        return bottom_saturation_difference(nodes, floor, floor_order);
    }
    if nodes.iter().all(|&x| x >= 0.0) {
        return 0.0;
    }
    let mut storage = [0.0; 4];
    let sorted = &mut storage[..nodes.len()];
    sorted.copy_from_slice(nodes);
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len() - 1;
    (bottom_remainder_difference(&sorted[1..], floor, floor_order)
        - bottom_remainder_difference(&sorted[..last], floor, floor_order))
        / (sorted[last] - sorted[0])
}

/// A complex number, for the partial fractions of the bottom profile.
#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self { re: self.re * factor, im: self.im * factor }
    }

    fn recip(self) -> Self {
        let norm = self.re.hypot(self.im);
        let (re, im) = (self.re / norm, self.im / norm);
        Self { re: re / norm, im: -im / norm }
    }
}

/// Complete homogeneous symmetric polynomials `h_k` of the nodes, advanced one
/// degree at a time: `h_k(x₀..x_l) = h_k(x₀..x_{l−1}) + x_l·h_{k−1}(x₀..x_l)`.
struct CompleteHomogeneous {
    nodes: [f64; 4],
    len: usize,
    /// `values[l] = h_k(x₀..x_l)` at the current degree `k`.
    values: [f64; 4],
}

impl CompleteHomogeneous {
    fn new(nodes: &[f64]) -> Self {
        let mut stored = [0.0; 4];
        stored[..nodes.len()].copy_from_slice(nodes);
        Self { nodes: stored, len: nodes.len(), values: [1.0; 4] }
    }

    fn advance(&mut self, degrees: usize) {
        for _ in 0..degrees {
            let mut previous = 0.0;
            for l in 0..self.len {
                previous += self.nodes[l] * self.values[l];
                self.values[l] = previous;
            }
        }
    }

    fn value(&self) -> f64 {
        self.values[self.len - 1]
    }
}

/// `Σ_{m≥1} term(m)` for a series whose terms fall geometrically, summed until a
/// term neither changes the sum nor exceeds its predecessor.
fn geometric_series_sum(mut term: impl FnMut(usize) -> f64) -> f64 {
    let mut sum = term(1);
    let mut previous = sum.abs();
    for m in 2.. {
        let next = term(m);
        let settled = sum + next == sum && next.abs() <= previous;
        sum += next;
        previous = next.abs();
        if settled {
            break;
        }
    }
    sum
}

/// Divided difference of `∂ʲ_floor d`, `d(λ) = w(λ/floor)/floor`, over two to four
/// negative nodes, `t_i = λ_i/floor`, with the floor partial taken at fixed `λ`.
///
/// `w = Σ_p a_p/(t − p)` over the roots of `p⁴ = −1`, `a_p = −p/4`, so the
/// difference is `2 Re Σ_{p = e^{iπ/4}, e^{3iπ/4}} (−1)ⁿ a_p pʲ Πζ · P_j/floor^{n+1+j}`
/// with `ζ_i = 1/(t_i − p)` and `P₀ = 1`, `P₁ = Σζ`, `P₂ = (Σζ)² + Σζ²`. That form
/// is exact at repeated nodes, but its two conjugate pairs cancel to the true
/// `O(t^{4−n})` when every `|t| ≪ 1` and to `O(|t|^{−n−4})` when every `|t| ≫ 1`, so
/// there the difference is the Taylor series of `w` about `0`, respectively `∞`:
///
/// * `max |t| ≤ ½`: `Σ_{m≥1} (−1)ᵐ ∏_{i<j}(−4m − 1 − i) h_{4m−n}(t)/floor^{n+1+j}`;
/// * `min |t| ≥ 2`: `(−1)ⁿ Π(1/t) Σ_{m≥1} (−1)^{m+1} ∏_{i<j}(4m − 1 − i) h_{4m−1}(1/t)/floor^{n+1+j}`.
///
/// The thresholds `½` and `2` are this method's choice of where to change forms:
/// they bound each series' term ratio by `2⁻⁴` up to its polynomial factors, and
/// the product form's cancellation between them by a fixed power of two.
fn bottom_saturation_difference(nodes: &[f64], floor: f64, floor_order: usize) -> f64 {
    let n = nodes.len() - 1;
    let mut storage = [0.0; 4];
    let t = &mut storage[..nodes.len()];
    for (slot, &lam) in t.iter_mut().zip(nodes) {
        *slot = lam / floor;
    }
    let scale = floor.powi(-((n + 1 + floor_order) as i32));
    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
    let falling = |start: f64, step: f64| (0..floor_order).map(|i| start + step * i as f64).product::<f64>();
    if t.iter().all(|x| x.abs() <= 0.5) {
        let mut h = CompleteHomogeneous::new(t);
        let mut degree = 0;
        return scale
            * geometric_series_sum(|m| {
                let target = 4 * m - n;
                h.advance(target - degree);
                degree = target;
                let alternating = if m % 2 == 0 { 1.0 } else { -1.0 };
                alternating * falling(-((4 * m + 1) as f64), -1.0) * h.value()
            });
    }
    if t.iter().all(|x| x.abs() >= 2.0) {
        let mut inverse = [0.0; 4];
        for (slot, &x) in inverse.iter_mut().zip(t.iter()) {
            *slot = x.recip();
        }
        let inverse = &inverse[..nodes.len()];
        let mut h = CompleteHomogeneous::new(inverse);
        let mut degree = 0;
        let prefactor = sign * inverse.iter().product::<f64>();
        return scale
            * prefactor
            * geometric_series_sum(|m| {
                let target = 4 * m - 1;
                h.advance(target - degree);
                degree = target;
                let alternating = if m % 2 == 0 { -1.0 } else { 1.0 };
                alternating * falling((4 * m - 1) as f64, -1.0) * h.value()
            });
    }
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let mut total = 0.0;
    for root in [Complex { re: half, im: half }, Complex { re: -half, im: half }] {
        let mut product = Complex { re: 1.0, im: 0.0 };
        let mut s1 = Complex { re: 0.0, im: 0.0 };
        let mut s2 = Complex { re: 0.0, im: 0.0 };
        for &x in t.iter() {
            let zeta = Complex { re: x - root.re, im: -root.im }.recip();
            product = product.mul(zeta);
            s1 = s1.add(zeta);
            s2 = s2.add(zeta.mul(zeta));
        }
        let (power, polynomial) = match floor_order {
            0 => (Complex { re: 1.0, im: 0.0 }, Complex { re: 1.0, im: 0.0 }),
            1 => (root, s1),
            _ => (root.mul(root), s1.mul(s1).add(s2)),
        };
        total += root.scale(-0.25).mul(power).mul(product).mul(polynomial).re;
    }
    2.0 * sign * scale * total
}

/// Which smooth piece of the capped inverse holds `x`: saturated below zero,
/// the floor plateau, the floored inverse, or the capped inverse.
fn inverse_kernel_piece(x: f64, floor: f64, cap: f64) -> u8 {
    if x >= cap {
        3
    } else if x >= floor {
        2
    } else if x >= 0.0 {
        1
    } else {
        0
    }
}

/// Divided differences of the capped inverse on one reduced spectrum, for every
/// node set the Fréchet rows read.
///
/// Each `inverse_frechet_rows` call evaluates the same `[λ_i, λ_k, λ_j]` triples,
/// and the second-order map reads `[λ_i, λ_k, λ_l, λ_j]` for all `m⁴` index tuples
/// of a pair. Recomputing every one by sort-and-recurse was the largest self time
/// of the survival marginal-slope outer Hessian (#979). The spectrum is fixed for a
/// drift base, so pairs and triples are tabulated once in exactly the node order the
/// rows ask for, and a four-node value is formed either from its own same-piece
/// closed form or from the two sorted triples `inverse_difference` would recurse
/// into — the same operations in the same order, so every value is unchanged.
pub(super) struct InverseDividedDifferences {
    m: usize,
    floor: f64,
    cap: f64,
    evals: Vec<f64>,
    pieces: Vec<u8>,
    /// `pairs[order][i·m + j] = inverse_difference(&[λ_i, λ_j], floor, order)`.
    pairs: [Vec<f64>; 3],
    /// `triples[order][(i·m + k)·m + j] = inverse_difference(&[λ_i, λ_k, λ_j], floor, order)`.
    triples: [Vec<f64>; 3],
    /// `λ_i⁻¹`, the factors of a four-node value on one branch above the floor.
    recips: Vec<f64>,
}

impl InverseDividedDifferences {
    pub(super) fn new(evals: &Array1<f64>, floor: f64) -> Self {
        let values: Vec<f64> = evals.iter().copied().collect();
        let m = values.len();
        let cap = floor.max(CONDITIONING_GATE_ABSOLUTE_CLEAR);
        let pieces = values
            .iter()
            .map(|&x| inverse_kernel_piece(x, floor, cap))
            .collect();
        let pairs = std::array::from_fn(|order| {
            let mut table = Vec::with_capacity(m * m);
            for &left in &values {
                for &right in &values {
                    table.push(inverse_difference(&[left, right], floor, order));
                }
            }
            table
        });
        let triples = std::array::from_fn(|order| {
            let mut table = Vec::with_capacity(m * m * m);
            for &first in &values {
                for &middle in &values {
                    for &last in &values {
                        table.push(inverse_difference(&[first, middle, last], floor, order));
                    }
                }
            }
            table
        });
        let recips = values.iter().map(|&x| x.recip()).collect();
        Self {
            m,
            floor,
            cap,
            evals: values,
            pieces,
            pairs,
            triples,
            recips,
        }
    }

    fn triple(&self, order: usize, first: usize, middle: usize, last: usize) -> f64 {
        self.triples[order][(first * self.m + middle) * self.m + last]
    }

    /// When every eigenvalue sits in one piece of the capped inverse, `quadruple([i, k, l, j])`
    /// is `scale · d_i d_k d_l d_j · τ`: `τ = 1` on the floored inverse, and
    /// `τ = t_i + t_k + t_l + t_j` with `t = d` on the capped inverse. These are the same
    /// closed forms `quadruple` evaluates. `None` when the spectrum spans pieces, and on the
    /// bottom saturation, whose differences do not factor node by node (gam#2982).
    fn separable_quadruple(&self) -> Option<SeparableQuadruple> {
        let branch = *self.pieces.first()?;
        if self.pieces.iter().any(|&piece| piece != branch) {
            return None;
        }
        let (scale, factors, tilted) = match branch {
            3 => (-self.cap, self.recips.clone(), true),
            2 => (-1.0, self.recips.clone(), false),
            1 => (0.0, vec![0.0; self.m], false),
            _ => return None,
        };
        Some(SeparableQuadruple {
            scale,
            factors,
            tilted,
        })
    }

    /// `inverse_difference(&[λ_a, λ_b, λ_c, λ_d], floor, 0)`.
    fn quadruple(&self, nodes: [usize; 4]) -> f64 {
        let branch = self.pieces[nodes[0]];
        if nodes.iter().all(|&index| self.pieces[index] == branch) {
            let sign = -1.0;
            return match branch {
                3 => {
                    let [a, b, c, d] = nodes.map(|index| self.recips[index]);
                    let product = a * b * c * d;
                    let sum = a + b + c + d;
                    sign * self.cap * product * sum
                }
                2 => {
                    let [a, b, c, d] = nodes.map(|index| self.recips[index]);
                    sign * (a * b * c * d)
                }
                1 => 0.0,
                _ => inverse_difference(&nodes.map(|index| self.evals[index]), self.floor, 0),
            };
        }
        let mut sorted = nodes;
        for position in 1..4 {
            let mut cursor = position;
            while cursor > 0
                && self.evals[sorted[cursor - 1]].total_cmp(&self.evals[sorted[cursor]])
                    == std::cmp::Ordering::Greater
            {
                sorted.swap(cursor - 1, cursor);
                cursor -= 1;
            }
        }
        (self.triple(0, sorted[1], sorted[2], sorted[3]) - self.triple(0, sorted[0], sorted[1], sorted[2]))
            / (self.evals[sorted[3]] - self.evals[sorted[0]])
    }
}

/// The factored four-node values of a spectrum inside one piece; see
/// [`InverseDividedDifferences::separable_quadruple`].
struct SeparableQuadruple {
    scale: f64,
    factors: Vec<f64>,
    /// Whether `τ = Σ` of the four node factors (capped inverse) rather than 1.
    tilted: bool,
}

/// `D³f[E, F, A]` for every axis row `A` through the `m⁴` coefficient loop: for one output
/// row at a time the linear map is assembled with `coefficient(i, k, l, j)` and contracted
/// with BLAS-3. The m²-by-m² Loewner map is never allocated.
fn loewner_second_rows(
    rows: &Array2<f64>,
    e: &Array2<f64>,
    f: &Array2<f64>,
    m: usize,
    coefficient: impl Fn(usize, usize, usize, usize) -> f64,
) -> Array2<f64> {
    let squared = m * m;
    let e = e.as_standard_layout();
    let f = f.as_standard_layout();
    let e = e.as_slice().expect("standard-layout spectral direction");
    let f = f.as_slice().expect("standard-layout spectral direction");
    let mut out = Array2::zeros(rows.raw_dim());
    let mut weights = Array2::<f64>::zeros((m, squared));
    for i in 0..m {
        weights.fill(0.0);
        let w = weights
            .as_slice_mut()
            .expect("freshly allocated weights are contiguous");
        for j in 0..m {
            for k in 0..m {
                for l in 0..m {
                    let c = coefficient(i, k, l, j);
                    let (ik, kl, lj) = (i * m + k, k * m + l, l * m + j);
                    w[j * squared + l * m + j] += c * (e[ik] * f[kl] + f[ik] * e[kl]);
                    w[j * squared + kl] += c * (e[ik] * f[lj] + f[ik] * e[lj]);
                    w[j * squared + ik] += c * (e[kl] * f[lj] + f[kl] * e[lj]);
                }
            }
        }
        out.slice_mut(ndarray::s![.., i * m..(i + 1) * m])
            .assign(&rows.dot(&weights.t()));
    }
    out
}

/// [`loewner_second_rows`] on a spectrum inside one piece. With `D = diag(d)` and the node
/// factors of `τ` on each of the four slots, row `r` is
/// `scale · D · Σ_σ X_σ D Y_σ D Z_σ · D` over the six orderings `(X, Y, Z)` of `(E, F, A_r)`
/// on the floored inverse, and `scale · D · (T S + S T + S₁ + S₂) · D` on the tilted capped inverse,
/// where `S₁`, `S₂` carry `D T` in the first and second interior slot. With
/// `P = E D F + F D E` formed once per call, the untilted row costs six `m × m` products.
/// No symmetry of `E`, `F` or `A` is assumed.
fn separable_second_frechet_rows(
    separable: &SeparableQuadruple,
    m: usize,
    rows: &Array2<f64>,
    e: &Array2<f64>,
    f: &Array2<f64>,
) -> Array2<f64> {
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
    let d = Array1::from(separable.factors.clone());
    let column_scaled = |matrix: &Array2<f64>, weights: &Array1<f64>| matrix * weights;
    let row_scaled = |weights: &Array1<f64>, matrix: &Array2<f64>| {
        &weights.view().insert_axis(ndarray::Axis(1)) * matrix
    };
    let ed = column_scaled(e, &d);
    let fd = column_scaled(f, &d);
    let p = ed.dot(f) + fd.dot(e);
    let dt = separable.tilted.then(|| &d * &d);
    let tilted_pair = dt.as_ref().map(|dt| {
        let edt = column_scaled(e, dt);
        let fdt = column_scaled(f, dt);
        let pt = edt.dot(f) + fdt.dot(e);
        (edt, fdt, pt)
    });
    let mut out = Array2::<f64>::zeros(rows.raw_dim());
    out.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .zip(rows.axis_iter(ndarray::Axis(0)).into_par_iter())
        .for_each(|(mut target, source)| {
            let a = source
                .to_owned()
                .into_shape_with_order((m, m))
                .expect("each axis row holds one m x m block");
            let da = row_scaled(&d, &a);
            let ad = column_scaled(&a, &d);
            let base = p.dot(&da) + ad.dot(&p) + ed.dot(&ad).dot(f) + fd.dot(&ad).dot(e);
            let total = match (dt.as_ref(), tilted_pair.as_ref()) {
                (Some(dt), Some((edt, fdt, pt))) => {
                    let dta = row_scaled(dt, &a);
                    let adt = column_scaled(&a, dt);
                    let first = pt.dot(&da) + adt.dot(&p) + edt.dot(&ad).dot(f) + fdt.dot(&ad).dot(e);
                    let second =
                        p.dot(&dta) + ad.dot(pt) + ed.dot(&adt).dot(f) + fd.dot(&adt).dot(e);
                    row_scaled(&d, &base) + column_scaled(&base, &d) + first + second
                }
                _ => base,
            };
            let values = target
                .as_slice_mut()
                .expect("owned axis rows are contiguous");
            for i in 0..m {
                for j in 0..m {
                    values[i * m + j] = separable.scale * d[i] * total[[i, j]] * d[j];
                }
            }
        });
    out
}

impl JeffreysHphiDriftBase {
    fn divided_differences(&self) -> &InverseDividedDifferences {
        self.divided_differences
            .get_or_init(|| InverseDividedDifferences::new(&self.evals, self.floor))
    }

    /// `aw_rows · a_rowsᵀ`, the Gram the gate's motion scales, formed on first use.
    fn weighted_gram(&self) -> &Array2<f64> {
        self.weighted_gram
            .get_or_init(|| self.aw_rows.dot(&self.a_rows.t()))
    }

    /// Apply the frozen-policy derivative of the omitted true-Hessian completion to a
    /// coefficient direction, on already-rotated objects: `e = Uᵀ H[u] U`, and `a`, `da`
    /// the rotated rows of `{H[v, e_a]}` and `{H[u, v, e_a]}`. This contracts the fifth
    /// likelihood derivative directly, without assembling a third coefficient tensor.
    pub(super) fn completion_drift_from_rows(
        &self,
        e: &Array2<f64>,
        a: &Array2<f64>,
        da: &Array2<f64>,
    ) -> Result<Array1<f64>, String> {
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let dg = g_min * e[[self.idx_min, self.idx_min]]
            + g_max * e[[self.idx_max, self.idx_max]];
        let dfloor = if self.floor_in_relative_regime {
            REDUCED_INFO_RELATIVE_FLOOR * e[[self.idx_max, self.idx_max]]
        } else {
            0.0
        };
        let mut result = Array1::<f64>::zeros(self.p);
        for i in 0..self.m {
            let kernel = inverse_difference(&[self.evals[i]], self.floor, 0);
            let floor_motion = dfloor * inverse_difference(&[self.evals[i]], self.floor, 1);
            for axis in 0..self.p {
                result[axis] -= 0.5 * (
                    self.gate_weight * (kernel * da[[axis, i * self.m + i]]
                        + floor_motion * a[[axis, i * self.m + i]])
                    + dg * kernel * a[[axis, i * self.m + i]]);
            }
            for j in 0..self.m {
                let dk = inverse_difference(&[self.evals[i], self.evals[j]], self.floor, 0)
                    * e[[i, j]];
                for axis in 0..self.p {
                    result[axis] -= 0.5 * self.gate_weight * dk * a[[axis, i * self.m + j]];
                }
            }
        }
        if result.iter().any(|v| !v.is_finite()) {
            return Err("Jeffreys completion drift produced a nonfinite response".into());
        }
        Ok(result)
    }

    /// The gate and floor channels along `u`, `w` and `(u, w)`, and the first and second
    /// Fréchet weights of the capped inverse in the base eigenbasis: everything the
    /// frozen-policy half of `completion_second_drift_matrix` reads from two directions
    /// (gam#2894).
    fn frozen_second_drift_weights(
        &self,
        e_u: &Array2<f64>,
        e_w: &Array2<f64>,
        e_uw: &Array2<f64>,
    ) -> FrozenSecondDriftWeights {
        let m = self.m;
        let (imin, imax) = (self.idx_min, self.idx_max);
        let spectral_scale = self.evals.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tie_tolerance = 64.0 * f64::EPSILON * spectral_scale;
        let second_extreme = |e: usize| {
            e_uw[[e, e]] + simple_eigenvalue_second_form(&self.evals, tie_tolerance, e, e_u, e_w)
        };
        let (lmin_u, lmax_u) = (e_u[[imin, imin]], e_u[[imax, imax]]);
        let (lmin_w, lmax_w) = (e_w[[imin, imin]], e_w[[imax, imax]]);
        let (lmin_uw, lmax_uw) = (second_extreme(imin), second_extreme(imax));
        let (g1, g2) = conditioning_gate_weight_grad(self.evals[imin], self.evals[imax]);
        let (g11, g12, g22) = conditioning_gate_weight_hess(self.evals[imin], self.evals[imax]);
        let gate_u = g1 * lmin_u + g2 * lmax_u;
        let gate_w = g1 * lmin_w + g2 * lmax_w;
        let gate_uw = g11 * lmin_u * lmin_w
            + g12 * (lmin_u * lmax_w + lmax_u * lmin_w)
            + g22 * lmax_u * lmax_w
            + g1 * lmin_uw
            + g2 * lmax_uw;
        let rate = if self.floor_in_relative_regime {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let (floor_u, floor_w, floor_uw) = (rate * lmax_u, rate * lmax_w, rate * lmax_uw);
        let divided = self.divided_differences();
        let kernel: Vec<f64> = (0..m).map(|i| inverse_difference(&[self.evals[i]], self.floor, 0)).collect();
        let kernel_floor: Vec<f64> =
            (0..m).map(|i| inverse_difference(&[self.evals[i]], self.floor, 1)).collect();
        let kernel_floor_floor: Vec<f64> =
            (0..m).map(|i| inverse_difference(&[self.evals[i]], self.floor, 2)).collect();
        // `D²f[E_u, E_w]_ij + Df[E_uw]_ij + floor channels`: the spectral weights the second
        // Fréchet contraction reads, formed once for every axis.
        let mut weight_uw = Array2::<f64>::zeros((m, m));
        let mut weight_u = Array2::<f64>::zeros((m, m));
        let mut weight_w = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let pair = divided.pairs[0][i * m + j];
                let pair_floor = divided.pairs[1][i * m + j];
                let mut second = pair * e_uw[[i, j]]
                    + pair_floor * (floor_u * e_w[[i, j]] + floor_w * e_u[[i, j]]);
                for k in 0..m {
                    second += divided.triple(0, i, k, j)
                        * (e_u[[i, k]] * e_w[[k, j]] + e_w[[i, k]] * e_u[[k, j]]);
                }
                weight_uw[[i, j]] = second;
                weight_u[[i, j]] = pair * e_u[[i, j]];
                weight_w[[i, j]] = pair * e_w[[i, j]];
            }
            weight_uw[[i, i]] += floor_u * floor_w * kernel_floor_floor[i] + floor_uw * kernel_floor[i];
            weight_u[[i, i]] += floor_u * kernel_floor[i];
            weight_w[[i, i]] += floor_w * kernel_floor[i];
        }
        FrozenSecondDriftWeights {
            gate_u,
            gate_w,
            gate_uw,
            kernel,
            weight_u,
            weight_w,
            weight_uw,
        }
    }

    /// The first β-drift of the complete second-order completion as a matrix,
    /// `D_u completion` (gam#2894). With `completion = −½·G·CTH(K) − CTH(E) − R`, where
    /// `CTH(W)_ab = ⟨W, H''[e_a, e_b]⟩`, `K` the capped inverse on the Jeffreys span and
    /// `(E, R)` the gate/floor motion of [`JointJeffreysHessianMotion`],
    ///
    /// ```text
    /// D_u completion = CTH(W₀) + CTH_u(W₁) − D_u R,
    /// W₀ = −½·G_u·K − ½·G·K_u − D_u E,    W₁ = −½·G·K − E,
    /// ```
    ///
    /// `CTH_u(W)_ab = ⟨W, H'''[u, e_a, e_b]⟩`. The two contractions are the caller's (a
    /// family's contracted-trace hooks); everything spectral is formed here from
    /// `H[u]` and the rotated `{H''[u, e_a]}`. The rotated axes feed only the gate and
    /// floor motion `(E, R)`, so they are read only where [`Self::hessian_motion_active`]
    /// holds, and are required there. Every column is the motion-completed
    /// [`Self::completion_drift_action_from_rotated`] along that axis.
    pub fn completion_drift_matrix(
        &self,
        pert_u: &Array2<f64>,
        second_u: Option<&JeffreysRotatedAxes>,
        contracted: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        contracted_along_u: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
    ) -> Result<Array2<f64>, String> {
        let (p, m) = (self.p, self.m);
        if pert_u.dim() != (p, p) || second_u.is_some_and(|axes| axes.rows.dim() != (p, m * m)) {
            return Err("Jeffreys completion drift matrix dimension mismatch".into());
        }
        let basis = &self.ambient_eigenbasis;
        let e_u = symmetric_basis_contraction(pert_u.view(), basis.view());
        let a_rows = &self.a_rows;
        let (imin, imax) = (self.idx_min, self.idx_max);
        let evals = &self.evals;
        let floor = self.floor;
        let gate = self.gate_weight;
        let spectral_scale = evals.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tie_tolerance = 64.0 * f64::EPSILON * spectral_scale;
        let rate = if self.floor_in_relative_regime {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let (lambda_min, lambda_max) = (evals[imin], evals[imax]);
        let (g1, g2) = conditioning_gate_weight_grad(lambda_min, lambda_max);
        let divided = self.divided_differences();
        let inverse: Vec<f64> = (0..m).map(|i| floored_inverse(evals[i], floor)).collect();
        let inverse_floor: Vec<f64> =
            (0..m).map(|i| floored_inverse_floor_sensitivity(evals[i], floor)).collect();
        let lambda_min_u = e_u[[imin, imin]];
        let lambda_max_u = e_u[[imax, imax]];
        let gate_u = g1 * lambda_min_u + g2 * lambda_max_u;
        let floor_u = rate * lambda_max_u;
        let ambient = |reduced: &Array2<f64>| basis.dot(reduced).dot(&basis.t());
        let kernel = ambient(&Array2::from_diag(&Array1::from_vec(inverse.clone())));
        let mut kernel_u_reduced = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                kernel_u_reduced[[i, j]] = divided.pairs[0][i * m + j] * e_u[[i, j]];
            }
            kernel_u_reduced[[i, i]] += floor_u * inverse_floor[i];
        }
        let kernel_u = ambient(&kernel_u_reduced);
        let mut weight_now = &kernel * (-0.5 * gate_u) + &kernel_u * (-0.5 * gate);
        let mut weight_along_u = &kernel * (-0.5 * gate);
        let motion = self.hessian_motion_active();
        let mut remainder_drift = Array2::<f64>::zeros((p, p));
        if motion {
            let b_u = &second_u
                .ok_or("Jeffreys completion drift matrix requires H²[u,·] where the gate or the floor moves")?
                .rows;
            let (g11, g12, g22) = conditioning_gate_weight_hess(lambda_min, lambda_max);
            let (g111, g112, g122, g222) = conditioning_gate_weight_third(lambda_min, lambda_max);
            let mut ungated = 0.0_f64;
            let (mut s_f, mut s_ff, mut s_fff) = (0.0_f64, 0.0_f64, 0.0_f64);
            let mut inverse_lambda_floor = vec![0.0_f64; m];
            let mut inverse_floor_floor = vec![0.0_f64; m];
            for i in 0..m {
                let lambda = evals[i];
                ungated += jeffreys_antiderivative(lambda, floor);
                s_f += jeffreys_antiderivative_floor_sensitivity(lambda, floor);
                s_ff += jeffreys_antiderivative_floor_second_sensitivity(lambda, floor);
                s_fff += jeffreys_antiderivative_floor_third_sensitivity(lambda, floor);
                inverse_lambda_floor[i] = floored_inverse_lambda_floor_sensitivity(lambda, floor);
                inverse_floor_floor[i] = floored_inverse_floor_second_sensitivity(lambda, floor);
            }
            ungated *= 0.5;
            let value_u = 0.5 * (0..m).map(|i| inverse[i] * e_u[[i, i]]).sum::<f64>()
                + 0.5 * s_f * rate * lambda_max_u;
            let floor_trace_u = (0..m).map(|i| inverse_floor[i] * e_u[[i, i]]).sum::<f64>();
            let s_f_u = floor_trace_u + s_ff * rate * lambda_max_u;
            let s_ff_u = s_fff * rate * lambda_max_u
                + (0..m).map(|i| inverse_floor_floor[i] * e_u[[i, i]]).sum::<f64>();
            // Extreme weights of `E` and their drift.
            let omega_min = ungated * g1;
            let omega_max = ungated * g2 + 0.5 * gate * s_f * rate;
            let omega_min_u = value_u * g1 + ungated * (g11 * lambda_min_u + g12 * lambda_max_u);
            let omega_max_u = value_u * g2
                + ungated * (g12 * lambda_min_u + g22 * lambda_max_u)
                + 0.5 * rate * (gate_u * s_f + gate * s_f_u);
            let gap = |e: usize, j: usize| simple_eigenvalue_gap_inverse(evals, tie_tolerance, e, j);
            let eigenvector_drift = |e: usize| {
                let reduced = Array1::from_shape_fn(m, |j| e_u[[j, e]] * gap(e, j));
                basis.dot(&reduced)
            };
            let outer = |x: &Array1<f64>, y: &Array1<f64>| {
                Array2::from_shape_fn((x.len(), y.len()), |(i, j)| x[i] * y[j])
            };
            let mut extreme_weight = Array2::<f64>::zeros((p, p));
            let mut extreme_weight_u = Array2::<f64>::zeros((p, p));
            for (e, omega, omega_u) in [(imin, omega_min, omega_min_u), (imax, omega_max, omega_max_u)] {
                let z = basis.column(e).to_owned();
                let z_u = eigenvector_drift(e);
                extreme_weight.scaled_add(omega, &outer(&z, &z));
                extreme_weight_u.scaled_add(omega_u, &outer(&z, &z));
                extreme_weight_u.scaled_add(omega, &(outer(&z_u, &z) + outer(&z, &z_u)));
            }
            weight_now -= &extreme_weight_u;
            weight_along_u -= &extreme_weight;
            // Axis objects of `R` and their drift along u.
            let row_entry = |rows: &Array2<f64>, a: usize, i: usize, j: usize| rows[[a, i * m + j]];
            let q_min = Array1::from_shape_fn(p, |a| row_entry(a_rows, a, imin, imin));
            let q_max = Array1::from_shape_fn(p, |a| row_entry(a_rows, a, imax, imax));
            let floor_trace =
                Array1::from_shape_fn(p, |a| (0..m).map(|i| inverse_floor[i] * row_entry(a_rows, a, i, i)).sum::<f64>());
            let grad_u = Array1::from_shape_fn(p, |a| {
                0.5 * (0..m).map(|i| inverse[i] * row_entry(a_rows, a, i, i)).sum::<f64>()
                    + 0.5 * s_f * rate * q_max[a]
            });
            let grad_g = &q_min * g1 + &q_max * g2;
            // `λ_i,au = B̃_au[i,i] + D²λ_i[P̃_a, e_u]` for every eigenvalue.
            let lambda_au = |i: usize, a: usize| {
                let mut sum = row_entry(b_u, a, i, i);
                for j in 0..m {
                    let inv = gap(i, j);
                    if inv != 0.0 {
                        sum += 2.0 * row_entry(a_rows, a, i, j) * e_u[[i, j]] * inv;
                    }
                }
                sum
            };
            let lambda_min_au = Array1::from_shape_fn(p, |a| lambda_au(imin, a));
            let lambda_max_au = Array1::from_shape_fn(p, |a| lambda_au(imax, a));
            let divided_u = self.aw_rows.dot(&Array1::from_iter(e_u.iter().copied()));
            let value_au = Array1::from_shape_fn(p, |a| {
                let mut kernel_trace = 0.0_f64;
                let mut floor_cross = 0.0_f64;
                for i in 0..m {
                    kernel_trace += inverse[i] * row_entry(b_u, a, i, i);
                    floor_cross += inverse_floor[i]
                        * (row_entry(a_rows, a, i, i) * lambda_max_u + e_u[[i, i]] * q_max[a]);
                }
                0.5 * divided_u[a]
                    + 0.5 * kernel_trace
                    + 0.5 * rate * floor_cross
                    + 0.5 * s_ff * rate * rate * q_max[a] * lambda_max_u
                    + 0.5 * s_f * rate * lambda_max_au[a]
            });
            let gate_au = Array1::from_shape_fn(p, |a| {
                g11 * q_min[a] * lambda_min_u
                    + g12 * (q_min[a] * lambda_max_u + q_max[a] * lambda_min_u)
                    + g22 * q_max[a] * lambda_max_u
                    + g1 * lambda_min_au[a]
                    + g2 * lambda_max_au[a]
            });
            remainder_drift += &(outer(&gate_au, &grad_u)
                + outer(&grad_g, &value_au)
                + outer(&value_au, &grad_g)
                + outer(&grad_u, &gate_au));
            let quadratic = |x_min: &Array1<f64>, x_max: &Array1<f64>, y_min: &Array1<f64>, y_max: &Array1<f64>| {
                outer(x_min, y_min) * g11
                    + (outer(x_min, y_max) + outer(x_max, y_min)) * g12
                    + outer(x_max, y_max) * g22
            };
            let q2 = quadratic(&q_min, &q_max, &q_min, &q_max);
            let q2_u = outer(&q_min, &q_min) * (g111 * lambda_min_u + g112 * lambda_max_u)
                + (outer(&q_min, &q_max) + outer(&q_max, &q_min)) * (g112 * lambda_min_u + g122 * lambda_max_u)
                + outer(&q_max, &q_max) * (g122 * lambda_min_u + g222 * lambda_max_u)
                + quadratic(&lambda_min_au, &lambda_max_au, &q_min, &q_max)
                + quadratic(&q_min, &q_max, &lambda_min_au, &lambda_max_au);
            remainder_drift.scaled_add(value_u, &q2);
            remainder_drift.scaled_add(ungated, &q2_u);
            if rate != 0.0 {
                let floor_trace_au = Array1::from_shape_fn(p, |a| {
                    (0..m)
                        .map(|i| {
                            (inverse_lambda_floor[i] * e_u[[i, i]] + inverse_floor_floor[i] * floor_u)
                                * row_entry(a_rows, a, i, i)
                                + inverse_floor[i] * lambda_au(i, a)
                        })
                        .sum::<f64>()
                });
                let floor_part = (outer(&floor_trace, &q_max) + outer(&q_max, &floor_trace)) * (0.5 * rate)
                    + outer(&q_max, &q_max) * (0.5 * s_ff * rate * rate);
                let floor_part_u = (outer(&floor_trace_au, &q_max)
                    + outer(&floor_trace, &lambda_max_au)
                    + outer(&lambda_max_au, &floor_trace)
                    + outer(&q_max, &floor_trace_au))
                    * (0.5 * rate)
                    + outer(&q_max, &q_max) * (0.5 * s_ff_u * rate * rate)
                    + (outer(&lambda_max_au, &q_max) + outer(&q_max, &lambda_max_au)) * (0.5 * s_ff * rate * rate);
                remainder_drift.scaled_add(gate_u, &floor_part);
                remainder_drift.scaled_add(gate, &floor_part_u);
            }
            // `ω_e·E_e` with `E_e[a,b] = D²λ_e[P̃_a, P̃_b]`, and its drift
            // `D²λ_e[B̃_au, P̃_b] + D²λ_e[P̃_a, B̃_bu] + D³λ_e[P̃_a, P̃_b, e_u]`.
            for (e, omega, omega_u) in [(imin, omega_min, omega_min_u), (imax, omega_max, omega_max_u)] {
                let gi = Array1::from_shape_fn(m, |j| gap(e, j));
                let x = Array2::from_shape_fn((p, m), |(a, j)| row_entry(a_rows, a, e, j));
                let y = Array2::from_shape_fn((p, m), |(a, j)| row_entry(b_u, a, e, j));
                let doubled = Array2::from_diag(&gi.mapv(|value| 2.0 * value));
                let extreme_second = x.dot(&doubled).dot(&x.t());
                let x_hat = Array2::from_shape_fn((p, m), |(a, j)| x[[a, j]] * gi[j]);
                let c_hat = Array1::from_shape_fn(m, |j| e_u[[e, j]] * gi[j]);
                let v = Array2::from_shape_fn((p, m), |(a, i)| {
                    (0..m).map(|k| row_entry(a_rows, a, i, k) * c_hat[k]).sum::<f64>()
                });
                let s = x_hat.dot(&c_hat);
                let q_e = Array1::from_shape_fn(p, |a| row_entry(a_rows, a, e, e));
                let third = (x_hat.dot(&v.t()) + v.dot(&x_hat.t())) * 2.0
                    + x_hat.dot(&e_u).dot(&x_hat.t()) * 2.0
                    - (outer(&q_e, &s) + outer(&s, &q_e)) * 2.0
                    - x_hat.dot(&x_hat.t()) * (2.0 * e_u[[e, e]]);
                let extreme_second_u = y.dot(&doubled).dot(&x.t()) + x.dot(&doubled).dot(&y.t()) + third;
                remainder_drift.scaled_add(omega_u, &extreme_second);
                remainder_drift.scaled_add(omega, &extreme_second_u);
            }
        }
        let mut result = contracted(&weight_now)? + contracted_along_u(&weight_along_u)?;
        if motion {
            result -= &remainder_drift;
        }
        let mut result = result.as_standard_layout().to_owned();
        symmetrize_contiguous(&mut result);
        if result.iter().any(|value| !value.is_finite()) {
            return Err("Jeffreys completion drift matrix produced nonfinite curvature".into());
        }
        Ok(result)
    }

    /// The complete second β-drift of the second-order completion as a matrix,
    /// `D² completion[u, w]` (gam#2905): the frozen policy together with the gate and floor
    /// motion. With `completion = −½·G·CTH(K) − CTH(E) − R` ([`JointJeffreysHessianMotion`]),
    ///
    /// ```text
    /// D² completion[u, w] = CTH(W₀₀ − D²E) + CTH_w(W₀ᵤ − D_uE) + CTH_u(W₁w − D_wE)
    ///                       + CTH_uw(W₁₁ − E) − D²R[u, w],
    /// ```
    ///
    /// with the frozen weights `W₀₀ = −½(G_uw·K + G_u·K_w + G_w·K_u + G·K_uw)`,
    /// `W₀ᵤ = −½(G_u·K + G·K_u)`, `W₁w = −½(G_w·K + G·K_w)` and `W₁₁ = −½·G·K`, where `K`,
    /// `K_u`, `K_uw` are the capped inverse and its first and second Fréchet derivatives,
    /// floor motion included. `E` reads
    /// the extreme eigenvectors to second order. `R = ∇G⊗∇U + ∇U⊗∇G + U·Q₂ + G·F + Σ_e ω_e·E_e`
    /// reads the gate to its fourth partials, the ungated value and its floor channels to third
    /// order, and `E_e[a,b] = D²λ_e[P̃_a, P̃_b]` through the fourth simple-eigenvalue form. The
    /// motion inputs are the rotated `{H²[u, e_a]}`, `{H²[w, e_a]}` and `{H³[u, w, e_a]}`; they
    /// are read only where [`Self::hessian_motion_active`] holds, and are required there.
    pub fn completion_second_drift_matrix(
        &self,
        pert_u: &Array2<f64>,
        pert_w: &Array2<f64>,
        pert_uw: &Array2<f64>,
        second_u: Option<&JeffreysRotatedAxes>,
        second_w: Option<&JeffreysRotatedAxes>,
        third_uw: Option<&JeffreysRotatedAxes>,
        contracted: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        contracted_along_u: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        contracted_along_w: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
        contracted_along_uw: &dyn Fn(&Array2<f64>) -> Result<Array2<f64>, String>,
    ) -> Result<Array2<f64>, String> {
        let (p, m) = (self.p, self.m);
        if [pert_u, pert_w, pert_uw].iter().any(|h| h.dim() != (p, p)) {
            return Err("Jeffreys complete second completion drift dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let basis = &self.ambient_eigenbasis;
        let e_u = symmetric_basis_contraction(pert_u.view(), basis.view());
        let e_w = symmetric_basis_contraction(pert_w.view(), basis.view());
        let e_uw = symmetric_basis_contraction(pert_uw.view(), basis.view());
        let FrozenSecondDriftWeights {
            gate_u,
            gate_w,
            gate_uw,
            kernel,
            weight_u,
            weight_w,
            weight_uw,
        } = self.frozen_second_drift_weights(&e_u, &e_w, &e_uw);
        let g = self.gate_weight;
        let ambient = |reduced: &Array2<f64>| basis.dot(reduced).dot(&basis.t());
        let kernel = ambient(&Array2::from_diag(&Array1::from_vec(kernel)));
        let kernel_u = ambient(&weight_u);
        let kernel_w = ambient(&weight_w);
        let kernel_uw = ambient(&weight_uw);
        let mut now = (&kernel * gate_uw + &kernel_w * gate_u + &kernel_u * gate_w + &kernel_uw * g) * -0.5;
        let mut along_w = (&kernel * gate_u + &kernel_u * g) * -0.5;
        let mut along_u = (&kernel * gate_w + &kernel_w * g) * -0.5;
        let mut along_uw = &kernel * (-0.5 * g);
        let motion = if self.hessian_motion_active() {
            let missing = || {
                "Jeffreys complete second completion drift requires H²[u,·], H²[w,·] and \
                 H³[u,w,·] where the gate or the floor moves"
                    .to_string()
            };
            let b_u = &second_u.ok_or_else(missing)?.rows;
            let b_w = &second_w.ok_or_else(missing)?.rows;
            let t_uw = &third_uw.ok_or_else(missing)?.rows;
            if [b_u, b_w, t_uw].iter().any(|rows| rows.dim() != (p, m * m)) {
                return Err("Jeffreys complete second completion drift axis dimension mismatch".into());
            }
            let parts = self.second_drift_motion(&e_u, &e_w, &e_uw, b_u, b_w, t_uw);
            now -= &parts.extreme_weight_uw;
            along_w -= &parts.extreme_weight_u;
            along_u -= &parts.extreme_weight_w;
            along_uw -= &parts.extreme_weight;
            Some(parts.remainder_uw)
        } else {
            None
        };
        let mut result = contracted(&now)?
            + contracted_along_w(&along_w)?
            + contracted_along_u(&along_u)?
            + contracted_along_uw(&along_uw)?;
        if let Some(remainder) = motion.as_ref() {
            result -= remainder;
        }
        let mut result = result.as_standard_layout().to_owned();
        symmetrize_contiguous(&mut result);
        if result.iter().any(|value| !value.is_finite()) {
            return Err("Jeffreys complete second completion drift produced nonfinite curvature".into());
        }
        Ok(result)
    }

    /// The gate and floor motion of [`Self::completion_second_drift_matrix`] (gam#2905): the
    /// ambient extreme-eigenvector weight `E`, its drifts `D_uE`, `D_wE`, `D²E`, and `D²R[u, w]`.
    /// Inputs are `e_x = Uᵀ H[x] U` and the rotated axis rows `b_u = {Uᵀ H²[u, e_a] U}`, `b_w`
    /// and `t_uw = {Uᵀ H³[u, w, e_a] U}`. The ungated value `U = ½ Σ_i ψ(λ_i; floor)` is
    /// differentiated as `Φ(β, t)` at `t = floor(β)`: spectral channels at a fixed floor use the
    /// divided differences of `f = ψ'` and of its floor partials, and the floor's own motion
    /// `floor = REL·λ_max` enters through Faà di Bruno.
    fn second_drift_motion(
        &self,
        e_u: &Array2<f64>,
        e_w: &Array2<f64>,
        e_uw: &Array2<f64>,
        b_u: &Array2<f64>,
        b_w: &Array2<f64>,
        t_uw: &Array2<f64>,
    ) -> SecondDriftMotion {
        let (p, m) = (self.p, self.m);
        let basis = &self.ambient_eigenbasis;
        let a_rows = &self.a_rows;
        let (imin, imax) = (self.idx_min, self.idx_max);
        let evals = &self.evals;
        let floor = self.floor;
        let gate = self.gate_weight;
        let spectral_scale = evals.iter().fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tie_tolerance = 64.0 * f64::EPSILON * spectral_scale;
        let rate = if self.floor_in_relative_regime {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let (lambda_min, lambda_max) = (evals[imin], evals[imax]);
        // Gate partials in `(λ_min, λ_max)`, stored by how many slots read `λ_max`.
        let (g1, g2) = conditioning_gate_weight_grad(lambda_min, lambda_max);
        let (g11, g12, g22) = conditioning_gate_weight_hess(lambda_min, lambda_max);
        let (g111, g112, g122, g222) = conditioning_gate_weight_third(lambda_min, lambda_max);
        let (g1111, g1112, g1122, g1222, g2222) = conditioning_gate_weight_fourth(lambda_min, lambda_max);
        let gate1 = [g1, g2];
        let gate2 = [g11, g12, g22];
        let gate3 = [g111, g112, g122, g222];
        let gate4 = [g1111, g1112, g1122, g1222, g2222];
        let tensor2 = |i: usize, j: usize| gate2[i + j];
        let tensor3 = |i: usize, j: usize, k: usize| gate3[i + j + k];
        let tensor4 = |i: usize, j: usize, k: usize, l: usize| gate4[i + j + k + l];
        let divided = self.divided_differences();
        let mut ungated = 0.0_f64;
        let (mut s_f, mut s_ff, mut s_fff, mut s_ffff) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let (mut f, mut ft, mut ftt, mut fttt) = (vec![0.0_f64; m], vec![0.0_f64; m], vec![0.0_f64; m], vec![0.0_f64; m]);
        for i in 0..m {
            let lambda = evals[i];
            ungated += jeffreys_antiderivative(lambda, floor);
            s_f += jeffreys_antiderivative_floor_sensitivity(lambda, floor);
            s_ff += jeffreys_antiderivative_floor_second_sensitivity(lambda, floor);
            s_fff += jeffreys_antiderivative_floor_third_sensitivity(lambda, floor);
            s_ffff += jeffreys_antiderivative_floor_fourth_sensitivity(lambda, floor);
            f[i] = floored_inverse(lambda, floor);
            ft[i] = floored_inverse_floor_sensitivity(lambda, floor);
            ftt[i] = floored_inverse_floor_second_sensitivity(lambda, floor);
            fttt[i] = floored_inverse_floor_third_sensitivity(lambda, floor);
        }
        ungated *= 0.5;
        let row_entry = |rows: &Array2<f64>, a: usize, i: usize, j: usize| rows[[a, i * m + j]];
        let square = |rows: &Array2<f64>, a: usize| Array2::from_shape_fn((m, m), |(i, j)| rows[[a, i * m + j]]);
        let flat = |x: &Array2<f64>| Array1::from_shape_fn(m * m, |k| x[[k / m, k % m]]);
        let diag = |x: &Array2<f64>, weights: &[f64]| (0..m).map(|i| weights[i] * x[[i, i]]).sum::<f64>();
        let axis_diag = |rows: &Array2<f64>, weights: &[f64]| {
            Array1::from_shape_fn(p, |a| (0..m).map(|i| weights[i] * row_entry(rows, a, i, i)).sum::<f64>())
        };
        // `Σ_ij f^{(order)}[λ_i, λ_j]·X_ij·Y_ij`, the second spectral channel at a fixed floor.
        let paired = |order: usize, x: &Array2<f64>, y: &Array2<f64>| {
            let table = &divided.pairs[order];
            let mut sum = 0.0_f64;
            for i in 0..m {
                for j in 0..m {
                    sum += table[i * m + j] * x[[i, j]] * y[[i, j]];
                }
            }
            sum
        };
        let gap = |e: usize, j: usize| simple_eigenvalue_gap_inverse(evals, tie_tolerance, e, j);
        let second_form = |e: usize, x: &Array2<f64>, y: &Array2<f64>| {
            simple_eigenvalue_second_form(evals, tie_tolerance, e, x, y)
        };
        let second_form_axis = |e: usize, rows: &Array2<f64>, y: &Array2<f64>| {
            Array1::from_shape_fn(p, |a| {
                (0..m).map(|j| 2.0 * row_entry(rows, a, e, j) * y[[e, j]] * gap(e, j)).sum::<f64>()
            })
        };
        let outer = |x: &Array1<f64>, y: &Array1<f64>| {
            Array2::from_shape_fn((x.len(), y.len()), |(i, j)| x[i] * y[j])
        };
        let sym_outer = |x: &Array1<f64>, y: &Array1<f64>| outer(x, y) + outer(y, x);

        // Extreme-eigenvalue channels along the directions and along every coefficient axis.
        let extremes = [imin, imax];
        let x_u = extremes.map(|e| e_u[[e, e]]);
        let x_w = extremes.map(|e| e_w[[e, e]]);
        let x_uw = extremes.map(|e| e_uw[[e, e]] + second_form(e, e_u, e_w));
        let x_a = extremes.map(|e| Array1::from_shape_fn(p, |a| row_entry(a_rows, a, e, e)));
        let x_au = extremes.map(|e| {
            Array1::from_shape_fn(p, |a| row_entry(b_u, a, e, e)) + second_form_axis(e, a_rows, e_u)
        });
        let x_aw = extremes.map(|e| {
            Array1::from_shape_fn(p, |a| row_entry(b_w, a, e, e)) + second_form_axis(e, a_rows, e_w)
        });
        let x_auw = extremes.map(|e| {
            Array1::from_shape_fn(p, |a| {
                let p_a = square(a_rows, a);
                row_entry(t_uw, a, e, e)
                    + second_form(e, &square(b_u, a), e_w)
                    + second_form(e, &square(b_w, a), e_u)
                    + second_form(e, &p_a, e_uw)
                    + simple_eigenvalue_third_form(evals, tie_tolerance, e, [&p_a, e_u, e_w])
            })
        });

        // Gate derivatives by Faà di Bruno over `(λ_min, λ_max)`.
        let slots = [(0usize, 0usize), (0, 1), (1, 0), (1, 1)];
        let gate_u: f64 = (0..2).map(|i| gate1[i] * x_u[i]).sum();
        let gate_w: f64 = (0..2).map(|i| gate1[i] * x_w[i]).sum();
        let gate_uw: f64 = slots.iter().map(|&(i, j)| tensor2(i, j) * x_u[i] * x_w[j]).sum::<f64>()
            + (0..2).map(|i| gate1[i] * x_uw[i]).sum::<f64>();
        let gate_a = &x_a[0] * g1 + &x_a[1] * g2;
        let gate_ax = |x_ax: &[Array1<f64>; 2], x_x: [f64; 2]| {
            let mut out = &x_ax[0] * g1 + &x_ax[1] * g2;
            for &(i, j) in &slots {
                out.scaled_add(tensor2(i, j) * x_x[j], &x_a[i]);
            }
            out
        };
        let gate_au = gate_ax(&x_au, x_u);
        let gate_aw = gate_ax(&x_aw, x_w);
        let mut gate_auw = &x_auw[0] * g1 + &x_auw[1] * g2;
        for &(i, j) in &slots {
            let coefficient = tensor2(i, j);
            gate_auw.scaled_add(coefficient * x_w[j], &x_au[i]);
            gate_auw.scaled_add(coefficient * x_u[j], &x_aw[i]);
            gate_auw.scaled_add(coefficient * x_uw[j], &x_a[i]);
            for k in 0..2 {
                gate_auw.scaled_add(tensor3(i, j, k) * x_u[j] * x_w[k], &x_a[i]);
            }
        }
        // `∂_x G_i` and `∂²_uw G_i` of the gate's first partials.
        let gate_partial_x = |i: usize, x_x: [f64; 2]| (0..2).map(|j| tensor2(i, j) * x_x[j]).sum::<f64>();
        let gate_partial_uw = |i: usize| {
            let mut sum = 0.0_f64;
            for j in 0..2 {
                sum += tensor2(i, j) * x_uw[j];
                for k in 0..2 {
                    sum += tensor3(i, j, k) * x_u[j] * x_w[k];
                }
            }
            sum
        };

        // Ungated value channels: `Φ_*` at a fixed floor, then the floor's motion.
        let (phi_t, phi_tt, phi_ttt) = (0.5 * s_f, 0.5 * s_ff, 0.5 * s_fff);
        let (fl_u, fl_w, fl_uw) = (rate * x_u[1], rate * x_w[1], rate * x_uw[1]);
        let fl_a = &x_a[1] * rate;
        let fl_au = &x_au[1] * rate;
        let fl_aw = &x_aw[1] * rate;
        let fl_auw = &x_auw[1] * rate;
        let (flat_u, flat_w, flat_uw) = (flat(e_u), flat(e_w), flat(e_uw));
        let phi_u = 0.5 * diag(e_u, &f);
        let phi_w = 0.5 * diag(e_w, &f);
        let phi_uw = 0.5 * (paired(0, e_u, e_w) + diag(e_uw, &f));
        let phi_ut = 0.5 * diag(e_u, &ft);
        let phi_wt = 0.5 * diag(e_w, &ft);
        let phi_uwt = 0.5 * (paired(1, e_u, e_w) + diag(e_uw, &ft));
        let phi_utt = 0.5 * diag(e_u, &ftt);
        let phi_wtt = 0.5 * diag(e_w, &ftt);
        let psi0_a = self.inverse_frechet_rows(a_rows, &[], 0);
        let psi1_a = self.inverse_frechet_rows(a_rows, &[], 1);
        let phi_a = axis_diag(a_rows, &f) * 0.5;
        let phi_at = axis_diag(a_rows, &ft) * 0.5;
        let phi_att = axis_diag(a_rows, &ftt) * 0.5;
        let phi_au = (psi0_a.dot(&flat_u) + axis_diag(b_u, &f)) * 0.5;
        let phi_aw = (psi0_a.dot(&flat_w) + axis_diag(b_w, &f)) * 0.5;
        let phi_aut = (psi1_a.dot(&flat_u) + axis_diag(b_u, &ft)) * 0.5;
        let phi_awt = (psi1_a.dot(&flat_w) + axis_diag(b_w, &ft)) * 0.5;
        let phi_auw = (self.inverse_frechet_rows(a_rows, &[e_u], 0).dot(&flat_w)
            + self.inverse_frechet_rows(b_u, &[], 0).dot(&flat_w)
            + self.inverse_frechet_rows(b_w, &[], 0).dot(&flat_u)
            + psi0_a.dot(&flat_uw)
            + axis_diag(t_uw, &f))
            * 0.5;
        let value_u = phi_u + phi_t * fl_u;
        let value_w = phi_w + phi_t * fl_w;
        let value_uw = phi_uw + phi_ut * fl_w + phi_wt * fl_u + phi_tt * fl_u * fl_w + phi_t * fl_uw;
        let value_a = &phi_a + &(&fl_a * phi_t);
        let axis_value_first = |phi_ax: &Array1<f64>, fl_ax: &Array1<f64>, phi_xt: f64, fl_x: f64| {
            let mut out = phi_ax.clone();
            out.scaled_add(fl_x, &phi_at);
            out.scaled_add(phi_xt + phi_tt * fl_x, &fl_a);
            out.scaled_add(phi_t, fl_ax);
            out
        };
        let value_au = axis_value_first(&phi_au, &fl_au, phi_ut, fl_u);
        let value_aw = axis_value_first(&phi_aw, &fl_aw, phi_wt, fl_w);
        let mut value_auw = phi_auw.clone();
        value_auw.scaled_add(fl_w, &phi_aut);
        value_auw.scaled_add(fl_u, &phi_awt);
        value_auw.scaled_add(fl_u * fl_w, &phi_att);
        value_auw.scaled_add(fl_uw, &phi_at);
        value_auw.scaled_add(
            phi_uwt + phi_utt * fl_w + phi_wtt * fl_u + phi_ttt * fl_u * fl_w + phi_tt * fl_uw,
            &fl_a,
        );
        value_auw.scaled_add(phi_ut + phi_tt * fl_u, &fl_aw);
        value_auw.scaled_add(phi_wt + phi_tt * fl_w, &fl_au);
        value_auw.scaled_add(phi_t, &fl_auw);
        let s_f_u = 2.0 * (phi_ut + phi_tt * fl_u);
        let s_f_w = 2.0 * (phi_wt + phi_tt * fl_w);
        let s_f_uw = 2.0 * (phi_uwt + phi_utt * fl_w + phi_wtt * fl_u + phi_ttt * fl_u * fl_w + phi_tt * fl_uw);

        // `E = Σ_e ω_e z_e z_eᵀ` and its drifts.
        let omega = [ungated * g1, ungated * g2 + 0.5 * gate * s_f * rate];
        let omega_x = |x_x: [f64; 2], value_x: f64, gate_x: f64, s_f_x: f64| {
            [
                value_x * g1 + ungated * gate_partial_x(0, x_x),
                value_x * g2
                    + ungated * gate_partial_x(1, x_x)
                    + 0.5 * rate * (gate_x * s_f + gate * s_f_x),
            ]
        };
        let omega_u = omega_x(x_u, value_u, gate_u, s_f_u);
        let omega_w = omega_x(x_w, value_w, gate_w, s_f_w);
        let omega_uw = [
            value_uw * g1
                + value_u * gate_partial_x(0, x_w)
                + value_w * gate_partial_x(0, x_u)
                + ungated * gate_partial_uw(0),
            value_uw * g2
                + value_u * gate_partial_x(1, x_w)
                + value_w * gate_partial_x(1, x_u)
                + ungated * gate_partial_uw(1)
                + 0.5 * rate * (gate_uw * s_f + gate_u * s_f_w + gate_w * s_f_u + gate * s_f_uw),
        ];
        let mut extreme_weight = Array2::<f64>::zeros((p, p));
        let mut extreme_weight_u = Array2::<f64>::zeros((p, p));
        let mut extreme_weight_w = Array2::<f64>::zeros((p, p));
        let mut extreme_weight_uw = Array2::<f64>::zeros((p, p));
        for (slot, &e) in extremes.iter().enumerate() {
            let z = basis.column(e).to_owned();
            let first_drift = |x: &Array2<f64>| basis.dot(&Array1::from_shape_fn(m, |j| x[[j, e]] * gap(e, j)));
            let z_u = first_drift(e_u);
            let z_w = first_drift(e_w);
            let z_uw = basis.dot(&simple_eigenvector_second_drift(evals, tie_tolerance, e, e_u, e_w, e_uw));
            let zz = outer(&z, &z);
            extreme_weight.scaled_add(omega[slot], &zz);
            extreme_weight_u.scaled_add(omega_u[slot], &zz);
            extreme_weight_u.scaled_add(omega[slot], &sym_outer(&z_u, &z));
            extreme_weight_w.scaled_add(omega_w[slot], &zz);
            extreme_weight_w.scaled_add(omega[slot], &sym_outer(&z_w, &z));
            extreme_weight_uw.scaled_add(omega_uw[slot], &zz);
            extreme_weight_uw.scaled_add(omega_u[slot], &sym_outer(&z_w, &z));
            extreme_weight_uw.scaled_add(omega_w[slot], &sym_outer(&z_u, &z));
            extreme_weight_uw.scaled_add(omega[slot], &(sym_outer(&z_uw, &z) + sym_outer(&z_u, &z_w)));
        }

        // `D²R[u, w]`. First `∇G⊗∇U + ∇U⊗∇G`.
        let gate_value = outer(&gate_auw, &value_a)
            + outer(&gate_au, &value_aw)
            + outer(&gate_aw, &value_au)
            + outer(&gate_a, &value_auw);
        let mut remainder = &gate_value + &gate_value.t();
        // `U·Q₂`, `Q₂[a,b] = Σ_ij G_ij λ_i,a λ_j,b`.
        let bilinear = |coefficient: &dyn Fn(usize, usize) -> f64, x: &[Array1<f64>; 2], y: &[Array1<f64>; 2]| {
            let mut out = Array2::<f64>::zeros((p, p));
            for &(i, j) in &slots {
                let value = coefficient(i, j);
                if value != 0.0 {
                    out.scaled_add(value, &outer(&x[i], &y[j]));
                }
            }
            out
        };
        let third_along = |x_x: [f64; 2]| move |i: usize, j: usize| (0..2).map(|k| tensor3(i, j, k) * x_x[k]).sum::<f64>();
        let q2 = bilinear(&tensor2, &x_a, &x_a);
        let q2_x = |x_ax: &[Array1<f64>; 2], x_x: [f64; 2]| {
            bilinear(&third_along(x_x), &x_a, &x_a) + bilinear(&tensor2, x_ax, &x_a) + bilinear(&tensor2, &x_a, x_ax)
        };
        let q2_u = q2_x(&x_au, x_u);
        let q2_w = q2_x(&x_aw, x_w);
        let fourth_along = |i: usize, j: usize| {
            let mut sum = 0.0_f64;
            for k in 0..2 {
                sum += tensor3(i, j, k) * x_uw[k];
                for l in 0..2 {
                    sum += tensor4(i, j, k, l) * x_u[k] * x_w[l];
                }
            }
            sum
        };
        let q2_uw = bilinear(&fourth_along, &x_a, &x_a)
            + bilinear(&third_along(x_u), &x_aw, &x_a)
            + bilinear(&third_along(x_u), &x_a, &x_aw)
            + bilinear(&third_along(x_w), &x_au, &x_a)
            + bilinear(&third_along(x_w), &x_a, &x_au)
            + bilinear(&tensor2, &x_auw, &x_a)
            + bilinear(&tensor2, &x_au, &x_aw)
            + bilinear(&tensor2, &x_aw, &x_au)
            + bilinear(&tensor2, &x_a, &x_auw);
        remainder.scaled_add(value_uw, &q2);
        remainder.scaled_add(value_u, &q2_w);
        remainder.scaled_add(value_w, &q2_u);
        remainder.scaled_add(ungated, &q2_uw);
        // `G·F`, `F[a,b] = X_a·floor_b + X_b·floor_a + Y·floor_a·floor_b`, with `X_a = ∂_floor U_a`
        // and `Y = ∂²_floor U` at the moving floor.
        if rate != 0.0 {
            let (phi_uwtt, phi_uttt, phi_wttt, phi_tttt) = (
                0.5 * (paired(2, e_u, e_w) + diag(e_uw, &ftt)),
                0.5 * diag(e_u, &fttt),
                0.5 * diag(e_w, &fttt),
                0.5 * s_ffff,
            );
            let psi2_a = self.inverse_frechet_rows(a_rows, &[], 2);
            let phi_attt = axis_diag(a_rows, &fttt) * 0.5;
            let phi_autt = (psi2_a.dot(&flat_u) + axis_diag(b_u, &ftt)) * 0.5;
            let phi_awtt = (psi2_a.dot(&flat_w) + axis_diag(b_w, &ftt)) * 0.5;
            let phi_auwt = (self.inverse_frechet_rows(a_rows, &[e_u], 1).dot(&flat_w)
                + self.inverse_frechet_rows(b_u, &[], 1).dot(&flat_w)
                + self.inverse_frechet_rows(b_w, &[], 1).dot(&flat_u)
                + psi1_a.dot(&flat_uw)
                + axis_diag(t_uw, &ft))
                * 0.5;
            let x_axis = &phi_at;
            let x_axis_u = &phi_aut + &(&phi_att * fl_u);
            let x_axis_w = &phi_awt + &(&phi_att * fl_w);
            let mut x_axis_uw = &phi_auwt + &(&phi_autt * fl_w);
            x_axis_uw.scaled_add(fl_u, &phi_awtt);
            x_axis_uw.scaled_add(fl_u * fl_w, &phi_attt);
            x_axis_uw.scaled_add(fl_uw, &phi_att);
            let y = phi_tt;
            let y_u = phi_utt + phi_ttt * fl_u;
            let y_w = phi_wtt + phi_ttt * fl_w;
            let y_uw = phi_uwtt + phi_uttt * fl_w + phi_wttt * fl_u + phi_tttt * fl_u * fl_w + phi_ttt * fl_uw;
            let floor_floor = outer(&fl_a, &fl_a);
            let floor_part = sym_outer(x_axis, &fl_a) + &floor_floor * y;
            let floor_part_x = |x_ax: &Array1<f64>, fl_ax: &Array1<f64>, y_x: f64| {
                sym_outer(x_ax, &fl_a) + sym_outer(x_axis, fl_ax) + &floor_floor * y_x + sym_outer(fl_ax, &fl_a) * y
            };
            let floor_part_u = floor_part_x(&x_axis_u, &fl_au, y_u);
            let floor_part_w = floor_part_x(&x_axis_w, &fl_aw, y_w);
            let floor_part_uw = sym_outer(&x_axis_uw, &fl_a)
                + sym_outer(&x_axis_u, &fl_aw)
                + sym_outer(&x_axis_w, &fl_au)
                + sym_outer(x_axis, &fl_auw)
                + &floor_floor * y_uw
                + sym_outer(&fl_aw, &fl_a) * y_u
                + sym_outer(&fl_au, &fl_a) * y_w
                + (sym_outer(&fl_auw, &fl_a) + sym_outer(&fl_au, &fl_aw)) * y;
            remainder.scaled_add(gate_uw, &floor_part);
            remainder.scaled_add(gate_u, &floor_part_w);
            remainder.scaled_add(gate_w, &floor_part_u);
            remainder.scaled_add(gate, &floor_part_uw);
        }
        // `Σ_e ω_e·E_e`, `E_e[a,b] = D²λ_e[P̃_a, P̃_b]`, differentiated twice: the partitions of
        // `{a, b, u, w}` that keep `a` and `b` in different blocks.
        let axis_squares: Vec<Array2<f64>> = (0..p).map(|a| square(a_rows, a)).collect();
        for (slot, &e) in extremes.iter().enumerate() {
            let gi = Array1::from_shape_fn(m, |j| gap(e, j));
            let doubled = Array2::from_diag(&gi.mapv(|value| 2.0 * value));
            let row_of = |rows: &Array2<f64>| Array2::from_shape_fn((p, m), |(a, j)| row_entry(rows, a, e, j));
            let (r_a, r_bu, r_bw, r_t) = (row_of(a_rows), row_of(b_u), row_of(b_w), row_of(t_uw));
            // `D²λ_e[X_a, Y_b]` for every axis pair.
            let form2 = |x: &Array2<f64>, y: &Array2<f64>| x.dot(&doubled).dot(&y.t());
            // `D³λ_e[X_a, Y_b, Z]` for every axis pair and one direction `Z`:
            // `2(x̂ᵀYẑ + x̂ᵀZŷ + ŷᵀXẑ) − 2(X_ee·ŷ·ẑ + Y_ee·x̂·ẑ + Z_ee·x̂·ŷ)`.
            let form3 = |x_rows: &Array2<f64>, y_rows: &Array2<f64>, z: &Array2<f64>| {
                let hat = |rows: &Array2<f64>| Array2::from_shape_fn((p, m), |(a, j)| row_entry(rows, a, e, j) * gi[j]);
                let z_hat = Array1::from_shape_fn(m, |j| z[[e, j]] * gi[j]);
                let apply = |rows: &Array2<f64>| {
                    Array2::from_shape_fn((p, m), |(a, i)| (0..m).map(|k| row_entry(rows, a, i, k) * z_hat[k]).sum::<f64>())
                };
                let (x_hat, y_hat) = (hat(x_rows), hat(y_rows));
                let (x_z, y_z) = (apply(x_rows), apply(y_rows));
                let x_diag = Array1::from_shape_fn(p, |a| row_entry(x_rows, a, e, e));
                let y_diag = Array1::from_shape_fn(p, |a| row_entry(y_rows, a, e, e));
                (x_hat.dot(&y_z.t()) + x_hat.dot(z).dot(&y_hat.t()) + x_z.dot(&y_hat.t())) * 2.0
                    - (outer(&x_diag, &y_hat.dot(&z_hat)) + outer(&x_hat.dot(&z_hat), &y_diag)) * 2.0
                    - x_hat.dot(&y_hat.t()) * (2.0 * z[[e, e]])
            };
            let e_base = form2(&r_a, &r_a);
            let e_along_u = form2(&r_bu, &r_a) + form2(&r_a, &r_bu) + form3(a_rows, a_rows, e_u);
            let e_along_w = form2(&r_bw, &r_a) + form2(&r_a, &r_bw) + form3(a_rows, a_rows, e_w);
            let third_bu_w = form3(b_u, a_rows, e_w);
            let third_bw_u = form3(b_w, a_rows, e_u);
            let mut e_along_uw = form2(&r_t, &r_a)
                + form2(&r_a, &r_t)
                + form2(&r_bu, &r_bw)
                + form2(&r_bw, &r_bu)
                + &third_bu_w
                + &third_bu_w.t()
                + &third_bw_u
                + &third_bw_u.t()
                + form3(a_rows, a_rows, e_uw);
            for a in 0..p {
                for b in a..p {
                    let value = simple_eigenvalue_fourth_form(
                        evals,
                        tie_tolerance,
                        e,
                        [&axis_squares[a], &axis_squares[b], e_u, e_w],
                    );
                    e_along_uw[[a, b]] += value;
                    if a != b {
                        e_along_uw[[b, a]] += value;
                    }
                }
            }
            remainder.scaled_add(omega_uw[slot], &e_base);
            remainder.scaled_add(omega_u[slot], &e_along_w);
            remainder.scaled_add(omega_w[slot], &e_along_u);
            remainder.scaled_add(omega[slot], &e_along_uw);
        }
        SecondDriftMotion {
            extreme_weight,
            extreme_weight_u,
            extreme_weight_w,
            extreme_weight_uw,
            remainder_uw: remainder,
        }
    }

    /// Differentiate the spectral matrix function in the fixed base frame.
    /// Divided differences include eigenvector motion without dividing by
    /// eigenvalue gaps, so repeated interior eigenvalues need no special case.
    pub(super) fn perturbation_derivative_from_axis_matrices(
        &self,
        pert_h: &Array2<f64>,
        pert_hdots: &[Array2<f64>],
    ) -> Result<Array2<f64>, String> {
        if pert_h.dim() != (self.p, self.p) {
            return Err("Jeffreys drift information dimension mismatch".into());
        }
        let da = self.rotate_axes(pert_hdots)?;
        self.perturbation_derivative_from_rotated_axes(pert_h, &da)
    }

    /// The perturbation derivative from coefficient-axis rows a family has already
    /// rotated into this base's eigenbasis, so the `p × p` axis matrices are never
    /// formed (#1082). The axis-matrix entry point rotates and then closes here.
    pub fn perturbation_derivative_from_rotated_axes(
        &self,
        pert_h: &Array2<f64>,
        axes: &JeffreysRotatedAxes,
    ) -> Result<Array2<f64>, String> {
        if pert_h.dim() != (self.p, self.p) {
            return Err("Jeffreys drift information dimension mismatch".into());
        }
        let e = symmetric_basis_contraction(pert_h.view(), self.ambient_eigenbasis.view());
        let da = &axes.rows;
        let mut dw = self.inverse_frechet_rows(&self.a_rows, &[&e], 0);
        if self.floor_in_relative_regime {
            let dfloor = REDUCED_INFO_RELATIVE_FLOOR * e[[self.idx_max, self.idx_max]];
            if dfloor != 0.0 {
                dw.scaled_add(dfloor, &self.inverse_frechet_rows(&self.a_rows, &[], 1));
            }
        }
        dw += &self.inverse_frechet_rows(&da, &[], 0);
        let mut result = (dw.dot(&self.a_rows.t()) + self.aw_rows.dot(&da.t()))
            * (-0.5 * self.gate_weight);
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let dg = g_min * e[[self.idx_min, self.idx_min]]
            + g_max * e[[self.idx_max, self.idx_max]];
        if dg != 0.0 {
            result.scaled_add(-0.5 * dg, self.weighted_gram());
        }
        let mut result = result.as_standard_layout().to_owned();
        symmetrize_contiguous(&mut result);
        if result.iter().any(|v| !v.is_finite()) {
            return Err("Jeffreys drift produced nonfinite curvature".into());
        }
        Ok(result)
    }

    /// The fixed ambient kernels `K_b = U·(Ψ∘Ṽ_b)·Uᵀ`, one per coefficient axis.
    /// For a symmetric axis matrix `A`, `⟨vec sym(UᵀAU), vec(Ψ∘Ṽ_b)⟩ = ⟨A, K_b⟩`, so
    /// [`Self::perturbation_derivative_from_axis_contractions`] needs only the
    /// Frobenius products of the axis matrices against these kernels.
    pub fn ambient_axis_kernels(&self) -> Vec<Array2<f64>> {
        let m = self.m;
        self.aw_rows
            .outer_iter()
            .map(|row| {
                let weighted = Array2::from_shape_vec((m, m), row.to_vec())
                    .expect("each weighted axis row holds one m x m block");
                self.ambient_eigenbasis
                    .dot(&weighted)
                    .dot(&self.ambient_eigenbasis.t())
            })
            .collect()
    }

    /// The explicit-perturbation drift from the contractions
    /// `contractions[[a, b]] = ⟨A_a, K_b⟩` of the symmetric perturbed axis matrices
    /// `A_a = ∂Hdot[e_a]` against [`Self::ambient_axis_kernels`], without forming or
    /// rotating any `A_a`. With `da` the rotated axis rows, the zero-direction
    /// Fréchet map is the Hadamard product with the same pair divided differences
    /// `Ψ` the weighted rows carry, so `L(da)·a_rowsᵀ = da·(Ψ∘a_rows)ᵀ` is
    /// `contractions` and `(Ψ∘a_rows)·daᵀ` is its transpose. Every other term is the
    /// axis-matrix drift's.
    pub fn perturbation_derivative_from_axis_contractions(
        &self,
        pert_h: &Array2<f64>,
        contractions: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        if pert_h.dim() != (self.p, self.p) || contractions.dim() != (self.p, self.p) {
            return Err("Jeffreys drift contraction dimension mismatch".into());
        }
        let e = symmetric_basis_contraction(pert_h.view(), self.ambient_eigenbasis.view());
        let mut dw = self.inverse_frechet_rows(&self.a_rows, &[&e], 0);
        if self.floor_in_relative_regime {
            let dfloor = REDUCED_INFO_RELATIVE_FLOOR * e[[self.idx_max, self.idx_max]];
            if dfloor != 0.0 {
                dw.scaled_add(dfloor, &self.inverse_frechet_rows(&self.a_rows, &[], 1));
            }
        }
        let mut result = (dw.dot(&self.a_rows.t()) + contractions + &contractions.t())
            * (-0.5 * self.gate_weight);
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let dg = g_min * e[[self.idx_min, self.idx_min]]
            + g_max * e[[self.idx_max, self.idx_max]];
        if dg != 0.0 {
            result.scaled_add(-0.5 * dg, self.weighted_gram());
        }
        let mut result = result.as_standard_layout().to_owned();
        symmetrize_contiguous(&mut result);
        if result.iter().any(|v| !v.is_finite()) {
            return Err("Jeffreys drift produced nonfinite curvature".into());
        }
        Ok(result)
    }

    pub(super) fn rotate_axis_rows(&self, axes: &[Array2<f64>]) -> Result<Array2<f64>, String> {
        if axes.len() != self.p || axes.iter().any(|a| a.dim() != (self.p, self.p)) {
            return Err("Jeffreys mixed drift requires one full information derivative per coefficient axis".into());
        }
        gam_model_api::jeffreys_rotated_axis_rows(axes, self.ambient_eigenbasis.view())
    }

    /// Apply Df, D²f[E,.], or D³f[E,F,.] to every axis matrix.
    /// Storage stays O(p m²); the fourth-order Loewner tensor is never stored.
    fn inverse_frechet_rows(
        &self,
        rows: &Array2<f64>,
        directions: &[&Array2<f64>],
        floor_order: usize,
    ) -> Array2<f64> {
        let m = self.m;
        let squared = m * m;
        let divided = self.divided_differences();
        let mut out = Array2::zeros(rows.raw_dim());
        if directions.len() == 2 {
            // D³f[E,F,A] is linear in A. On a spectrum inside one piece of the capped
            // inverse the four-node values factor and the map closes as `m × m` products
            // per axis row (#1082); otherwise `loewner_second_rows` assembles it one output
            // row at a time and contracts with BLAS-3.
            if floor_order == 0
                && let Some(separable) = divided.separable_quadruple()
            {
                return separable_second_frechet_rows(&separable, m, rows, directions[0], directions[1]);
            }
            return loewner_second_rows(rows, directions[0], directions[1], m, |i, k, l, j| {
                if floor_order == 0 {
                    divided.quadruple([i, k, l, j])
                } else {
                    inverse_difference(
                        &[self.evals[i], self.evals[k], self.evals[l], self.evals[j]],
                        self.floor,
                        floor_order,
                    )
                }
            });
        }
        let input = rows.as_standard_layout();
        let input = input.as_slice().expect("standard-layout axis rows");
        let output = out.as_slice_mut().expect("freshly allocated rows are contiguous");
        if directions.is_empty() {
            let table = &divided.pairs[floor_order];
            for (target, source) in output.chunks_exact_mut(squared).zip(input.chunks_exact(squared)) {
                for ((value, &coefficient), &entry) in
                    target.iter_mut().zip(table.iter()).zip(source.iter())
                {
                    *value = coefficient * entry;
                }
            }
            return out;
        }
        let e = directions[0].as_standard_layout();
        let e = e.as_slice().expect("standard-layout spectral direction");
        let table = &divided.triples[floor_order];
        // `D²f[E, A]_ij = Σ_k T_ikj (E_ik A_kj + A_ik E_kj)` for every axis row `A`.
        // Each row writes only its own output chunk, so the rows fan over rayon
        // with no cross-row reduction. Inside a row the loop order is (i, k, j):
        // for a fixed (i, k) the j sweep reads three contiguous rows and
        // vectorizes. Every output entry still starts at 0.0 and adds its k terms
        // in increasing k with the same operands, so each value is bit-identical
        // to the strided (i, j, k) scalar loop this replaces, the largest
        // drift-base self time on the rigid marginal-slope ψ gradient (#979).
        use rayon::iter::{IndexedParallelIterator, ParallelIterator};
        use rayon::slice::{ParallelSlice, ParallelSliceMut};
        let rows_per_task = (1usize << 15).div_ceil(squared.saturating_mul(m)).max(1);
        output
            .par_chunks_exact_mut(squared)
            .zip(input.par_chunks_exact(squared))
            .with_min_len(rows_per_task)
            .for_each(|(target, source)| {
                for i in 0..m {
                    let table_i = &table[i * squared..(i + 1) * squared];
                    let target_i = &mut target[i * m..(i + 1) * m];
                    for k in 0..m {
                        let (e_ik, source_ik) = (e[i * m + k], source[i * m + k]);
                        for (((value, &coefficient), &source_kj), &e_kj) in target_i
                            .iter_mut()
                            .zip(&table_i[k * m..(k + 1) * m])
                            .zip(&source[k * m..(k + 1) * m])
                            .zip(&e[k * m..(k + 1) * m])
                        {
                            *value += coefficient * (e_ik * source_kj + source_ik * e_kj);
                        }
                    }
                }
            });
        out
    }

    /// Exact mixed derivative `D² H_Φ[u,v]` on the current spectral stratum.
    /// The inputs are H_u, H_v, H_uv and their coefficient-axis derivatives.
    /// No mode second response is included; the caller adds `D H_Φ[β_uv]`.
    pub fn mixed_perturbation_derivative_batched_axes(
        &self,
        pert_u: &Array2<f64>,
        pert_v: &Array2<f64>,
        pert_uv: &Array2<f64>,
        axes_u: &[Array2<f64>],
        axes_v: &[Array2<f64>],
        axes_uv: &[Array2<f64>],
    ) -> Result<Array2<f64>, String> {
        if [pert_u, pert_v, pert_uv]
            .iter()
            .any(|h| h.dim() != (self.p, self.p))
        {
            return Err("Jeffreys mixed drift information dimension mismatch".into());
        }
        let u = self.direction_frame(pert_u, axes_u)?;
        let v = self.direction_frame(pert_v, axes_v)?;
        self.mixed_perturbation_derivative_from_frames(&u, &v, pert_uv, &self.rotate_axes(axes_uv)?)
    }

    /// Everything the mixed derivative reads from ONE direction: its rotated
    /// information derivative, its rotated coefficient-axis derivatives, their
    /// first Fréchet rows, and the gate and floor channels they drive. An outer
    /// Hessian over `k` coordinates requests `k(k+1)/2` pairs drawn from `k`
    /// mode responses, so a caller batching pairs builds one frame per distinct
    /// direction and closes every pair from two frames.
    pub fn direction_frame(
        &self,
        pert: &Array2<f64>,
        axes: &[Array2<f64>],
    ) -> Result<JeffreysDirectionFrame, String> {
        if pert.dim() != (self.p, self.p) {
            return Err("Jeffreys mixed drift information dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let rows = self.rotate_axis_rows(axes)?;
        self.direction_frame_from_rotated(pert, JeffreysRotatedAxes { rows })
    }

    /// [`Self::direction_frame`] from coefficient-axis rows a family has already rotated
    /// into this base's eigenbasis (#1082).
    pub fn direction_frame_from_rotated(
        &self,
        pert: &Array2<f64>,
        axes: JeffreysRotatedAxes,
    ) -> Result<JeffreysDirectionFrame, String> {
        if pert.dim() != (self.p, self.p) {
            return Err("Jeffreys mixed drift information dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let e = symmetric_basis_contraction(pert.view(), self.ambient_eigenbasis.view());
        let rows = axes.rows;
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let min = e[[self.idx_min, self.idx_min]];
        let max = e[[self.idx_max, self.idx_max]];
        let floor_scale = if self.moving_relative_floor() {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let floor_motion = floor_scale * max;
        let gate = g_min * min + g_max * max;
        let mut first = self.first_frechet_rows(&self.a_rows, &e, floor_motion);
        first += &self.inverse_frechet_rows(&rows, &[], 0);
        let raw = (first.dot(&self.a_rows.t()) + self.aw_rows.dot(&rows.t())) * -0.5;
        Ok(JeffreysDirectionFrame {
            e,
            rows,
            first,
            raw,
            min,
            max,
            floor_motion,
            gate,
        })
    }

    /// `D² H_Φ[u,v]` closed from the two directions' frames. Only the pair's own
    /// objects — `H_uv`, its rotated coefficient-axis derivatives and the second-order
    /// spectral rows — are formed here.
    pub fn mixed_perturbation_derivative_from_frames(
        &self,
        u: &JeffreysDirectionFrame,
        v: &JeffreysDirectionFrame,
        pert_uv: &Array2<f64>,
        axes_uv: &JeffreysRotatedAxes,
    ) -> Result<Array2<f64>, String> {
        if pert_uv.dim() != (self.p, self.p) {
            return Err("Jeffreys mixed drift information dimension mismatch".into());
        }
        self.refuse_inverse_kernel_branch_boundary()?;
        let e = &u.e;
        let f = &v.e;
        let ef = symmetric_basis_contraction(pert_uv.view(), self.ambient_eigenbasis.view());
        let auv = &axes_uv.rows;
        let a = &self.a_rows;
        let (g_min, g_max) =
            conditioning_gate_weight_grad(self.evals[self.idx_min], self.evals[self.idx_max]);
        let (g_mm, g_mx, g_xx) =
            conditioning_gate_weight_hess(self.evals[self.idx_min], self.evals[self.idx_max]);
        let eigen_mixed = |idx: usize, needed: bool| -> Result<f64, String> {
            if !needed {
                return Ok(0.0);
            }
            let mut result = ef[[idx, idx]];
            for j in 0..self.m {
                if j == idx {
                    continue;
                }
                let gap = self.evals[idx] - self.evals[j];
                if gap.abs() <= f64::EPSILON * self.evals[idx].abs().max(self.evals[j].abs()) * 16.0
                {
                    return Err(
                        "Jeffreys mixed drift is undefined at a repeated active extreme eigenvalue"
                            .into(),
                    );
                }
                result += (e[[idx, j]] * f[[j, idx]] + f[[idx, j]] * e[[j, idx]]) / gap;
            }
            Ok(result)
        };
        let min_uv = eigen_mixed(self.idx_min, g_min != 0.0)?;
        let moving_floor = self.moving_relative_floor();
        let max_uv = eigen_mixed(self.idx_max, g_max != 0.0 || moving_floor)?;
        let floor_scale = if moving_floor {
            REDUCED_INFO_RELATIVE_FLOOR
        } else {
            0.0
        };
        let floor_uv = floor_scale * max_uv;
        let guv = g_min * min_uv
            + g_max * max_uv
            + g_mm * u.min * v.min
            + g_mx * (u.min * v.max + u.max * v.min)
            + g_xx * u.max * v.max;
        let mut wuv = self.inverse_frechet_rows(a, &[e, f], 0);
        wuv += &self.inverse_frechet_rows(a, &[&ef], 0);
        if v.floor_motion != 0.0 {
            wuv.scaled_add(v.floor_motion, &self.inverse_frechet_rows(a, &[e], 1));
        }
        if u.floor_motion != 0.0 {
            wuv.scaled_add(u.floor_motion, &self.inverse_frechet_rows(a, &[f], 1));
        }
        if u.floor_motion * v.floor_motion != 0.0 {
            wuv.scaled_add(
                u.floor_motion * v.floor_motion,
                &self.inverse_frechet_rows(a, &[], 2),
            );
        }
        if floor_uv != 0.0 {
            wuv.scaled_add(floor_uv, &self.inverse_frechet_rows(a, &[], 1));
        }
        wuv += &self.first_frechet_rows(&v.rows, e, u.floor_motion);
        wuv += &self.first_frechet_rows(&u.rows, f, v.floor_motion);
        wuv += &self.inverse_frechet_rows(auv, &[], 0);
        let w = &self.aw_rows;
        let raw = self.weighted_gram() * -0.5;
        let mut result = (wuv.dot(&a.t())
            + u.first.dot(&v.rows.t())
            + v.first.dot(&u.rows.t())
            + w.dot(&auv.t()))
            * (-0.5 * self.gate_weight);
        result.scaled_add(u.gate, &v.raw);
        result.scaled_add(v.gate, &u.raw);
        result.scaled_add(guv, &raw);
        let mut result = result.as_standard_layout().to_owned();
        symmetrize_contiguous(&mut result);
        if result.iter().any(|value| !value.is_finite()) {
            return Err("Jeffreys mixed drift produced nonfinite curvature".into());
        }
        Ok(result)
    }

    /// `Df[A]` plus the relative-floor motion `dfloor · ∂_floor f[A]` along one
    /// direction, for every axis row.
    fn first_frechet_rows(
        &self,
        rows: &Array2<f64>,
        direction: &Array2<f64>,
        dfloor: f64,
    ) -> Array2<f64> {
        let mut out = self.inverse_frechet_rows(rows, &[direction], 0);
        if dfloor != 0.0 {
            out.scaled_add(dfloor, &self.inverse_frechet_rows(rows, &[], 1));
        }
        out
    }

    /// The relative spectral floor moves with the information along a direction.
    fn moving_relative_floor(&self) -> bool {
        self.floor_in_relative_regime
            && (self.evals.iter().any(|&x| x < self.floor)
                || self.floor > CONDITIONING_GATE_ABSOLUTE_CLEAR)
    }

    fn refuse_inverse_kernel_branch_boundary(&self) -> Result<(), String> {
        let cap = self.floor.max(CONDITIONING_GATE_ABSOLUTE_CLEAR);
        // `λ = 0` is no knot: the bottom saturation joins the plateau C⁴ (gam#2982).
        for &value in &self.evals {
            for knot in [self.floor, cap] {
                let resolution = 16.0 * f64::EPSILON * value.abs().max(knot.abs());
                if (value - knot).abs() <= resolution {
                    return Err(format!(
                        "Jeffreys mixed drift is undefined at an inverse-kernel branch boundary: \
                         reduced information eigenvalue {value:e} sits on the knot {knot:e} \
                         (floor {floor:e}, cap {cap:e}, resolution {resolution:e}) of the \
                         {m} eigenvalues",
                        floor = self.floor,
                        m = self.evals.len(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The gate and floor motion the complete second completion drift reads: the ambient
/// extreme-eigenvector weight `E`, its first and second drifts, and `D²R[u, w]`; see
/// [`JeffreysHphiDriftBase::second_drift_motion`].
struct SecondDriftMotion {
    extreme_weight: Array2<f64>,
    extreme_weight_u: Array2<f64>,
    extreme_weight_w: Array2<f64>,
    extreme_weight_uw: Array2<f64>,
    remainder_uw: Array2<f64>,
}

/// Gate and floor channels along two directions and the capped inverse's first and second
/// Fréchet weights in the base eigenbasis; see
/// [`JeffreysHphiDriftBase::frozen_second_drift_weights`].
struct FrozenSecondDriftWeights {
    gate_u: f64,
    gate_w: f64,
    gate_uw: f64,
    /// `f(λ_i)` of the capped inverse.
    kernel: Vec<f64>,
    /// `Df[E_u]` plus the floor channel along `u`.
    weight_u: Array2<f64>,
    /// `Df[E_w]` plus the floor channel along `w`.
    weight_w: Array2<f64>,
    /// `D²f[E_u, E_w] + Df[E_uw]` plus the floor channels along `(u, w)`.
    weight_uw: Array2<f64>,
}

/// The per-direction half of `D² H_Φ[u,v]`; see
/// [`JeffreysHphiDriftBase::direction_frame`].
pub struct JeffreysDirectionFrame {
    /// Rotated information derivative `Uᵀ H[u] U`.
    e: Array2<f64>,
    /// Rotated coefficient-axis derivatives `vec(Uᵀ H²[u,e_a] U)`.
    rows: Array2<f64>,
    /// First Fréchet rows of the capped inverse along `u`, floor motion included.
    first: Array2<f64>,
    /// `D H_Φ_raw[u]` before the gate weight.
    raw: Array2<f64>,
    min: f64,
    max: f64,
    floor_motion: f64,
    gate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// #979: the tabulated divided differences, including the four-node values
    /// closed from sorted triples, are bit-identical to `inverse_difference` on the
    /// same nodes, across all four pieces of the capped inverse and a repeated
    /// eigenvalue.
    #[test]
    fn tabulated_divided_differences_match_inverse_difference_bitwise_979() {
        let floor = 1e-3;
        let evals = array![-0.3, 2e-4, 0.5, 0.5, 7.0, 40.0];
        let table = InverseDividedDifferences::new(&evals, floor);
        let m = evals.len();
        let same = |left: f64, right: f64| left.to_bits() == right.to_bits();
        for order in 0..3 {
            for i in 0..m {
                for j in 0..m {
                    let direct = inverse_difference(&[evals[i], evals[j]], floor, order);
                    assert!(same(table.pairs[order][i * m + j], direct), "pair ({i},{j}) order {order}");
                    for k in 0..m {
                        let direct = inverse_difference(&[evals[i], evals[k], evals[j]], floor, order);
                        assert!(same(table.triple(order, i, k, j), direct), "triple ({i},{k},{j}) order {order}");
                    }
                }
            }
        }
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    for d in 0..m {
                        let direct =
                            inverse_difference(&[evals[a], evals[b], evals[c], evals[d]], floor, 0);
                        let tabulated = table.quadruple([a, b, c, d]);
                        assert!(
                            same(tabulated, direct),
                            "quadruple ({a},{b},{c},{d}): tabulated {tabulated} vs direct {direct}"
                        );
                    }
                }
            }
        }
    }

    /// gam#2982: divided differences on the bottom saturation `d = w(λ/floor)/floor` keep
    /// their relative accuracy at tied nodes, near `0` where `d` leaves the plateau as
    /// `O(t⁴)`, far below it where `d = O(t⁻⁴)`, across the series/partial-fraction
    /// thresholds, and on node sets that straddle `0`.
    #[test]
    fn bottom_saturation_divided_differences_are_accurate_2982() {
        let relative = |actual: f64, expected: f64| {
            if actual == expected { 0.0 } else { (actual - expected).abs() / expected.abs() }
        };
        let floor = 2.0;
        // Tied nodes: the confluent value `d⁽ⁿ⁾/n!` and its floor partials, from the
        // pointwise profile.
        let mut worst_tied = 0.0_f64;
        for t in [-1e-5, -1e-3, -0.3, -0.5, -0.500001, -0.8, -1.0, -1.999999, -2.0, -30.0, -1e4] {
            let lam = t * floor;
            let profile = BottomProfile::new(lam, floor);
            let m = |p, q| profile.monomial(p, q);
            let f = floor;
            let cases = [
                (vec![lam, lam], 0, m(0, 1) / (f * f)),
                (vec![lam, lam], 1, floored_inverse_lambda_floor_sensitivity(lam, f)),
                (vec![lam, lam], 2, (6.0 * m(0, 1) + 6.0 * m(1, 2) + m(2, 3)) / f.powi(4)),
                (vec![lam, lam, lam], 0, m(0, 2) / (2.0 * f.powi(3))),
                (vec![lam, lam, lam, lam], 0, m(0, 3) / (6.0 * f.powi(4))),
            ];
            for (nodes, order, expected) in cases {
                let actual = inverse_difference(&nodes, floor, order);
                let error = relative(actual, expected);
                worst_tied = worst_tied.max(error);
                assert!(
                    error < 1e-12,
                    "t={t}, {} tied nodes, floor order {order}: {actual:e} vs {expected:e} \
                     (relative {error:e})",
                    nodes.len()
                );
            }
        }
        // Separated nodes: the recursion on pointwise floor partials, which is accurate
        // when every node gap is O(floor) and the values do not cancel.
        let pointwise = |lam: f64, order: usize| match order {
            0 => floored_inverse(lam, floor),
            1 => floored_inverse_floor_sensitivity(lam, floor),
            _ => floored_inverse_floor_second_sensitivity(lam, floor),
        };
        fn naive(nodes: &[f64], order: usize, pointwise: &dyn Fn(f64, usize) -> f64) -> f64 {
            if let [lam] = nodes {
                return pointwise(*lam, order);
            }
            let last = nodes.len() - 1;
            (naive(&nodes[1..], order, pointwise) - naive(&nodes[..last], order, pointwise))
                / (nodes[last] - nodes[0])
        }
        let mut worst_separated = 0.0_f64;
        for (region, t_nodes) in [
            ("near", [-0.5, -0.4, -0.3, -0.2]),
            ("product", [-1.9, -1.4, -1.0, -0.6]),
            ("across regions", [-8.0, -3.0, -1.0, -0.2]),
            ("far", [-12.0, -7.0, -4.0, -2.5]),
            ("plateau and bottom", [-1.2, -0.7, 0.3, 0.8]),
        ] {
            let nodes = t_nodes.map(|t| t * floor);
            for len in 2..=4 {
                for order in 0..3 {
                    let actual = inverse_difference(&nodes[..len], floor, order);
                    let expected = naive(&nodes[..len], order, &pointwise);
                    let error = relative(actual, expected);
                    worst_separated = worst_separated.max(error);
                    assert!(
                        error < 1e-9,
                        "{region}, {len} nodes, floor order {order}: {actual:e} vs {expected:e} \
                         (relative {error:e})"
                    );
                }
            }
        }
        // A pair straddling 0 that the plateau-sized recursion rounds to zero.
        let t0 = -1e-6;
        let pair = [t0 * floor, 3e-7 * floor];
        let expected = t0.powi(4) / (1.0 + t0.powi(4)) / floor / (pair[1] - pair[0]);
        let straddling = relative(inverse_difference(&pair, floor, 0), expected);
        assert!(straddling < 1e-14, "straddling pair: relative {straddling:e}");
        eprintln!(
            "[2982] worst relative error: tied {worst_tied:e}, separated {worst_separated:e}, \
             straddling {straddling:e}"
        );
    }

    #[test]
    fn mixed_jeffreys_drift_matches_first_drift_difference_979() {
        // Gate transition, repeated interior spectrum, moving relative floor,
        // and the signed continuation all exercise different spectral channels.
        for diagonal in [
            [3.0, 7.0, 30.0],
            [0.4, 2.0, 2.0],
            [1e-4, 4e8, 5e8],
            [-0.2, 0.4, 3.0],
        ] {
            let h = Array2::from_diag(&Array1::from_vec(diagonal.to_vec()));
            let z = Array2::eye(3);
            let e = array![[0.2, 0.03, -0.04], [0.03, -0.1, 0.02], [-0.04, 0.02, 0.15]];
            let f = array![[0.1, -0.02, 0.01], [-0.02, 0.2, 0.03], [0.01, 0.03, -0.1]];
            let ef = &e * 0.13 + &f * 0.07;
            let axes = vec![e.clone(), f.clone(), &e + &f];
            let au: Vec<_> = axes.iter().map(|a| a * 0.11).collect();
            let av: Vec<_> = axes.iter().map(|a| a * -0.08).collect();
            let auv: Vec<_> = axes.iter().map(|a| a * 0.03).collect();
            let base = JeffreysHphiDriftBase::prepare_with_axes(h.view(), z.view(), axes.clone())
                .unwrap()
                .unwrap();
            let completion_actual = base
                .completion_drift_from_rows(
                    &symmetric_basis_contraction(e.view(), base.ambient_eigenbasis.view()),
                    &base.rotate_axis_rows(&axes).unwrap(),
                    &base.rotate_axis_rows(&au).unwrap(),
                )
                .unwrap();
            let score = base.explicit_score_pair(&e, &f, &ef, &au, &av, &auv).unwrap();
            for axis in 0..3 {
                let at = |t: f64| joint_jeffreys_phi_explicit_param_second_derivative(
                    (&h + &axes[axis]*t).view(), z.view(),
                    &(&e + &au[axis]*t), &(&f + &av[axis]*t), &(&ef + &auv[axis]*t)).unwrap();
                let step = 1e-5;
                let fd = (at(step)-at(-step))/(2.0*step);
                let error = (score[axis]-fd).abs()/(1.0+score[axis].abs().max(fd.abs()));
                assert!(error < 2e-5, "scalar third spectrum={diagonal:?}, axis={axis}, analytic={}, fd={fd}, error={error}", score[axis]);
            }
            let completion_at = |t: f64| {
                let ht = &h + &e * t;
                let at: Vec<_> = axes.iter().zip(&au).map(|(a, d)| a + &(d * t)).collect();
                let point = JeffreysHphiDriftBase::prepare_with_axes(ht.view(), z.view(), at.clone())
                    .unwrap().unwrap();
                let rows = point.rotate_axis_rows(&at).unwrap();
                Array1::from_shape_fn(3, |a| {
                    -0.5 * point.gate_weight * (0..3).map(|i| {
                        floored_inverse(point.evals[i], point.floor) * rows[[a, i * 3 + i]]
                    }).sum::<f64>()
                })
            };
            let completion_step = 1e-5;
            let completion_fd = (completion_at(completion_step) - completion_at(-completion_step)) / (2.0 * completion_step);
            for a in 0..3 {
                let scale = 1.0 + completion_actual[a].abs().max(completion_fd[a].abs());
                assert!((completion_actual[a] - completion_fd[a]).abs() < 2e-6 * scale,
                    "completion spectrum={diagonal:?}, axis={a}, analytic={}, fd={}", completion_actual[a], completion_fd[a]);
            }
            let actual = base
                .mixed_perturbation_derivative_batched_axes(
                    &e,
                    &f,
                    &ef,
                    &au,
                    &av,
                    &auv,
                )
                .unwrap();
            let first_at = |t: f64| {
                let ht = &h + &e * t;
                let at = axes.iter().zip(&au).map(|(a, d)| a + &(d * t)).collect();
                let avt = av.iter().zip(&auv).map(|(a, d)| a + &(d * t)).collect();
                JeffreysHphiDriftBase::prepare_with_axes(ht.view(), z.view(), at)
                    .unwrap()
                    .unwrap()
                    .perturbation_derivative_batched_axes(&(&f + &ef * t), Some(avt))
                    .unwrap()
            };
            let step = 1e-5;
            let expected = (first_at(step) - first_at(-step)) / (2.0 * step);
            let scale = expected
                .iter()
                .chain(actual.iter())
                .fold(1.0_f64, |m, x| m.max(x.abs()));
            let error = (&actual - &expected)
                .iter()
                .fold(0.0_f64, |m, x| m.max(x.abs()));
            assert!(
                error < 2e-4 * scale,
                "spectrum={diagonal:?}, relative error={} actual={actual:?} expected={expected:?}",
                error / scale
            );
            let swapped = base
                .mixed_perturbation_derivative_batched_axes(&f, &e, &ef, &av, &au, &auv)
                .unwrap();
            assert!((&actual - &swapped).iter().all(|x| x.abs() < 1e-12 * scale));
        }
    }

    /// #1082: on a spectrum inside one piece of the capped inverse, the factored map equals the
    /// coefficient loop on every piece; a spectrum that spans pieces, or sits on the bottom
    /// saturation (gam#2982), is never factored.
    #[test]
    fn separable_second_frechet_rows_match_the_coefficient_loop_1082() {
        let floor = 1e-3;
        let e = array![[0.2, 0.03, -0.04, 0.01], [0.05, -0.1, 0.02, 0.07], [-0.04, 0.02, 0.15, -0.03], [0.02, 0.06, -0.01, 0.09]];
        let f = array![[0.1, -0.02, 0.01, 0.04], [-0.03, 0.2, 0.03, -0.05], [0.01, 0.03, -0.1, 0.02], [0.06, -0.02, 0.05, 0.12]];
        let rows = Array2::<f64>::from_shape_fn((3, 16), |(r, c)| 0.11 * ((3 * r + c + 1) as f64).sin() - 0.05 * ((r + 2 * c) as f64).cos());
        for spectrum in [
            array![0.3, 1.7, 5.0, 11.0],
            array![18.0, 25.0, 40.0, 90.0],
            array![0.0, 2e-4, 5e-4, 9e-4],
        ] {
            let table = InverseDividedDifferences::new(&spectrum, floor);
            let separable = table
                .separable_quadruple()
                .expect("a spectrum inside one piece factors");
            let expected = loewner_second_rows(&rows, &e, &f, 4, |i, k, l, j| table.quadruple([i, k, l, j]));
            let actual = separable_second_frechet_rows(&separable, 4, &rows, &e, &f);
            let scale = expected.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
            for ((index, &want), &got) in expected.indexed_iter().zip(actual.iter()) {
                assert!(
                    (want - got).abs() <= 1e-12 * (1.0 + scale),
                    "spectrum {spectrum:?} entry {index:?}: loop {want} vs factored {got}"
                );
            }
            // The agreement bar is only informative where the compared values dwarf it. The
            // capped-inverse spectrum's coefficients are `16·Π(1/λ)·Σ(1/λ) ≈ 1.2e-6`, so its
            // map is O(1e-8): a fixed floor would reject a sound fixture.
            let tolerance = 1e-12 * (1.0 + scale);
            if spectrum[3] >= floor {
                assert!(
                    scale > 1e3 * tolerance,
                    "positive control: the map on {spectrum:?} must exceed the agreement bar by three \
                     orders (scale {scale:e}, bar {tolerance:e})"
                );
            }
            if separable.tilted {
                let untilted = SeparableQuadruple {
                    scale: separable.scale,
                    factors: separable.factors.clone(),
                    tilted: false,
                };
                let wrong = separable_second_frechet_rows(&untilted, 4, &rows, &e, &f);
                let gap = expected
                    .iter()
                    .zip(wrong.iter())
                    .fold(0.0_f64, |acc, (want, got)| acc.max((want - got).abs()));
                assert!(
                    gap > 1e3 * tolerance,
                    "negative control: dropping the tilt on {spectrum:?} must break agreement \
                     (gap {gap:e}, bar {tolerance:e})"
                );
            }
        }
        for spectrum in [
            array![0.3, 20.0],
            array![-0.2, 0.5],
            array![5e-4, 3.0],
            array![-0.4, -1.1, -2.5, -6.0],
        ] {
            assert!(
                InverseDividedDifferences::new(&spectrum, floor).separable_quadruple().is_none(),
                "a spectrum spanning pieces or on the bottom saturation must keep the coefficient \
                 loop: {spectrum:?}"
            );
        }
    }

    /// #1082: the cached gate-drift Gram is `aw_rows · a_rowsᵀ` bit for bit, so every
    /// drift that reads it closes on the matrix it used to re-form on each call.
    #[test]
    fn weighted_gram_matches_the_gate_drift_gram_bitwise_1082() {
        let h = Array2::from_diag(&array![3.0, 7.0, 30.0]);
        let z = Array2::eye(3);
        let e = array![[0.2, 0.03, -0.04], [0.03, -0.1, 0.02], [-0.04, 0.02, 0.15]];
        let f = array![[0.1, -0.02, 0.01], [-0.02, 0.2, 0.03], [0.01, 0.03, -0.1]];
        let axes = vec![e.clone(), f.clone(), &e + &f];
        let base = JeffreysHphiDriftBase::prepare_with_axes(h.view(), z.view(), axes)
            .unwrap()
            .unwrap();
        let direct = base.aw_rows.dot(&base.a_rows.t());
        let cached = base.weighted_gram();
        assert_eq!(cached.dim(), direct.dim());
        assert!(
            cached
                .iter()
                .zip(direct.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits()),
            "cached {cached:?} vs direct {direct:?}"
        );
        assert!(
            direct.iter().any(|value| *value != 0.0),
            "positive control: the fixture's Gram does not vanish"
        );
    }
}
