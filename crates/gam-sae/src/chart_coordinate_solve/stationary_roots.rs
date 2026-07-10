//! Complete stationary-point solves for period-one harmonic curves.
//!
//! A real trigonometric polynomial
//! `f(t) = sum_{h=-H}^H c_h exp(i tau h t)` has stationary points at the
//! unit-circle roots of the ordinary polynomial obtained from
//! `z^H f'(z)`, `z = exp(i tau t)`.  We enumerate those roots through the
//! companion-matrix eigenproblem, polish them against the real Laurent
//! polynomial, and compare the objective at every verified stationary point.
//! There is no sampling lattice and therefore no coordinate quantisation.

use faer::prelude::*;
use ndarray::ArrayView2;

/// The global extremum of a period-one curve objective.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PeriodicExtremum {
    pub(crate) coordinate: f64,
    pub(crate) value: f64,
}

/// Fourier representation of `phi(t)^T G phi(t)` for the standard harmonic
/// basis `[1, sin(tau t), cos(tau t), ...]`.
///
/// The quadratic term is target-independent, so callers solving many rows
/// against one decoder build this once and reuse it.
#[derive(Clone, Debug)]
pub(crate) struct PeriodicCurveExtrema {
    width: usize,
    quadratic: LaurentPolynomial,
}

impl PeriodicCurveExtrema {
    pub(crate) fn from_gram(gram: ArrayView2<'_, f64>) -> Result<Self, String> {
        if gram.nrows() != gram.ncols() {
            return Err(format!(
                "PeriodicCurveExtrema: Gram matrix must be square, got {:?}",
                gram.dim()
            ));
        }
        validate_harmonic_width(gram.nrows())?;
        if let Some((index, value)) = gram
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "PeriodicCurveExtrema: Gram entry {index} is not finite ({value})"
            ));
        }
        Ok(Self {
            width: gram.nrows(),
            quadratic: harmonic_quadratic(gram),
        })
    }

    /// Minimise `phi(t)^T G phi(t) - 2 c^T phi(t)`, which differs from
    /// `||x - phi(t) D||^2` only by the target-only constant `||x||^2` when
    /// `G = D D^T` and `c = D x`.
    pub(crate) fn minimize_squared_distance(
        &self,
        linear: &[f64],
    ) -> Result<PeriodicExtremum, String> {
        let linear = self.linear_polynomial(linear)?;
        let objective = self.quadratic.add_scaled(&linear, -2.0);
        global_extremum(
            &objective,
            &objective.derivative(),
            ExtremumKind::Minimum,
            |t| objective.eval_real(t),
        )
    }

    /// Maximise `(c^T phi(t))^2 / (phi(t)^T G phi(t))`, the decoder-amplitude
    /// profiled reconstruction gain.  Its nonzero-score stationary points are
    /// the roots of `2 q' r - q r'`, with `q = c^T phi` and
    /// `r = phi^T G phi`.
    pub(crate) fn maximize_profiled_score(
        &self,
        linear: &[f64],
    ) -> Result<PeriodicExtremum, String> {
        let q = self.linear_polynomial(linear)?;
        if q.is_numerically_zero() || self.quadratic.is_numerically_zero() {
            return Ok(PeriodicExtremum {
                coordinate: 0.0,
                value: 0.0,
            });
        }
        let q_prime = q.derivative();
        let r_prime = self.quadratic.derivative();
        let stationarity = q_prime
            .multiply(&self.quadratic)
            .scaled(2.0)
            .add_scaled(&q.multiply(&r_prime), -1.0);
        global_extremum(&stationarity, &stationarity, ExtremumKind::Maximum, |t| {
            profiled_score_at(&q, &self.quadratic, t)
        })
    }

    fn linear_polynomial(&self, linear: &[f64]) -> Result<LaurentPolynomial, String> {
        if linear.len() != self.width {
            return Err(format!(
                "PeriodicCurveExtrema: linear width {} != harmonic width {}",
                linear.len(),
                self.width
            ));
        }
        if let Some((index, value)) = linear
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "PeriodicCurveExtrema: linear entry {index} is not finite ({value})"
            ));
        }
        Ok(harmonic_linear(linear))
    }
}

#[derive(Clone, Debug)]
struct LaurentPolynomial {
    /// Coefficient at exponent `h` is stored at `h + degree`.
    coefficients: Vec<c64>,
    degree: usize,
}

impl LaurentPolynomial {
    fn zeros(degree: usize) -> Self {
        Self {
            coefficients: vec![c64::new(0.0, 0.0); 2 * degree + 1],
            degree,
        }
    }

    fn coefficient(&self, exponent: isize) -> c64 {
        if exponent.unsigned_abs() > self.degree {
            c64::new(0.0, 0.0)
        } else {
            self.coefficients[(exponent + self.degree as isize) as usize]
        }
    }

    fn coefficient_mut(&mut self, exponent: isize) -> &mut c64 {
        &mut self.coefficients[(exponent + self.degree as isize) as usize]
    }

    fn add_scaled(&self, rhs: &Self, scale: f64) -> Self {
        let degree = self.degree.max(rhs.degree);
        let mut out = Self::zeros(degree);
        for h in -(degree as isize)..=degree as isize {
            *out.coefficient_mut(h) = self.coefficient(h) + rhs.coefficient(h) * scale;
        }
        out
    }

    fn scaled(mut self, scale: f64) -> Self {
        for coefficient in &mut self.coefficients {
            *coefficient *= scale;
        }
        self
    }

    fn multiply(&self, rhs: &Self) -> Self {
        let degree = self.degree + rhs.degree;
        let mut out = Self::zeros(degree);
        for left in -(self.degree as isize)..=self.degree as isize {
            let a = self.coefficient(left);
            if a == c64::new(0.0, 0.0) {
                continue;
            }
            for right in -(rhs.degree as isize)..=rhs.degree as isize {
                *out.coefficient_mut(left + right) += a * rhs.coefficient(right);
            }
        }
        out
    }

    fn derivative(&self) -> Self {
        let mut out = Self::zeros(self.degree);
        for h in -(self.degree as isize)..=self.degree as isize {
            *out.coefficient_mut(h) =
                self.coefficient(h) * c64::new(0.0, std::f64::consts::TAU * h as f64);
        }
        out
    }

    fn eval_real(&self, t: f64) -> f64 {
        self.eval_complex(t).re
    }

    fn eval_complex(&self, t: f64) -> c64 {
        let theta = std::f64::consts::TAU * t;
        let mut value = c64::new(0.0, 0.0);
        for h in -(self.degree as isize)..=self.degree as isize {
            let angle = theta * h as f64;
            value += self.coefficient(h) * c64::new(angle.cos(), angle.sin());
        }
        value
    }

    fn coefficient_norm(&self) -> f64 {
        self.coefficients
            .iter()
            .map(|coefficient| coefficient.re.hypot(coefficient.im))
            .sum()
    }

    fn numerical_degree(&self) -> usize {
        let scale = self.coefficient_norm();
        if scale == 0.0 {
            return 0;
        }
        let backward_error = f64::EPSILON * self.coefficients.len() as f64 * scale;
        (1..=self.degree)
            .rev()
            .find(|&h| {
                let positive = self.coefficient(h as isize);
                let negative = self.coefficient(-(h as isize));
                positive
                    .re
                    .hypot(positive.im)
                    .max(negative.re.hypot(negative.im))
                    > backward_error
            })
            .unwrap_or(0)
    }

    fn is_numerically_zero(&self) -> bool {
        let scale = self.coefficient_norm();
        scale == 0.0
            || self
                .coefficients
                .iter()
                .all(|coefficient| coefficient.re == 0.0 && coefficient.im == 0.0)
    }

    fn derivative_bound(&self, order: usize) -> f64 {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(index, coefficient)| {
                let h = index as isize - self.degree as isize;
                let frequency = std::f64::consts::TAU * h.unsigned_abs() as f64;
                coefficient.re.hypot(coefficient.im) * frequency.powi(order as i32)
            })
            .sum()
    }

    fn derivative_value(&self, t: f64, order: usize) -> f64 {
        let theta = std::f64::consts::TAU * t;
        let mut value = c64::new(0.0, 0.0);
        for h in -(self.degree as isize)..=self.degree as isize {
            let frequency = std::f64::consts::TAU * h as f64;
            let multiplier = c64::new(0.0, frequency).powi(order as i32);
            let angle = theta * h as f64;
            value += self.coefficient(h) * multiplier * c64::new(angle.cos(), angle.sin());
        }
        value.re
    }
}

fn validate_harmonic_width(width: usize) -> Result<(), String> {
    if width == 0 || width % 2 == 0 {
        return Err(format!(
            "periodic harmonic basis width must be odd and positive, got {width}"
        ));
    }
    Ok(())
}

fn harmonic_terms(column: usize) -> [(isize, c64); 2] {
    if column == 0 {
        return [(0, c64::new(1.0, 0.0)), (0, c64::new(0.0, 0.0))];
    }
    let harmonic = column.div_ceil(2) as isize;
    if column % 2 == 1 {
        // sin(h theta) = -i z^h / 2 + i z^-h / 2.
        [
            (harmonic, c64::new(0.0, -0.5)),
            (-harmonic, c64::new(0.0, 0.5)),
        ]
    } else {
        [
            (harmonic, c64::new(0.5, 0.0)),
            (-harmonic, c64::new(0.5, 0.0)),
        ]
    }
}

fn harmonic_linear(weights: &[f64]) -> LaurentPolynomial {
    let degree = (weights.len() - 1) / 2;
    let mut out = LaurentPolynomial::zeros(degree);
    for (column, &weight) in weights.iter().enumerate() {
        for (exponent, coefficient) in harmonic_terms(column) {
            *out.coefficient_mut(exponent) += coefficient * weight;
        }
    }
    out
}

fn harmonic_quadratic(gram: ArrayView2<'_, f64>) -> LaurentPolynomial {
    let degree = gram.nrows() - 1;
    let mut out = LaurentPolynomial::zeros(degree);
    for left in 0..gram.nrows() {
        for right in 0..gram.ncols() {
            let weight = gram[[left, right]];
            if weight == 0.0 {
                continue;
            }
            for (left_exponent, left_coefficient) in harmonic_terms(left) {
                for (right_exponent, right_coefficient) in harmonic_terms(right) {
                    *out.coefficient_mut(left_exponent + right_exponent) +=
                        left_coefficient * right_coefficient * weight;
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
enum ExtremumKind {
    Minimum,
    Maximum,
}

fn global_extremum(
    objective: &LaurentPolynomial,
    stationarity: &LaurentPolynomial,
    kind: ExtremumKind,
    mut evaluate: impl FnMut(f64) -> Result<f64, String>,
) -> Result<PeriodicExtremum, String> {
    if stationarity.numerical_degree() == 0 {
        let value = evaluate(0.0)?;
        if !value.is_finite() {
            return Err("periodic extremum: constant objective is not finite".to_string());
        }
        return Ok(PeriodicExtremum {
            coordinate: 0.0,
            value,
        });
    }
    let candidates = stationary_coordinates(stationarity)?;
    if candidates.is_empty() {
        return Err(
            "periodic extremum: companion solve found no verified stationary point".to_string(),
        );
    }
    let mut best: Option<PeriodicExtremum> = None;
    for coordinate in candidates {
        let value = evaluate(coordinate)?;
        if !value.is_finite() {
            continue;
        }
        let candidate = PeriodicExtremum { coordinate, value };
        let replace = match best {
            None => true,
            Some(incumbent) => {
                let scale = 1.0 + incumbent.value.abs().max(value.abs());
                let tie = f64::EPSILON * objective.coefficients.len() as f64 * scale;
                match kind {
                    ExtremumKind::Minimum => {
                        value < incumbent.value - tie
                            || ((value - incumbent.value).abs() <= tie
                                && coordinate < incumbent.coordinate)
                    }
                    ExtremumKind::Maximum => {
                        value > incumbent.value + tie
                            || ((value - incumbent.value).abs() <= tie
                                && coordinate < incumbent.coordinate)
                    }
                }
            }
        };
        if replace {
            best = Some(candidate);
        }
    }
    best.ok_or_else(|| "periodic extremum: every stationary objective value was non-finite".into())
}

fn stationary_coordinates(polynomial: &LaurentPolynomial) -> Result<Vec<f64>, String> {
    let degree = polynomial.numerical_degree();
    if degree == 0 {
        return Ok(Vec::new());
    }
    let order = 2 * degree;
    let leading = polynomial.coefficient(degree as isize);
    let leading_norm = leading.re.hypot(leading.im);
    if !(leading_norm > 0.0 && leading_norm.is_finite()) {
        return Err("periodic stationary roots: non-finite companion leading coefficient".into());
    }
    let mut companion = Mat::<c64>::zeros(order, order);
    for row in 1..order {
        companion[(row, row - 1)] = c64::new(1.0, 0.0);
    }
    for row in 0..order {
        let exponent = row as isize - degree as isize;
        companion[(row, order - 1)] = -polynomial.coefficient(exponent) / leading;
    }
    let roots = companion
        .eigenvalues()
        .map_err(|error| format!("periodic stationary companion eigenproblem failed: {error:?}"))?;
    let derivative = polynomial.derivative();
    let residual_scale = polynomial.coefficient_norm();
    let residual_tolerance =
        f64::EPSILON.sqrt() * polynomial.coefficients.len() as f64 * residual_scale;
    let mut coordinates = Vec::with_capacity(roots.len());
    for root in roots {
        if !(root.re.is_finite() && root.im.is_finite()) || (root.re == 0.0 && root.im == 0.0) {
            continue;
        }
        let raw = (root.im.atan2(root.re) / std::f64::consts::TAU).rem_euclid(1.0);
        let polished = polish_real_root(polynomial, &derivative, raw);
        for coordinate in [polished, raw] {
            let residual = polynomial.eval_complex(coordinate);
            if residual.re.hypot(residual.im) <= residual_tolerance {
                coordinates.push(canonical_coordinate(coordinate));
                break;
            }
        }
    }
    coordinates.sort_by(f64::total_cmp);
    let separation = f64::EPSILON.sqrt() * (1.0 + degree as f64);
    coordinates.dedup_by(|left, right| circular_distance(*left, *right) <= separation);
    if coordinates.len() > 1
        && circular_distance(coordinates[0], *coordinates.last().unwrap()) <= separation
    {
        coordinates.pop();
        coordinates[0] = 0.0;
    }
    Ok(coordinates)
}

fn polish_real_root(
    polynomial: &LaurentPolynomial,
    derivative: &LaurentPolynomial,
    mut coordinate: f64,
) -> f64 {
    let residual_scale = polynomial.coefficient_norm();
    let residual_tolerance = f64::EPSILON * polynomial.coefficients.len() as f64 * residual_scale;
    // Each successful Newton step doubles the resolved bits near a simple root;
    // the mantissa width is therefore a machine-derived absolute ceiling, not an
    // answer-changing iteration knob.
    for _ in 0..f64::MANTISSA_DIGITS {
        let residual = polynomial.eval_real(coordinate);
        if residual.abs() <= residual_tolerance {
            break;
        }
        let slope = derivative.eval_real(coordinate);
        if !(slope.is_finite() && slope != 0.0) {
            break;
        }
        let next = canonical_coordinate(coordinate - residual / slope);
        if circular_distance(next, coordinate) <= f64::EPSILON * (1.0 + coordinate.abs()) {
            coordinate = next;
            break;
        }
        coordinate = next;
    }
    coordinate
}

fn profiled_score_at(
    linear: &LaurentPolynomial,
    quadratic: &LaurentPolynomial,
    coordinate: f64,
) -> Result<f64, String> {
    let numerator = linear.eval_real(coordinate);
    let denominator = quadratic.eval_real(coordinate);
    let denominator_tolerance =
        f64::EPSILON * quadratic.coefficients.len() as f64 * quadratic.coefficient_norm();
    if denominator > denominator_tolerance {
        return Ok(numerator * numerator / denominator);
    }
    if denominator < -denominator_tolerance {
        return Err(format!(
            "periodic profiled score: decoder Gram produced negative norm {denominator}"
        ));
    }

    // At an exact zero of the decoded curve, q^2/r can have a removable limit.
    // Recover it from the first nonzero Taylor coefficients instead of choosing
    // an arbitrary norm cutoff or discarding a legitimate limiting direction.
    let numerator_order = first_nonzero_derivative(linear, coordinate);
    let denominator_order = first_nonzero_derivative(quadratic, coordinate);
    let Some((q_order, q_derivative)) = numerator_order else {
        return Ok(0.0);
    };
    let Some((r_order, r_derivative)) = denominator_order else {
        return Ok(0.0);
    };
    match (2 * q_order).cmp(&r_order) {
        std::cmp::Ordering::Greater => Ok(0.0),
        std::cmp::Ordering::Less => {
            Err("periodic profiled score: numerator/Gram zero orders are inconsistent".into())
        }
        std::cmp::Ordering::Equal => {
            let q_taylor = q_derivative / factorial(q_order);
            let r_taylor = r_derivative / factorial(r_order);
            if r_taylor <= 0.0 {
                return Err(format!(
                    "periodic profiled score: non-positive removable-limit denominator {r_taylor}"
                ));
            }
            Ok(q_taylor * q_taylor / r_taylor)
        }
    }
}

fn first_nonzero_derivative(
    polynomial: &LaurentPolynomial,
    coordinate: f64,
) -> Option<(usize, f64)> {
    for order in 0..=2 * polynomial.degree {
        let value = polynomial.derivative_value(coordinate, order);
        let tolerance = f64::EPSILON
            * polynomial.coefficients.len() as f64
            * polynomial.derivative_bound(order);
        if value.abs() > tolerance {
            return Some((order, value));
        }
    }
    None
}

fn factorial(order: usize) -> f64 {
    (1..=order).fold(1.0, |value, factor| value * factor as f64)
}

fn canonical_coordinate(coordinate: f64) -> f64 {
    let wrapped = coordinate.rem_euclid(1.0);
    if wrapped <= f64::EPSILON || 1.0 - wrapped <= f64::EPSILON {
        0.0
    } else {
        wrapped
    }
}

fn circular_distance(left: f64, right: f64) -> f64 {
    let difference = (left - right).abs().rem_euclid(1.0);
    difference.min(1.0 - difference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn distance_projection_recovers_off_lattice_phase() {
        let gram = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let solver = PeriodicCurveExtrema::from_gram(gram.view()).unwrap();
        let planted = 0.173_205_080_756_887_73;
        let angle = std::f64::consts::TAU * planted;
        let linear = [0.0, angle.sin(), angle.cos()];
        let solved = solver.minimize_squared_distance(&linear).unwrap();
        assert!(circular_distance(solved.coordinate, planted) < 1e-11);
    }

    #[test]
    fn profiled_score_beats_dense_grid_oracle() {
        let gram = array![
            [1.3, 0.1, -0.2, 0.05, 0.03],
            [0.1, 1.1, 0.07, -0.1, 0.02],
            [-0.2, 0.07, 0.9, 0.04, -0.06],
            [0.05, -0.1, 0.04, 0.8, 0.03],
            [0.03, 0.02, -0.06, 0.03, 0.7],
        ];
        let solver = PeriodicCurveExtrema::from_gram(gram.view()).unwrap();
        let linear = [0.4, -0.7, 0.2, 1.1, -0.3];
        let solved = solver.maximize_profiled_score(&linear).unwrap();
        let q = harmonic_linear(&linear);
        let r = harmonic_quadratic(gram.view());
        let dense_best = (0..200_003)
            .map(|index| index as f64 / 200_003.0)
            .map(|t| profiled_score_at(&q, &r, t).unwrap())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(solved.value + 1e-10 >= dense_best);
    }

    #[test]
    fn constant_and_zero_curve_choose_canonical_origin() {
        let gram = array![[4.0]];
        let solver = PeriodicCurveExtrema::from_gram(gram.view()).unwrap();
        let distance = solver.minimize_squared_distance(&[3.0]).unwrap();
        let score = solver.maximize_profiled_score(&[3.0]).unwrap();
        assert_eq!(distance.coordinate, 0.0);
        assert_eq!(score.coordinate, 0.0);
        assert!((score.value - 9.0 / 4.0).abs() < 1e-14);

        let zero = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let zero_solver = PeriodicCurveExtrema::from_gram(zero.view()).unwrap();
        let zero_score = zero_solver
            .maximize_profiled_score(&[0.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(zero_score.coordinate, 0.0);
        assert_eq!(zero_score.value, 0.0);
    }
}
