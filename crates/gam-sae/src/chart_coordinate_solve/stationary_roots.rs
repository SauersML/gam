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
        global_minimum(&objective.derivative(), |t| {
            Ok(objective.eval_real(t))
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
        let scale = self
            .coefficients
            .iter()
            .map(|coefficient| coefficient.re.hypot(coefficient.im))
            .fold(0.0_f64, f64::max);
        if scale == 0.0 || !scale.is_finite() {
            return scale;
        }
        scale
            * self
                .coefficients
                .iter()
                .map(|coefficient| coefficient.re.hypot(coefficient.im) / scale)
                .sum::<f64>()
    }

    fn normalized(&self) -> Result<(Self, f64), String> {
        let scale = self
            .coefficients
            .iter()
            .map(|coefficient| coefficient.re.hypot(coefficient.im))
            .fold(0.0_f64, f64::max);
        if !(scale.is_finite() && scale > 0.0) {
            return Err(format!(
                "periodic polynomial normalization requires a finite positive scale, got {scale}"
            ));
        }
        let mut normalized = self.clone();
        for coefficient in &mut normalized.coefficients {
            *coefficient /= scale;
        }
        Ok((normalized, scale))
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

fn global_minimum(
    stationarity: &LaurentPolynomial,
    mut evaluate: impl FnMut(f64) -> Result<f64, String>,
) -> Result<PeriodicExtremum, String> {
    if stationarity.is_numerically_zero() {
        let value = evaluate(0.0)?;
        if !value.is_finite() {
            return Err("periodic extremum: constant objective is not finite".to_string());
        }
        return Ok(PeriodicExtremum {
            coordinate: 0.0,
            value,
        });
    }
    // Root locations are invariant to a scalar multiple.  Normalizing here
    // also protects the distance objective (not only the profiled objective)
    // from coefficient-norm overflow during degree trimming and residual
    // verification.
    let (stationarity, _) = stationarity.normalized()?;
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
    let candidates = stationary_coordinates(&stationarity)?;
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
                value < incumbent.value
                    || (value == incumbent.value && coordinate < incumbent.coordinate)
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
    // Companion eigenvalues belonging to a repeated root may yield duplicate
    // angles.  Keeping duplicates is harmless when candidates are scored, while
    // tolerance-based merging is not: distinct stationary points can be
    // arbitrarily close, including on opposite sides of the period seam.  Only
    // remove bit-identical coordinates; never synthesize a seam coordinate that
    // was not itself verified as a root.
    coordinates.dedup_by(|left, right| left.to_bits() == right.to_bits());
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
    use ndarray::{Array2, array};

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
        let solved = solver
            .maximize_nonnegative_profiled_score(&linear)
            .unwrap();
        let q = harmonic_linear(&linear);
        let r = harmonic_quadratic(gram.view());
        let dense_best = (0..200_003)
            .map(|index| index as f64 / 200_003.0)
            .map(|t| nonnegative_profiled_score_at(&q, &r, t).unwrap())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(solved.value + 1e-10 >= dense_best);
    }

    #[test]
    fn constant_and_zero_curve_choose_canonical_origin() {
        let gram = array![[4.0]];
        let solver = PeriodicCurveExtrema::from_gram(gram.view()).unwrap();
        let distance = solver.minimize_squared_distance(&[3.0]).unwrap();
        let score = solver
            .maximize_nonnegative_profiled_score(&[3.0])
            .unwrap();
        assert_eq!(distance.coordinate, 0.0);
        assert_eq!(score.coordinate, 0.0);
        assert!((score.value - 9.0 / 4.0).abs() < 1e-14);

        let zero = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let zero_solver = PeriodicCurveExtrema::from_gram(zero.view()).unwrap();
        let zero_score = zero_solver
            .maximize_nonnegative_profiled_score(&[0.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(zero_score.coordinate, 0.0);
        assert_eq!(zero_score.value, 0.0);
    }

    #[test]
    fn profiled_coordinate_is_invariant_to_extreme_decoder_scale() {
        let planted = 0.173_205_080_756_887_73;
        let angle = std::f64::consts::TAU * planted;
        for amplitude in [1.0e-120_f64, 1.0, 1.0e120_f64] {
            let gram = array![
                [0.0, 0.0, 0.0],
                [0.0, amplitude * amplitude, 0.0],
                [0.0, 0.0, amplitude * amplitude],
            ];
            let linear = [0.0, amplitude * angle.sin(), amplitude * angle.cos()];
            let solver = PeriodicCurveExtrema::from_gram(gram.view()).unwrap();
            let solved = solver
                .maximize_nonnegative_profiled_score(&linear)
                .unwrap();
            let error = circular_distance(solved.coordinate, planted);
            assert!(
                error < 1.0e-10,
                "decoder scale {amplitude:e} returned {}, circular error {error}",
                solved.coordinate
            );
        }
    }

    #[test]
    fn nonzero_projection_against_zero_gram_is_rejected() {
        let gram = Array2::<f64>::zeros((3, 3));
        let solver = PeriodicCurveExtrema::from_gram(gram.view()).unwrap();
        let error = solver
            .maximize_nonnegative_profiled_score(&[0.0, 1.0, 0.0])
            .unwrap_err();
        assert!(error.contains("zero decoder Gram"), "{error}");
    }

    #[test]
    fn strict_objective_ordering_beats_coordinate_tie_break() {
        // sin(2*pi*t) has stationary candidates at 0 and 1/2.  Their supplied
        // objective values differ by exactly one ulp at 1.0: the larger value
        // must win even though its coordinate is larger.
        let stationarity = harmonic_linear(&[0.0, 1.0, 0.0]);
        let solved = global_extremum(&stationarity, ExtremumKind::Maximum, |coordinate| {
            Ok::<f64, String>(if (coordinate - 0.5).abs() < f64::EPSILON.sqrt() {
                1.0 + f64::EPSILON
            } else {
                1.0
            })
        })
        .unwrap();
        assert!((solved.coordinate - 0.5).abs() < f64::EPSILON.sqrt());
    }

    #[test]
    fn distinct_roots_are_not_merged_across_the_period_seam() {
        let offset = 2.0e-8_f64;
        let sine_with_root = |root: f64| {
            let angle = std::f64::consts::TAU * root;
            harmonic_linear(&[0.0, angle.cos(), -angle.sin()])
        };
        let polynomial = sine_with_root(offset).multiply(&sine_with_root(1.0 - offset));
        let roots = stationary_coordinates(&polynomial).unwrap();
        for expected in [offset, 1.0 - offset, 0.5 - offset, 0.5 + offset] {
            assert!(
                roots
                    .iter()
                    .any(|&root| circular_distance(root, expected) < 1.0e-8),
                "missing close stationary root {expected}; got {roots:?}"
            );
        }
        assert!(
            roots.iter().all(|&root| root != 0.0),
            "the seam canonicalizer fabricated t=0 from distinct roots: {roots:?}"
        );
    }
}
