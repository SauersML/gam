//! Closed-form derivatives of normalized cross-Gram energies.
//!
//! Both decoder coherence and subspace overlap are a scalar monomial of three
//! polynomial statistics. Keeping that composition explicit provides their
//! gradient, Hessian diagonal, Hessian action and contracted third derivative
//! without coordinate probes, a third-order tensor, or automatic differentiation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

#[derive(Clone, Copy, Debug)]
pub enum GramNormalization {
    /// ||X Y'||² / (||X||² ||Y||²).
    DecoderNorm,
    /// ||X Y'||² / (||X X'|| ||Y Y'||).
    SelfGramNorm,
}

pub struct NormalizedCrossGram {
    left: Array2<f64>,
    right: Array2<f64>,
    cross: Array2<f64>,
    left_gram: Array2<f64>,
    right_gram: Array2<f64>,
    normalization: GramNormalization,
    gradients: [Array1<f64>; 3],
    diagonals: [Array1<f64>; 3],
    scalar_first: [f64; 3],
    scalar_second: [[f64; 3]; 3],
    scalar_third: [[[f64; 3]; 3]; 3],
}

fn flatten_pair(left: Array2<f64>, right: Array2<f64>) -> Array1<f64> {
    Array1::from_iter(left.iter().chain(right.iter()).copied())
}

impl NormalizedCrossGram {
    /// Positive normalizers define this smooth stratum. A zero decoder has no
    /// normalized direction, and is reported explicitly to the owning prior.
    pub fn new(
        left: ArrayView2<'_, f64>,
        right: ArrayView2<'_, f64>,
        normalization: GramNormalization,
    ) -> Option<Self> {
        assert_eq!(left.ncols(), right.ncols());
        let cross = left.dot(&right.t());
        let left_gram = left.dot(&left.t());
        let right_gram = right.dot(&right.t());
        let (left_stat, right_stat, exponent) = match normalization {
            GramNormalization::DecoderNorm => (
                left.iter().map(|x| x * x).sum::<f64>(),
                right.iter().map(|x| x * x).sum::<f64>(),
                -1.0,
            ),
            GramNormalization::SelfGramNorm => (
                left_gram.iter().map(|x| x * x).sum::<f64>(),
                right_gram.iter().map(|x| x * x).sum::<f64>(),
                -0.5,
            ),
        };
        if !(left_stat > 0.0 && right_stat > 0.0) {
            return None;
        }
        let energy = cross.iter().map(|x| x * x).sum::<f64>();
        // Derivatives of phi(E,u,v)=E*u^p*v^p. Multi-index orders above
        // one in E vanish. Falling factorials give all remaining derivatives.
        let scalar_derivative = |indices: &[usize]| {
            let mut orders = [0_usize; 3];
            for &index in indices {
                orders[index] += 1;
            }
            if orders[0] > 1 {
                return 0.0;
            }
            let power_derivative = |value: f64, order: usize| {
                let coefficient = (0..order).map(|i| exponent - i as f64).product::<f64>();
                coefficient * value.powf(exponent - order as f64)
            };
            (if orders[0] == 0 { energy } else { 1.0 })
                * power_derivative(left_stat, orders[1])
                * power_derivative(right_stat, orders[2])
        };
        let scalar_first = std::array::from_fn(|i| scalar_derivative(&[i]));
        let scalar_second =
            std::array::from_fn(|i| std::array::from_fn(|j| scalar_derivative(&[i, j])));
        let scalar_third = std::array::from_fn(|i| {
            std::array::from_fn(|j| std::array::from_fn(|k| scalar_derivative(&[i, j, k])))
        });
        let energy_gradient = flatten_pair(cross.dot(&right) * 2.0, cross.t().dot(&left) * 2.0);
        let left_gradient = match normalization {
            GramNormalization::DecoderNorm => left.to_owned() * 2.0,
            GramNormalization::SelfGramNorm => left_gram.dot(&left) * 4.0,
        };
        let right_gradient = match normalization {
            GramNormalization::DecoderNorm => right.to_owned() * 2.0,
            GramNormalization::SelfGramNorm => right_gram.dot(&right) * 4.0,
        };
        let left_columns =
            Array1::from_shape_fn(left.ncols(), |col| left.column(col).dot(&left.column(col)));
        let right_columns = Array1::from_shape_fn(right.ncols(), |col| {
            right.column(col).dot(&right.column(col))
        });
        let energy_diagonal = flatten_pair(
            Array2::from_shape_fn(left.dim(), |(_, col)| 2.0 * right_columns[col]),
            Array2::from_shape_fn(right.dim(), |(_, col)| 2.0 * left_columns[col]),
        );
        let left_diagonal = Array2::from_shape_fn(left.dim(), |(row, col)| match normalization {
            GramNormalization::DecoderNorm => 2.0,
            GramNormalization::SelfGramNorm => {
                4.0 * (left_columns[col]
                    + left[[row, col]] * left[[row, col]]
                    + left_gram[[row, row]])
            }
        });
        let right_diagonal = Array2::from_shape_fn(right.dim(), |(row, col)| match normalization {
            GramNormalization::DecoderNorm => 2.0,
            GramNormalization::SelfGramNorm => {
                4.0 * (right_columns[col]
                    + right[[row, col]] * right[[row, col]]
                    + right_gram[[row, row]])
            }
        });
        Some(Self {
            left: left.to_owned(),
            right: right.to_owned(),
            cross,
            left_gram,
            right_gram,
            normalization,
            gradients: [
                energy_gradient,
                flatten_pair(left_gradient, Array2::zeros(right.dim())),
                flatten_pair(Array2::zeros(left.dim()), right_gradient),
            ],
            diagonals: [
                energy_diagonal,
                flatten_pair(left_diagonal, Array2::zeros(right.dim())),
                flatten_pair(Array2::zeros(left.dim()), right_diagonal),
            ],
            scalar_first,
            scalar_second,
            scalar_third,
        })
    }

    pub fn gradient(&self) -> Array1<f64> {
        let mut out = Array1::zeros(self.gradients[0].len());
        for i in 0..3 {
            out.scaled_add(self.scalar_first[i], &self.gradients[i]);
        }
        out
    }

    pub fn diagonal(&self) -> Array1<f64> {
        let mut out = Array1::zeros(self.gradients[0].len());
        for i in 0..3 {
            out.scaled_add(self.scalar_first[i], &self.diagonals[i]);
            for j in 0..3 {
                out.scaled_add(
                    self.scalar_second[i][j],
                    &(&self.gradients[i] * &self.gradients[j]),
                );
            }
        }
        out
    }

    /// Gradient of left' B right for the frozen-normalizer cross-Gram
    /// Gauss–Newton majorizer B. Its live normalizer is differentiated too; the
    /// directional vectors stay fixed.
    pub(crate) fn gauss_newton_bilinear_gradient(
        &self,
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let (lx, ly) = self.split(left);
        let (rx, ry) = self.split(right);
        let cl = lx.dot(&self.right.t()) + self.left.dot(&ly.t());
        let cr = rx.dot(&self.right.t()) + self.left.dot(&ry.t());
        let inner = cl.iter().zip(cr.iter()).map(|(&a, &b)| a * b).sum::<f64>();
        let mut out = flatten_pair(cr.dot(&ly) + cl.dot(&ry), cr.t().dot(&lx) + cl.t().dot(&rx))
            * (2.0 * self.scalar_first[0]);
        for j in 0..3 {
            out.scaled_add(2.0 * inner * self.scalar_second[0][j], &self.gradients[j]);
        }
        out
    }

    fn split<'a>(
        &self,
        direction: ArrayView1<'a, f64>,
    ) -> (ArrayView2<'a, f64>, ArrayView2<'a, f64>) {
        assert_eq!(direction.len(), self.left.len() + self.right.len());
        let left = direction
            .slice_move(s![..self.left.len()])
            .into_shape_with_order(self.left.dim())
            .expect("validated left direction length matches the captured decoder layout");
        let right = direction
            .slice_move(s![self.left.len()..])
            .into_shape_with_order(self.right.dim())
            .expect("validated right direction length matches the captured decoder layout");
        (left, right)
    }

    fn statistic_hessians(&self, direction: ArrayView1<'_, f64>) -> [Array1<f64>; 3] {
        let (l, r) = self.split(direction);
        let dc = l.dot(&self.right.t()) + self.left.dot(&r.t());
        let energy = flatten_pair(
            (dc.dot(&self.right) + self.cross.dot(&r)) * 2.0,
            (dc.t().dot(&self.left) + self.cross.t().dot(&l)) * 2.0,
        );
        let normalizer_action = |x: &Array2<f64>, gram: &Array2<f64>, v: ArrayView2<'_, f64>| {
            match self.normalization {
                GramNormalization::DecoderNorm => v.to_owned() * 2.0,
                GramNormalization::SelfGramNorm => {
                    let one_leg = v.dot(&x.t());
                    ((&one_leg + &one_leg.t()).dot(x) + gram.dot(&v)) * 4.0
                }
            }
        };
        [
            energy,
            flatten_pair(
                normalizer_action(&self.left, &self.left_gram, l),
                Array2::zeros(self.right.dim()),
            ),
            flatten_pair(
                Array2::zeros(self.left.dim()),
                normalizer_action(&self.right, &self.right_gram, r),
            ),
        ]
    }

    pub fn hessian_action(&self, direction: ArrayView1<'_, f64>) -> Array1<f64> {
        let actions = self.statistic_hessians(direction);
        let mut out = Array1::zeros(direction.len());
        for i in 0..3 {
            out.scaled_add(self.scalar_first[i], &actions[i]);
            for j in 0..3 {
                out.scaled_add(
                    self.scalar_second[i][j] * self.gradients[j].dot(&direction),
                    &self.gradients[i],
                );
            }
        }
        out
    }

    /// Gradient of left' H right, evaluated by the analytic third-order chain
    /// rule of the three polynomial statistics. Both direction vectors are fixed.
    pub fn third_bilinear(
        &self,
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        let (lx, ly) = self.split(left);
        let (rx, ry) = self.split(right);
        let cl = lx.dot(&self.right.t()) + self.left.dot(&ly.t());
        let cr = rx.dot(&self.right.t()) + self.left.dot(&ry.t());
        let clr = lx.dot(&ry.t()) + rx.dot(&ly.t());
        let energy_third = flatten_pair(
            (clr.dot(&self.right) + cl.dot(&ry) + cr.dot(&ly)) * 2.0,
            (clr.t().dot(&self.left) + cl.t().dot(&rx) + cr.t().dot(&lx)) * 2.0,
        );
        let norm_third =
            |x: &Array2<f64>, l: ArrayView2<'_, f64>, r: ArrayView2<'_, f64>| match self
                .normalization
            {
                GramNormalization::DecoderNorm => Array2::zeros(x.dim()),
                GramNormalization::SelfGramNorm => {
                    let dl = l.dot(&x.t());
                    let dr = r.dot(&x.t());
                    let mixed = l.dot(&r.t());
                    ((&mixed + &mixed.t()).dot(x)
                        + (&dl + &dl.t()).dot(&r)
                        + (&dr + &dr.t()).dot(&l))
                        * 4.0
                }
            };
        let thirds = [
            energy_third,
            flatten_pair(
                norm_third(&self.left, lx, rx),
                Array2::zeros(self.right.dim()),
            ),
            flatten_pair(
                Array2::zeros(self.left.dim()),
                norm_third(&self.right, ly, ry),
            ),
        ];
        let h_left = self.statistic_hessians(left);
        let h_right = self.statistic_hessians(right);
        let dl: [f64; 3] = std::array::from_fn(|i| self.gradients[i].dot(&left));
        let dr: [f64; 3] = std::array::from_fn(|i| self.gradients[i].dot(&right));
        let mixed: [f64; 3] = std::array::from_fn(|i| left.dot(&h_right[i]));
        let mut out = Array1::zeros(left.len());
        for i in 0..3 {
            out.scaled_add(self.scalar_first[i], &thirds[i]);
            for j in 0..3 {
                let coefficient = self.scalar_second[i][j];
                out.scaled_add(coefficient * mixed[i], &self.gradients[j]);
                out.scaled_add(coefficient * dr[j], &h_left[i]);
                out.scaled_add(coefficient * dl[j], &h_right[i]);
                for k in 0..3 {
                    out.scaled_add(
                        self.scalar_third[i][j][k] * dl[i] * dr[j],
                        &self.gradients[k],
                    );
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    const LEFT: (usize, usize) = (2, 3);
    const RIGHT: (usize, usize) = (3, 3);

    fn split(theta: &Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        let cut = LEFT.0 * LEFT.1;
        let left = theta
            .slice(s![..cut])
            .to_owned()
            .into_shape_with_order(LEFT)
            .expect("left decoder shape");
        let right = theta
            .slice(s![cut..])
            .to_owned()
            .into_shape_with_order(RIGHT)
            .expect("right decoder shape");
        (left, right)
    }

    /// `u^p v^p`, the normalizer factor of `φ = E u^p v^p`, computed directly
    /// from the decoders.
    fn normalizer(
        left: &Array2<f64>,
        right: &Array2<f64>,
        normalization: GramNormalization,
    ) -> f64 {
        let squared = |m: &Array2<f64>| m.iter().map(|x| x * x).sum::<f64>();
        match normalization {
            GramNormalization::DecoderNorm => 1.0 / (squared(left) * squared(right)),
            GramNormalization::SelfGramNorm => {
                1.0 / (squared(&left.dot(&left.t())) * squared(&right.dot(&right.t()))).sqrt()
            }
        }
    }

    fn value(theta: &Array1<f64>, normalization: GramNormalization) -> f64 {
        let (left, right) = split(theta);
        let energy = left.dot(&right.t()).iter().map(|x| x * x).sum::<f64>();
        energy * normalizer(&left, &right, normalization)
    }

    /// `lᵀ B r = 2 u^p v^p ⟨J l, J r⟩`, where `J` is the Jacobian of the
    /// cross-Gram `X Yᵀ` and the directions stay fixed.
    fn gauss_newton_form(
        theta: &Array1<f64>,
        l: &Array1<f64>,
        r: &Array1<f64>,
        normalization: GramNormalization,
    ) -> f64 {
        let (left, right) = split(theta);
        let (lx, ly) = split(l);
        let (rx, ry) = split(r);
        let jl = lx.dot(&right.t()) + left.dot(&ly.t());
        let jr = rx.dot(&right.t()) + left.dot(&ry.t());
        2.0 * normalizer(&left, &right, normalization) * (&jl * &jr).sum()
    }

    /// Nothing tied the closed-form derivatives of the normalized cross-Gram to
    /// its value. This pins them against central differences with `h = 1e-5`,
    /// for both normalizations:
    /// - `gradient` against the value
    /// - `diagonal` against the gradient
    /// - `hessian_action` against the directional derivative of the gradient
    /// - `third_bilinear(l, r)` against the gradient of `lᵀ H r`
    /// - `gauss_newton_bilinear_gradient(l, r)` against the gradient of the
    ///   frozen-direction form `lᵀ B r`
    ///
    /// Tolerance. The stencil error is `h²/6 |f'''| + ε_mach |f| / h`. Every
    /// differenced quantity is smooth on this fixture, and its third derivative
    /// along each coordinate (and along `l`) is at most about `17`. That bound
    /// was estimated with a five-point stencil of step `2e-3`, and `lᵀ H r`
    /// under `SelfGramNorm` is the largest. The truncation error is therefore
    /// at most about `3e-10`. Every value is `O(1)`, so the roundoff is about
    /// `5e-11`. A `1e-8` tolerance leaves at least 25× margin.
    #[test]
    fn normalized_cross_gram_derivatives_match_central_differences() {
        let theta = array![
            0.7_f64, -0.3, 0.5, 0.2, 0.9, -0.4, 0.6, 0.1, -0.8, -0.5, 0.4, 0.3, 0.3, -0.7, 0.2
        ];
        let n = theta.len();
        let l = Array1::from_shape_fn(n, |i| 0.6 * (0.9 * i as f64 + 0.4).cos());
        let r = Array1::from_shape_fn(n, |i| 0.5 * (1.3 * i as f64 + 0.2).sin());
        let h = 1e-5;
        let tol = 1e-8;
        for normalization in [
            GramNormalization::DecoderNorm,
            GramNormalization::SelfGramNorm,
        ] {
            let at = |point: &Array1<f64>| {
                let (left, right) = split(point);
                NormalizedCrossGram::new(left.view(), right.view(), normalization)
                    .expect("both normalizers are positive")
            };
            let geometry = at(&theta);
            let gradient = geometry.gradient();
            let diagonal = geometry.diagonal();
            let third = geometry.third_bilinear(l.view(), r.view());
            let gauss_newton = geometry.gauss_newton_bilinear_gradient(l.view(), r.view());
            for i in 0..n {
                let mut plus = theta.clone();
                let mut minus = theta.clone();
                plus[i] += h;
                minus[i] -= h;
                let (upper, lower) = (at(&plus), at(&minus));
                let fd_value =
                    (value(&plus, normalization) - value(&minus, normalization)) / (2.0 * h);
                assert_abs_diff_eq!(gradient[i], fd_value, epsilon = tol);
                let fd_diagonal = (upper.gradient()[i] - lower.gradient()[i]) / (2.0 * h);
                assert_abs_diff_eq!(diagonal[i], fd_diagonal, epsilon = tol);
                let fd_third = (l.dot(&upper.hessian_action(r.view()))
                    - l.dot(&lower.hessian_action(r.view())))
                    / (2.0 * h);
                assert_abs_diff_eq!(third[i], fd_third, epsilon = tol);
                let fd_gauss_newton = (gauss_newton_form(&plus, &l, &r, normalization)
                    - gauss_newton_form(&minus, &l, &r, normalization))
                    / (2.0 * h);
                assert_abs_diff_eq!(gauss_newton[i], fd_gauss_newton, epsilon = tol);
            }
            let action = geometry.hessian_action(l.view());
            let fd_action = (at(&(&theta + &(h * &l))).gradient()
                - at(&(&theta - &(h * &l))).gradient())
                / (2.0 * h);
            for i in 0..n {
                assert_abs_diff_eq!(action[i], fd_action[i], epsilon = tol);
            }
            assert_abs_diff_eq!(
                l.dot(&geometry.hessian_action(r.view())),
                r.dot(&action),
                epsilon = 1e-12
            );
        }
    }
}
