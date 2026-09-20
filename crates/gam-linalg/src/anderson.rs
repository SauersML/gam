//! Anderson acceleration of a fixed-point map.
//!
//! For an iteration `x_{k+1} = G(x_k)` whose residual `f(x) = G(x) - x`
//! contracts linearly, the plain iteration needs `log(tol)/log(ρ)` steps. When
//! `ρ` is close to one — an alternating block minimisation on a coupled problem
//! typically lands at `ρ ≈ 0.95`–`0.99` — that is hundreds of steps spent on
//! the tail, and each one costs a full pass over the data.
//!
//! Anderson acceleration (Anderson, *JACM* 1965; Walker & Ni, *SINUM* 2011)
//! replaces the next iterate with the affine combination of the last `m + 1`
//! images whose residual combination is smallest:
//!
//! ```text
//! γ  = argmin_γ ‖ f_k − ΔF_k γ ‖₂ ,   ΔF_k = [f_{k-m+1}−f_{k-m}, …, f_k−f_{k-1}]
//! x⁺ = g_k − ΔG_k γ ,                 ΔG_k = [g_{k-m+1}−g_{k-m}, …, g_k−g_{k-1}]
//! ```
//!
//! It is a multisecant quasi-Newton step on `f`: with `m` stored differences it
//! matches the secant condition on the `m`-dimensional subspace they span, and
//! it needs nothing from the map beyond its images — no Jacobian, no
//! directional derivative, no extra evaluation.
//!
//! # Why this speaks only in differences
//!
//! [`AndersonAccelerator::propose`] never sees an iterate. It takes the current
//! residual `f_k` and the STEP the caller actually took to reach `x_k`, and it
//! returns the step to take next. Everything it forms is a difference:
//!
//! ```text
//! Δf_k = f_k − f_{k-1}
//! Δg_k = (x_k − x_{k-1}) + Δf_k        // = g_k − g_{k-1}, by definition of g
//! step⁺ = f_k − ΔG_k γ                 // so x⁺ = x_k + step⁺ = g_k − ΔG_k γ
//! ```
//!
//! That is not a stylistic choice; it is what makes the method usable on a
//! state with a MANIFOLD structure. On a periodic coordinate the iteration
//! projects each image back to a principal branch, so absolute values live in
//! whatever lift the last projection left them in and `g_k − g_{k-1}` computed
//! from stored absolutes can be a whole period where the motion was
//! infinitesimal. Differences taken by the caller — which knows each axis's
//! topology — are unambiguous, and returning a step means the caller applies it
//! through its own retraction rather than having an extrapolated absolute
//! position forced back onto the manifold by projection.
//!
//! The caller also owns the safeguard. A proposal is a candidate, not a step:
//! Anderson has no descent guarantee, and the honest contract is that the caller
//! evaluates its own merit function at the candidate and keeps the plain iterate
//! when the candidate does not improve on it (Walker & Ni §5; Toth & Kelley,
//! *SINUM* 2015, for the safeguarded convergence theory). `Self::reset` is
//! there for the rejection path.

use crate::LinalgError;
use crate::faer_ndarray::FaerEigh;
use crate::roundoff::accumulation_growth;
use faer::Side;
use ndarray::{Array1, Array2};
use std::collections::VecDeque;

/// Bounded-history Anderson accelerator over a fixed-point map.
#[derive(Debug, Clone)]
pub struct AndersonAccelerator {
    depth: usize,
    dimension: Option<usize>,
    /// `f_{k-1}` — the previous residual, kept so the next call can form one new
    /// difference column. No absolute iterate is ever stored.
    previous_residual: Option<Vec<f64>>,
    /// Difference columns, oldest first, at most `depth` of each.
    image_differences: VecDeque<Vec<f64>>,
    residual_differences: VecDeque<Vec<f64>>,
}

impl AndersonAccelerator {
    /// `depth` is the number of stored difference columns `m` — the order of
    /// the multisecant model, and the method's only knob.
    ///
    /// There is deliberately no regularisation parameter. `ΔFᵀΔF` is
    /// rank-deficient by construction whenever the map has an exactly flat
    /// direction — a gauge orbit, for instance — because consecutive residuals
    /// then differ by nothing along it, and it is near-deficient whenever two
    /// stored differences are nearly parallel, which is what a long history on a
    /// slowly-contracting map produces. Rather than pick a shift, the solve
    /// DERIVES the floor below which an eigenvalue is indistinguishable from the
    /// Gram's own accumulation roundoff (see `Self::solve_multisecant`) and
    /// drops those modes. A flat direction then contributes nothing, instead of
    /// contributing a large coefficient that a chosen shift merely bounds.
    pub fn new(depth: usize) -> Result<Self, LinalgError> {
        if depth == 0 {
            return Err(LinalgError::InvalidInput(
                "Anderson acceleration requires a positive history depth".to_string(),
            ));
        }
        Ok(Self {
            depth,
            dimension: None,
            previous_residual: None,
            image_differences: VecDeque::with_capacity(depth),
            residual_differences: VecDeque::with_capacity(depth),
        })
    }

    /// Number of stored difference columns — the effective order of the current
    /// multisecant model.
    pub fn history_len(&self) -> usize {
        self.residual_differences.len()
    }

    /// Forget the history, keeping the configuration.
    ///
    /// The caller resets after rejecting a proposal: the rejected candidate is
    /// not the iterate the next residual will be measured from, so keeping
    /// differences taken across that break would fit a secant model to a
    /// trajectory that never happened.
    pub fn reset(&mut self) {
        self.previous_residual = None;
        self.image_differences.clear();
        self.residual_differences.clear();
    }

    /// Offer the current residual and the step that reached the current
    /// iterate; return the accelerated STEP to take from it.
    ///
    /// `taken_step` is `x_k − x_{k-1}`, expressed by the caller in whatever
    /// geometry its state has — it is ignored on the first call, where there is
    /// no previous iterate. Pass the plain step when the previous cycle took
    /// one, or the accepted extrapolated step when it took that; passing
    /// anything else fits the secant model to a trajectory that did not happen,
    /// which is what [`Self::reset`] is for.
    ///
    /// `Ok(None)` on the first call (no difference column yet) and whenever the
    /// least squares carries no usable information — an all-zero `ΔF`, or a
    /// step that is not finite. Those are ordinary states of a converging
    /// iteration, not errors: the caller simply takes the plain step `f_k`.
    pub fn propose(
        &mut self,
        residual: &[f64],
        taken_step: &[f64],
    ) -> Result<Option<Vec<f64>>, LinalgError> {
        if residual.len() != taken_step.len() {
            return Err(LinalgError::InvalidInput(format!(
                "Anderson residual width {} != step width {}",
                residual.len(),
                taken_step.len()
            )));
        }
        if residual.is_empty() {
            return Err(LinalgError::InvalidInput(
                "Anderson acceleration requires a non-empty state".to_string(),
            ));
        }
        match self.dimension {
            None => self.dimension = Some(residual.len()),
            Some(dimension) if dimension != residual.len() => {
                return Err(LinalgError::InvalidInput(format!(
                    "Anderson state width changed from {dimension} to {}",
                    residual.len()
                )));
            }
            Some(_) => {}
        }
        if residual
            .iter()
            .chain(taken_step)
            .any(|value| !value.is_finite())
        {
            return Err(LinalgError::InvalidInput(
                "Anderson acceleration requires a finite residual and step".to_string(),
            ));
        }

        if let Some(previous_residual) = self.previous_residual.as_ref() {
            let residual_difference: Vec<f64> = residual
                .iter()
                .zip(previous_residual)
                .map(|(new, old)| new - old)
                .collect();
            // g_k − g_{k-1} = (x_k − x_{k-1}) + (f_k − f_{k-1}).
            let image_difference: Vec<f64> = taken_step
                .iter()
                .zip(&residual_difference)
                .map(|(step, difference)| step + difference)
                .collect();
            if self.residual_differences.len() == self.depth {
                self.image_differences.pop_front();
                self.residual_differences.pop_front();
            }
            self.image_differences.push_back(image_difference);
            self.residual_differences.push_back(residual_difference);
        }
        self.previous_residual = Some(residual.to_vec());

        let order = self.residual_differences.len();
        if order == 0 {
            return Ok(None);
        }
        let Some(coefficients) = self.solve_multisecant(residual, order)? else {
            return Ok(None);
        };
        let mut step = residual.to_vec();
        for (column, weight) in self.image_differences.iter().zip(coefficients.iter()) {
            for (value, difference) in step.iter_mut().zip(column) {
                *value -= weight * difference;
            }
        }
        if step.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        Ok(Some(step))
    }

    /// `γ = (ΔFᵀΔF)⁺ ΔFᵀ f_k` on the modes the Gram can actually resolve.
    ///
    /// An eigendecomposition of the `order × order` normal matrix (`order ≤
    /// depth`, so a handful of flops next to one evaluation of the map), with
    /// every mode at or below a DERIVED floor set to zero rather than inverted.
    ///
    /// The floor is the Gram's own backward-error band. Each entry is an inner
    /// product over the full state width, so its computed value carries an
    /// absolute error up to `γ_dimension · Σ|terms|`; summed over the diagonal
    /// that is `γ_dimension · trace`. An eigenvalue below that is not
    /// distinguishable from the roundoff of forming the matrix, and inverting it
    /// amplifies noise by its reciprocal. This is a bound, not a tuned shift:
    /// a genuinely informative secant direction has an eigenvalue of order
    /// `‖Δf‖²`, which is enormous next to `γ_n·trace`, so the floor never
    /// touches one.
    fn solve_multisecant(
        &self,
        residual: &[f64],
        order: usize,
    ) -> Result<Option<Array1<f64>>, LinalgError> {
        let mut normal = Array2::<f64>::zeros((order, order));
        for (left, left_column) in self.residual_differences.iter().enumerate() {
            for (right, right_column) in self.residual_differences.iter().enumerate().skip(left) {
                let entry: f64 = left_column
                    .iter()
                    .zip(right_column)
                    .map(|(a, b)| a * b)
                    .sum();
                normal[[left, right]] = entry;
                normal[[right, left]] = entry;
            }
        }
        let trace: f64 = (0..order).map(|index| normal[[index, index]]).sum();
        if !(trace.is_finite() && trace > 0.0) {
            return Ok(None);
        }
        let floor = accumulation_growth(self.dimension.unwrap_or(0).max(1)) * trace;
        if !floor.is_finite() {
            // A state so wide that its inner products carry no error bound at
            // all: the multisecant model cannot be trusted, so take the plain
            // step rather than extrapolate on unbounded noise.
            return Ok(None);
        }
        let rhs = Array1::from_iter(self.residual_differences.iter().map(|column| {
            column
                .iter()
                .zip(residual)
                .map(|(difference, value)| difference * value)
                .sum::<f64>()
        }));
        let (eigenvalues, eigenvectors) = normal.eigh(Side::Lower).map_err(|error| {
            LinalgError::InvalidInput(format!(
                "Anderson multisecant eigendecomposition failed: {error}"
            ))
        })?;
        let projected = eigenvectors.t().dot(&rhs);
        let mut spectral = Array1::<f64>::zeros(order);
        for mode in 0..order {
            if eigenvalues[mode] > floor {
                spectral[mode] = projected[mode] / eigenvalues[mode];
            }
        }
        let coefficients = eigenvectors.dot(&spectral);
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        Ok(Some(coefficients))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A linear contraction `G(x) = A x + b` with spectral radius 0.975 — the
    /// rate measured on the support-sparse inner fixed point (#2575). Anderson
    /// with a history at least the problem's dimension is EXACT on an affine
    /// map after that many steps, so this pins both the algebra and the
    /// convention (`propose` takes the residual, not the image).
    #[test]
    fn an_affine_contraction_is_solved_in_dimension_steps() {
        // Row sums 0.995, 0.930, 0.810, so the map is a genuine contraction in
        // the max norm; its spectral radius is ≈ 0.975, the rate #2575 measured.
        let a = [[0.975_f64, 0.02, 0.0], [0.0, 0.90, 0.03], [0.01, 0.0, 0.80]];
        let b = [1.0_f64, -2.0, 0.5];
        let map = |x: &[f64]| -> Vec<f64> {
            (0..3)
                .map(|row| (0..3).map(|col| a[row][col] * x[col]).sum::<f64>() + b[row])
                .collect()
        };
        // Fixed point by plain iteration, run long enough to be exact to 1e-14.
        let mut reference = vec![0.0_f64; 3];
        for _ in 0..20_000 {
            reference = map(&reference);
        }
        assert!(
            reference.iter().all(|value| value.is_finite()),
            "the fixture must contract: {reference:?}"
        );

        let mut accelerator = AndersonAccelerator::new(3).expect("accelerator");
        let mut x = vec![0.0_f64; 3];
        let mut taken_step = vec![0.0_f64; 3];
        let mut accelerated_error = f64::INFINITY;
        for _ in 0..6 {
            let g = map(&x);
            let residual: Vec<f64> = g.iter().zip(&x).map(|(g, x)| g - x).collect();
            let step = accelerator
                .propose(&residual, &taken_step)
                .expect("propose")
                .unwrap_or_else(|| residual.clone());
            for (value, delta) in x.iter_mut().zip(&step) {
                *value += delta;
            }
            taken_step = step;
            accelerated_error = x
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
        }
        assert!(
            accelerated_error < 1.0e-10,
            "six accelerated steps must solve a 3-dimensional affine map; error {accelerated_error:.3e}"
        );

        // The plain iteration at the same budget is nowhere near.
        let mut plain = vec![0.0_f64; 3];
        for _ in 0..6 {
            plain = map(&plain);
        }
        let plain_error = plain
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            plain_error > 1.0e3 * accelerated_error,
            "the comparison is only meaningful if the plain rate is genuinely slow: \
             plain {plain_error:.3e} vs accelerated {accelerated_error:.3e}"
        );
    }

    /// An exactly flat direction — a gauge orbit — makes `ΔF` rank-deficient.
    /// The accelerator must still produce a finite, bounded proposal, and must
    /// not extrapolate along the flat direction just because the least squares
    /// cannot see it.
    #[test]
    fn a_flat_gauge_direction_does_not_blow_up_the_extrapolation() {
        // Coordinate 2 is invariant under the map and carries no residual.
        let map = |x: &[f64]| -> Vec<f64> { vec![0.98 * x[0] + 1.0, 0.95 * x[1] - 0.5, x[2]] };
        let mut accelerator = AndersonAccelerator::new(4).expect("accelerator");
        let mut x = vec![0.0_f64, 0.0, 7.0];
        let mut taken_step = vec![0.0_f64; 3];
        for _ in 0..8 {
            let g = map(&x);
            let residual: Vec<f64> = g.iter().zip(&x).map(|(g, x)| g - x).collect();
            let step = accelerator
                .propose(&residual, &taken_step)
                .expect("propose")
                .unwrap_or_else(|| residual.clone());
            for (value, delta) in x.iter_mut().zip(&step) {
                *value += delta;
            }
            taken_step = step;
            assert!(x.iter().all(|value| value.is_finite()), "{x:?}");
            assert_eq!(x[2], 7.0, "the flat coordinate must not move");
        }
        assert!((x[0] - 50.0).abs() < 1.0e-8, "x0 -> 1/(1-0.98) = 50");
        assert!((x[1] + 10.0).abs() < 1.0e-8, "x1 -> -0.5/(1-0.95) = -10");
    }

    #[test]
    fn the_configuration_and_the_state_width_are_validated() {
        assert!(AndersonAccelerator::new(0).is_err());
                let mut accelerator = AndersonAccelerator::new(2).expect("accelerator");
        assert!(accelerator.propose(&[], &[]).is_err());
        assert!(accelerator.propose(&[1.0], &[1.0, 2.0]).is_err());
        assert!(accelerator.propose(&[0.1, 0.2], &[0.0, 0.0]).is_ok());
        assert!(
            accelerator.propose(&[0.1], &[0.0]).is_err(),
            "a state that changes width is a caller error, not a silent restart"
        );
        assert!(accelerator.propose(&[0.1, f64::NAN], &[0.0, 0.0]).is_err());
    }

    /// The first call has no difference column, so there is nothing to
    /// extrapolate from; `reset` returns to that state.
    #[test]
    fn the_history_starts_and_resets_empty() {
        let mut accelerator = AndersonAccelerator::new(2).expect("accelerator");
        assert_eq!(accelerator.history_len(), 0);
        assert!(
            accelerator
                .propose(&[0.1, 0.2], &[0.0, 0.0])
                .expect("propose")
                .is_none()
        );
        assert!(
            accelerator
                .propose(&[0.05, 0.1], &[0.1, 0.2])
                .expect("propose")
                .is_some()
        );
        assert_eq!(accelerator.history_len(), 1);
        accelerator.reset();
        assert_eq!(accelerator.history_len(), 0);
        assert!(
            accelerator
                .propose(&[0.02, 0.05], &[0.05, 0.1])
                .expect("propose")
                .is_none()
        );
    }
}
