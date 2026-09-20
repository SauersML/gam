//! Newton step on the profiled decoder objective (#3193).
//!
//! At fixed supports the codes of a row are the ridge solve `c = (G + ρI)⁻¹ A x`,
//! with `A` the row's active decoder rows and `G = A Aᵀ`, so the penalized loss
//! `Σ_i ‖x_i − A_iᵀ c_i‖² + ρ‖c_i‖²` is a function `F(D)` of the decoder alone
//! (variable projection). Every decoder row lives on the unit sphere. The epoch map
//! of [`super::update`] is alternating minimization of that objective, and it
//! contracts only linearly: where coherent atoms rotate inside their span the ridge
//! is the only term that picks the rotation, and the map crawls at a rate near one.
//! A per-epoch displacement then says nothing about the distance to the fixed point,
//! which is `step/(1 − r)`.
//!
//! This module measures that distance and closes it. The Newton decrement
//! `½ gᵀH⁻¹g` of `F` on the product of spheres is the loss the local quadratic model
//! still promises, and the step `−H⁻¹g` reaches its minimizer. Both come from one
//! conjugate-gradient solve on the tangent space with Hessian-vector products.
//!
//! **Gradient.** By the envelope theorem the codes' own sensitivity drops out, and
//! `∂F/∂d_a = −2 Σ_{(i, j): a_ij = a} c_ij r_i` with `r_i = x_i − A_iᵀ c_i`.
//!
//! **Hessian-vector product.** For a direction `V` (rows `v_a`), write `V_i` for
//! the row's active directions and `w_i = V_iᵀ c_i`. Differentiating
//! `(G + ρI) c = A x` gives `δc = (G + ρI)⁻¹(V_i r_i − A_i w_i)`, and the residual
//! moves by `δr = −(w_i + A_iᵀ δc)`. Differentiating the gradient then gives
//! `−2 Σ (δc_ij r_i + c_ij δr_i)` per atom. The Riemannian Hessian on the sphere
//! is that product projected to the tangent space minus `⟨d_a, ∂F/∂d_a⟩ v_a`.
//!
//! Every per-atom sum runs over the atom's firings in row order and every inner
//! product over atoms in atom order, so the step does not depend on the thread
//! count.

use super::codes::{ResolvedActiveGram, SparseCode};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// One row's profiled state at fixed support: its live atoms, the f64 ridge codes
/// at the decoder, the residual, and the resolved Gram factor.
struct ProfiledRow {
    atoms: Vec<usize>,
    codes: Array1<f64>,
    residual: Array1<f64>,
    gram: ResolvedActiveGram,
}

/// What one decoder Newton solve measured and proposes.
pub(super) struct DecoderNewtonStep {
    /// `½ gᵀH⁻¹g` in loss units, the decrease the quadratic model promises.
    pub(super) decrement: f64,
    /// The solve resolved the decrement: the Hessian was positive definite on every
    /// Krylov direction and conjugate gradients converged.
    pub(super) resolved: bool,
    /// The decoder the Newton step reaches, retracted to unit rows. `None` when the
    /// first Krylov direction already had nonpositive curvature.
    pub(super) candidate: Option<Array2<f32>>,
    /// Hessian-vector products spent.
    pub(super) hessian_products: usize,
}

/// The profiled objective `F(D)` at one decoder, with its Riemannian gradient and
/// Hessian-vector product on the product of unit spheres over the movable atoms.
struct ProfiledDecoder {
    decoder: Array2<f64>,
    moves: Vec<bool>,
    rows: Vec<Option<ProfiledRow>>,
    /// Firings of each movable atom as `(row, slot)`, in row order.
    firings: Vec<Vec<(usize, usize)>>,
    /// `⟨d_a, ∂F/∂d_a⟩`, the radial part of the Euclidean gradient.
    radial: Vec<f64>,
    /// The tangent gradient.
    gradient: Array2<f64>,
}

impl ProfiledDecoder {
    /// Profile `decoder` (unit rows) at the supports `codes` fire. Only atoms marked
    /// in `movable` with a nonzero row are coordinates.
    fn new(
        x: ArrayView2<'_, f32>,
        decoder: Array2<f64>,
        codes: &[SparseCode],
        code_ridge: f64,
        movable: &[bool],
    ) -> Self {
        let (k, p) = decoder.dim();
        let moves: Vec<bool> = (0..k)
            .map(|atom| movable[atom] && decoder.row(atom).iter().any(|&value| value != 0.0))
            .collect();
        let rows: Vec<Option<ProfiledRow>> = codes
            .par_iter()
            .enumerate()
            .map(|(row, code)| profile_row(x.row(row), decoder.view(), code, code_ridge))
            .collect();
        let mut firings: Vec<Vec<(usize, usize)>> = vec![Vec::new(); k];
        for (row, profiled) in rows.iter().enumerate() {
            if let Some(profiled) = profiled {
                for (slot, &atom) in profiled.atoms.iter().enumerate() {
                    if moves[atom] {
                        firings[atom].push((row, slot));
                    }
                }
            }
        }
        let euclidean_gradient: Array2<f64> = per_atom(k, p, |atom, out| {
            for &(row, slot) in &firings[atom] {
                let profiled = rows[row].as_ref().expect("a firing row is profiled");
                out.scaled_add(-2.0 * profiled.codes[slot], &profiled.residual);
            }
        });
        let radial: Vec<f64> = (0..k)
            .map(|atom| decoder.row(atom).dot(&euclidean_gradient.row(atom)))
            .collect();
        let mut gradient = euclidean_gradient;
        project_tangent(&mut gradient, decoder.view(), &moves);
        Self {
            decoder,
            moves,
            rows,
            firings,
            radial,
            gradient,
        }
    }

    /// The Riemannian Hessian applied to the tangent direction `direction`.
    fn hessian_product(&self, direction: &Array2<f64>) -> Array2<f64> {
        let (k, p) = self.decoder.dim();
        let sensitivities: Vec<Option<(Array1<f64>, Array1<f64>)>> = self
            .rows
            .par_iter()
            .map(|profiled| {
                let profiled = profiled.as_ref()?;
                let mut along = Array1::<f64>::zeros(p);
                for (slot, &atom) in profiled.atoms.iter().enumerate() {
                    along.scaled_add(profiled.codes[slot], &direction.row(atom));
                }
                let rhs: Array1<f64> = profiled
                    .atoms
                    .iter()
                    .map(|&atom| {
                        direction.row(atom).dot(&profiled.residual)
                            - self.decoder.row(atom).dot(&along)
                    })
                    .collect();
                let code_change = profiled.gram.solve(rhs.view());
                let mut residual_change = along;
                for (slot, &atom) in profiled.atoms.iter().enumerate() {
                    residual_change.scaled_add(code_change[slot], &self.decoder.row(atom));
                }
                residual_change.mapv_inplace(|value| -value);
                Some((code_change, residual_change))
            })
            .collect();
        let mut product = per_atom(k, p, |atom, out| {
            for &(row, slot) in &self.firings[atom] {
                let profiled = self.rows[row].as_ref().expect("a firing row is profiled");
                let (code_change, residual_change) = sensitivities[row]
                    .as_ref()
                    .expect("a firing row has sensitivities");
                out.scaled_add(-2.0 * code_change[slot], &profiled.residual);
                out.scaled_add(-2.0 * profiled.codes[slot], residual_change);
            }
        });
        project_tangent(&mut product, self.decoder.view(), &self.moves);
        for atom in 0..k {
            if self.moves[atom] {
                product
                    .row_mut(atom)
                    .scaled_add(-self.radial[atom], &direction.row(atom));
            }
        }
        product
    }
}

/// The decoder Newton step at `decoder`, holding the supports of `codes` fixed.
///
/// Only atoms marked in `movable` move: those the epoch map refreshes. An atom the
/// map holds is not a coordinate of its fixed point, so it is not one of the
/// objective's coordinates here either. Zero rows are dormant capacity and never
/// move.
///
/// Conjugate gradients stops when the residual falls to `relative_tolerance` of the
/// gradient, or when an iteration no longer moves the decrement at f64 resolution.
/// A direction of nonpositive curvature ends the solve unresolved: the point is not
/// a local minimum of the profiled objective, and the step keeps the iterate built
/// before that direction.
pub(super) fn decoder_newton_step(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    code_ridge: f32,
    movable: &[bool],
    relative_tolerance: f64,
) -> DecoderNewtonStep {
    let (k, p) = decoder.dim();
    let profiled = ProfiledDecoder::new(
        x,
        decoder.mapv(f64::from),
        codes,
        f64::from(code_ridge),
        movable,
    );
    let gradient = &profiled.gradient;
    let gradient_norm2 = inner(gradient, gradient);
    if gradient_norm2 == 0.0 {
        return DecoderNewtonStep {
            decrement: 0.0,
            resolved: true,
            candidate: None,
            hessian_products: 0,
        };
    }
    let stop = relative_tolerance * gradient_norm2.sqrt();
    let mut step = Array2::<f64>::zeros((k, p));
    let mut residual = gradient.mapv(|value| -value);
    let mut search = residual.clone();
    let mut residual_norm2 = gradient_norm2;
    let mut decrement = 0.0f64;
    let mut resolved = false;
    let mut hessian_products = 0usize;
    loop {
        let curved = profiled.hessian_product(&search);
        hessian_products += 1;
        let curvature = inner(&search, &curved);
        if !(curvature > 0.0) {
            log::debug!(
                "[SAE decoder Newton] nonpositive curvature {curvature:.3e} on Krylov direction \
                 {hessian_products}: |g|={:.3e} |r|={:.3e} decrement so far {decrement:.3e}",
                gradient_norm2.sqrt(),
                residual_norm2.sqrt(),
            );
            break;
        }
        let length = residual_norm2 / curvature;
        step.scaled_add(length, &search);
        residual.scaled_add(-length, &curved);
        // Each CG iteration lowers the quadratic model by `½·‖r‖⁴/(sᵀHs)`, and those
        // decreases sum to the decrement.
        let gain = 0.5 * length * residual_norm2;
        decrement += gain;
        let next_norm2 = inner(&residual, &residual);
        if next_norm2.sqrt() <= stop || gain <= f64::EPSILON * decrement {
            resolved = true;
            break;
        }
        let beta = next_norm2 / residual_norm2;
        residual_norm2 = next_norm2;
        search.mapv_inplace(|value| beta * value);
        search += &residual;
    }

    let candidate = (decrement > 0.0).then(|| {
        let mut moved = decoder.to_owned();
        for atom in 0..k {
            if !profiled.moves[atom] {
                continue;
            }
            let mut row = profiled.decoder.row(atom).to_owned();
            row += &step.row(atom);
            let norm = row.dot(&row).sqrt();
            for (slot, &value) in moved.row_mut(atom).iter_mut().zip(row.iter()) {
                *slot = (value / norm) as f32;
            }
        }
        moved
    });
    DecoderNewtonStep {
        decrement: if resolved { decrement } else { f64::INFINITY },
        resolved,
        candidate,
        hessian_products,
    }
}

/// The profiled state of one row, or `None` when its code fires no atom.
fn profile_row(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f64>,
    code: &SparseCode,
    code_ridge: f64,
) -> Option<ProfiledRow> {
    let atoms: Vec<usize> = code
        .indices
        .iter()
        .zip(code.codes.iter())
        .filter(|entry| *entry.1 != 0.0)
        .map(|entry| *entry.0 as usize)
        .collect();
    if atoms.is_empty() {
        return None;
    }
    let p = row.len();
    let target: Array1<f64> = row.mapv(f64::from);
    let active: Array2<f64> = decoder.select(Axis(0), &atoms);
    let gram_matrix = active.dot(&active.t());
    let gram = ResolvedActiveGram::new(&gram_matrix, code_ridge, p);
    let codes = gram.solve(active.dot(&target).view());
    let residual = &target - &active.t().dot(&codes);
    Some(ProfiledRow {
        atoms,
        codes,
        residual,
        gram,
    })
}

/// A `k × p` array whose rows are filled independently by `fill`.
fn per_atom(k: usize, p: usize, fill: impl Fn(usize, &mut Array1<f64>) + Sync) -> Array2<f64> {
    let rows: Vec<Array1<f64>> = (0..k)
        .into_par_iter()
        .map(|atom| {
            let mut out = Array1::<f64>::zeros(p);
            fill(atom, &mut out);
            out
        })
        .collect();
    let mut out = Array2::<f64>::zeros((k, p));
    for (atom, row) in rows.into_iter().enumerate() {
        out.row_mut(atom).assign(&row);
    }
    out
}

/// Project each movable row onto the tangent space at its (unit) decoder row, and
/// zero the rest.
fn project_tangent(values: &mut Array2<f64>, unit: ArrayView2<'_, f64>, moves: &[bool]) {
    for (atom, mut row) in values.axis_iter_mut(Axis(0)).enumerate() {
        if !moves[atom] {
            row.fill(0.0);
            continue;
        }
        let along = row.dot(&unit.row(atom));
        row.scaled_add(-along, &unit.row(atom));
    }
}

/// Frobenius inner product, reduced per atom and then in atom order.
fn inner(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    let per_atom: Vec<f64> = left
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(right.axis_iter(Axis(0)).into_par_iter())
        .map(|(a, b)| a.dot(&b))
        .collect();
    per_atom.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic spread of values in `[-1, 1]`.
    fn spread(index: usize) -> f64 {
        ((index as f64 + 1.0) * 0.618_033_988_75).fract() * 2.0 - 1.0
    }

    fn unit_rows(mut rows: Array2<f64>) -> Array2<f64> {
        for mut row in rows.axis_iter_mut(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            row.mapv_inplace(|value| value / norm);
        }
        rows
    }

    /// `F(D) = Σ ‖x_i − A_iᵀc_i‖² + ρ‖c_i‖²` at fixed supports.
    fn profiled_loss(
        x: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f64>,
        codes: &[SparseCode],
        code_ridge: f64,
    ) -> f64 {
        codes
            .iter()
            .enumerate()
            .filter_map(|(row, code)| profile_row(x.row(row), decoder, code, code_ridge))
            .map(|profiled| {
                profiled.residual.dot(&profiled.residual)
                    + code_ridge * profiled.codes.dot(&profiled.codes)
            })
            .sum()
    }

    /// Rows drawn from three unit atoms in `R⁴`, each row firing one of the three
    /// atom pairs in turn, with a small part off every pair's span. No row fits
    /// exactly, so the profiled objective has a strict local minimum near the
    /// generating atoms, and the returned decoder is a perturbation of them inside
    /// its basin. Every code carries a zero-valued entry on the third atom, which
    /// the profile must drop from the support.
    fn fixture() -> (Array2<f32>, Array2<f64>, Vec<SparseCode>) {
        let n = 24;
        let truth = unit_rows(ndarray::array![
            [1.0, 0.2, 0.0, 0.1],
            [0.3, 1.0, 0.1, 0.0],
            [0.1, -0.2, 1.0, 0.2]
        ]);
        let (k, p) = truth.dim();
        let pairs = [[0u32, 1], [1, 2], [0, 2]];
        let mut x = Array2::<f32>::zeros((n, p));
        let mut codes = Vec::with_capacity(n);
        for row in 0..n {
            let pair = pairs[row % pairs.len()];
            let mut value = Array1::from_shape_fn(p, |entry| 0.02 * spread(7 * row + 2 + entry));
            for (slot, &atom) in pair.iter().enumerate() {
                let draw = spread(5 * row + slot);
                value.scaled_add(
                    draw.signum() * (0.5 + draw.abs()),
                    &truth.row(atom as usize),
                );
            }
            let idle = (0..k as u32)
                .find(|atom| !pair.contains(atom))
                .expect("three atoms, two fire");
            x.row_mut(row).assign(&value.mapv(|entry| entry as f32));
            codes.push(SparseCode {
                indices: vec![pair[0], pair[1], idle],
                codes: vec![1.0, 1.0, 0.0],
            });
        }
        let perturbed = unit_rows(
            &truth
                + &ndarray::array![
                    [0.0, 0.15, -0.1, 0.0],
                    [-0.2, 0.0, 0.1, 0.05],
                    [0.1, 0.05, 0.0, -0.1]
                ],
        );
        (x, perturbed, codes)
    }

    fn tangent_direction(decoder: &Array2<f64>, seed: usize) -> Array2<f64> {
        let (k, p) = decoder.dim();
        let mut direction =
            Array2::from_shape_fn((k, p), |(atom, entry)| spread(seed + atom * p + entry));
        project_tangent(&mut direction, decoder.view(), &vec![true; k]);
        direction
    }

    fn retract(decoder: &Array2<f64>, direction: &Array2<f64>, t: f64) -> Array2<f64> {
        unit_rows(decoder + &direction.mapv(|value| t * value))
    }

    #[test]
    fn profiled_gradient_matches_central_difference_3193() {
        let (x, decoder, codes) = fixture();
        let ridge = 0.05;
        let profiled = ProfiledDecoder::new(x.view(), decoder.clone(), &codes, ridge, &[true; 3]);
        let direction = tangent_direction(&decoder, 7);
        let t = 1e-5;
        let difference = (profiled_loss(
            x.view(),
            retract(&decoder, &direction, t).view(),
            &codes,
            ridge,
        ) - profiled_loss(
            x.view(),
            retract(&decoder, &direction, -t).view(),
            &codes,
            ridge,
        )) / (2.0 * t);
        let analytic = inner(&profiled.gradient, &direction);
        assert!(
            (difference - analytic).abs() <= 1e-6 * analytic.abs().max(1.0),
            "directional derivative {analytic:e} vs central difference {difference:e}"
        );
    }

    #[test]
    fn profiled_hessian_matches_second_difference_3193() {
        let (x, decoder, codes) = fixture();
        let ridge = 0.05;
        let profiled = ProfiledDecoder::new(x.view(), decoder.clone(), &codes, ridge, &[true; 3]);
        for seed in [3, 11, 29] {
            let direction = tangent_direction(&decoder, seed);
            let t = 1e-4;
            let center = profiled_loss(x.view(), decoder.view(), &codes, ridge);
            let difference = (profiled_loss(
                x.view(),
                retract(&decoder, &direction, t).view(),
                &codes,
                ridge,
            ) - 2.0 * center
                + profiled_loss(
                    x.view(),
                    retract(&decoder, &direction, -t).view(),
                    &codes,
                    ridge,
                ))
                / (t * t);
            let analytic = inner(&direction, &profiled.hessian_product(&direction));
            assert!(
                (difference - analytic).abs() <= 1e-4 * analytic.abs().max(1.0),
                "seed {seed}: curvature {analytic:e} vs second difference {difference:e}"
            );
            let other = tangent_direction(&decoder, seed + 101);
            let forward = inner(&other, &profiled.hessian_product(&direction));
            let backward = inner(&direction, &profiled.hessian_product(&other));
            assert!(
                (forward - backward).abs() <= 1e-10 * forward.abs().max(1.0),
                "seed {seed}: Hessian not symmetric, {forward:e} vs {backward:e}"
            );
        }
    }

    #[test]
    fn newton_step_descends_and_its_decrement_contracts_quadratically_3193() {
        let (x, decoder, codes) = fixture();
        let ridge = 0.05f32;
        let loss = |decoder: ArrayView2<'_, f32>| {
            profiled_loss(
                x.view(),
                decoder.mapv(f64::from).view(),
                &codes,
                f64::from(ridge),
            )
        };
        let start = decoder.mapv(|value| value as f32);
        let first = decoder_newton_step(x.view(), start.view(), &codes, ridge, &[true; 3], 1e-10);
        assert!(first.resolved, "first solve unresolved");
        let moved = first
            .candidate
            .expect("a nonzero decrement proposes a step");
        let second = decoder_newton_step(x.view(), moved.view(), &codes, ridge, &[true; 3], 1e-10);
        let third = decoder_newton_step(
            x.view(),
            second.candidate.as_ref().expect("second step").view(),
            &codes,
            ridge,
            &[true; 3],
            1e-10,
        );
        eprintln!(
            "loss {:e} -> {:e}; decrements {:e} {:e} {:e}",
            loss(start.view()),
            loss(moved.view()),
            first.decrement,
            second.decrement,
            third.decrement
        );
        assert!(second.resolved && third.resolved, "later solves unresolved");
        assert!(loss(moved.view()) < loss(start.view()));
        // Quadratic convergence makes the contraction ratio `λ_{j+1}/λ_j` itself
        // shrink along the iterates; a linear rate would hold it constant.
        assert!(second.decrement < first.decrement);
        assert!(
            third.decrement * first.decrement <= second.decrement * second.decrement,
            "decrements {:e} {:e} {:e} do not contract quadratically",
            first.decrement,
            second.decrement,
            third.decrement
        );
    }
}
