//! gam#3452 — the equal-mass grid's OWN sampling influence, checked against the
//! production builder.
//!
//! `node_zeta_vjp` carries rows moving through the grid. What gam#3452 found
//! missing is the other half: at the true first stage the grid is still built
//! from `n` draws, so its standardized nodes carry sampling error, and that error
//! is a function of the same rows the first stage's influence is.
//! `node_sampling_influence` returns it per row. Everything here is judged
//! against `build_empirical_z_grid_with_alpha` itself, never a second copy of
//! the formula.
//!
//! The acceptance gate is a Monte Carlo one. For a fixed adjoint `v` the
//! functional `T = Σ_b v_b x_b` of the standardized nodes has sampling variance
//! `Var T`, and the influence's claim is `Var T = E[Σ_i IF_i²]`. The test builds
//! the grid on many independent samples and requires the empirical variance of
//! `T` to sit inside its two-sided χ² band around the influence's variance. It
//! fails on an influence that is too small (the grid's error omitted, as before
//! gam#3452) exactly as on one too large.

#![cfg(test)]

use super::empirical_measure_sensitivity::build_empirical_z_grid_with_alpha;
use ndarray::{Array1, Array2};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit(state: &mut u64) -> f64 {
    ((splitmix(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn gauss(state: &mut u64) -> f64 {
    let u1 = unit(state);
    let u2 = unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The latent laws of the gam#3452 receipt: a Gaussian control and the
/// scale mixture (`σ² = 0.5286` w.p. 0.9, `5.243` w.p. 0.1, `κ₄ = 9`).
#[derive(Clone, Copy, Debug)]
enum Law {
    Gaussian,
    ScaleMixture,
}

impl Law {
    fn draw(self, state: &mut u64) -> f64 {
        match self {
            Law::Gaussian => gauss(state),
            Law::ScaleMixture => {
                let variance: f64 = if unit(state) < 0.9 { 0.5286 } else { 5.243 };
                variance.sqrt() * gauss(state)
            }
        }
    }
}

/// One sample of `n` rows with weights in `[0.5, 1.5)` and every seventh row
/// carrying zero weight, so the influence's weighting and its filter are both on
/// the path.
fn sample(law: Law, n: usize, state: &mut u64) -> (Array1<f64>, Array1<f64>) {
    let zeta = Array1::from_shape_fn(n, |_| law.draw(state));
    let weights = Array1::from_shape_fn(n, |i| if i % 7 == 3 { 0.0 } else { 0.5 + unit(state) });
    (zeta, weights)
}

const GRID_SIZE: usize = 65;

/// A fixed smooth adjoint on the sorted nodes. `M` projects out its constant
/// and linear parts; the curvature left over is what reaches the rows.
fn adjoint() -> Array2<f64> {
    Array2::from_shape_fn((GRID_SIZE, 1), |(b, _)| {
        let t = (b as f64 + 0.5) / GRID_SIZE as f64;
        (3.0 * t).cos() + 2.0 * (t - 0.5).powi(3)
    })
}

fn functional(nodes: &[f64], v: &Array2<f64>) -> f64 {
    nodes.iter().enumerate().map(|(b, &x)| v[[b, 0]] * x).sum()
}

#[test]
fn grid_sampling_influence_is_centered_and_zero_on_zero_weight_rows_3452() {
    let mut state = 0x3452_0000_0000_0001;
    let (zeta, weights) = sample(Law::ScaleMixture, 1_500, &mut state);
    let build = build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), GRID_SIZE, "test")
        .expect("grid builds");
    assert_eq!(build.grid.nodes.len(), GRID_SIZE);
    let v = Array2::from_shape_fn((GRID_SIZE, 3), |(b, j)| ((b * (j + 2)) as f64 * 0.37).sin());
    let influence = build.node_sampling_influence(v.view()).expect("influence");
    assert_eq!(influence.dim(), (zeta.len(), 3));
    for j in 0..3 {
        let column = influence.column(j);
        let total: f64 = column.sum();
        let scale = column.iter().map(|x| x.abs()).sum::<f64>();
        assert!(
            total.abs() <= 1.0e-12 * scale,
            "gam#3452: an influence function is centered under the empirical law, so the rows \
             must sum to zero; column {j} sums to {total:.3e} against Σ|IF| = {scale:.3e}"
        );
        for (i, &value) in column.iter().enumerate() {
            if weights[i] == 0.0 {
                assert_eq!(value, 0.0, "gam#3452: zero-weight row {i} carries influence {value}");
            }
        }
    }
    // Every row the grid saw moves it: a zero influence would be the pre-fix
    // state, where the grid's sampling error was absent from the correction.
    let norm = influence.iter().map(|x| x * x).sum::<f64>();
    assert!(norm > 0.0, "gam#3452: the grid's sampling influence came out identically zero");
}

#[test]
fn grid_sampling_influence_is_a_location_scale_invariant_3452() {
    // The standardized nodes do not move under ζ → a + bζ (b > 0), so neither
    // may their sampling influence: the quantile breaks, the trimmed means and
    // `sd` all rescale together and `M` absorbs the shift.
    let mut state = 0x3452_0000_0000_0002;
    let (zeta, weights) = sample(Law::ScaleMixture, 1_200, &mut state);
    let v = adjoint();
    let base = build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), GRID_SIZE, "test")
        .expect("grid builds")
        .node_sampling_influence(v.view())
        .expect("influence");
    let moved_zeta = zeta.mapv(|t| 0.7 + 2.5 * t);
    let moved =
        build_empirical_z_grid_with_alpha(moved_zeta.view(), weights.view(), GRID_SIZE, "test")
            .expect("grid builds")
            .node_sampling_influence(v.view())
            .expect("influence");
    let scale = base.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    for (i, (a, b)) in base.iter().zip(moved.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1.0e-10 * scale,
            "gam#3452: row {i}'s grid influence changed under an affine map of ζ: {a:.6e} vs {b:.6e}"
        );
    }
}

/// `R` independent samples: the empirical variance of `T` and the mean of the
/// influence's `Σ_i IF_i²`.
fn monte_carlo(law: Law, n: usize, replicates: usize, seed: u64) -> (f64, f64) {
    let v = adjoint();
    let mut state = seed;
    let mut values = Vec::with_capacity(replicates);
    let mut predicted = 0.0;
    for _ in 0..replicates {
        let (zeta, weights) = sample(law, n, &mut state);
        let build =
            build_empirical_z_grid_with_alpha(zeta.view(), weights.view(), GRID_SIZE, "mc")
                .expect("grid builds");
        assert_eq!(build.grid.nodes.len(), GRID_SIZE);
        values.push(functional(&build.grid.nodes, &v));
        let influence = build.node_sampling_influence(v.view()).expect("influence");
        predicted += influence.iter().map(|x| x * x).sum::<f64>();
    }
    let mean = values.iter().sum::<f64>() / replicates as f64;
    let variance =
        values.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (replicates - 1) as f64;
    (variance, predicted / replicates as f64)
}

/// Wilson–Hilferty quantile of `χ²_k / k` at standard-normal quantile `z`.
fn chi2_over_df(k: f64, z: f64) -> f64 {
    let c = 2.0 / (9.0 * k);
    (1.0 - c + z * c.sqrt()).powi(3)
}

#[test]
fn grid_sampling_influence_reproduces_the_grids_monte_carlo_variance_3452() {
    // Two arms, two-sided each; Bonferroni over the arms at α = 10⁻³ puts each
    // tail at 2.5·10⁻⁴, standard-normal quantile 3.4808.
    const Z_TAIL: f64 = 3.480_756;
    const REPLICATES: usize = 2_000;
    for (law, seed) in [(Law::Gaussian, 0x3452_0000_0000_00A1), (Law::ScaleMixture, 0x3452_0000_0000_00A2)] {
        let (empirical, predicted) = monte_carlo(law, 2_000, REPLICATES, seed);
        let k = (REPLICATES - 1) as f64;
        let low = predicted * chi2_over_df(k, -Z_TAIL);
        let high = predicted * chi2_over_df(k, Z_TAIL);
        eprintln!(
            "[3452 grid IF] {law:?}: Monte Carlo Var T = {empirical:.4e}, influence Σ IF² = \
             {predicted:.4e}, ratio {:.4}, band [{low:.4e}, {high:.4e}]",
            empirical / predicted
        );
        assert!(
            empirical >= low && empirical <= high,
            "gam#3452 ({law:?}): the grid's Monte Carlo variance {empirical:.4e} is outside the \
             two-sided χ² band [{low:.4e}, {high:.4e}] around the influence's {predicted:.4e}"
        );
    }
}
