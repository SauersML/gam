//! The resolution of the estimated local latent law (gam#3610): its kernel
//! width `h` and the pooled share `ε` of every row's law,
//!
//! ```text
//! law(x) = (Σ_c u_c(x)·F_c + ε·F_pooled) / (Σ_c u_c(x) + ε),
//! u_c = K(d_c) − K(d_{top_k+1}),   K(d) = exp(−d²/2h²),
//! ```
//!
//! chosen from the data instead of fixed. They are the minimiser of the
//! cross-fitted continuous ranked probability score of the score itself,
//!
//! ```text
//! J(log h, log ε) = Σ_i w_i · CRPS(law_{−f(i)}(x_i), z_i),
//! CRPS(F, z) = E_F|X − z| − ½·E_F|X − X′|,
//! ```
//!
//! where `law_{−f(i)}` is the law built without row `i`'s fold: the held-out
//! laws the moving-law certificate already builds. CRPS is a strictly proper
//! score for a law of `z` whether the law is discrete or not, so `J` measures
//! how well the local law estimates the conditional law of `z` given the
//! covariates, the object it stands for. It is not the certificate's anchoring
//! loss, so the certificate still compares the arms on a loss nothing was tuned
//! on.
//!
//! A row's law is linear in its mixture weights, and which centres a row mixes
//! depends only on its distances, not on `(h, ε)`. So with `A_ij = E_{F_j}|X − z_i|`
//! and `B_jl = E|X_j − X_l|` computed once, `CRPS_i = πᵀA_i − ½·πᵀBπ` in the
//! row's normalised weights `π(log h, log ε)`, and `J` with its exact gradient and
//! Hessian costs `O(top_k²)` per row. `J` is minimised by Newton's method with a
//! halving line search. Where the Hessian is indefinite its eigenvalues enter by
//! magnitude, and along an eigenvector whose curvature is inside the Hessian's
//! own rounding band the step moves `log h` or `log ε` by one unit, a factor of
//! `e`. Every accepted step lowers `J` by more than `J`'s rounding band. `J` is
//! non-negative, so the iteration ends. It stops when the Newton decrement is
//! inside that band, or when the halved step's first-order change in `J` is
//! inside it before any halving lowered `J` by more than it.
//!
//! As `h → 0` or `h → ∞` every `u_c` vanishes, as it does when `ε → ∞`, and the
//! law is the pooled law. So the search has no box. When the conditional law
//! does not move it drifts toward that limit until the gain falls inside `J`'s
//! rounding band, and the local arm then scores as the pooled law it has become.

use super::estimated_latent_law::{HeldOutLocalLaw, LocalLawResolution};
use super::*;

/// `E|X − Y|` for independent `X ~ left` and `Y ~ right`:
/// `∫ F + G − 2·F·G dt` over the merged nodes, where the CDFs are constant
/// between nodes.
fn expected_absolute_difference(left: &EmpiricalZGrid, right: &EmpiricalZGrid) -> f64 {
    let (mut i, mut j) = (0, 0);
    let (mut f, mut g) = (0.0_f64, 0.0_f64);
    let mut previous: Option<f64> = None;
    let mut total = 0.0;
    loop {
        let next = match (left.nodes.get(i), right.nodes.get(j)) {
            (Some(&a), Some(&b)) => a.min(b),
            (Some(&a), None) => a,
            (None, Some(&b)) => b,
            (None, None) => break,
        };
        if let Some(previous) = previous {
            total += (next - previous) * (f + g - 2.0 * f * g);
        }
        while left.nodes.get(i) == Some(&next) {
            f += left.weights[i];
            i += 1;
        }
        while right.nodes.get(j) == Some(&next) {
            g += right.weights[j];
            j += 1;
        }
        previous = Some(next);
    }
    total
}

/// `E|X − z|` for `X ~ grid`.
fn expected_distance(grid: &EmpiricalZGrid, z: f64) -> f64 {
    grid.pairs()
        .map(|(node, weight)| weight * (node - z).abs())
        .sum()
}

/// One scored row: its weight, which held-out law it is read under, its
/// contexts' `(grid, squared distance, A)` in `entries[start..end]`, the squared
/// distance of the centre that truncates the mixture, and `A` of the pooled law.
struct ScoredRow {
    law: usize,
    weight: f64,
    start: usize,
    end: usize,
    truncation: Option<f64>,
    pooled_distance: f64,
}

/// `J`, its gradient and Hessian in `(log h, log ε)`, and the absolute sums
/// their rounding bands are read from.
#[derive(Clone, Copy, Default)]
struct Objective {
    value: f64,
    value_magnitude: f64,
    gradient: [f64; 2],
    hessian: [[f64; 2]; 2],
    hessian_magnitude: [[f64; 2]; 2],
}

impl std::ops::AddAssign for Objective {
    fn add_assign(&mut self, other: Self) {
        self.value += other.value;
        self.value_magnitude += other.value_magnitude;
        for a in 0..2 {
            self.gradient[a] += other.gradient[a];
            for b in 0..2 {
                self.hessian[a][b] += other.hessian[a][b];
                self.hessian_magnitude[a][b] += other.hessian_magnitude[a][b];
            }
        }
    }
}

/// A mixture component of one row at one resolution: its grid, its `A`, its
/// unnormalised weight `v` and `v`'s derivatives in `(log h, log ε)`.
/// `∂²v/∂log h ∂log ε = 0` for every component.
struct Component {
    grid: usize,
    distance: f64,
    v: f64,
    dv: [f64; 2],
    d2v: [f64; 2],
}

struct Selection<'a> {
    laws: Vec<&'a HeldOutLocalLaw>,
    /// Per law, `B` over its grids, row-major.
    differences: Vec<Vec<f64>>,
    rows: Vec<ScoredRow>,
    entries: Vec<(usize, f64, f64)>,
    /// The rounded operations on the longest accumulation path of `J`: the rows
    /// summed, one row's `A` over the longest grid, and its quadratic form.
    depth: usize,
}

impl Selection<'_> {
    fn difference(&self, law: usize, left: usize, right: usize) -> f64 {
        self.differences[law][left * self.laws[law].grids().len() + right]
    }

    /// The row's components at `(log h, log ε)`, in the order
    /// [`estimated_latent_law::mixture_from_neighbours`] forms them.
    fn components(&self, row: &ScoredRow, theta: [f64; 2]) -> Vec<Component> {
        let bandwidth = theta[0].exp();
        let bw2 = bandwidth * bandwidth;
        let floor = theta[1].exp();
        let kernel = |d2: f64| (-0.5 * d2 / bw2).exp();
        // K = e^{−a} with a = d²/2h², so ∂K/∂log h = 2a·K and
        // ∂²K/∂log h² = −4a(1 − a)·K.
        let slope = |d2: f64| {
            let a = 0.5 * d2 / bw2;
            let value = kernel(d2);
            (2.0 * a * value, -4.0 * a * (1.0 - a) * value)
        };
        let (truncation, truncation_slope) = row
            .truncation
            .map_or((0.0, (0.0, 0.0)), |d2| (kernel(d2), slope(d2)));
        let mut components = Vec::with_capacity(row.end - row.start + 1);
        for &(grid, d2, distance) in &self.entries[row.start..row.end] {
            let v = kernel(d2) - truncation;
            if v > 0.0 {
                let (first, second) = slope(d2);
                components.push(Component {
                    grid,
                    distance,
                    v,
                    dv: [first - truncation_slope.0, 0.0],
                    d2v: [second - truncation_slope.1, 0.0],
                });
            }
        }
        components.push(Component {
            grid: self.laws[row.law].grids().len() - 1,
            distance: row.pooled_distance,
            v: floor,
            dv: [0.0, floor],
            d2v: [0.0, floor],
        });
        components
    }

    /// One row's weighted `CRPS`, with its derivatives when `derivatives`.
    fn row_objective(&self, row: &ScoredRow, theta: [f64; 2], derivatives: bool) -> Objective {
        let components = self.components(row, theta);
        let m = components.len();
        let total: f64 = components.iter().map(|c| c.v).sum();
        let pi: Vec<f64> = components.iter().map(|c| c.v / total).collect();
        let b =
            |j: usize, l: usize| self.difference(row.law, components[j].grid, components[l].grid);
        let quadratic = |x: &[f64], y: &[f64], absolute: bool| {
            let mut sum = 0.0;
            for j in 0..m {
                for l in 0..m {
                    let term = x[j] * b(j, l) * y[l];
                    sum += if absolute { term.abs() } else { term };
                }
            }
            sum
        };
        let linear = |x: &[f64], absolute: bool| {
            components
                .iter()
                .zip(x)
                .map(|(c, &p)| {
                    if absolute {
                        (p * c.distance).abs()
                    } else {
                        p * c.distance
                    }
                })
                .sum::<f64>()
        };
        let w = row.weight;
        let mut objective = Objective {
            value: w * (linear(&pi, false) - 0.5 * quadratic(&pi, &pi, false)),
            value_magnitude: w * (linear(&pi, true) + 0.5 * quadratic(&pi, &pi, true)),
            ..Objective::default()
        };
        if !derivatives {
            return objective;
        }
        // π = v/V: ∂_a π = (∂_a v − π·∂_a V)/V and
        // ∂_ab π = (∂_ab v − ∂_a π·∂_b V − ∂_b π·∂_a V − π·∂_ab V)/V.
        let dtotal: [f64; 2] =
            std::array::from_fn(|a| components.iter().map(|c| c.dv[a]).sum::<f64>());
        let d2total: [f64; 2] =
            std::array::from_fn(|a| components.iter().map(|c| c.d2v[a]).sum::<f64>());
        let dpi: [Vec<f64>; 2] = std::array::from_fn(|a| {
            components
                .iter()
                .zip(&pi)
                .map(|(c, &p)| (c.dv[a] - p * dtotal[a]) / total)
                .collect()
        });
        let d2pi = |a: usize, b: usize| -> Vec<f64> {
            components
                .iter()
                .enumerate()
                .map(|(j, c)| {
                    let own = if a == b { c.d2v[a] } else { 0.0 };
                    let own_total = if a == b { d2total[a] } else { 0.0 };
                    (own - dpi[a][j] * dtotal[b] - dpi[b][j] * dtotal[a] - pi[j] * own_total)
                        / total
                })
                .collect()
        };
        for a in 0..2 {
            objective.gradient[a] = w * (linear(&dpi[a], false) - quadratic(&dpi[a], &pi, false));
            for b in a..2 {
                let second = d2pi(a, b);
                let value = linear(&second, false)
                    - quadratic(&second, &pi, false)
                    - quadratic(&dpi[a], &dpi[b], false);
                let magnitude = linear(&second, true)
                    + quadratic(&second, &pi, true)
                    + quadratic(&dpi[a], &dpi[b], true);
                objective.hessian[a][b] = w * value;
                objective.hessian[b][a] = w * value;
                objective.hessian_magnitude[a][b] = w * magnitude;
                objective.hessian_magnitude[b][a] = w * magnitude;
            }
        }
        objective
    }

    /// `J` at `theta`, summed in row order so it does not depend on the thread
    /// count.
    fn objective(&self, theta: [f64; 2], derivatives: bool) -> Objective {
        let per_row: Vec<Objective> = self
            .rows
            .par_iter()
            .map(|row| self.row_objective(row, theta, derivatives))
            .collect();
        let mut total = Objective::default();
        for row in per_row {
            total += row;
        }
        total
    }
}

impl<'a> Selection<'a> {
    /// `A` and `B` of every scored row of `laws`, each held-out law with the
    /// rows it is read at, in the order it keeps them.
    fn new(
        z: ArrayView1<'_, f64>,
        weights: &Array1<f64>,
        laws: &[(&'a HeldOutLocalLaw, &[usize])],
    ) -> Result<Self, String> {
        let mut rows = Vec::new();
        let mut entries = Vec::new();
        let mut longest_grid = 0usize;
        let mut widest_mixture = 0usize;
        for (index, &(law, law_rows)) in laws.iter().enumerate() {
            if law.rows() != law_rows.len() {
                return Err(format!(
                    "local law resolution: held-out law {index} keeps {} rows but {} are named",
                    law.rows(),
                    law_rows.len()
                ));
            }
            let grids = law.grids();
            longest_grid = grids
                .iter()
                .map(|g| g.nodes.len())
                .fold(longest_grid, usize::max);
            let pooled = &grids[grids.len() - 1];
            for (position, &row) in law_rows.iter().enumerate() {
                let neighbours = law.neighbours(position)?;
                let k = law.top_k().min(neighbours.len());
                widest_mixture = widest_mixture.max(k + 1);
                let start = entries.len();
                for &(grid, d2) in &neighbours[..k] {
                    entries.push((grid, d2, expected_distance(&grids[grid], z[row])));
                }
                rows.push(ScoredRow {
                    law: index,
                    weight: weights[row],
                    start,
                    end: entries.len(),
                    truncation: neighbours.get(k).map(|&(_, d2)| d2),
                    pooled_distance: expected_distance(pooled, z[row]),
                });
            }
        }
        if rows.is_empty() {
            return Err("local law resolution: no held-out row to score".to_string());
        }
        let differences = laws
            .par_iter()
            .map(|&(law, _)| {
                let grids = law.grids();
                let mut matrix = vec![0.0; grids.len() * grids.len()];
                for (j, left) in grids.iter().enumerate() {
                    for (l, right) in grids.iter().enumerate().skip(j) {
                        let value = expected_absolute_difference(left, right);
                        matrix[j * grids.len() + l] = value;
                        matrix[l * grids.len() + j] = value;
                    }
                }
                matrix
            })
            .collect();
        Ok(Selection {
            laws: laws.iter().map(|&(law, _)| law).collect(),
            differences,
            depth: rows.len() + longest_grid + widest_mixture * widest_mixture,
            rows,
            entries,
        })
    }
}

/// The Newton step on `objective`, with the eigenvalues of an indefinite Hessian
/// taken by magnitude and a unit step along an eigenvector whose curvature is
/// inside `curvature_band`; and the Newton decrement when the Hessian is
/// positive definite outside that band.
fn newton_step(objective: &Objective, curvature_band: f64) -> ([f64; 2], Option<f64>) {
    let [[p, q], [_, r]] = objective.hessian;
    let g = objective.gradient;
    let mean = 0.5 * (p + r);
    let radius = (0.5 * (p - r)).hypot(q);
    let angle = 0.5 * (2.0 * q).atan2(p - r);
    let eigen = [
        (mean + radius, [angle.cos(), angle.sin()]),
        (mean - radius, [-angle.sin(), angle.cos()]),
    ];
    let mut step = [0.0; 2];
    let mut decrement = Some(0.0);
    for (lambda, vector) in eigen {
        let component = vector[0] * g[0] + vector[1] * g[1];
        let length = if lambda.abs() > curvature_band {
            -component / lambda.abs()
        } else if component != 0.0 {
            -component.signum()
        } else {
            0.0
        };
        decrement = match decrement {
            Some(sum) if lambda > curvature_band => {
                Some(sum + 0.5 * component * component / lambda)
            }
            _ => None,
        };
        step[0] += length * vector[0];
        step[1] += length * vector[1];
    }
    (step, decrement)
}

/// The resolution minimising the cross-fitted CRPS of `z` under `laws`: each
/// held-out law with the rows it is read at, in the order it keeps them.
pub(crate) fn select_local_law_resolution(
    z: ArrayView1<'_, f64>,
    weights: &Array1<f64>,
    laws: &[(&HeldOutLocalLaw, &[usize])],
) -> Result<LocalLawResolution, String> {
    let selection = Selection::new(z, weights, laws)?;
    let growth = gam_linalg::roundoff::accumulation_growth(selection.depth);

    // Start at one standard deviation of every scaled covariate and a pooled
    // share equal to a context's kernel at its own centre, `K(0) = 1`.
    let mut theta = [0.0_f64; 2];
    let mut current = selection.objective(theta, true);
    if !current.value.is_finite() {
        return Err(format!(
            "local law resolution: the cross-fitted CRPS is not finite ({})",
            current.value
        ));
    }
    loop {
        let band = growth * current.value_magnitude;
        let m = current.hessian_magnitude;
        let curvature_band =
            growth * (m[0][0] * m[0][0] + m[1][1] * m[1][1] + 2.0 * m[0][1] * m[0][1]).sqrt();
        let (step, decrement) = newton_step(&current, curvature_band);
        if decrement.is_some_and(|decrement| decrement <= band) {
            break;
        }
        let slope = current.gradient[0] * step[0] + current.gradient[1] * step[1];
        let mut length = 1.0;
        let accepted = loop {
            let candidate = [theta[0] + length * step[0], theta[1] + length * step[1]];
            // A step whose first-order change is inside `J`'s rounding band
            // cannot lower `J` by more than it, and neither can a shorter one.
            if candidate == theta || (length * slope).abs() <= band {
                break None;
            }
            let value = selection.objective(candidate, false).value;
            if value.is_finite() && value < current.value - band {
                break Some(candidate);
            }
            length *= 0.5;
        };
        match accepted {
            Some(next) => {
                theta = next;
                current = selection.objective(theta, true);
            }
            None => break,
        }
    }
    let resolution = LocalLawResolution {
        bandwidth: theta[0].exp(),
        floor: theta[1].exp(),
    };
    if !(resolution.bandwidth.is_finite()
        && resolution.bandwidth > 0.0
        && resolution.floor.is_finite()
        && resolution.floor > 0.0)
    {
        return Err(format!(
            "local law resolution left the representable range: bandwidth {}, pooled share {}",
            resolution.bandwidth, resolution.floor
        ));
    }
    Ok(resolution)
}

#[cfg(test)]
mod tests {
    use super::super::estimated_latent_law::{
        LocalLawContext, local_law_parts, mixture_from_neighbours,
    };
    use super::*;

    /// SplitMix64 uniforms on `(0, 1)`.
    fn uniforms(seed: u64, count: usize) -> Vec<f64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut x = state;
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                x ^= x >> 31;
                ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            })
            .collect()
    }

    /// Standard normals by Box–Muller.
    fn normals(seed: u64, count: usize) -> Vec<f64> {
        let u = uniforms(seed, 2 * count);
        (0..count)
            .map(|i| (-2.0 * u[2 * i].ln()).sqrt() * (std::f64::consts::TAU * u[2 * i + 1]).cos())
            .collect()
    }

    /// A score over one covariate, its three-fold held-out local laws, and each
    /// fold's rows.
    struct Fixture {
        z: Array1<f64>,
        weights: Array1<f64>,
        laws: Vec<HeldOutLocalLaw>,
        folds: Vec<Vec<usize>>,
    }

    impl Fixture {
        fn new(n: usize, slope: f64) -> Self {
            let x = uniforms(3610, n);
            let noise = normals(0x3610, n);
            let z = Array1::from_iter(x.iter().zip(&noise).map(|(&x, &e)| slope * x + e));
            let weights = Array1::from_elem(n, 1.0);
            let features = Array2::from_shape_vec((n, 1), x).expect("features");
            let context = LocalLawContext {
                features: features.view(),
                feature_cols: vec![0],
            };
            let grid_size = 9;
            let contexts = local_law_parts(&z, &weights, &context, grid_size, None)
                .expect("full-data local law")
                .contexts();
            let folds: Vec<Vec<usize>> = (0..3).map(|f| (f..n).step_by(3).collect()).collect();
            let laws = folds
                .iter()
                .map(|rows| {
                    let mut fold_weights = weights.clone();
                    for &row in rows {
                        fold_weights[row] = 0.0;
                    }
                    local_law_parts(&z, &fold_weights, &context, grid_size, Some(contexts))
                        .and_then(|parts| parts.held_out(rows))
                        .expect("held-out local law")
                })
                .collect();
            Self {
                z,
                weights,
                laws,
                folds,
            }
        }

        fn held_out(&self) -> Vec<(&HeldOutLocalLaw, &[usize])> {
            self.laws
                .iter()
                .zip(&self.folds)
                .map(|(law, rows)| (law, rows.as_slice()))
                .collect()
        }
    }

    fn theta(resolution: LocalLawResolution) -> [f64; 2] {
        [resolution.bandwidth.ln(), resolution.floor.ln()]
    }

    /// `J` of the pooled law alone, the limit of every rail.
    fn pooled_objective(selection: &Selection<'_>) -> f64 {
        selection
            .rows
            .iter()
            .map(|row| {
                let pooled = selection.laws[row.law].grids().len() - 1;
                row.weight
                    * (row.pooled_distance - 0.5 * selection.difference(row.law, pooled, pooled))
            })
            .sum()
    }

    #[test]
    fn expected_absolute_difference_is_the_double_sum_3610() {
        let left =
            EmpiricalZGrid::new(vec![-1.0, 0.25, 0.5, 2.0], vec![0.1, 0.4, 0.3, 0.2], "left")
                .expect("grid");
        let right = EmpiricalZGrid::new(vec![-2.0, 0.25, 3.0], vec![0.5, 0.25, 0.25], "right")
            .expect("grid");
        for (a, b) in [
            (&left, &right),
            (&right, &left),
            (&left, &left),
            (&right, &right),
        ] {
            let direct: f64 = a
                .pairs()
                .flat_map(|(x, v)| b.pairs().map(move |(y, w)| v * w * (x - y).abs()))
                .sum();
            let merged = expected_absolute_difference(a, b);
            let band = gam_linalg::roundoff::accumulation_growth(a.nodes.len() * b.nodes.len())
                * a.pairs()
                    .flat_map(|(x, v)| b.pairs().map(move |(y, w)| v * w * (x.abs() + y.abs())))
                    .sum::<f64>();
            assert!(
                (merged - direct).abs() <= band,
                "merged-CDF E|X - Y| = {merged}, double sum = {direct}"
            );
        }
    }

    /// The CRPS a row is scored by is the CRPS of the law the fit mixes for it.
    #[test]
    fn row_crps_is_the_crps_of_the_mixed_law_3610() {
        let fixture = Fixture::new(600, 3.0);
        let held_out = fixture.held_out();
        let selection =
            Selection::new(fixture.z.view(), &fixture.weights, &held_out).expect("selection");
        let mut index = 0;
        for (law, rows) in &held_out {
            for (position, &row) in rows.iter().enumerate() {
                for resolution in [
                    LocalLawResolution {
                        bandwidth: 0.3,
                        floor: 0.05,
                    },
                    LocalLawResolution {
                        bandwidth: 2.0,
                        floor: 1.5,
                    },
                ] {
                    let grid = law.grid(position, resolution).expect("row law");
                    let direct = expected_distance(&grid, fixture.z[row])
                        - 0.5 * expected_absolute_difference(&grid, &grid);
                    let scored =
                        selection.row_objective(&selection.rows[index], theta(resolution), false);
                    let band = gam_linalg::roundoff::accumulation_growth(
                        selection.depth + grid.nodes.len() * grid.nodes.len(),
                    ) * 2.0
                        * scored.value_magnitude;
                    assert!(
                        (scored.value - direct).abs() <= band,
                        "row {row}: scored CRPS {} vs the mixed law's {direct}",
                        scored.value
                    );
                }
                index += 1;
            }
        }
    }

    #[test]
    fn objective_derivatives_match_central_differences_3610() {
        let fixture = Fixture::new(600, 3.0);
        let held_out = fixture.held_out();
        let selection =
            Selection::new(fixture.z.view(), &fixture.weights, &held_out).expect("selection");
        let step = 1.0e-4;
        for at in [[-0.7, -1.5], [0.4, 0.8], [-2.0, 0.0]] {
            let exact = selection.objective(at, true);
            let shifted = |a: usize, sign: f64| {
                let mut point = at;
                point[a] += sign * step;
                selection.objective(point, true)
            };
            for a in 0..2 {
                let (up, down) = (shifted(a, 1.0), shifted(a, -1.0));
                let gradient = (up.value - down.value) / (2.0 * step);
                assert!(
                    (gradient - exact.gradient[a]).abs() <= 1.0e-6 * exact.value_magnitude,
                    "at {at:?}: dJ/dθ{a} = {} but central difference {gradient}",
                    exact.gradient[a]
                );
                for b in 0..2 {
                    let hessian = (up.gradient[b] - down.gradient[b]) / (2.0 * step);
                    assert!(
                        (hessian - exact.hessian[a][b]).abs() <= 1.0e-6 * exact.value_magnitude,
                        "at {at:?}: d²J/dθ{a}dθ{b} = {} but central difference {hessian}",
                        exact.hessian[a][b]
                    );
                }
            }
        }
    }

    #[test]
    fn a_moving_law_is_resolved_locally_3610() {
        let fixture = Fixture::new(1200, 4.0);
        let held_out = fixture.held_out();
        let resolution = select_local_law_resolution(fixture.z.view(), &fixture.weights, &held_out)
            .expect("resolution");
        let selection =
            Selection::new(fixture.z.view(), &fixture.weights, &held_out).expect("selection");
        let chosen = selection.objective(theta(resolution), false).value;
        let pooled = pooled_objective(&selection);
        eprintln!("moving: {resolution:?}, J = {chosen}, pooled J = {pooled}");
        assert!(
            chosen < 0.75 * pooled,
            "the local law ({resolution:?}) scores {chosen}, barely under the pooled law's {pooled}"
        );
        for a in 0..2 {
            for sign in [1.0, -1.0] {
                let mut nearby = theta(resolution);
                nearby[a] += sign * 0.1;
                let value = selection.objective(nearby, false).value;
                assert!(
                    chosen <= value,
                    "{resolution:?} is not a minimum: J = {chosen} but {value} at {nearby:?}"
                );
            }
        }
    }

    #[test]
    fn a_law_that_does_not_move_drifts_to_the_pooled_law_3610() {
        let fixture = Fixture::new(1200, 0.0);
        let held_out = fixture.held_out();
        let resolution = select_local_law_resolution(fixture.z.view(), &fixture.weights, &held_out)
            .expect("resolution");
        let selection =
            Selection::new(fixture.z.view(), &fixture.weights, &held_out).expect("selection");
        let chosen = selection.objective(theta(resolution), false).value;
        let pooled = pooled_objective(&selection);
        let mut local_share = 0.0;
        let mut rows = 0.0;
        for (law, law_rows) in &held_out {
            for position in 0..law_rows.len() {
                let mixture = mixture_from_neighbours(
                    law.neighbours(position).expect("neighbours"),
                    law.grids().len() - 1,
                    law.top_k(),
                    resolution.bandwidth,
                    LocalLawMixture::VanishingAtTruncation {
                        floor: resolution.floor,
                    },
                )
                .expect("mixture");
                local_share += mixture
                    .iter()
                    .filter(|&&(grid, _)| grid != law.grids().len() - 1)
                    .map(|&(_, weight)| weight)
                    .sum::<f64>();
                rows += 1.0;
            }
        }
        local_share /= rows;
        eprintln!(
            "unmoving: {resolution:?}, J = {chosen}, pooled J = {pooled}, local share {local_share}"
        );
        assert!(
            chosen <= pooled * (1.0 + 1.0e-3),
            "the local law ({resolution:?}) scores {chosen}, over the pooled law's {pooled}"
        );
        assert!(
            local_share < 0.1,
            "a law that does not move keeps a local share of {local_share} at {resolution:?}"
        );
    }
}
