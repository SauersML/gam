use crate::faer_ndarray::{FaerCholesky, FaerEigh};
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

const DEFAULT_MAX_ITER: usize = 30;
const DEFAULT_TOP_K: usize = 1;
const DEFAULT_TEMPERATURE: f64 = 0.25;
const DEFAULT_CODE_RIDGE: f64 = 1.0e-8;
const DEFAULT_TOLERANCE: f64 = 1.0e-7;
const INACTIVE_LAMBDA: f64 = 1.0e30;
const MIN_NORM2: f64 = 1.0e-24;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearDictionaryAssignment {
    TopK,
    Softmax,
}

impl LinearDictionaryAssignment {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "top_k" | "topk" | "hard" => Ok(Self::TopK),
            "softmax" | "soft" => Ok(Self::Softmax),
            other => Err(format!(
                "linear dictionary assignment must be 'top_k' or 'softmax'; got {other:?}"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopK => "top_k",
            Self::Softmax => "softmax",
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearDictionaryConfig {
    pub n_atoms: usize,
    pub max_iter: usize,
    pub top_k: usize,
    pub assignment: LinearDictionaryAssignment,
    pub temperature: f64,
    pub code_ridge: f64,
    pub tolerance: f64,
}

impl LinearDictionaryConfig {
    pub fn new(n_atoms: usize) -> Self {
        Self {
            n_atoms,
            ..Self::default()
        }
    }
}

impl Default for LinearDictionaryConfig {
    fn default() -> Self {
        Self {
            n_atoms: 1,
            max_iter: DEFAULT_MAX_ITER,
            top_k: DEFAULT_TOP_K,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearDictionaryFit {
    pub atoms: Array2<f64>,
    pub assignments: Array2<f64>,
    pub fitted: Array2<f64>,
    pub lambdas: Array1<f64>,
    pub reml_scores: Array1<f64>,
    pub explained_variance: f64,
    pub iterations: usize,
    pub converged: bool,
    pub assignment: LinearDictionaryAssignment,
    pub top_k: usize,
}

pub fn fit_linear_dictionary(
    x: ArrayView2<'_, f64>,
    config: &LinearDictionaryConfig,
) -> Result<LinearDictionaryFit, String> {
    validate_inputs(x, config)?;
    if config.n_atoms == 1 {
        return fit_rank_one_pca_lane(x, config);
    }

    let top_k = config.top_k.min(config.n_atoms).max(1);
    let mut atoms = initialize_atoms(x, config.n_atoms);
    let mut assignments = Array2::<f64>::zeros((x.nrows(), config.n_atoms));
    let mut fitted = Array2::<f64>::zeros(x.dim());
    let mut lambdas = Array1::<f64>::from_elem(config.n_atoms, INACTIVE_LAMBDA);
    let mut reml_scores = Array1::<f64>::zeros(config.n_atoms);
    let mut previous_ev = f64::NEG_INFINITY;
    let mut converged = false;
    let mut completed_iterations = 0usize;

    for iter in 0..config.max_iter {
        assignments = match config.assignment {
            LinearDictionaryAssignment::TopK => {
                top_k_assignments(x, atoms.view(), top_k, config.code_ridge)?
            }
            LinearDictionaryAssignment::Softmax => softmax_assignments(
                x,
                atoms.view(),
                top_k,
                config.temperature,
                config.code_ridge,
            )?,
        };

        fitted = assignments.dot(&atoms);
        let mut any_reseeded = false;
        for atom_idx in 0..config.n_atoms {
            any_reseeded |= fit_one_atom_penalized_ls(
                x,
                &mut atoms,
                &mut assignments,
                &mut fitted,
                &mut lambdas,
                &mut reml_scores,
                atom_idx,
                config.code_ridge,
            )?;
        }

        completed_iterations = iter + 1;
        let ev = explained_variance(x, fitted.view());
        // #1500: never declare convergence on an iteration that re-seeded a dead
        // atom — its revived direction carries no code yet, so EV is momentarily
        // flat; one more sweep lets the assignment step route rows to it.
        if !any_reseeded && (ev - previous_ev).abs() <= config.tolerance.max(0.0) {
            converged = true;
            break;
        }
        previous_ev = ev;
    }

    let final_ev = explained_variance(x, fitted.view());
    Ok(LinearDictionaryFit {
        atoms,
        assignments,
        fitted,
        lambdas,
        reml_scores,
        explained_variance: final_ev,
        iterations: completed_iterations,
        converged,
        assignment: config.assignment,
        top_k,
    })
}

fn validate_inputs(x: ArrayView2<'_, f64>, config: &LinearDictionaryConfig) -> Result<(), String> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("linear_dictionary_fit requires a non-empty 2-D matrix".to_string());
    }
    if !x.iter().all(|value| value.is_finite()) {
        return Err("linear_dictionary_fit input must be finite".to_string());
    }
    if config.n_atoms == 0 {
        return Err("linear_dictionary_fit requires K >= 1".to_string());
    }
    if config.max_iter == 0 {
        return Err("linear_dictionary_fit requires max_iter >= 1".to_string());
    }
    if config.top_k == 0 || config.top_k > config.n_atoms {
        return Err(format!(
            "linear_dictionary_fit top_k must be in [1, K={}]; got {}",
            config.n_atoms, config.top_k
        ));
    }
    if !(config.temperature.is_finite() && config.temperature > 0.0) {
        return Err(format!(
            "linear_dictionary_fit temperature must be finite and positive; got {}",
            config.temperature
        ));
    }
    if !(config.code_ridge.is_finite() && config.code_ridge > 0.0) {
        return Err(format!(
            "linear_dictionary_fit code_ridge must be finite and positive; got {}",
            config.code_ridge
        ));
    }
    if !config.tolerance.is_finite() {
        return Err("linear_dictionary_fit tolerance must be finite".to_string());
    }
    Ok(())
}

fn fit_rank_one_pca_lane(
    x: ArrayView2<'_, f64>,
    config: &LinearDictionaryConfig,
) -> Result<LinearDictionaryFit, String> {
    let covariance = x.t().dot(&x);
    let (evals, evecs) = covariance
        .eigh(Side::Lower)
        .map_err(|err| format!("linear_dictionary_fit PCA eigensolve failed: {err}"))?;
    let last = evals.len() - 1;
    let mut atom = evecs.column(last).to_owned();
    orient_vector(&mut atom);
    let mut assignments = Array2::<f64>::zeros((x.nrows(), 1));
    for row in 0..x.nrows() {
        assignments[[row, 0]] = x.row(row).dot(&atom) / (1.0 + config.code_ridge);
    }
    let mut atoms = atom.insert_axis(Axis(0)).to_owned();
    normalize_atom_and_assignments(&mut atoms, &mut assignments, 0);
    let fitted = assignments.dot(&atoms);
    let score = penalized_reconstruction_loss(x, fitted.view(), config.code_ridge, atoms.view());
    Ok(LinearDictionaryFit {
        atoms,
        assignments,
        fitted: fitted.clone(),
        lambdas: Array1::from_elem(1, config.code_ridge),
        reml_scores: Array1::from_elem(1, score),
        explained_variance: explained_variance(x, fitted.view()),
        iterations: 1.min(config.max_iter),
        converged: true,
        assignment: config.assignment,
        top_k: 1,
    })
}

fn initialize_atoms(x: ArrayView2<'_, f64>, n_atoms: usize) -> Array2<f64> {
    let mut atoms = Array2::<f64>::zeros((n_atoms, x.ncols()));
    let first = max_norm_row(x);
    atoms.row_mut(0).assign(&x.row(first));
    normalize_row(atoms.slice_mut(s![0, ..]));
    let mut min_dist2 = Array1::<f64>::from_elem(x.nrows(), f64::INFINITY);

    for atom_idx in 1..n_atoms {
        let prev = atoms.row(atom_idx - 1);
        for row in 0..x.nrows() {
            let dist2 = squared_distance(x.row(row), prev);
            if dist2 < min_dist2[row] {
                min_dist2[row] = dist2;
            }
        }
        let chosen = if atom_idx < x.nrows() {
            max_index(min_dist2.view())
        } else {
            atom_idx % x.nrows()
        };
        atoms.row_mut(atom_idx).assign(&x.row(chosen));
        normalize_row(atoms.slice_mut(s![atom_idx, ..]));
    }
    atoms
}

fn fit_one_atom_penalized_ls(
    x: ArrayView2<'_, f64>,
    atoms: &mut Array2<f64>,
    assignments: &mut Array2<f64>,
    fitted: &mut Array2<f64>,
    lambdas: &mut Array1<f64>,
    reml_scores: &mut Array1<f64>,
    atom_idx: usize,
    atom_ridge: f64,
) -> Result<bool, String> {
    let code = assignments.column(atom_idx).to_owned();
    let code_norm2 = code.dot(&code);
    if code_norm2 <= MIN_NORM2 {
        // #1500: this atom's cluster is EMPTY (no rows routed to it by the
        // assignment step). Zeroing it here made the atom permanently DEAD — a
        // zero atom has zero similarity to every row, so `top_k_assignments`
        // never routes anything back to it, the dictionary collapses to < K live
        // atoms, and it under-explains variance even when the data is exactly K
        // rank-1 atoms a K-atom dictionary could reconstruct perfectly. Instead
        // RE-SEED the atom into the worst-currently-reconstructed direction (the
        // standard k-means empty-cluster cure): point it at the largest-residual
        // row's UNEXPLAINED component so the next assignment sweep can route that
        // row's cluster to it and revive it. Returns `true` so the outer loop
        // suppresses convergence this iteration (the revived atom has no code
        // yet, so EV is momentarily flat — converging now would strand it).
        let mut worst_row = 0usize;
        let mut worst_res2 = -1.0_f64;
        for row in 0..x.nrows() {
            let mut res2 = 0.0_f64;
            for col in 0..x.ncols() {
                let d = x[[row, col]] - fitted[[row, col]];
                res2 += d * d;
            }
            if res2 > worst_res2 {
                worst_res2 = res2;
                worst_row = row;
            }
        }
        if worst_res2 <= MIN_NORM2 {
            // Every row is already fully reconstructed by the other atoms: there
            // is no unexplained direction to seed, so this atom is genuinely
            // redundant capacity. Leave it inactive (this is not the bug).
            atoms.row_mut(atom_idx).fill(0.0);
            lambdas[atom_idx] = INACTIVE_LAMBDA;
            reml_scores[atom_idx] = 0.0;
            return Ok(false);
        }
        for col in 0..x.ncols() {
            atoms[[atom_idx, col]] = x[[worst_row, col]] - fitted[[worst_row, col]];
        }
        normalize_row(atoms.slice_mut(s![atom_idx, ..]));
        lambdas[atom_idx] = atom_ridge;
        reml_scores[atom_idx] =
            penalized_reconstruction_loss(x, fitted.view(), atom_ridge, atoms.view());
        return Ok(true);
    }

    let old_atom = atoms.row(atom_idx).to_owned();
    let mut residual = x.to_owned() - fitted.view();
    residual += &code
        .view()
        .insert_axis(Axis(1))
        .dot(&old_atom.view().insert_axis(Axis(0)));

    let denominator = code_norm2 + atom_ridge;
    for col in 0..x.ncols() {
        atoms[[atom_idx, col]] = code.dot(&residual.column(col)) / denominator;
    }
    lambdas[atom_idx] = atom_ridge;
    normalize_atom_and_assignments(atoms, assignments, atom_idx);
    let updated_code = assignments.column(atom_idx).to_owned();
    fitted.assign(&x);
    *fitted -= &residual;
    *fitted += &updated_code
        .view()
        .insert_axis(Axis(1))
        .dot(&atoms.row(atom_idx).insert_axis(Axis(0)));
    reml_scores[atom_idx] =
        penalized_reconstruction_loss(x, fitted.view(), atom_ridge, atoms.view());
    Ok(false)
}

fn top_k_assignments(
    x: ArrayView2<'_, f64>,
    atoms: ArrayView2<'_, f64>,
    top_k: usize,
    code_ridge: f64,
) -> Result<Array2<f64>, String> {
    let cross = x.dot(&atoms.t());
    let mut assignments = Array2::<f64>::zeros((x.nrows(), atoms.nrows()));
    for row in 0..x.nrows() {
        let active = top_indices_by_abs(cross.row(row), top_k);
        let coeffs = solve_active_coefficients(atoms, cross.row(row), &active, code_ridge)?;
        for pos in 0..active.len() {
            assignments[[row, active[pos]]] = coeffs[pos];
        }
    }
    Ok(assignments)
}

fn softmax_assignments(
    x: ArrayView2<'_, f64>,
    atoms: ArrayView2<'_, f64>,
    top_k: usize,
    temperature: f64,
    code_ridge: f64,
) -> Result<Array2<f64>, String> {
    let cross = x.dot(&atoms.t());
    let atom_norm2 = atoms.map_axis(Axis(1), |row| row.dot(&row).max(MIN_NORM2));
    let mut assignments = Array2::<f64>::zeros((x.nrows(), atoms.nrows()));
    for row in 0..x.nrows() {
        let active = top_indices_by_abs(cross.row(row), top_k);
        let mut max_score = f64::NEG_INFINITY;
        for &atom_idx in &active {
            let score = cross[[row, atom_idx]].abs() / (atom_norm2[atom_idx].sqrt() * temperature);
            if score > max_score {
                max_score = score;
            }
        }
        let mut denom = 0.0;
        for &atom_idx in &active {
            let score = cross[[row, atom_idx]].abs() / (atom_norm2[atom_idx].sqrt() * temperature);
            let mass = (score - max_score).exp();
            assignments[[row, atom_idx]] = mass;
            denom += mass;
        }
        if denom <= 0.0 || !denom.is_finite() {
            return Err("linear_dictionary_fit softmax assignment underflowed".to_string());
        }
        for &atom_idx in &active {
            let projection = cross[[row, atom_idx]] / (atom_norm2[atom_idx] + code_ridge);
            assignments[[row, atom_idx]] = assignments[[row, atom_idx]] * projection / denom;
        }
    }
    Ok(assignments)
}

fn solve_active_coefficients(
    atoms: ArrayView2<'_, f64>,
    cross_row: ArrayView1<'_, f64>,
    active: &[usize],
    code_ridge: f64,
) -> Result<Array1<f64>, String> {
    let m = active.len();
    let mut system = Array2::<f64>::zeros((m, m));
    let mut rhs = Array2::<f64>::zeros((m, 1));
    for i in 0..m {
        rhs[[i, 0]] = cross_row[active[i]];
        for j in 0..m {
            system[[i, j]] = atoms.row(active[i]).dot(&atoms.row(active[j]));
        }
        system[[i, i]] += code_ridge;
    }
    let factor = system
        .cholesky(Side::Lower)
        .map_err(|err| format!("linear_dictionary_fit sparse-code solve failed: {err}"))?;
    let mut solution = rhs;
    factor.solve_mat_in_place(&mut solution);
    Ok(solution.column(0).to_owned())
}

fn top_indices_by_abs(row: ArrayView1<'_, f64>, top_k: usize) -> Vec<usize> {
    let mut selected: Vec<(usize, f64)> = Vec::with_capacity(top_k);
    for idx in 0..row.len() {
        let score = row[idx].abs();
        if selected.len() < top_k {
            selected.push((idx, score));
            continue;
        }
        let mut worst_pos = 0usize;
        for pos in 1..selected.len() {
            if selected[pos].1 < selected[worst_pos].1
                || (selected[pos].1 == selected[worst_pos].1
                    && selected[pos].0 > selected[worst_pos].0)
            {
                worst_pos = pos;
            }
        }
        let worst = selected[worst_pos];
        if score > worst.1 || (score == worst.1 && idx < worst.0) {
            selected[worst_pos] = (idx, score);
        }
    }
    selected.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    selected.into_iter().map(|(idx, _)| idx).collect()
}

fn normalize_atom_and_assignments(
    atoms: &mut Array2<f64>,
    assignments: &mut Array2<f64>,
    atom_idx: usize,
) {
    let norm = atoms.row(atom_idx).dot(&atoms.row(atom_idx)).sqrt();
    if norm > MIN_NORM2.sqrt() {
        atoms.row_mut(atom_idx).mapv_inplace(|value| value / norm);
        assignments
            .column_mut(atom_idx)
            .mapv_inplace(|value| value * norm);
    }
    orient_atom_and_code(atoms, assignments, atom_idx);
}

fn orient_atom_and_code(atoms: &mut Array2<f64>, assignments: &mut Array2<f64>, atom_idx: usize) {
    let sign = first_nonzero_sign(atoms.row(atom_idx));
    if sign < 0.0 {
        atoms.row_mut(atom_idx).mapv_inplace(|value| -value);
        assignments
            .column_mut(atom_idx)
            .mapv_inplace(|value| -value);
    }
}

fn orient_vector(vector: &mut Array1<f64>) {
    if first_nonzero_sign(vector.view()) < 0.0 {
        vector.mapv_inplace(|value| -value);
    }
}

fn first_nonzero_sign(row: ndarray::ArrayView1<'_, f64>) -> f64 {
    for &value in row {
        if value.abs() > 1.0e-12 {
            return value.signum();
        }
    }
    1.0
}

fn normalize_row(mut row: ndarray::ArrayViewMut1<'_, f64>) {
    let norm = row.dot(&row).sqrt();
    if norm > MIN_NORM2.sqrt() {
        row.mapv_inplace(|value| value / norm);
    }
}

fn max_norm_row(x: ArrayView2<'_, f64>) -> usize {
    let mut best = 0usize;
    let mut best_norm = f64::NEG_INFINITY;
    for row in 0..x.nrows() {
        let norm = x.row(row).dot(&x.row(row));
        if norm > best_norm {
            best = row;
            best_norm = norm;
        }
    }
    best
}

fn max_index(values: ndarray::ArrayView1<'_, f64>) -> usize {
    let mut best = 0usize;
    let mut best_value = f64::NEG_INFINITY;
    for idx in 0..values.len() {
        if values[idx] > best_value {
            best = idx;
            best_value = values[idx];
        }
    }
    best
}

fn squared_distance(a: ndarray::ArrayView1<'_, f64>, b: ndarray::ArrayView1<'_, f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(av, bv)| {
            let diff = av - bv;
            diff * diff
        })
        .sum()
}

fn explained_variance(x: ArrayView2<'_, f64>, fitted: ArrayView2<'_, f64>) -> f64 {
    let mut rss = 0.0;
    for row in 0..x.nrows() {
        for col in 0..x.ncols() {
            let residual = x[[row, col]] - fitted[[row, col]];
            rss += residual * residual;
        }
    }
    let means = x.mean_axis(Axis(0)).expect("non-empty input has means");
    let mut tss = 0.0;
    for row in 0..x.nrows() {
        for col in 0..x.ncols() {
            let centered = x[[row, col]] - means[col];
            tss += centered * centered;
        }
    }
    if tss <= MIN_NORM2 {
        if rss <= MIN_NORM2 { 1.0 } else { 0.0 }
    } else {
        1.0 - rss / tss
    }
}

fn penalized_reconstruction_loss(
    x: ArrayView2<'_, f64>,
    fitted: ArrayView2<'_, f64>,
    ridge: f64,
    atoms: ArrayView2<'_, f64>,
) -> f64 {
    let mut loss = 0.0;
    for row in 0..x.nrows() {
        for col in 0..x.ncols() {
            let residual = x[[row, col]] - fitted[[row, col]];
            loss += residual * residual;
        }
    }
    loss + ridge * atoms.iter().map(|value| value * value).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};

    #[test]
    fn planted_sparse_linear_dictionary_reaches_high_explained_variance() {
        let truth = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut assignments = Array2::<f64>::zeros((160, 4));
        for row in 0..160 {
            let atom = row % 4;
            assignments[[row, atom]] = 0.7 + 0.01 * ((row / 4) as f64);
            assignments[[row, (atom + 1) % 4]] = 0.2;
        }
        let x = assignments.dot(&truth);
        let config = LinearDictionaryConfig {
            n_atoms: 4,
            max_iter: 40,
            top_k: 2,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: 1.0e-9,
        };

        let fit = fit_linear_dictionary(x.view(), &config).expect("linear dictionary fit");

        assert!(
            fit.explained_variance > 0.95,
            "expected EV > 0.95, got {}",
            fit.explained_variance
        );
    }

    #[test]
    fn single_atom_matches_penalized_pca_oracle() {
        let mut x = Array2::<f64>::zeros((80, 3));
        for row in 0..80 {
            let t = (row as f64 - 39.5) / 20.0;
            x[[row, 0]] = 2.0 * t;
            x[[row, 1]] = -t;
            x[[row, 2]] = 0.05 * (row as f64).sin();
        }
        let config = LinearDictionaryConfig {
            n_atoms: 1,
            max_iter: 5,
            top_k: 1,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: DEFAULT_TOLERANCE,
        };

        let fit = fit_linear_dictionary(x.view(), &config).expect("rank-one fit");
        let covariance = x.t().dot(&x);
        let (evals, _) = covariance.eigh(Side::Lower).expect("PCA eigensolve");
        let shrink = 1.0 / (1.0 + DEFAULT_CODE_RIDGE);
        let oracle_ev = 1.0
            - ((1.0 - shrink) * (1.0 - shrink) * evals[evals.len() - 1]
                + evals.slice(s![..evals.len() - 1]).sum())
                / evals.sum();

        assert!(fit.explained_variance > 0.99);
        assert_abs_diff_eq!(fit.explained_variance, oracle_ev, epsilon = 2.0e-4);
    }

    #[test]
    fn sparse_assignment_scales_to_thousand_atom_dictionary() {
        let active_atoms = array![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        let mut x = Array2::<f64>::zeros((256, 8));
        for row in 0..x.nrows() {
            let atom = row % active_atoms.nrows();
            let scale = 0.7 + 0.003 * row as f64;
            x.row_mut(row).assign(&(&active_atoms.row(atom) * scale));
        }
        let config = LinearDictionaryConfig {
            n_atoms: 1024,
            max_iter: 8,
            top_k: 1,
            assignment: LinearDictionaryAssignment::TopK,
            temperature: DEFAULT_TEMPERATURE,
            code_ridge: DEFAULT_CODE_RIDGE,
            tolerance: 1.0e-9,
        };

        let fit = fit_linear_dictionary(x.view(), &config).expect("large-K linear dictionary fit");
        let max_active = fit
            .assignments
            .axis_iter(Axis(0))
            .map(|row| row.iter().filter(|value| value.abs() > 1.0e-10).count())
            .max()
            .unwrap();

        assert_eq!(max_active, 1);
        assert!(
            fit.explained_variance > 0.95,
            "expected EV > 0.95 at K=1024, got {}",
            fit.explained_variance
        );
    }
}
