use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
};
use crate::custom_family::LinearInequalityConstraints;
use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_ab, fast_atb, rrqr_nullspace_basis};
use crate::families::gamlss::{ParameterBlockInput, initializewiggle_knots_from_seed};
use crate::matrix::DesignMatrix;
use ndarray::{Array1, Array2};

#[derive(Clone, Debug)]
pub struct AnchoredDeviationBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub derivative_grid_size: usize,
    pub monotonicity_eps: f64,
}

impl Default for AnchoredDeviationBlockConfig {
    fn default() -> Self {
        Self {
            degree: 3,
            num_internal_knots: 8,
            penalty_order: 2,
            penalty_orders: vec![1, 2],
            double_penalty: true,
            derivative_grid_size: 64,
            monotonicity_eps: 1e-4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnchoredDeviationRuntime {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub transform: Array2<f64>,
}

#[derive(Clone)]
pub struct AnchoredDeviationPrepared {
    pub block: ParameterBlockInput,
    pub runtime: AnchoredDeviationRuntime,
    pub constraints: LinearInequalityConstraints,
}

impl AnchoredDeviationRuntime {
    fn constrained_basis(
        &self,
        values: &Array1<f64>,
        basis_options: BasisOptions,
    ) -> Result<Array2<f64>, String> {
        let (basis, _) = create_basis::<Dense>(
            values.view(),
            KnotSource::Provided(self.knots.view()),
            self.degree,
            basis_options,
        )
        .map_err(|e| e.to_string())?;
        let full = basis.as_ref().clone();
        if full.ncols() != self.transform.nrows() {
            return Err(format!(
                "anchored deviation basis/transform mismatch: basis has {} columns but transform has {} rows",
                full.ncols(),
                self.transform.nrows()
            ));
        }
        Ok(full.dot(&self.transform))
    }

    pub fn design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::value())
    }

    pub fn first_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::first_derivative())
    }

    pub fn second_derivative_design(&self, values: &Array1<f64>) -> Result<Array2<f64>, String> {
        self.constrained_basis(values, BasisOptions::second_derivative())
    }

    pub fn apply(&self, values: &Array1<f64>, beta: &Array1<f64>) -> Result<Array1<f64>, String> {
        let design = self.design(values)?;
        if design.ncols() != beta.len() {
            return Err(format!(
                "anchored deviation apply mismatch: design has {} columns but beta has {} entries",
                design.ncols(),
                beta.len()
            ));
        }
        Ok(design.dot(beta))
    }

    pub fn derivative(
        &self,
        values: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let design = self.first_derivative_design(values)?;
        if design.ncols() != beta.len() {
            return Err(format!(
                "anchored deviation derivative mismatch: design has {} columns but beta has {} entries",
                design.ncols(),
                beta.len()
            ));
        }
        Ok(design.dot(beta))
    }

    pub fn second_derivative(
        &self,
        values: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let design = self.second_derivative_design(values)?;
        if design.ncols() != beta.len() {
            return Err(format!(
                "anchored deviation second-derivative mismatch: design has {} columns but beta has {} entries",
                design.ncols(),
                beta.len()
            ));
        }
        Ok(design.dot(beta))
    }
}

fn derivative_grid_from_knots(
    knots: &Array1<f64>,
    degree: usize,
    grid_size: usize,
) -> Result<Array1<f64>, String> {
    if knots.len() < degree + 2 {
        return Err(format!(
            "anchored deviation derivative grid needs at least {} knots for degree {}, got {}",
            degree + 2,
            degree,
            knots.len()
        ));
    }
    let left = knots[degree];
    let right = knots[knots.len() - degree - 1];
    if !left.is_finite() || !right.is_finite() || right < left {
        return Err(format!(
            "anchored deviation invalid knot domain [{left}, {right}]"
        ));
    }
    let n = grid_size.max(2);
    if (right - left).abs() < 1e-12 {
        return Ok(Array1::from_vec(vec![left; n]));
    }
    Ok(Array1::from_iter((0..n).map(|i| {
        let t = i as f64 / ((n - 1) as f64);
        left + (right - left) * t
    })))
}

fn homogeneous_anchor_transform(
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, String> {
    let anchor = Array1::from_vec(vec![0.0]);
    let (value_basis, _) = create_basis::<Dense>(
        anchor.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| e.to_string())?;
    let (d1_basis, _) = create_basis::<Dense>(
        anchor.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .map_err(|e| e.to_string())?;
    let k = value_basis.ncols();
    if d1_basis.ncols() != k {
        return Err(format!(
            "anchored deviation anchor derivative width mismatch: value has {} columns, derivative has {}",
            k,
            d1_basis.ncols()
        ));
    }
    let mut c = Array2::<f64>::zeros((2, k));
    c.row_mut(0).assign(&value_basis.row(0));
    c.row_mut(1).assign(&d1_basis.row(0));
    let (z, rank) = rrqr_nullspace_basis(&c.t(), default_rrqr_rank_alpha())
        .map_err(|e| format!("anchored deviation RRQR failed: {e}"))?;
    if rank >= k || z.ncols() == 0 {
        return Err(
            "anchored deviation anchor constraints removed all columns; increase basis richness"
                .to_string(),
        );
    }
    Ok(z)
}

pub fn build_anchored_deviation_block_from_seed(
    seed: &Array1<f64>,
    cfg: &AnchoredDeviationBlockConfig,
) -> Result<AnchoredDeviationPrepared, String> {
    let knots = initializewiggle_knots_from_seed(
        seed.view(),
        cfg.degree,
        cfg.num_internal_knots,
    )?;
    let runtime = AnchoredDeviationRuntime {
        knots: knots.clone(),
        degree: cfg.degree,
        transform: homogeneous_anchor_transform(&knots, cfg.degree)?,
    };

    let design = runtime.design(seed)?;
    let raw_dim = runtime.transform.nrows();
    let dim = runtime.transform.ncols();
    if dim == 0 {
        return Err("anchored deviation transform has zero columns".to_string());
    }

    let mut penalties = Vec::new();
    let base_penalty = create_difference_penalty_matrix(raw_dim, cfg.penalty_order, None)
        .map_err(|e| e.to_string())?;
    penalties.push(fast_ab(
        &fast_atb(&runtime.transform, &base_penalty),
        &runtime.transform,
    ));
    for &order in &cfg.penalty_orders {
        if order == cfg.penalty_order || order == 0 || order >= raw_dim {
            continue;
        }
        let raw = create_difference_penalty_matrix(raw_dim, order, None).map_err(|e| e.to_string())?;
        penalties.push(fast_ab(
            &fast_atb(&runtime.transform, &raw),
            &runtime.transform,
        ));
    }
    if cfg.double_penalty {
        penalties.push(Array2::<f64>::eye(dim));
    }

    let derivative_grid = derivative_grid_from_knots(&knots, cfg.degree, cfg.derivative_grid_size)?;
    let derivative_design = runtime.first_derivative_design(&derivative_grid)?;
    let constraints = LinearInequalityConstraints {
        a: derivative_design,
        b: Array1::from_elem(derivative_grid.len(), cfg.monotonicity_eps - 1.0),
    };

    Ok(AnchoredDeviationPrepared {
        block: ParameterBlockInput {
            design: DesignMatrix::Dense(design),
            offset: Array1::zeros(seed.len()),
            penalties,
            initial_log_lambdas: None,
            initial_beta: Some(Array1::zeros(dim)),
        },
        runtime,
        constraints,
    })
}
