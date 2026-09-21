//! Affine-Gaussian latent state law and the factorization of its conditional
//! posterior precision.
//!
//! Integration coordinates are the missing genetic scores `g` and the node
//! states `x_n`. At fixed coefficients the state law is linear-Gaussian and
//! affine in `g`:
//!
//! ```text
//! g   ~ N(mu, Lambda^-1)
//! x_0 = a_0 + B_0 g + L_0 e_0
//! x_n = Phi_n x_(n-1) + c_n + D_n g + L_n e_n,        e_n ~ N(0, I).
//! ```
//!
//! Node-local curvatures `W_n` of the observation factors add to its precision,
//! which is block tridiagonal in the states with a dense genetic border. The
//! factorization is a covariance-form forward filter with an information update
//! and a modified Bryson-Frazier backward pass, conditional on `g`, followed by
//! the genetic Schur complement. No step divides by an innovation covariance
//! `V_n = L_n L_n'`. Static signatures (`V_n = 0`), tied event times and nearly
//! static transitions are therefore representable. An explicit precision `V_n^-1`
//! would lose about `eps / V_n` of relative accuracy in its Schur chain.
//!
//! The predicted covariance is propagated as a square root: `C_n` comes from a
//! QR factorization of `[Phi_n F_(n-1), L_n]`, where `F` is the filtered root.
//! So a sharply identified static axis, whose covariance shrinks toward roundoff,
//! stays definite by construction instead of being refactored from a formed
//! `Phi P Phi'`.
//!
//! Work is `O(N (K^3 + K^2 G) + G^3)` and storage `O(N (K^2 + K G) + G^2)`.
//! Neither a dense trajectory covariance nor a state grid is formed.
//!
//! The determinant is taken in standardized coordinates `(g, e_0, ..., e_(N-1))`.
//! Their map to `(g, x)` is affine at fixed coefficients, so a Laplace
//! approximation is the same in either chart, and the path normalizer
//! `-1/2 log det V_n` is never formed.
use super::law::numerical;
use crate::EventHistoryError;
use crate::scalar::{div, exp, ln, recip, sqrt};
use gam_math::nested_dual::JetField;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

/// A dense lower triangular factor of a small symmetric positive-definite
/// block.
pub(super) struct Cholesky {
    lower: Array2<f64>,
    pub(super) log_determinant: f64,
    /// Sum of the magnitudes of the pivot logarithms summed into
    /// `log_determinant`: the scale its rounding is measured against.
    pub(super) log_determinant_magnitude: f64,
}

impl Cholesky {
    /// `None` when the block is not numerically positive definite.
    pub(super) fn new(matrix: &Array2<f64>) -> Option<Self> {
        let n = matrix.nrows();
        let mut lower = Array2::<f64>::zeros((n, n));
        let mut log_determinant = 0.0;
        let mut log_determinant_magnitude = 0.0;
        for i in 0..n {
            for j in 0..=i {
                let value =
                    matrix[[i, j]] - (0..j).map(|q| lower[[i, q]] * lower[[j, q]]).sum::<f64>();
                if i == j {
                    if !value.is_finite() || value <= 0.0 {
                        return None;
                    }
                    lower[[i, i]] = value.sqrt();
                    log_determinant += value.ln();
                    log_determinant_magnitude += value.ln().abs();
                } else {
                    lower[[i, j]] = value / lower[[j, j]];
                }
            }
        }
        Some(Self {
            lower,
            log_determinant,
            log_determinant_magnitude,
        })
    }

    /// The square root `C` with `C C' = M M'` of a `K x c` matrix `M`, from
    /// Householder reflections of `M'`. `M M'` is never formed. `None` when the
    /// root is singular.
    fn from_columns(matrix: &Array2<f64>) -> Option<Self> {
        let (k, columns) = matrix.dim();
        let mut reflected = matrix.t().to_owned();
        for j in 0..k.min(columns) {
            let norm = (j..columns).fold(0.0_f64, |acc, i| acc.hypot(reflected[[i, j]]));
            if norm == 0.0 {
                continue;
            }
            let alpha = if reflected[[j, j]] > 0.0 { -norm } else { norm };
            let mut vector: Vec<f64> = (j..columns).map(|i| reflected[[i, j]]).collect();
            vector[0] -= alpha;
            let length = vector.iter().fold(0.0_f64, |acc, v| acc.hypot(*v));
            if length == 0.0 {
                continue;
            }
            for v in &mut vector {
                *v /= length;
            }
            for c in j..k {
                let dot: f64 = (j..columns).map(|i| vector[i - j] * reflected[[i, c]]).sum();
                for i in j..columns {
                    reflected[[i, c]] -= 2.0 * dot * vector[i - j];
                }
            }
        }
        // C = R' with the signs of R's rows chosen so that C has a positive
        // diagonal; C D D C' = C C' for any diagonal D of signs.
        let mut lower = Array2::<f64>::zeros((k, k));
        let mut log_determinant = 0.0;
        for i in 0..k {
            let diagonal = if i < columns { reflected[[i, i]] } else { 0.0 };
            if !diagonal.is_finite() || diagonal == 0.0 {
                return None;
            }
            let sign = diagonal.signum();
            for j in i..k {
                if i < columns {
                    lower[[j, i]] = sign * reflected[[i, j]];
                }
            }
            log_determinant += 2.0 * diagonal.abs().ln();
        }
        let log_determinant_magnitude = (0..k)
            .map(|i| 2.0 * lower[[i, i]].ln().abs())
            .sum();
        Some(Self {
            lower,
            log_determinant,
            log_determinant_magnitude,
        })
    }

    /// `L^-1 rhs`, column by column.
    fn forward(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let n = self.lower.nrows();
        let mut out = rhs.clone();
        for col in 0..out.ncols() {
            for i in 0..n {
                let mut value = out[[i, col]];
                for j in 0..i {
                    value -= self.lower[[i, j]] * out[[j, col]];
                }
                out[[i, col]] = value / self.lower[[i, i]];
            }
        }
        out
    }

    /// `L^-T rhs`, column by column.
    fn backward(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let n = self.lower.nrows();
        let mut out = rhs.clone();
        for col in 0..out.ncols() {
            for i in (0..n).rev() {
                let mut value = out[[i, col]];
                for j in i + 1..n {
                    value -= self.lower[[j, i]] * out[[j, col]];
                }
                out[[i, col]] = value / self.lower[[i, i]];
            }
        }
        out
    }

    pub(super) fn solve_vector(&self, rhs: ArrayView1<'_, f64>) -> Array1<f64> {
        let column = rhs.to_owned().insert_axis(Axis(1));
        self.backward(&self.forward(&column)).remove_axis(Axis(1))
    }

    /// `L^-T z`: covariance `(L L')^-1` for standard normal `z`.
    fn whiten(&self, z: ArrayView1<'_, f64>) -> Array1<f64> {
        self.backward(&z.to_owned().insert_axis(Axis(1)))
            .remove_axis(Axis(1))
    }

    pub(super) fn inverse(&self) -> Array2<f64> {
        let mut inverse = self.backward(&self.forward(&Array2::eye(self.lower.nrows())));
        symmetrize(&mut inverse);
        inverse
    }
}

fn symmetrize(matrix: &mut Array2<f64>) {
    let n = matrix.nrows();
    for i in 0..n {
        for j in 0..i {
            let mean = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
            matrix[[i, j]] = mean;
            matrix[[j, i]] = mean;
        }
    }
}

/// The affine part of one node's state map: `c + D g + L e`.
#[derive(Clone, Debug)]
pub(super) struct Step {
    pub offset: Array1<f64>,
    /// `K x G` loading of the missing genetic scores.
    pub genes: Array2<f64>,
    /// `L`, with innovation covariance `L L'`. A zero column is a static axis.
    pub innovation: Array2<f64>,
}

#[derive(Clone, Debug)]
pub(super) struct StateLaw {
    pub gene_mean: Array1<f64>,
    pub gene_precision: Array2<f64>,
    pub entry: Step,
    /// `transitions[n - 1]` propagates node `n - 1` to node `n`.
    pub transitions: Vec<Array2<f64>>,
    pub steps: Vec<Step>,
}

impl StateLaw {
    pub fn nodes(&self) -> usize {
        self.steps.len() + 1
    }

    pub fn signatures(&self) -> usize {
        self.entry.offset.len()
    }

    pub fn genes(&self) -> usize {
        self.gene_mean.len()
    }

    fn step(&self, node: usize) -> &Step {
        match node.checked_sub(1) {
            Some(previous) => &self.steps[previous],
            None => &self.entry,
        }
    }

    pub fn validate(&self) -> Result<(), EventHistoryError> {
        let k = self.signatures();
        let g = self.genes();
        let step_valid = |step: &Step| {
            step.offset.len() == k
                && step.genes.dim() == (k, g)
                && step.innovation.dim() == (k, k)
                && step
                    .offset
                    .iter()
                    .chain(step.genes.iter())
                    .chain(step.innovation.iter())
                    .all(|v| v.is_finite())
        };
        if self.gene_precision.dim() != (g, g)
            || self
                .gene_mean
                .iter()
                .chain(self.gene_precision.iter())
                .any(|v| !v.is_finite())
            || self.transitions.len() != self.steps.len()
            || self
                .transitions
                .iter()
                .any(|phi| phi.dim() != (k, k) || phi.iter().any(|v| !v.is_finite()))
            || !step_valid(&self.entry)
            || !self.steps.iter().all(step_valid)
        {
            return Err(numerical(
                "joint state law has inconsistent dimensions or non-finite coefficients",
            ));
        }
        Ok(())
    }

    /// States reached from genetic scores and standardized innovations.
    /// `affine = false` drops every offset: the linear part of the map.
    pub fn transport(
        &self,
        genes: ArrayView1<'_, f64>,
        innovations: ArrayView2<'_, f64>,
        affine: bool,
    ) -> Array2<f64> {
        let mut states = Array2::<f64>::zeros((self.nodes(), self.signatures()));
        for n in 0..self.nodes() {
            let step = self.step(n);
            let mut state = step.genes.dot(&genes) + step.innovation.dot(&innovations.row(n));
            if affine {
                state += &step.offset;
            }
            if n > 0 {
                state += &self.transitions[n - 1].dot(&states.row(n - 1));
            }
            states.row_mut(n).assign(&state);
        }
        states
    }

    /// Transpose of the linear transport: node forces to genetic and
    /// innovation forces, by one reverse recursion.
    pub fn transport_adjoint(&self, forces: ArrayView2<'_, f64>) -> (Array1<f64>, Array2<f64>) {
        let mut genes = Array1::<f64>::zeros(self.genes());
        let mut innovations = Array2::<f64>::zeros(forces.dim());
        let mut carried = Array1::<f64>::zeros(self.signatures());
        for n in (0..self.nodes()).rev() {
            carried += &forces.row(n);
            let step = self.step(n);
            genes += &step.genes.t().dot(&carried);
            innovations
                .row_mut(n)
                .assign(&step.innovation.t().dot(&carried));
            if n > 0 {
                carried = self.transitions[n - 1].t().dot(&carried);
            }
        }
        (genes, innovations)
    }
}

struct Node {
    curvature: Array2<f64>,
    /// `P-`, the predicted covariance, formed from its root only for the lag
    /// blocks, and the root `C` itself.
    predicted: Array2<f64>,
    predicted_root: Cholesky,
    /// `F = C R^-T`, with filtered covariance `P = F F'`.
    spread: Array2<f64>,
    filtered: Array2<f64>,
    /// `T = I - P W`: the complement of the information gain.
    complement: Array2<f64>,
    genes_predicted: Array2<f64>,
    genes_filtered: Array2<f64>,
    smoothed_genes: Array2<f64>,
    adjoint_genes: Array2<f64>,
}

/// Selected blocks of the inverse precision, and the moments of the scaled
/// innovations `nu_n = V_n^-1 (x_n - E[x_n | x_(n-1), g])` that differentiate
/// the state law without dividing by `V_n`.
#[derive(Clone, Debug)]
pub(super) struct SelectedMoments {
    pub states: Vec<Array2<f64>>,
    /// `lags[n - 1] = Cov(x_n, x_(n-1))`.
    pub lags: Vec<Array2<f64>>,
    /// `Cov(x_n, g)`.
    pub state_genes: Vec<Array2<f64>>,
    pub genes: Array2<f64>,
    /// `Cov(nu_n) - V_n^-1`.
    pub innovation_excess: Vec<Array2<f64>>,
    /// `innovation_past[n - 1] = Cov(nu_n, x_(n-1))`.
    pub innovation_past: Vec<Array2<f64>>,
    /// `Cov(nu_n, g)`.
    pub innovation_genes: Vec<Array2<f64>>,
}

/// Posterior means of the Gaussian problem with node information `b_n`:
/// maximize `log p(g, x) + sum_n (b_n' x_n - x_n' W_n x_n / 2)`.
#[derive(Clone, Debug)]
pub(super) struct Solution {
    pub genes: Array1<f64>,
    pub states: Array2<f64>,
    /// `E[nu_n]`, negated: the modified Bryson-Frazier adjoint at node `n`.
    pub adjoints: Array2<f64>,
    /// `E[e_n] = -L_n' adjoint_n`, standardized innovations of the mean path.
    pub innovations: Array2<f64>,
}

pub(super) struct Factorization {
    nodes: Vec<Node>,
    gene_factor: Cholesky,
    /// `log det` of the posterior precision in `(g, e)` coordinates.
    pub log_determinant: f64,
    /// Sum of the magnitudes of every pivot logarithm in `log_determinant`.
    pub log_determinant_magnitude: f64,
    pub moments: SelectedMoments,
}

impl Factorization {
    /// Bytes retained by a factorization of `nodes` nodes with its selected
    /// moments, counting the transient backward blocks: twelve `K x K` blocks
    /// and one `K x 2K` reflection block per node, seven `K x G` blocks per
    /// node, and three `G x G` blocks.
    pub fn working_bytes(nodes: usize, signatures: usize, genes: usize) -> Option<usize> {
        let square = signatures.checked_mul(signatures)?;
        let border = signatures.checked_mul(genes)?;
        let per_node = square.checked_mul(14)?.checked_add(border.checked_mul(7)?)?;
        nodes
            .checked_mul(per_node)?
            .checked_add(genes.checked_mul(genes)?.checked_mul(3)?)?
            .checked_mul(std::mem::size_of::<f64>())
    }

    /// `Ok(None)` when `Q + W` is not positive definite, a property of the
    /// supplied curvature rather than a numerical failure.
    pub fn new(law: &StateLaw, curvature: &[Array2<f64>]) -> Result<Option<Self>, EventHistoryError> {
        law.validate()?;
        let k = law.signatures();
        let g = law.genes();
        let count = law.nodes();
        if curvature.len() != count
            || curvature
                .iter()
                .any(|w| w.dim() != (k, k) || w.iter().any(|v| !v.is_finite()))
        {
            return Err(numerical(
                "joint posterior curvature does not match the state law",
            ));
        }
        let identity = Array2::<f64>::eye(k);
        let mut nodes: Vec<Node> = Vec::with_capacity(count);
        let mut log_determinant = 0.0;
        let mut log_determinant_magnitude = 0.0;
        let mut schur = law.gene_precision.clone();
        for n in 0..count {
            let step = law.step(n);
            let mut genes_predicted = step.genes.clone();
            let columns = if n > 0 {
                let phi = &law.transitions[n - 1];
                genes_predicted += &phi.dot(&nodes[n - 1].genes_filtered);
                ndarray::concatenate(
                    Axis(1),
                    &[phi.dot(&nodes[n - 1].spread).view(), step.innovation.view()],
                )
                .map_err(|error| numerical(format!("joint predicted root: {error}")))?
            } else {
                step.innovation.clone()
            };
            let root = Cholesky::from_columns(&columns).ok_or_else(|| {
                numerical("joint predicted state covariance is singular")
            })?;
            let predicted = root.lower.dot(&root.lower.t());
            let w = &curvature[n];
            let mut information = root.lower.t().dot(w).dot(&root.lower) + &identity;
            symmetrize(&mut information);
            let Some(update) = Cholesky::new(&information) else {
                return Ok(None);
            };
            log_determinant += update.log_determinant;
            log_determinant_magnitude += update.log_determinant_magnitude;
            // P = C S^-1 C' = F F' with F = C R^-T.
            let spread = update.forward(&root.lower.t().to_owned()).t().to_owned();
            let mut filtered = spread.dot(&spread.t());
            symmetrize(&mut filtered);
            let complement = &identity - &filtered.dot(w);
            let genes_filtered = complement.dot(&genes_predicted);
            schur += &genes_predicted
                .t()
                .dot(&w.dot(&complement))
                .dot(&genes_predicted);
            nodes.push(Node {
                curvature: w.clone(),
                predicted,
                predicted_root: root,
                spread,
                filtered,
                complement,
                genes_predicted,
                genes_filtered,
                smoothed_genes: Array2::zeros((k, g)),
                adjoint_genes: Array2::zeros((k, g)),
            });
        }
        symmetrize(&mut schur);
        let Some(gene_factor) = Cholesky::new(&schur) else {
            return Ok(None);
        };
        log_determinant += gene_factor.log_determinant;
        log_determinant_magnitude += gene_factor.log_determinant_magnitude;
        let gene_covariance = gene_factor.inverse();

        let mut adjoints: Vec<Array2<f64>> = vec![Array2::zeros((k, k)); count];
        let mut smoothed: Vec<Array2<f64>> = vec![Array2::zeros((k, k)); count];
        let mut future = Array2::<f64>::zeros((k, k));
        let mut future_genes = Array2::<f64>::zeros((k, g));
        for n in (0..count).rev() {
            let node = &mut nodes[n];
            let gain = node.curvature.dot(&node.complement);
            let mut adjoint = gain + node.complement.t().dot(&future).dot(&node.complement);
            symmetrize(&mut adjoint);
            node.adjoint_genes = node
                .complement
                .t()
                .dot(&(&future_genes + &node.curvature.dot(&node.genes_predicted)));
            node.smoothed_genes = &node.genes_filtered - &node.filtered.dot(&future_genes);
            let mut state = &node.filtered - &node.filtered.dot(&future).dot(&node.filtered);
            symmetrize(&mut state);
            smoothed[n] = state;
            if n > 0 {
                let phi = &law.transitions[n - 1];
                future = phi.t().dot(&adjoint).dot(phi);
                future_genes = phi.t().dot(&node.adjoint_genes);
            }
            adjoints[n] = adjoint;
        }

        let mut moments = SelectedMoments {
            states: Vec::with_capacity(count),
            lags: Vec::with_capacity(count.saturating_sub(1)),
            state_genes: Vec::with_capacity(count),
            genes: gene_covariance.clone(),
            innovation_excess: Vec::with_capacity(count),
            innovation_past: Vec::with_capacity(count.saturating_sub(1)),
            innovation_genes: Vec::with_capacity(count),
        };
        for n in 0..count {
            let node = &nodes[n];
            let cross = node.smoothed_genes.dot(&gene_covariance);
            let mut state = &smoothed[n] + &cross.dot(&node.smoothed_genes.t());
            symmetrize(&mut state);
            moments.states.push(state);
            let adjoint_cross = node.adjoint_genes.dot(&gene_covariance);
            let mut excess = adjoint_cross.dot(&node.adjoint_genes.t()) - &adjoints[n];
            symmetrize(&mut excess);
            moments.innovation_excess.push(excess);
            if n > 0 {
                let previous = &nodes[n - 1];
                let propagated = law.transitions[n - 1].dot(&previous.filtered);
                let history = cross.dot(&previous.smoothed_genes.t());
                moments.lags.push(
                    &propagated - &node.predicted.dot(&adjoints[n]).dot(&propagated) + &history,
                );
                moments.innovation_past.push(
                    -(adjoints[n].dot(&propagated) + adjoint_cross.dot(&previous.smoothed_genes.t())),
                );
            }
            moments.innovation_genes.push(-adjoint_cross);
            moments.state_genes.push(cross);
        }
        if !log_determinant.is_finite()
            || moments
                .states
                .iter()
                .chain(&moments.lags)
                .chain(&moments.state_genes)
                .chain(std::iter::once(&moments.genes))
                .chain(&moments.innovation_excess)
                .chain(&moments.innovation_past)
                .chain(&moments.innovation_genes)
                .any(|block| block.iter().any(|v| !v.is_finite()))
        {
            return Err(numerical(
                "joint posterior factorization produced non-finite moments",
            ));
        }
        Ok(Some(Self {
            nodes,
            gene_factor,
            log_determinant,
            log_determinant_magnitude,
            moments,
        }))
    }

    /// Posterior means for node information `b_n` (rows of `information`).
    /// `affine = false` solves the centred problem: every offset and the
    /// genetic mean are dropped, as for a linear sensitivity.
    pub fn solve(
        &self,
        law: &StateLaw,
        information: ArrayView2<'_, f64>,
        affine: bool,
    ) -> Result<Solution, EventHistoryError> {
        let k = law.signatures();
        let gene_information = if affine {
            law.gene_precision.dot(&law.gene_mean)
        } else {
            Array1::zeros(law.genes())
        };
        let offset = |n: usize| {
            if affine {
                law.step(n).offset.clone()
            } else {
                Array1::zeros(k)
            }
        };
        self.solve_given(law, information, gene_information, &offset)
    }

    /// `Q^-1 r` for a general right-hand side `r = (r_g, r_e)` in `(g, e)`
    /// coordinates. The same Gaussian problem with prior information `r_g` on
    /// the genes and innovation means `r_e` has exactly this solution, so no
    /// further factorization is needed and static axes pass `r_e` through.
    pub fn solve_latent(
        &self,
        law: &StateLaw,
        genes: ArrayView1<'_, f64>,
        innovations: ArrayView2<'_, f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), EventHistoryError> {
        let count = self.nodes.len();
        let k = law.signatures();
        if genes.len() != law.genes() || innovations.dim() != (count, k) {
            return Err(numerical(
                "joint latent right-hand side does not match its factorization",
            ));
        }
        let offset = |n: usize| law.step(n).innovation.dot(&innovations.row(n));
        let solution = self.solve_given(
            law,
            Array2::<f64>::zeros((count, k)).view(),
            genes.to_owned(),
            &offset,
        )?;
        Ok((solution.genes, solution.innovations + &innovations))
    }

    /// Forward filtered means conditional on `g` and the genetic posterior
    /// information, for node information `b_n` and node offsets.
    fn filter_means(
        &self,
        law: &StateLaw,
        information: ArrayView2<'_, f64>,
        mut gene_information: Array1<f64>,
        offset: &dyn Fn(usize) -> Array1<f64>,
    ) -> Result<(Vec<Array1<f64>>, Vec<Array1<f64>>, Array1<f64>), EventHistoryError> {
        let count = self.nodes.len();
        let k = law.signatures();
        if law.nodes() != count || information.dim() != (count, k) {
            return Err(numerical(
                "joint posterior information does not match its factorization",
            ));
        }
        let mut residuals: Vec<Array1<f64>> = Vec::with_capacity(count);
        let mut filtered: Vec<Array1<f64>> = Vec::with_capacity(count);
        for (n, node) in self.nodes.iter().enumerate() {
            let mut predicted = offset(n);
            if n > 0 {
                predicted += &law.transitions[n - 1].dot(&filtered[n - 1]);
            }
            let residual = &information.row(n) - &node.curvature.dot(&predicted);
            gene_information += &node
                .genes_predicted
                .t()
                .dot(&node.complement.t().dot(&residual));
            filtered.push(&predicted + &node.filtered.dot(&residual));
            residuals.push(residual);
        }
        Ok((filtered, residuals, gene_information))
    }

    fn solve_given(
        &self,
        law: &StateLaw,
        information: ArrayView2<'_, f64>,
        gene_information: Array1<f64>,
        offset: &dyn Fn(usize) -> Array1<f64>,
    ) -> Result<Solution, EventHistoryError> {
        let count = self.nodes.len();
        let k = law.signatures();
        let (filtered, residuals, gene_information) =
            self.filter_means(law, information, gene_information, offset)?;
        let genes = self.gene_factor.solve_vector(gene_information.view());
        let mut states = Array2::<f64>::zeros((count, k));
        let mut adjoints = Array2::<f64>::zeros((count, k));
        let mut innovations = Array2::<f64>::zeros((count, k));
        let mut future = Array1::<f64>::zeros(k);
        for n in (0..count).rev() {
            let node = &self.nodes[n];
            let adjoint = node.complement.t().dot(&(&future - &residuals[n]));
            let state =
                &filtered[n] - &node.filtered.dot(&future) + &node.smoothed_genes.dot(&genes);
            let total = &adjoint + &node.adjoint_genes.dot(&genes);
            innovations
                .row_mut(n)
                .assign(&(-law.step(n).innovation.t().dot(&total)));
            states.row_mut(n).assign(&state);
            adjoints.row_mut(n).assign(&total);
            if n > 0 {
                future = law.transitions[n - 1].t().dot(&adjoint);
            }
        }
        if genes
            .iter()
            .chain(states.iter())
            .chain(adjoints.iter())
            .chain(innovations.iter())
            .any(|v| !v.is_finite())
        {
            return Err(numerical("joint posterior solve produced non-finite means"));
        }
        Ok(Solution {
            genes,
            states,
            adjoints,
            innovations,
        })
    }

    /// A draw from the Gaussian posterior with node information `b_n`, in
    /// `(g, e)` coordinates, from `G + 2 N K` independent standard normals:
    /// the genes, then per node a filtered-state and an innovation block.
    ///
    /// The genes are drawn from their posterior, then the states backwards.
    /// Each pair `(x_(n-1), e_n)` is drawn from its filtered law and
    /// conditioned exactly on the transition constraint
    /// `Phi_n x_(n-1) + L_n e_n = x_n - c_n - D_n g`. No step inverts `L_n` or
    /// subtracts covariances, so static axes keep their prior innovations.
    /// The draw is affine in the normals; with all normals zero it is the
    /// posterior mean.
    pub fn draw(
        &self,
        law: &StateLaw,
        information: ArrayView2<'_, f64>,
        standard: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, Array2<f64>), EventHistoryError> {
        let count = self.nodes.len();
        let k = law.signatures();
        let g = law.genes();
        if standard.len() != g + 2 * count * k {
            return Err(numerical(
                "joint posterior draw needs G + 2 N K standard normals",
            ));
        }
        let offset = |n: usize| law.step(n).offset.clone();
        let (filtered, residuals, gene_information) = self.filter_means(
            law,
            information,
            law.gene_precision.dot(&law.gene_mean),
            &offset,
        )?;
        drop(residuals);
        let genes = self.gene_factor.solve_vector(gene_information.view())
            + self.gene_factor.whiten(standard.slice(s![..g]));
        let state_noise = |n: usize| standard.slice(s![g + 2 * n * k..g + 2 * n * k + k]);
        let innovation_noise =
            |n: usize| standard.slice(s![g + 2 * n * k + k..g + 2 * (n + 1) * k]);
        let unconditional = |n: usize| {
            let node = &self.nodes[n];
            &filtered[n] + &node.genes_filtered.dot(&genes) + &node.spread.dot(&state_noise(n))
        };
        let mut states = Array2::<f64>::zeros((count, k));
        let mut innovations = Array2::<f64>::zeros((count, k));
        states.row_mut(count - 1).assign(&unconditional(count - 1));
        for n in (0..count).rev() {
            let step = law.step(n);
            let noise = innovation_noise(n);
            let mut propagated =
                &step.offset + &step.genes.dot(&genes) + &step.innovation.dot(&noise);
            let previous = (n > 0).then(|| unconditional(n - 1));
            if let Some(state) = &previous {
                propagated += &law.transitions[n - 1].dot(state);
            }
            let gain = self.nodes[n]
                .predicted_root
                .solve_vector((&states.row(n) - &propagated).view());
            innovations
                .row_mut(n)
                .assign(&(&noise + &step.innovation.t().dot(&gain)));
            if let Some(state) = previous {
                let moved = self.nodes[n - 1]
                    .filtered
                    .dot(&law.transitions[n - 1].t().dot(&gain));
                states.row_mut(n - 1).assign(&(state + moved));
            }
        }
        if genes
            .iter()
            .chain(innovations.iter())
            .any(|v| !v.is_finite())
        {
            return Err(numerical("joint posterior draw is not finite"));
        }
        Ok((genes, innovations))
    }

    /// Normalized log density of `N(center, Q^-1)` at `point`, both in
    /// `(g, e)` coordinates, with `Q` applied matrix-free through the
    /// transport.
    pub fn log_density(
        &self,
        law: &StateLaw,
        center: (ArrayView1<'_, f64>, ArrayView2<'_, f64>),
        point: (ArrayView1<'_, f64>, ArrayView2<'_, f64>),
    ) -> Result<f64, EventHistoryError> {
        let count = self.nodes.len();
        let k = law.signatures();
        if center.0.len() != law.genes()
            || point.0.len() != law.genes()
            || center.1.dim() != (count, k)
            || point.1.dim() != (count, k)
        {
            return Err(numerical(
                "joint posterior density point does not match its factorization",
            ));
        }
        let gene_delta = &point.0 - &center.0;
        let innovation_delta = &point.1 - &center.1;
        let mut forces = law.transport(gene_delta.view(), innovation_delta.view(), false);
        for (n, node) in self.nodes.iter().enumerate() {
            let force = node.curvature.dot(&forces.row(n));
            forces.row_mut(n).assign(&force);
        }
        let (gene_product, innovation_product) = law.transport_adjoint(forces.view());
        let quadratic = gene_delta.dot(&(gene_product + law.gene_precision.dot(&gene_delta)))
            + (&innovation_delta * &(innovation_product + &innovation_delta)).sum();
        let dimension = (law.genes() + count * k) as f64;
        let value = 0.5
            * (self.log_determinant - dimension * (2.0 * std::f64::consts::PI).ln() - quadratic);
        if !value.is_finite() {
            return Err(numerical("joint posterior density is not finite"));
        }
        Ok(value)
    }
}

/// Row-major `rows x inner` times `inner x cols` over any scalar, for the
/// certificate's small blocks.
fn block_product<S: JetField>(a: &[S], b: &[S], rows: usize, inner: usize, cols: usize, like: &S) -> Vec<S> {
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            out.push(
                (0..inner).fold(like.constant_like(0.0), |sum, q| sum.add(&a[i * inner + q].mul(&b[q * cols + j]))),
            );
        }
    }
    out
}

/// The transpose of a row-major `rows x cols` block.
fn block_transpose<S: JetField>(a: &[S], rows: usize, cols: usize) -> Vec<S> {
    (0..cols * rows)
        .map(|index| a[(index % rows) * cols + index / rows].clone())
        .collect()
}

fn block_sum<S: JetField>(a: &[S], b: &[S]) -> Vec<S> {
    a.iter().zip(b).map(|(x, y)| x.add(y)).collect()
}

/// A law matrix entering a certificate as exact input.
fn block_input<S: JetField>(matrix: &Array2<f64>, like: &S) -> Vec<S> {
    matrix.iter().map(|&v| like.constant_like(v)).collect()
}

/// The lower end of a running-error value's bracket; deleted once `Running` carries
/// its own accessor.
fn running_lower(x: &numerical::Running) -> f64 {
    x.value - x.rounding()
}

/// The upper end of a running-error value's bracket; deleted once `Running` carries
/// its own accessor.
fn running_upper(x: &numerical::Running) -> f64 {
    x.value + x.rounding()
}

impl StateLaw {
    /// `S_n = (J J')_nn`, the covariance of the node states when the genes and
    /// the innovations are independent standard normal, over any scalar:
    /// `S_n = Phi S_(n-1) Phi' + Phi C_(n-1) D' + D C_(n-1)' Phi' + D D' + L L'`
    /// with `C_n = Cov(x_n, g) = Phi C_(n-1) + D`, and `S_0 = B B' + L L'`.
    fn transport_gram<S: JetField>(&self, like: &S) -> Vec<Vec<S>> {
        let k = self.signatures();
        let g = self.genes();
        let mut gram: Vec<Vec<S>> = Vec::with_capacity(self.nodes());
        let mut cross: Vec<S> = Vec::new();
        for n in 0..self.nodes() {
            let step = self.step(n);
            let loading = block_input(&step.genes, like);
            let root = block_input(&step.innovation, like);
            let mut block = block_sum(
                &block_product(&loading, &block_transpose(&loading, k, g), k, g, k, like),
                &block_product(&root, &block_transpose(&root, k, k), k, k, k, like),
            );
            if n > 0 {
                let phi = block_input(&self.transitions[n - 1], like);
                let carried = block_product(&phi, &cross, k, k, g, like);
                let spread = block_product(
                    &block_product(&phi, &gram[n - 1], k, k, k, like),
                    &block_transpose(&phi, k, k),
                    k,
                    k,
                    k,
                    like,
                );
                let coupling = block_product(&carried, &block_transpose(&loading, k, g), k, g, k, like);
                block = block_sum(
                    &block_sum(&block, &spread),
                    &block_sum(&coupling, &block_transpose(&coupling, k, k)),
                );
                cross = block_sum(&carried, &loading);
            } else {
                cross = loading;
            }
            gram.push(block);
        }
        gram
    }

    /// `Q z = J' W (J z) + diag(Lambda_mm, I) z`, matrix-free over any scalar,
    /// with the law's entries and the node curvatures as exact inputs: the
    /// definition of the precision a structured solve inverts.
    pub fn precision_product<S: JetField>(
        &self,
        curvature: &[Array2<f64>],
        genes: &[S],
        innovations: &[S],
        like: &S,
    ) -> (Vec<S>, Vec<S>) {
        let k = self.signatures();
        let g = self.genes();
        let nodes = self.nodes();
        let zero = like.constant_like(0.0);
        let mut states: Vec<S> = Vec::with_capacity(nodes * k);
        for n in 0..nodes {
            let step = self.step(n);
            for i in 0..k {
                let mut value = (0..g).fold(zero.clone(), |sum, j| sum.add(&genes[j].scale(step.genes[[i, j]])));
                value = (0..k).fold(value, |sum, j| {
                    sum.add(&innovations[n * k + j].scale(step.innovation[[i, j]]))
                });
                if n > 0 {
                    value = (0..k).fold(value, |sum, j| {
                        sum.add(&states[(n - 1) * k + j].scale(self.transitions[n - 1][[i, j]]))
                    });
                }
                states.push(value);
            }
        }
        let forces: Vec<S> = (0..nodes * k)
            .map(|index| {
                let (n, i) = (index / k, index % k);
                (0..k).fold(zero.clone(), |sum, j| sum.add(&states[n * k + j].scale(curvature[n][[i, j]])))
            })
            .collect();
        let mut gene_product = vec![zero.clone(); g];
        let mut innovation_product = vec![zero.clone(); nodes * k];
        let mut carried = vec![zero.clone(); k];
        for n in (0..nodes).rev() {
            let step = self.step(n);
            for i in 0..k {
                carried[i] = carried[i].add(&forces[n * k + i]);
            }
            for (j, product) in gene_product.iter_mut().enumerate() {
                *product = (0..k).fold(product.clone(), |sum, i| sum.add(&carried[i].scale(step.genes[[i, j]])));
            }
            for j in 0..k {
                innovation_product[n * k + j] =
                    (0..k).fold(zero.clone(), |sum, i| sum.add(&carried[i].scale(step.innovation[[i, j]])));
            }
            if n > 0 {
                carried = (0..k)
                    .map(|j| (0..k).fold(zero.clone(), |sum, i| sum.add(&carried[i].scale(self.transitions[n - 1][[i, j]]))))
                    .collect();
            }
        }
        for (j, product) in gene_product.iter_mut().enumerate() {
            *product = (0..g).fold(product.clone(), |sum, q| sum.add(&genes[q].scale(self.gene_precision[[j, q]])));
        }
        for (product, innovation) in innovation_product.iter_mut().zip(innovations) {
            *product = product.add(innovation);
        }
        (gene_product, innovation_product)
    }
}

/// A certified lower bound on the smallest eigenvalue of the conditional
/// precision `Q = J' W J + diag(Lambda_mm, I)` in the coordinates the
/// structured solves use, and the forward error it gives those solves.
///
/// Each computed node curvature splits as `W_n = M_n - N_n + E_n`. `M_n` sums
/// nonnegative multiples of outer products and nonnegative diagonals. `N_n`, the
/// observation's `excess`, is built from the computed nonnegative multipliers
/// and vectors of the negative parts the metric drops. So both are positive
/// semidefinite exactly. `E_n` is the rounding of the curvature's own
/// accumulation, `|E_n| <= eps mu_W`. For every `z = (g, e)`,
/// `z' Q z = sum_n (J z)_n' M_n (J z)_n - z' J' N J z + z' J' E J z + g' Lambda_mm g + |e|^2`,
/// and the first sum is nonnegative whatever the gene-innovation cross terms
/// inside `J` are. The innovations' standardized prior is the identity block,
/// so the spectrum of `diag(Lambda_mm, I)` is that of `Lambda_mm` together with
/// ones, and its smallest eigenvalue is `min(lambda_min(Lambda_mm), 1)`. By
/// Weyl's inequality
/// `lambda_min(Q) >= min(lambda_min(Lambda_mm), 1) - ||J' N J||_2 - ||J' E J||_2`.
/// Here `||J' N J||_2 <= ||N^(1/2) J||_F^2 = sum_n tr(N_n S_n)` with
/// `S_n = (J J')_nn`, bounded by the upper bound of its computed value, which
/// carries `mu_N` and `mu_S`. And `||J' E J||_2 <= sum_n ||E_n||_F tr(S_n)`.
///
/// The genetic border's `lambda_min(Lambda_mm)` is bounded below by the larger of
/// its Gershgorin floor, which proves definiteness by itself when positive, and
/// `det(Lambda_mm) / ||Lambda_mm||_inf^(G - 1)`. The determinant form holds only
/// for a positive definite matrix: one with two negative eigenvalues also has a
/// positive determinant. So it is taken only after every Cholesky pivot of
/// `Lambda_mm` is certified positive by its lower bound; otherwise the Gershgorin
/// floor stands alone. Then every eigenvalue lies in `(0, rho(Lambda_mm)]` with
/// `rho(Lambda_mm) <= ||Lambda_mm||_inf`, and they multiply to the determinant. A
/// border whose bound is not positive is refused.
///
/// Every quantity is evaluated over the running-error scalar, and the floor is
/// its lower bound. The sum of the per-node contributions, the floor's
/// subtraction, and the forward error's norms and division are first order. The
/// certificate bounds solves of the operator defined by the stored f64 law
/// entries and curvature blocks. A decision band on the posterior mean must
/// charge those entries' own assembly error separately.
pub(super) enum PrecisionFloor {
    Certified {
        floor: f64,
    },
    /// The floor is not resolved above zero. `node` carries the largest
    /// negative-curvature contribution `tr(N_n S_n)` plus its rounding charge.
    Refused {
        node: usize,
        floor: f64,
    },
}

impl PrecisionFloor {
    /// `curvature` is the node curvature the factorization used and `excess` its
    /// dropped negative part, row-major `K x K` per node, both evaluated over the
    /// running-error scalar. A factorization of the metric passes a zero excess.
    pub(super) fn new(
        law: &StateLaw,
        curvature: &[Vec<numerical::Running>],
        excess: &[Vec<numerical::Running>],
    ) -> Result<Self, EventHistoryError> {
        law.validate()?;
        let k = law.signatures();
        let g = law.genes();
        if curvature.len() != law.nodes()
            || excess.len() != law.nodes()
            || curvature.iter().chain(excess).any(|block| block.len() != k * k)
        {
            return Err(numerical(
                "joint solve certificate blocks do not match the state law",
            ));
        }
        let like = numerical::Running::exact(0.0);
        let gram = law.transport_gram(&like);
        let mut node = 0;
        let mut largest = f64::NEG_INFINITY;
        let mut subtracted = numerical::RunningSum::default();
        for n in 0..law.nodes() {
            // tr(N S) = sum_ab N_ab S_ba.
            let pull = (0..k * k).fold(like.clone(), |sum, index| {
                sum.add(&excess[n][index].mul(&gram[n][(index % k) * k + index / k]))
            });
            let trace = (0..k).fold(like.clone(), |sum, a| sum.add(&gram[n][a * k + a]));
            // The excess's own rounding is carried inside `pull`.
            let accumulation = f64::EPSILON * numerical::frobenius((0..k * k).map(|index| curvature[n][index].mu));
            let spread = running_upper(&trace);
            let reach = running_upper(&pull);
            let contribution = reach + accumulation * spread;
            subtracted.add(reach, 0.0);
            subtracted.add(
                accumulation * spread,
                numerical::product_error(accumulation, 0.0, spread, 0.0),
            );
            if contribution > largest {
                largest = contribution;
                node = n;
            }
        }
        let border = if g == 0 {
            1.0
        } else {
            let precision = block_input(&law.gene_precision, &like);
            let gershgorin = (0..g)
                .map(|i| {
                    running_lower(&(0..g).filter(|&j| j != i).fold(precision[i * g + i].clone(), |acc, j| {
                        acc.sub(&Self::absolute(&precision[i * g + j]))
                    }))
                })
                .fold(f64::INFINITY, f64::min);
            gershgorin.max(Self::determinant_floor(&precision, g).unwrap_or(f64::NEG_INFINITY))
        };
        if !(border > 0.0) {
            return Err(numerical(
                "conditional genetic precision is not certified positive definite",
            ));
        }
        let floor = border.min(1.0) - numerical::upper(subtracted.value + subtracted.error);
        if floor > 0.0 {
            Ok(Self::Certified { floor })
        } else {
            Ok(Self::Refused { node, floor })
        }
    }

    fn absolute(x: &numerical::Running) -> numerical::Running {
        if x.value < 0.0 { x.neg() } else { x.clone() }
    }

    /// `det(Lambda_mm) / ||Lambda_mm||_inf^(G - 1)` as a lower bound, or `None` when
    /// a Cholesky pivot is not certified positive by its lower bound, where the
    /// determinant form does not apply.
    fn determinant_floor(precision: &[numerical::Running], g: usize) -> Option<f64> {
        let like = numerical::Running::exact(0.0);
        let mut factor = vec![like.clone(); g * g];
        let mut log_determinant = like.clone();
        for j in 0..g {
            let pivot = (0..j).fold(precision[j * g + j].clone(), |acc, q| {
                acc.sub(&factor[j * g + q].mul(&factor[j * g + q]))
            });
            if !(pivot.value.is_finite() && running_lower(&pivot) > 0.0) {
                return None;
            }
            log_determinant = log_determinant.add(&ln(&pivot));
            let root = sqrt(&pivot);
            for i in j + 1..g {
                let entry = (0..j).fold(precision[i * g + j].clone(), |acc, q| {
                    acc.sub(&factor[i * g + q].mul(&factor[j * g + q]))
                });
                factor[i * g + j] = div(&entry, &root);
            }
            factor[j * g + j] = root;
        }
        let norm = (0..g)
            .map(|i| (0..g).fold(like.clone(), |acc, j| acc.add(&Self::absolute(&precision[i * g + j]))))
            .fold(like.clone(), |widest, row| if running_upper(&row) > running_upper(&widest) { row } else { widest });
        let power = (1..g).fold(like.constant_like(1.0), |acc, _| acc.mul(&norm));
        Some(running_lower(&exp(&log_determinant).mul(&recip(&power))))
    }

    /// `||x^ - x||_2 <= (||r^||_2 + eps ||mu_r||_2) / floor` for the solve
    /// `Q x = b` at the computed `x^`, where `r^ = b^ - Q x^` is formed
    /// matrix-free over the running-error scalar, so `mu_r` carries the
    /// right-hand side's own bounds and the product's rounding. The norms of the
    /// values and of the roundings are taken separately: a squared residual's
    /// running error is first order and would drop `(eps mu_r)^2`, vanishing at
    /// an exact residual whatever `mu_r` is. The final norms and division are
    /// first order. A refused certificate bounds nothing.
    pub(super) fn forward_error(
        &self,
        law: &StateLaw,
        curvature: &[Array2<f64>],
        rhs_genes: &[numerical::Running],
        rhs_innovations: &[numerical::Running],
        genes: &[f64],
        innovations: &[f64],
    ) -> Result<f64, EventHistoryError> {
        let Self::Certified { floor } = self else {
            return Err(numerical(
                "joint solve certificate was refused; no forward error bound follows",
            ));
        };
        if rhs_genes.len() != law.genes()
            || genes.len() != law.genes()
            || rhs_innovations.len() != law.nodes() * law.signatures()
            || innovations.len() != rhs_innovations.len()
            || curvature.len() != law.nodes()
        {
            return Err(numerical(
                "joint solve certificate residual does not match the state law",
            ));
        }
        let like = numerical::Running::exact(0.0);
        let exact = |values: &[f64]| -> Vec<numerical::Running> {
            values.iter().map(|&v| numerical::Running::exact(v)).collect()
        };
        let (gene_product, innovation_product) =
            law.precision_product(curvature, &exact(genes), &exact(innovations), &like);
        let residuals: Vec<numerical::Running> = rhs_genes
            .iter()
            .chain(rhs_innovations)
            .zip(gene_product.iter().chain(&innovation_product))
            .map(|(b, product)| b.sub(product))
            .collect();
        let values = numerical::frobenius(residuals.iter().map(|r| r.value));
        let bounds = numerical::frobenius(residuals.iter().map(|r| r.mu));
        Ok(numerical::upper(values + f64::EPSILON * bounds) / floor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{Bound, agrees};
    use gam_math::nested_dual::JetField;

    /// K = 2, G = 2, N = 5 with dense transitions, correlated innovations, a
    /// static axis at node 2, and a fully static transition at node 3 (the
    /// identity transition of a tied event time).
    fn fixture() -> (StateLaw, Vec<Array2<f64>>) {
        let k = 2;
        let g = 2;
        let step = |n: usize, innovation: Array2<f64>| Step {
            offset: Array1::from_shape_fn(k, |i| 0.3 * ((n + 2 * i) as f64).sin()),
            genes: Array2::from_shape_fn((k, g), |(i, j)| 0.2 * ((n + i + 3 * j) as f64).cos()),
            innovation,
        };
        let law = StateLaw {
            gene_mean: ndarray::arr1(&[0.4, -0.3]),
            gene_precision: ndarray::arr2(&[[2.0, 0.6], [0.6, 1.5]]),
            entry: step(0, Array2::eye(k)),
            transitions: vec![
                ndarray::arr2(&[[0.8, 0.1], [-0.05, 0.9]]),
                ndarray::arr2(&[[0.95, 0.0], [0.02, 1.0]]),
                Array2::eye(k),
                ndarray::arr2(&[[0.6, -0.2], [0.1, 0.7]]),
            ],
            steps: vec![
                step(1, ndarray::arr2(&[[0.5, 0.0], [0.2, 0.4]])),
                step(2, ndarray::arr2(&[[0.3, 0.0], [0.0, 0.0]])),
                step(3, Array2::zeros((k, k))),
                step(4, ndarray::arr2(&[[0.7, 0.0], [-0.1, 0.6]])),
            ],
        };
        let curvature = (0..5)
            .map(|n| match n {
                1 => ndarray::arr2(&[[1.3, 0.4], [0.4, 0.2]]),
                // An indefinite node curvature; the prior keeps Q + W definite.
                3 => ndarray::arr2(&[[-0.2, 0.05], [0.05, 0.9]]),
                _ => ndarray::arr2(&[[0.6, -0.1], [-0.1, 0.8]]) * (n as f64 + 1.0) / 3.0,
            })
            .collect();
        (law, curvature)
    }

    /// The dense innovation posterior, assembled over [`Bound`]. Solves go through
    /// the crate's one dense Cholesky oracle, `test_support::cholesky_forward_error`
    /// (Higham, 2nd ed., Theorem 10.4, with an a posteriori inverse certificate and
    /// a typed refusal), so every oracle entry carries its forward error bound.
    ///
    /// The structured route is a covariance-form filter with square-root
    /// propagation and a backward pass, not a block Cholesky factorization of
    /// the block-tridiagonal precision, so no stated bound covers it. It is
    /// seeded exact: the bar is the dense route's forward error alone, which is
    /// stricter than a sum of two routes' bounds, and a failure is a finding.
    struct Dense {
        /// Rows: node-major states. Columns: genes, then node-major innovations.
        jacobian: Array2<Bound>,
        offset: Array1<Bound>,
        precision: Array2<Bound>,
        /// `Q^-1`: column `j` is the oracle's solve against `e_j`.
        covariance: Array2<Bound>,
    }

    fn exact(matrix: &Array2<f64>) -> Array2<Bound> {
        matrix.mapv(Bound::exact)
    }

    fn column(vector: &Array1<f64>) -> Array2<Bound> {
        vector.mapv(Bound::exact).insert_axis(Axis(1))
    }

    fn sum(terms: impl Iterator<Item = Bound>) -> Bound {
        terms.fold(Bound::exact(0.0), |acc, term| acc.add(&term))
    }

    fn product(a: ArrayView2<'_, Bound>, b: ArrayView2<'_, Bound>) -> Array2<Bound> {
        Array2::from_shape_fn((a.nrows(), b.ncols()), |(i, j)| {
            sum((0..a.ncols()).map(|q| a[[i, q]].mul(&b[[q, j]])))
        })
    }

    /// States of the transport by the recursion of `StateLaw::transport`, with
    /// running rounding bounds.
    fn bounded_transport(
        law: &StateLaw,
        genes: &[Bound],
        innovations: ArrayView2<'_, Bound>,
        affine: bool,
    ) -> Array2<Bound> {
        let k = law.signatures();
        let mut states = Array2::from_elem((law.nodes(), k), Bound::exact(0.0));
        for n in 0..law.nodes() {
            let step = law.step(n);
            for i in 0..k {
                let mut terms: Vec<Bound> = (0..law.genes())
                    .map(|j| Bound::exact(step.genes[[i, j]]).mul(&genes[j]))
                    .chain((0..k).map(|j| Bound::exact(step.innovation[[i, j]]).mul(&innovations[[n, j]])))
                    .collect();
                if affine {
                    terms.push(Bound::exact(step.offset[i]));
                }
                if n > 0 {
                    terms.extend(
                        (0..k).map(|j| Bound::exact(law.transitions[n - 1][[i, j]]).mul(&states[[n - 1, j]])),
                    );
                }
                states[[n, i]] = sum(terms.into_iter());
            }
        }
        states
    }

    fn dense(law: &StateLaw, curvature: &[Array2<f64>]) -> Dense {
        let (jacobian, offset, precision) = dense_precision(law, curvature);
        let d = precision.nrows();
        let mut covariance = Array2::from_elem((d, d), Bound::exact(0.0));
        for j in 0..d {
            let unit: Vec<Bound> = (0..d).map(|i| Bound::exact(f64::from(i == j))).collect();
            let solved = crate::test_support::cholesky_forward_error(&precision, &unit).unwrap();
            for (i, entry) in solved.into_iter().enumerate() {
                covariance[[i, j]] = entry;
            }
        }
        Dense {
            jacobian,
            offset,
            precision,
            covariance,
        }
    }

    fn dense_precision(
        law: &StateLaw,
        curvature: &[Array2<f64>],
    ) -> (Array2<Bound>, Array1<Bound>, Array2<Bound>) {
        let k = law.signatures();
        let g = law.genes();
        let count = law.nodes();
        let dimension = g + count * k;
        let mut jacobian = Array2::from_elem((count * k, dimension), Bound::exact(0.0));
        for index in 0..dimension {
            let genes: Vec<Bound> = (0..g).map(|i| Bound::exact(f64::from(i == index))).collect();
            let innovations =
                Array2::from_shape_fn((count, k), |(n, i)| Bound::exact(f64::from(g + n * k + i == index)));
            let states = bounded_transport(law, &genes, innovations.view(), false);
            for (row, value) in states.iter().enumerate() {
                jacobian[[row, index]] = *value;
            }
        }
        let offset = bounded_transport(
            law,
            &vec![Bound::exact(0.0); g],
            Array2::from_elem((count, k), Bound::exact(0.0)).view(),
            true,
        )
        .into_shape_with_order(count * k)
        .unwrap();
        let weighted = Array2::from_shape_fn((count * k, dimension), |(row, index)| {
            let (n, a) = (row / k, row % k);
            sum((0..k).map(|b| Bound::exact(curvature[n][[a, b]]).mul(&jacobian[[n * k + b, index]])))
        });
        let mut precision = product(jacobian.t(), weighted.view());
        for i in 0..dimension {
            for j in 0..=i {
                let prior = if i < g && j < g {
                    law.gene_precision[[i, j]]
                } else {
                    f64::from(i == j)
                };
                // Both triangles take the lower entry: `J' W J` rounds its two
                // triangles along different operation orders, and the oracle
                // factors one symmetric matrix.
                let entry = precision[[i, j]].add(&Bound::exact(prior));
                precision[[i, j]] = entry;
                precision[[j, i]] = entry;
            }
        }
        (jacobian, offset, precision)
    }

    /// `Q^-1 b` through the crate's dense Cholesky oracle, as a column.
    fn solve(oracle: &Dense, rhs: &Array2<Bound>) -> Array2<Bound> {
        let solved =
            crate::test_support::cholesky_forward_error(&oracle.precision, &rhs.column(0).to_vec()).unwrap();
        Array2::from_shape_vec((solved.len(), 1), solved).unwrap()
    }

    /// `J' (b - W offset)`, plus the genetic prior's pull in the affine problem.
    fn information_rhs(
        law: &StateLaw,
        curvature: &[Array2<f64>],
        oracle: &Dense,
        information: &Array2<f64>,
        affine: bool,
    ) -> Array2<Bound> {
        let k = law.signatures();
        let g = law.genes();
        let residual = Array2::from_shape_fn((law.nodes() * k, 1), |(row, _)| {
            let (n, a) = (row / k, row % k);
            let information = Bound::exact(information[[n, a]]);
            if affine {
                information.sub(&sum(
                    (0..k).map(|b| Bound::exact(curvature[n][[a, b]]).mul(&oracle.offset[n * k + b])),
                ))
            } else {
                information
            }
        });
        let mut rhs = product(oracle.jacobian.t(), residual.view());
        if affine {
            let prior = product(exact(&law.gene_precision).view(), column(&law.gene_mean).view());
            for i in 0..g {
                rhs[[i, 0]] = rhs[[i, 0]].add(&prior[[i, 0]]);
            }
        }
        rhs
    }

    /// Every entry within its bar. The block is resolved: some oracle entry is
    /// above its bar, unless the production block is exactly zero (a static
    /// axis).
    fn close(production: &Array2<Bound>, oracle: &Array2<Bound>, label: &str) {
        assert_eq!(production.dim(), oracle.dim(), "{label}");
        let mut resolved = production.iter().all(|p| p.value == 0.0);
        for ((i, j), expected) in oracle.indexed_iter() {
            let actual = &production[[i, j]];
            let bar = actual.bar(expected);
            assert!(
                (actual.value - expected.value).abs() <= bar,
                "{label} [{i},{j}]: {} vs {}, bar {bar}",
                actual.value,
                expected.value
            );
            resolved |= expected.value.abs() > bar;
        }
        assert!(resolved, "{label}: every entry is below its bar");
    }

    fn latent(genes: &Array1<f64>, innovations: &Array2<f64>) -> Array1<f64> {
        genes.iter().chain(innovations.iter()).copied().collect()
    }

    fn assert_matches_dense(law: &StateLaw, curvature: &[Array2<f64>]) {
        let k = law.signatures();
        let g = law.genes();
        let count = law.nodes();
        let dimension = g + count * k;
        let factor = Factorization::new(law, curvature).unwrap().unwrap();
        let oracle = dense(law, curvature);
        agrees(
            &Bound::exact(factor.log_determinant),
            &crate::test_support::cholesky_log_det(&oracle.precision).unwrap(),
            "log det",
        );
        let sigma = &oracle.covariance;
        let transport = &oracle.jacobian;
        close(
            &exact(&factor.moments.genes),
            &sigma.slice(s![..g, ..g]).to_owned(),
            "genes",
        );
        for n in 0..count {
            let rows = transport.slice(s![n * k..(n + 1) * k, ..]);
            let spread = product(rows, sigma.view());
            close(
                &exact(&factor.moments.states[n]),
                &product(spread.view(), rows.t()),
                "states",
            );
            close(
                &exact(&factor.moments.state_genes[n]),
                &spread.slice(s![.., ..g]).to_owned(),
                "state genes",
            );
            if n > 0 {
                let previous = transport.slice(s![(n - 1) * k..n * k, ..]);
                close(
                    &exact(&factor.moments.lags[n - 1]),
                    &product(spread.view(), previous.t()),
                    "lags",
                );
            }
        }

        // Scaled-innovation moments, checked as L' nu = e for any L, including
        // static axes where V_n is singular.
        for n in 0..count {
            let e = g + n * k..g + (n + 1) * k;
            let root = exact(&law.step(n).innovation);
            let excess = exact(&factor.moments.innovation_excess[n]);
            let identity = Array2::from_shape_fn((k, k), |(i, j)| {
                sigma[[g + n * k + i, g + n * k + j]].sub(&Bound::exact(f64::from(i == j)))
            });
            close(
                &product(product(root.t(), excess.view()).view(), root.view()),
                &identity,
                "innovation excess",
            );
            close(
                &product(root.t(), exact(&factor.moments.innovation_genes[n]).view()),
                &sigma.slice(s![e.clone(), ..g]).to_owned(),
                "innovation genes",
            );
            if n > 0 {
                let past = product(
                    sigma.slice(s![e.clone(), ..]),
                    transport.slice(s![(n - 1) * k..n * k, ..]).t(),
                );
                close(
                    &product(root.t(), exact(&factor.moments.innovation_past[n - 1]).view()),
                    &past,
                    "innovation past",
                );
            }
        }

        // Means of the affine and the centred problems, and a general
        // right-hand side.
        let information =
            Array2::from_shape_fn((count, k), |(n, i)| 0.4 * ((3 * n + i) as f64).cos());
        for affine in [true, false] {
            let expected = solve(
                &oracle,
                &information_rhs(law, curvature, &oracle, &information, affine),
            );
            let solution = factor.solve(law, information.view(), affine).unwrap();
            let states = Array2::from_shape_fn((count * k, 1), |(row, _)| {
                let linear = sum((0..dimension).map(|j| transport[[row, j]].mul(&expected[[j, 0]])));
                if affine {
                    linear.add(&oracle.offset[row])
                } else {
                    linear
                }
            });
            close(
                &column(&solution.genes),
                &expected.slice(s![..g, ..]).to_owned(),
                "mean genes",
            );
            close(
                &exact(&solution.states.clone().into_shape_with_order((count * k, 1)).unwrap()),
                &states,
                "mean states",
            );
            close(
                &exact(&solution.innovations.clone().into_shape_with_order((count * k, 1)).unwrap()),
                &expected.slice(s![g.., ..]).to_owned(),
                "mean innovations",
            );
        }
        let rhs = Array1::from_shape_fn(dimension, |i| (1.7 * i as f64).sin());
        let (gene_solve, innovation_solve) = factor
            .solve_latent(
                law,
                rhs.slice(s![..g]),
                rhs.slice(s![g..]).into_shape_with_order((count, k)).unwrap(),
            )
            .unwrap();
        close(
            &column(&latent(&gene_solve, &innovation_solve)),
            &solve(&oracle, &column(&rhs)),
            "latent solve",
        );
    }

    #[test]
    fn structured_factorization_matches_the_dense_innovation_posterior() {
        let (law, curvature) = fixture();
        assert_matches_dense(&law, &curvature);
    }

    /// Axis 0 is static after entry and observed with a large curvature at
    /// every node, so its posterior variance shrinks by 1e6 per node over 40
    /// nodes while axis 1 keeps an ordinary OU law. `loading` is axis 0's entry
    /// loading on the one genetic score.
    fn sharp_static_axis(loading: f64) -> (StateLaw, Vec<Array2<f64>>) {
        let k = 2;
        let count = 40;
        let law = StateLaw {
            gene_mean: ndarray::arr1(&[0.2]),
            gene_precision: ndarray::arr2(&[[1.0]]),
            entry: Step {
                offset: ndarray::arr1(&[0.1, -0.2]),
                genes: ndarray::arr2(&[[loading], [-0.3]]),
                innovation: Array2::eye(k),
            },
            transitions: vec![ndarray::arr2(&[[1.0, 0.0], [0.0, 0.9]]); count - 1],
            steps: (1..count)
                .map(|n| Step {
                    offset: ndarray::arr1(&[0.0, 0.05 * (n as f64).sin()]),
                    genes: ndarray::arr2(&[[0.0], [0.02]]),
                    innovation: ndarray::arr2(&[[0.0, 0.0], [0.0, 0.43]]),
                })
                .collect(),
        };
        let curvature: Vec<Array2<f64>> = (0..count)
            .map(|n| ndarray::arr2(&[[1e6, 0.0], [0.0, 0.5 + 0.1 * (n as f64).cos()]]))
            .collect();
        (law, curvature)
    }

    /// Every production moment is finite, and every state block and the gene
    /// block factor as positive definite.
    fn assert_definite_moments(factor: &Factorization) {
        let moments = &factor.moments;
        assert!(
            moments
                .states
                .iter()
                .chain(&moments.lags)
                .chain(&moments.state_genes)
                .chain(std::iter::once(&moments.genes))
                .all(|block| block.iter().all(|v| v.is_finite())),
            "a production moment is not finite"
        );
        assert!(
            moments.states.iter().all(|block| Cholesky::new(block).is_some())
                && Cholesky::new(&moments.genes).is_some(),
            "a production state or gene block is not positive definite"
        );
    }

    #[test]
    fn a_sharply_identified_static_axis_stays_representable() {
        // Axis 0 loads no genetic score, so the dense oracle certifies every block.
        let (law, curvature) = sharp_static_axis(0.0);
        assert_matches_dense(&law, &curvature);
        // Axis 0's state variance shrinks toward roundoff; the structured route
        // keeps it positive by construction.
        let factor = Factorization::new(&law, &curvature).unwrap().unwrap();
        assert!(factor.moments.states.iter().all(|block| block[[0, 0]] > 0.0));
        assert_definite_moments(&factor);
    }

    #[test]
    fn a_static_axis_on_a_genetic_ridge_is_a_documented_limit_of_the_dense_oracle() {
        // Axis 0 loads the genetic score, so the gene and the entry innovation
        // share one sharply identified direction: a ridge. Production still
        // returns finite, definite moments. The dense oracle does not certify the
        // gene-state border there: it refuses a gene column, or it returns bounds
        // too wide to resolve Cov(x_n, g) at some node (job 1220069 measured the
        // second). If the oracle ever resolves the whole border, this limit has
        // moved and the documentation must follow.
        let (law, curvature) = sharp_static_axis(0.5);
        let factor = Factorization::new(&law, &curvature).unwrap().unwrap();
        assert_definite_moments(&factor);
        let (jacobian, _, precision) = dense_precision(&law, &curvature);
        let k = law.signatures();
        let g = law.genes();
        let d = precision.nrows();
        let mut sigma_genes = Array2::from_elem((d, g), Bound::exact(0.0));
        for j in 0..g {
            let unit: Vec<Bound> = (0..d).map(|i| Bound::exact(f64::from(i == j))).collect();
            let solved = crate::test_support::cholesky_forward_error(&precision, &unit);
            if let Err(crate::test_support::UnresolvedSolve::Certificate(rho)) = solved {
                eprintln!("ridge: the dense oracle refuses gene column {j}, certificate rho {rho}");
                return;
            }
            // Any other refusal is an assembly defect, not the documented limit.
            for (i, entry) in solved.unwrap().into_iter().enumerate() {
                sigma_genes[[i, j]] = entry;
            }
        }
        let border = product(jacobian.view(), sigma_genes.view());
        let mut unresolved = Vec::new();
        for n in 0..law.nodes() {
            let mut largest = 0.0_f64;
            for a in 0..k {
                for j in 0..g {
                    let oracle = &border[[n * k + a, j]];
                    let production = Bound::exact(factor.moments.state_genes[n][[a, j]]);
                    let bar = production.bar(oracle);
                    assert!(
                        (production.value - oracle.value).abs() <= bar,
                        "ridge state genes {n},{a},{j}: {} vs {}, bar {bar}",
                        production.value,
                        oracle.value
                    );
                    largest = largest.max(oracle.value.abs() / bar);
                }
            }
            if largest <= 1.0 {
                unresolved.push(n);
                eprintln!("ridge: node {n} gene-state border unresolved, largest |oracle| / bar {largest}");
            }
        }
        assert!(
            !unresolved.is_empty(),
            "the dense oracle now resolves the ridge's gene-state border at every node"
        );
    }

    #[test]
    fn structured_draws_have_the_posterior_law_and_its_density() {
        let (law, curvature) = fixture();
        let k = law.signatures();
        let g = law.genes();
        let count = law.nodes();
        let dimension = g + count * k;
        let factor = Factorization::new(&law, &curvature).unwrap().unwrap();
        let oracle = dense(&law, &curvature);
        let information =
            Array2::from_shape_fn((count, k), |(n, i)| 0.4 * ((3 * n + i) as f64).cos());
        let mean = factor.solve(&law, information.view(), true).unwrap();
        let normals = g + 2 * count * k;
        let draw = |z: &Array1<f64>| {
            let (genes, innovations) = factor.draw(&law, information.view(), z.view()).unwrap();
            latent(&genes, &innovations)
        };
        let centre = draw(&Array1::zeros(normals));
        close(
            &column(&centre),
            &solve(
                &oracle,
                &information_rhs(&law, &curvature, &oracle, &information, true),
            ),
            "draw centre",
        );
        let mut map = Array2::from_elem((dimension, normals), Bound::exact(0.0));
        for j in 0..normals {
            let mut unit = Array1::<f64>::zeros(normals);
            unit[j] = 1.0;
            let shifted = draw(&unit);
            for i in 0..dimension {
                map[[i, j]] = Bound::exact(shifted[i]).sub(&Bound::exact(centre[i]));
            }
        }
        close(&product(map.view(), map.t()), &oracle.covariance, "draw covariance");

        let expected = latent(&mean.genes, &mean.innovations);
        let point = Array1::from_shape_fn(dimension, |i| expected[i] + 0.3 * (0.9 * i as f64).cos());
        let delta: Vec<Bound> = (0..dimension)
            .map(|i| Bound::exact(point[i]).sub(&Bound::exact(expected[i])))
            .collect();
        let quadratic = sum((0..dimension).flat_map(|i| {
            let delta = &delta;
            let precision = &oracle.precision;
            (0..dimension).map(move |j| delta[i].mul(&precision[[i, j]]).mul(&delta[j]))
        }));
        let tau = Bound::exact(0.0).constant_like(std::f64::consts::TAU);
        let log_tau = tau.compose_unary([tau.value.ln(), tau.value.recip(), 0.0, 0.0, 0.0]);
        let direct = crate::test_support::cholesky_log_det(&oracle.precision)
            .unwrap()
            .sub(&log_tau.mul(&Bound::exact(dimension as f64)))
            .sub(&quadratic)
            .scale(0.5);
        let actual = factor
            .log_density(
                &law,
                (mean.genes.view(), mean.innovations.view()),
                (
                    point.slice(s![..g]),
                    point.slice(s![g..]).into_shape_with_order((count, k)).unwrap(),
                ),
            )
            .unwrap();
        agrees(&Bound::exact(actual), &direct, "log density");
    }

    /// Node curvatures `W_n = M_n - N_n` over the running-error scalar from their
    /// summands, `M_n = A_n A_n' + diagonal I` and `N_n = sum c v v'` for the listed
    /// `(node, c, v)`, with the f64 blocks a factorization reads.
    fn split_curvature(
        count: usize,
        diagonal: f64,
        excess: &[(usize, f64, [f64; 2])],
    ) -> (Vec<Vec<numerical::Running>>, Vec<Vec<numerical::Running>>, Vec<Array2<f64>>) {
        let k = 2;
        let like = numerical::Running::exact(0.0);
        let mut curvature = Vec::with_capacity(count);
        let mut dropped = Vec::with_capacity(count);
        let mut blocks = Vec::with_capacity(count);
        for n in 0..count {
            let root: Vec<numerical::Running> = (0..k * k)
                .map(|index| like.constant_like(0.5 * ((n + 3 * index) as f64).sin()))
                .collect();
            let mut metric = block_product(&root, &block_transpose(&root, k, k), k, k, k, &like);
            for a in 0..k {
                metric[a * k + a] = metric[a * k + a].add(&like.constant_like(diagonal));
            }
            let mut negative = vec![like.clone(); k * k];
            for &(node, c, direction) in excess.iter().filter(|entry| entry.0 == n) {
                let v: Vec<numerical::Running> = direction.iter().map(|&x| like.constant_like(x)).collect();
                let outer: Vec<numerical::Running> =
                    block_product(&v, &v, k, 1, k, &like).iter().map(|x| x.scale(c)).collect();
                assert_eq!(node, n);
                negative = block_sum(&negative, &outer);
            }
            let observed: Vec<numerical::Running> = metric.iter().zip(&negative).map(|(m, x)| m.sub(x)).collect();
            blocks.push(Array2::from_shape_fn((k, k), |(a, b)| observed[a * k + b].value));
            curvature.push(observed);
            dropped.push(negative);
        }
        (curvature, dropped, blocks)
    }

    fn certified_floor(certificate: &PrecisionFloor) -> Option<f64> {
        match certificate {
            PrecisionFloor::Certified { floor } => Some(*floor),
            PrecisionFloor::Refused { .. } => None,
        }
    }

    #[test]
    fn the_solve_certificate_floor_bounds_the_precision_and_its_solves() {
        let (law, _) = fixture();
        let k = law.signatures();
        let g = law.genes();
        let count = law.nodes();
        let dimension = g + count * k;
        // Node 0 carries a rank-two excess larger than its metric, so its observed
        // curvature is negative definite, and node 2 carries a rank-one excess:
        // the floor subtracts a sum over nodes and still certifies the solve.
        let (curvature, excess, blocks) = split_curvature(
            count,
            0.05,
            &[(0, 0.3, [1.0, 0.0]), (0, 0.2, [0.0, 1.0]), (2, 0.1, [0.6, 0.8])],
        );
        assert!(Cholesky::new(&blocks[0]).is_none(), "node 0 is meant to be indefinite");
        let certificate = PrecisionFloor::new(&law, &curvature, &excess).unwrap();
        let floor = certified_floor(&certificate).unwrap();

        // lambda_min(Q) >= floor: Q - floor I factors.
        let (jacobian, _, precision) = dense_precision(&law, &blocks);
        let values = precision.mapv(|q| q.value);
        let shifted = Array2::from_shape_fn((dimension, dimension), |(i, j)| {
            values[[i, j]] - if i == j { floor } else { 0.0 }
        });
        assert!(Cholesky::new(&shifted).is_some(), "lambda_min(Q) is below the certified floor {floor}");

        // S_n = (J J')_nn: the recursion over Bound against the dense jacobian.
        let bounded_gram = law.transport_gram(&Bound::exact(0.0));
        for n in 0..count {
            let rows = jacobian.slice(s![n * k..(n + 1) * k, ..]);
            close(
                &Array2::from_shape_vec((k, k), bounded_gram[n].clone()).unwrap(),
                &product(rows, rows.t()),
                "transport gram",
            );
        }

        // ||J' N J||_F <= sum_n tr(N_n S_n), on the dense product.
        let gram = law.transport_gram(&numerical::Running::exact(0.0));
        let traces: f64 = (0..count)
            .map(|n| {
                (0..k * k)
                    .map(|index| excess[n][index].value * gram[n][(index % k) * k + index / k].value)
                    .sum::<f64>()
            })
            .sum();
        let mut dense_excess = Array2::<f64>::zeros((count * k, count * k));
        for n in 0..count {
            for a in 0..k {
                for b in 0..k {
                    dense_excess[[n * k + a, n * k + b]] = excess[n][a * k + b].value;
                }
            }
        }
        let transport = jacobian.mapv(|b| b.value);
        let pulled = transport.t().dot(&dense_excess).dot(&transport);
        let frobenius = pulled.iter().fold(0.0_f64, |acc, v| acc.hypot(*v));
        assert!(frobenius <= traces, "||J' N J||_F {frobenius} exceeds sum tr(N S) {traces}");

        // The structured solve lies within the certificate's bound of the dense
        // oracle, up to the oracle's own bars.
        let factor = Factorization::new(&law, &blocks).unwrap().unwrap();
        let rhs = Array1::from_shape_fn(dimension, |i| (1.3 * i as f64).cos());
        let (gene_solve, innovation_solve) = factor
            .solve_latent(
                &law,
                rhs.slice(s![..g]),
                rhs.slice(s![g..]).into_shape_with_order((count, k)).unwrap(),
            )
            .unwrap();
        let running: Vec<numerical::Running> = rhs.iter().map(|&v| numerical::Running::exact(v)).collect();
        let bound = certificate
            .forward_error(
                &law,
                &blocks,
                &running[..g],
                &running[g..],
                gene_solve.as_slice().unwrap(),
                innovation_solve.as_slice().unwrap(),
            )
            .unwrap();
        let oracle = crate::test_support::cholesky_forward_error(
            &precision,
            &rhs.iter().map(|&v| Bound::exact(v)).collect::<Vec<_>>(),
        )
        .unwrap();
        let computed: Vec<f64> = gene_solve.iter().chain(innovation_solve.iter()).copied().collect();
        let distance = computed.iter().zip(&oracle).fold(0.0_f64, |acc, (x, o)| acc.hypot(x - o.value));
        let oracle_bar = oracle.iter().fold(0.0_f64, |acc, o| acc.hypot(o.rounding()));
        assert!(
            distance <= bound + oracle_bar,
            "structured solve {distance} from the oracle, certificate {bound}, oracle bar {oracle_bar}"
        );
    }

    #[test]
    fn the_solve_certificate_floor_is_at_most_the_innovation_prior() {
        // With zero curvature and Lambda_mm = 4 I, Q = diag(4 I, I) has
        // lambda_min(Q) = 1 exactly, so the floor must not exceed 1.
        let (mut law, _) = fixture();
        law.gene_precision = Array2::eye(law.genes()) * 4.0;
        let k = law.signatures();
        let zero = numerical::Running::exact(0.0);
        let blocks = vec![vec![zero.clone(); k * k]; law.nodes()];
        let floor = certified_floor(&PrecisionFloor::new(&law, &blocks, &blocks).unwrap()).unwrap();
        assert!(floor > 0.0 && floor <= 1.0, "floor {floor} with lambda_min(Q) = 1");
    }

    #[test]
    fn the_solve_certificate_covers_an_injected_error_and_charges_an_exact_residual() {
        let (law, _) = fixture();
        let k = law.signatures();
        let g = law.genes();
        let count = law.nodes();
        // Node 0's rank-two excess makes its curvature negative definite, so Q has
        // directions it shrinks below the innovation prior.
        let (curvature, excess, blocks) =
            split_curvature(count, 0.05, &[(0, 0.3, [1.0, 0.0]), (0, 0.2, [0.0, 1.0])]);
        let certificate = PrecisionFloor::new(&law, &curvature, &excess).unwrap();
        assert!(certified_floor(&certificate).is_some());
        let exact = |values: &[f64]| -> Vec<numerical::Running> {
            values.iter().map(|&v| numerical::Running::exact(v)).collect()
        };
        let genes: Vec<f64> = (0..g).map(|j| 0.5 * j as f64 - 0.25).collect();
        let innovations: Vec<f64> = (0..count * k).map(|i| (i % 5) as f64 * 0.5 - 1.0).collect();
        let (rhs_genes, rhs_innovations) = law.precision_product(
            &blocks,
            &exact(&genes),
            &exact(&innovations),
            &numerical::Running::exact(0.0),
        );
        // x^ solves the computed b^ exactly: the bound is the right-hand side's
        // rounding, never zero.
        let exact_residual = certificate
            .forward_error(&law, &blocks, &rhs_genes, &rhs_innovations, &genes, &innovations)
            .unwrap();
        assert!(
            exact_residual > 0.0,
            "an exact residual gave a zero bound despite a rounded right-hand side"
        );
        // An injected error along the dense precision's lowest direction, found by
        // inverse iteration on the dense factor, is covered:
        // ||Q delta|| >= lambda_min ||delta|| >= floor ||delta||. The premise
        // ||Q delta|| < ||delta|| makes the division by the floor load-bearing.
        let (_, _, precision) = dense_precision(&law, &blocks);
        let values = precision.mapv(|q| q.value);
        let dense = Cholesky::new(&values).unwrap();
        let dimension = g + count * k;
        let norm = |v: &Array1<f64>| v.iter().fold(0.0_f64, |acc, x| acc.hypot(*x));
        let mut direction = Array1::from_shape_fn(dimension, |i| 1.0 + 0.1 * i as f64);
        for _ in 0..dimension {
            direction = dense.solve_vector(direction.view());
            let length = norm(&direction);
            direction.mapv_inplace(|v| 0.25 * v / length);
        }
        let stretched = norm(&values.dot(&direction));
        assert!(
            stretched < norm(&direction),
            "the fixture needs a direction Q shrinks: ||Q delta|| {stretched} vs ||delta|| {}",
            norm(&direction)
        );
        let moved_genes: Vec<f64> = (0..g).map(|j| genes[j] + direction[j]).collect();
        let moved_innovations: Vec<f64> = (0..count * k).map(|i| innovations[i] + direction[g + i]).collect();
        let injected = certificate
            .forward_error(&law, &blocks, &rhs_genes, &rhs_innovations, &moved_genes, &moved_innovations)
            .unwrap();
        assert!(
            injected >= norm(&direction),
            "the certificate bound {injected} misses an injected error of {}",
            norm(&direction)
        );
    }

    #[test]
    fn a_dominant_negative_curvature_refuses_the_solve_certificate_at_its_node() {
        // A measurement in its tails mid-trajectory, at node 2, contributes a large
        // c l l'; smaller excesses sit at an earlier node and at the last one, so
        // the refusal must name the dominant node, not the first or last excess.
        let (law, _) = fixture();
        let (curvature, excess, blocks) = split_curvature(
            law.nodes(),
            0.05,
            &[(1, 0.1, [0.0, 1.0]), (2, 40.0, [0.6, 0.8]), (4, 0.2, [1.0, 0.0])],
        );
        let certificate = PrecisionFloor::new(&law, &curvature, &excess).unwrap();
        assert!(
            matches!(certificate, PrecisionFloor::Refused { node: 2, .. }),
            "the certificate did not refuse at node 2"
        );
        let running = vec![numerical::Running::exact(1.0); law.nodes() * law.signatures()];
        let genes = vec![numerical::Running::exact(1.0); law.genes()];
        assert!(
            certificate
                .forward_error(
                    &law,
                    &blocks,
                    &genes,
                    &running,
                    &vec![0.0; law.genes()],
                    &vec![0.0; running.len()],
                )
                .is_err()
        );
    }

    #[test]
    fn an_indefinite_genetic_border_with_a_positive_determinant_refuses_the_certificate() {
        // Two negative eigenvalues give a positive determinant, so the determinant
        // floor would "certify" this border if it were taken without a definiteness
        // proof.
        let (mut law, _) = fixture();
        let border = ndarray::arr2(&[[-1.0, 0.2, 0.0], [0.2, -2.0, 0.1], [0.0, 0.1, 3.0]]);
        let determinant = border[[0, 0]] * (border[[1, 1]] * border[[2, 2]] - border[[1, 2]] * border[[2, 1]])
            - border[[0, 1]] * (border[[1, 0]] * border[[2, 2]] - border[[1, 2]] * border[[2, 0]])
            + border[[0, 2]] * (border[[1, 0]] * border[[2, 1]] - border[[1, 1]] * border[[2, 0]]);
        assert!(determinant > 0.0, "the fixture must have a positive determinant, got {determinant}");
        law.gene_mean = ndarray::arr1(&[0.1, 0.2, 0.3]);
        law.gene_precision = border;
        let k = law.signatures();
        let loading = |seed: usize| Array2::from_shape_fn((k, 3), |(i, j)| 0.1 * ((seed + i + 2 * j) as f64).cos());
        law.entry.genes = loading(0);
        for (n, step) in law.steps.iter_mut().enumerate() {
            step.genes = loading(n + 1);
        }
        // The determinant form is gated: the first pivot is not certified positive.
        let precision: Vec<numerical::Running> =
            law.gene_precision.iter().map(|&v| numerical::Running::exact(v)).collect();
        assert!(
            PrecisionFloor::determinant_floor(&precision, 3).is_none(),
            "the determinant form was taken on an indefinite border"
        );
        // The Gershgorin floor is negative too, so the border is refused.
        let (curvature, excess, _) = split_curvature(law.nodes(), 0.05, &[]);
        let result = PrecisionFloor::new(&law, &curvature, &excess);
        assert!(result.is_err(), "the certificate certified an indefinite genetic border");
        if let Err(error) = result {
            assert!(
                format!("{error:?}").contains("not certified positive definite"),
                "the certificate refused for another reason: {error:?}"
            );
        }
    }

    #[test]
    fn an_indefinite_conditional_precision_is_reported_not_damped() {
        let (law, mut curvature) = fixture();
        assert!(Factorization::new(&law, &curvature).unwrap().is_some());
        curvature[2] = ndarray::arr2(&[[-40.0, 0.0], [0.0, 0.5]]);
        let oracle = dense_precision(&law, &curvature).2.mapv(|q| q.value);
        assert!(Cholesky::new(&oracle).is_none());
        assert!(Factorization::new(&law, &curvature).unwrap().is_none());
    }

    #[test]
    fn the_propagated_root_is_triangular_and_refuses_a_rank_deficient_block() {
        // C C' = M M' is checked through the dense comparison, whose filter
        // propagates every predicted root this way. Householder QR's stated
        // backward error carries an unspecified constant (Higham, chapter 19),
        // so no direct bar is derived here.
        let columns = ndarray::arr2(&[[0.3, -1.2, 0.0, 0.7], [2.0, 0.1, 0.0, -0.4]]);
        let root = Cholesky::from_columns(&columns).unwrap();
        assert!(root.lower[[0, 1]] == 0.0 && root.lower[[0, 0]] > 0.0 && root.lower[[1, 1]] > 0.0);
        // An exactly rank-deficient root is refused rather than divided by.
        assert!(Cholesky::from_columns(&ndarray::arr2(&[[1.0, 0.0], [0.0, 0.0]])).is_none());
    }
}
