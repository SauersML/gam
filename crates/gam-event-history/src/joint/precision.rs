//! Block-tridiagonal state precision with a dense genetic border.
use super::numerical;
use crate::EventHistoryError;
use ndarray::Array2;

#[derive(Clone)]
pub(super) struct Precision {
    pub diagonal: Vec<Array2<f64>>,
    pub lower: Vec<Array2<f64>>,
    pub border: Vec<Array2<f64>>,
    pub corner: Array2<f64>,
    pub signatures: usize,
    pub genes: usize,
}

impl Precision {
    pub fn new(nodes: usize, signatures: usize, genes: usize) -> Self {
        Self {
            diagonal: vec![Array2::zeros((signatures, signatures)); nodes],
            lower: vec![Array2::zeros((signatures, signatures)); nodes.saturating_sub(1)],
            border: vec![Array2::zeros((signatures, genes)); nodes],
            corner: Array2::zeros((genes, genes)),
            signatures,
            genes,
        }
    }

    pub fn dimension(&self) -> usize {
        self.genes + self.diagonal.len() * self.signatures
    }

    /// The upper temporal triangle and upper genetic-border triangle are
    /// represented by their stored transpose; all other entries are stored.
    pub fn add(&mut self, i: usize, j: usize, value: f64) -> Result<(), EventHistoryError> {
        if i < self.genes && j < self.genes {
            self.corner[[i, j]] += value;
        } else if i >= self.genes && j < self.genes {
            let node = (i - self.genes) / self.signatures;
            let axis = (i - self.genes) % self.signatures;
            self.border[node][[axis, j]] += value;
        } else if i >= self.genes && j >= self.genes {
            let ni = (i - self.genes) / self.signatures;
            let nj = (j - self.genes) / self.signatures;
            let ai = (i - self.genes) % self.signatures;
            let aj = (j - self.genes) % self.signatures;
            if ni == nj {
                self.diagonal[ni][[ai, aj]] += value;
            } else if ni == nj + 1 {
                self.lower[nj][[ai, aj]] += value;
            } else if nj != ni + 1 {
                return Err(numerical(
                    "joint prior has a nonlocal temporal precision entry",
                ));
            }
        }
        Ok(())
    }

    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        for i in 0..self.genes {
            for j in 0..self.genes {
                out[i] += self.corner[[i, j]] * x[j];
            }
        }
        for n in 0..self.diagonal.len() {
            let base = self.genes + n * self.signatures;
            for i in 0..self.signatures {
                for j in 0..self.signatures {
                    out[base + i] += self.diagonal[n][[i, j]] * x[base + j];
                    if n > 0 {
                        out[base + i] += self.lower[n - 1][[i, j]] * x[base - self.signatures + j];
                        out[base - self.signatures + j] += self.lower[n - 1][[i, j]] * x[base + i];
                    }
                }
                for j in 0..self.genes {
                    out[base + i] += self.border[n][[i, j]] * x[j];
                    out[j] += self.border[n][[i, j]] * x[base + i];
                }
            }
        }
        out
    }

    pub fn damping_scale(&self) -> f64 {
        self.diagonal
            .iter()
            .chain(std::iter::once(&self.corner))
            .flat_map(|m| m.diag().to_vec())
            .fold(1.0_f64, |a, b| a.max(b.abs()))
    }

    pub fn add_damping(&mut self, amount: f64) {
        for matrix in self
            .diagonal
            .iter_mut()
            .chain(std::iter::once(&mut self.corner))
        {
            for i in 0..matrix.nrows() {
                matrix[[i, i]] += amount;
            }
        }
    }

    pub fn stored_entries(&self) -> usize {
        self.diagonal
            .iter()
            .chain(&self.lower)
            .chain(&self.border)
            .map(|a| a.len())
            .sum::<usize>()
            + self.corner.len()
    }
}

struct Cholesky {
    lower: Array2<f64>,
    log_determinant: f64,
}

impl Cholesky {
    fn new(matrix: &Array2<f64>) -> Result<Self, EventHistoryError> {
        let n = matrix.nrows();
        let mut lower = Array2::<f64>::zeros((n, n));
        let mut log_determinant = 0.0;
        for i in 0..n {
            for j in 0..=i {
                let value =
                    matrix[[i, j]] - (0..j).map(|k| lower[[i, k]] * lower[[j, k]]).sum::<f64>();
                if i == j {
                    if !value.is_finite() || value <= 0.0 {
                        return Err(numerical("joint precision is not positive definite"));
                    }
                    lower[[i, j]] = value.sqrt();
                    log_determinant += value.ln();
                } else {
                    lower[[i, j]] = value / lower[[j, j]];
                }
            }
        }
        Ok(Self {
            lower,
            log_determinant,
        })
    }

    fn solve(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let n = self.lower.nrows();
        let mut out = rhs.clone();
        for col in 0..out.ncols() {
            for i in 0..n {
                for j in 0..i {
                    out[[i, col]] -= self.lower[[i, j]] * out[[j, col]];
                }
                out[[i, col]] /= self.lower[[i, i]];
            }
            for i in (0..n).rev() {
                for j in i + 1..n {
                    out[[i, col]] -= self.lower[[j, i]] * out[[j, col]];
                }
                out[[i, col]] /= self.lower[[i, i]];
            }
        }
        out
    }

    /// L^{-T} z has covariance (L L')^{-1} for standard normal z.
    fn whiten(&self, z: &[f64]) -> Vec<f64> {
        let mut out = z.to_vec();
        for i in (0..out.len()).rev() {
            for j in i + 1..out.len() {
                out[i] -= self.lower[[j, i]] * out[j];
            }
            out[i] /= self.lower[[i, i]];
        }
        out
    }
}

pub(super) struct Factorization {
    chain: Vec<Cholesky>,
    lower: Vec<Array2<f64>>,
    border: Vec<Array2<f64>>,
    response: Vec<Array2<f64>>,
    schur: Cholesky,
    genes: usize,
    signatures: usize,
    pub log_determinant: f64,
}

impl Factorization {
    pub fn new(p: &Precision) -> Result<Self, EventHistoryError> {
        let mut chain: Vec<Cholesky> = Vec::with_capacity(p.diagonal.len());
        let mut log_determinant = 0.0;
        for n in 0..p.diagonal.len() {
            let mut diagonal = p.diagonal[n].clone();
            if n > 0 {
                let solved = chain[n - 1].solve(&p.lower[n - 1].t().to_owned());
                diagonal -= &p.lower[n - 1].dot(&solved);
            }
            let factor = Cholesky::new(&diagonal)?;
            log_determinant += factor.log_determinant;
            chain.push(factor);
        }
        let response = Self::chain_solve(&chain, &p.lower, &p.border);
        let mut schur = p.corner.clone();
        for (border, response) in p.border.iter().zip(&response) {
            schur -= &border.t().dot(response);
        }
        let schur = Cholesky::new(&schur)?;
        log_determinant += schur.log_determinant;
        Ok(Self {
            chain,
            lower: p.lower.clone(),
            border: p.border.clone(),
            response,
            schur,
            genes: p.genes,
            signatures: p.signatures,
            log_determinant,
        })
    }

    fn chain_solve(
        chain: &[Cholesky],
        lower: &[Array2<f64>],
        rhs: &[Array2<f64>],
    ) -> Vec<Array2<f64>> {
        let mut out = rhs.to_vec();
        for n in 1..out.len() {
            let previous = chain[n - 1].solve(&out[n - 1]);
            out[n] -= &lower[n - 1].dot(&previous);
        }
        for n in (0..out.len()).rev() {
            if n + 1 < out.len() {
                let correction = lower[n].t().dot(&out[n + 1]);
                out[n] -= &correction;
            }
            out[n] = chain[n].solve(&out[n]);
        }
        out
    }

    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let state_rhs: Vec<Array2<f64>> = (0..self.chain.len())
            .map(|n| {
                let base = self.genes + n * self.signatures;
                Array2::from_shape_fn((self.signatures, 1), |(i, _)| rhs[base + i])
            })
            .collect();
        let state = Self::chain_solve(&self.chain, &self.lower, &state_rhs);
        let mut gene_rhs = Array2::from_shape_fn((self.genes, 1), |(i, _)| rhs[i]);
        for (border, value) in self.border.iter().zip(&state) {
            gene_rhs -= &border.t().dot(value);
        }
        let genes = self.schur.solve(&gene_rhs);
        let mut out: Vec<f64> = genes.column(0).to_vec();
        for (value, response) in state.iter().zip(&self.response) {
            out.extend((value - &response.dot(&genes)).column(0).iter().copied());
        }
        out
    }

    /// Map independent standard normals to a centered joint draw. Conditional
    /// state simulation runs backwards through the Markov blocks; the genetic
    /// Schur draw induces the state/genetic dependence via -A^{-1} C g.
    pub fn gaussian_draw(&self, z: &[f64]) -> Vec<f64> {
        assert_eq!(z.len(), self.genes + self.chain.len() * self.signatures);
        let genes = self.schur.whiten(&z[..self.genes]);
        let mut states = vec![Array2::zeros((self.signatures, 1)); self.chain.len()];
        for n in (0..self.chain.len()).rev() {
            let base = self.genes + n * self.signatures;
            let noise = self.chain[n].whiten(&z[base..base + self.signatures]);
            states[n] = Array2::from_shape_fn((self.signatures, 1), |(i, _)| noise[i]);
            if n + 1 < self.chain.len() {
                let correction = self.chain[n].solve(&self.lower[n].t().dot(&states[n + 1]));
                states[n] -= &correction;
            }
        }
        let g = Array2::from_shape_fn((self.genes, 1), |(i, _)| genes[i]);
        let mut out = genes;
        for (state, response) in states.iter().zip(&self.response) {
            out.extend((state - &response.dot(&g)).column(0).iter().copied());
        }
        out
    }

    /// Selected covariance blocks, never a dense trajectory covariance.
    pub fn covariance(&self) -> (Vec<Array2<f64>>, Array2<f64>, Vec<Array2<f64>>) {
        let genes = self.schur.solve(&Array2::eye(self.genes));
        let mut states = vec![Array2::zeros((self.signatures, self.signatures)); self.chain.len()];
        for n in (0..self.chain.len()).rev() {
            let inverse = self.chain[n].solve(&Array2::eye(self.signatures));
            let mut covariance = inverse.clone();
            if n + 1 < self.chain.len() {
                let response = inverse.dot(&self.lower[n].t());
                covariance += &response.dot(&states[n + 1]).dot(&response.t());
            }
            states[n] = covariance;
        }
        let mut cross = Vec::with_capacity(states.len());
        for (state, response) in states.iter_mut().zip(&self.response) {
            let response_covariance = response.dot(&genes);
            *state += &response_covariance.dot(&response.t());
            cross.push(-response_covariance);
        }
        (states, genes, cross)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_draw_map_has_the_inverse_precision_covariance() {
        let mut p = Precision::new(4, 2, 2);
        p.corner = Array2::from_shape_vec((2, 2), vec![3.0, 0.4, 0.4, 2.0]).unwrap();
        for n in 0..4 {
            p.diagonal[n] = Array2::from_shape_vec((2, 2), vec![2.0, 0.3, 0.3, 1.7]).unwrap();
            p.border[n] = Array2::from_shape_fn((2, 2), |(i, j)| 0.03 * (n + 1 + i + 2 * j) as f64);
            if n > 0 {
                p.lower[n - 1] =
                    Array2::from_shape_vec((2, 2), vec![-0.4, 0.1, 0.05, -0.3]).unwrap();
            }
        }
        let f = Factorization::new(&p).unwrap();
        let dim = p.dimension();
        let mut draw_map = Array2::zeros((dim, dim));
        for j in 0..dim {
            let mut unit = vec![0.0; dim];
            unit[j] = 1.0;
            let draw = f.gaussian_draw(&unit);
            for i in 0..dim {
                draw_map[[i, j]] = draw[i];
            }
        }
        let covariance = draw_map.dot(&draw_map.t());
        for j in 0..dim {
            let applied = p.apply(&covariance.column(j).to_vec());
            for i in 0..dim {
                assert!(
                    (applied[i] - f64::from(i == j)).abs() < 1e-13,
                    "draw covariance {i},{j}"
                );
            }
        }
        let (states, genes, cross) = f.covariance();
        for i in 0..2 {
            for j in 0..2 {
                assert!((genes[[i, j]] - covariance[[i, j]]).abs() < 1e-13);
                for n in 0..4 {
                    let base = 2 + n * 2;
                    assert!((states[n][[i, j]] - covariance[[base + i, base + j]]).abs() < 1e-13);
                    assert!((cross[n][[i, j]] - covariance[[base + i, j]]).abs() < 1e-13);
                }
            }
        }
    }
}
