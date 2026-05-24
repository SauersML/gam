use num_traits::{Float, FromPrimitive};

#[derive(Clone, Debug)]
pub(crate) struct RunningVariance<S> {
    dim: usize,
    n: usize,
    mean: Vec<S>,
    m2: Vec<S>,
}

impl<S> RunningVariance<S>
where
    S: Float + FromPrimitive,
{
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            n: 0,
            mean: vec![S::zero(); dim],
            m2: vec![S::zero(); dim],
        }
    }

    pub(crate) fn update(&mut self, x: &[S]) {
        assert_eq!(x.len(), self.dim, "running variance update shape mismatch");
        self.n += 1;
        let n_s = S::from_usize(self.n).unwrap();
        for (i, &value) in x.iter().enumerate().take(self.dim) {
            let delta = value - self.mean[i];
            self.mean[i] = self.mean[i] + delta / n_s;
            let delta2 = value - self.mean[i];
            self.m2[i] = self.m2[i] + delta * delta2;
        }
    }

    pub(crate) fn update_batch(&mut self, flat: &[S]) {
        assert_eq!(
            flat.len() % self.dim,
            0,
            "running variance batch update shape mismatch"
        );
        for chunk in flat.chunks(self.dim) {
            self.update(chunk);
        }
    }

    pub(crate) fn sample_count(&self) -> usize {
        self.n
    }

    pub(crate) fn regularized_variance(&self, regularize: S, jitter: S) -> Option<Vec<S>> {
        if self.n < 2 {
            return None;
        }
        let n_denom = S::from_usize(self.n - 1).unwrap();
        let one_minus_reg = S::one() - regularize;
        Some(
            self.m2
                .iter()
                .map(|&m2| ((one_minus_reg * (m2 / n_denom)) + regularize).max(jitter))
                .collect(),
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DiagMass<S> {
    inv: Vec<S>,
    sqrt: Vec<S>,
}

impl<S> DiagMass<S>
where
    S: Float + FromPrimitive,
{
    pub(crate) fn identity(dim: usize) -> Self {
        Self {
            inv: vec![S::one(); dim],
            sqrt: vec![S::one(); dim],
        }
    }

    pub(crate) fn from_variance(mut var: Vec<S>, jitter: S) -> Self {
        let mut inv = vec![S::zero(); var.len()];
        let mut sqrt = vec![S::zero(); var.len()];
        for i in 0..var.len() {
            let v = var[i].max(jitter);
            var[i] = v;
            inv[i] = S::one() / v;
            sqrt[i] = v.sqrt();
        }
        Self { inv, sqrt }
    }

    pub(crate) fn inv(&self) -> &[S] {
        &self.inv
    }

    pub(crate) fn sqrt(&self) -> &[S] {
        &self.sqrt
    }
}
