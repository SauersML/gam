use ndarray::{Array1, Array2};
use opt::{FirstOrderObjective, ObjectiveEvalError, SecondOrderObjective, SymmetricHessianMut};

fn approx_same_point(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(&lhs, &rhs)| (lhs - rhs).abs() <= 1e-12)
}

#[derive(Clone)]
struct CachedFirstOrderSample {
    x: Array1<f64>,
    cost: f64,
    grad: Array1<f64>,
}

pub(in crate::solver) struct CachedFirstOrderObjective<F> {
    inner: F,
    last: Option<CachedFirstOrderSample>,
}

impl<F> CachedFirstOrderObjective<F> {
    pub(in crate::solver) fn new(inner: F) -> Self {
        Self { inner, last: None }
    }

    fn evaluate(&mut self, x: &Array1<f64>) -> Result<(f64, Array1<f64>), ObjectiveEvalError>
    where
        F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), ObjectiveEvalError>,
    {
        if let Some(sample) = &self.last
            && approx_same_point(x, &sample.x)
        {
            return Ok((sample.cost, sample.grad.clone()));
        }

        let (cost, grad) = (self.inner)(x)?;
        self.last = Some(CachedFirstOrderSample {
            x: x.clone(),
            cost,
            grad: grad.clone(),
        });
        Ok((cost, grad))
    }
}

impl<F> FirstOrderObjective for CachedFirstOrderObjective<F>
where
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), ObjectiveEvalError>,
{
    fn eval(
        &mut self,
        x: &Array1<f64>,
        grad_out: &mut Array1<f64>,
    ) -> Result<f64, ObjectiveEvalError> {
        let (cost, grad) = self.evaluate(x)?;
        grad_out.assign(&grad);
        Ok(cost)
    }
}

#[derive(Clone)]
struct CachedSecondOrderSample {
    x: Array1<f64>,
    cost: f64,
    grad: Array1<f64>,
    hessian: Option<Array2<f64>>,
}

pub(in crate::solver) struct CachedSecondOrderObjective<F> {
    inner: F,
    finite_diff_step: f64,
    last: Option<CachedSecondOrderSample>,
}

impl<F> CachedSecondOrderObjective<F> {
    pub(in crate::solver) fn new(inner: F, finite_diff_step: f64) -> Self {
        Self {
            inner,
            finite_diff_step,
            last: None,
        }
    }

    fn evaluate(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), ObjectiveEvalError>
    where
        F: FnMut(
            &Array1<f64>,
        ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), ObjectiveEvalError>,
    {
        if let Some(sample) = &self.last
            && approx_same_point(x, &sample.x)
        {
            return Ok((sample.cost, sample.grad.clone(), sample.hessian.clone()));
        }

        let (cost, grad, hessian) = (self.inner)(x)?;
        self.last = Some(CachedSecondOrderSample {
            x: x.clone(),
            cost,
            grad: grad.clone(),
            hessian: hessian.clone(),
        });
        Ok((cost, grad, hessian))
    }

    fn finite_differencehessian(
        &mut self,
        x: &Array1<f64>,
    ) -> Result<Array2<f64>, ObjectiveEvalError>
    where
        F: FnMut(
            &Array1<f64>,
        ) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), ObjectiveEvalError>,
    {
        let p = x.len();
        let h = self.finite_diff_step;
        if !h.is_finite() || h <= 0.0 {
            return Err(ObjectiveEvalError::fatal(
                "finite-difference Hessian step must be positive and finite",
            ));
        }

        let mut hessian = Array2::<f64>::zeros((p, p));
        let mut xp = x.clone();
        let mut xm = x.clone();
        for j in 0..p {
            xp[j] += h;
            let (_, gp, _) = self.evaluate(&xp)?;
            xp[j] = x[j];

            xm[j] -= h;
            let (_, gm, _) = self.evaluate(&xm)?;
            xm[j] = x[j];

            let column = (&gp - &gm) / (2.0 * h);
            hessian.column_mut(j).assign(&column);
        }

        let symmetric = 0.5 * (&hessian + &hessian.t().to_owned());
        Ok(symmetric)
    }
}

impl<F> SecondOrderObjective for CachedSecondOrderObjective<F>
where
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>, Option<Array2<f64>>), ObjectiveEvalError>,
{
    fn eval_grad(
        &mut self,
        x: &Array1<f64>,
        grad_out: &mut Array1<f64>,
    ) -> Result<f64, ObjectiveEvalError> {
        let (cost, grad, _) = self.evaluate(x)?;
        grad_out.assign(&grad);
        Ok(cost)
    }

    fn eval_hessian(
        &mut self,
        x: &Array1<f64>,
        grad_out: &mut Array1<f64>,
        mut hess_out: SymmetricHessianMut<'_>,
    ) -> Result<f64, ObjectiveEvalError> {
        let (cost, grad, hessian_opt) = self.evaluate(x)?;
        grad_out.assign(&grad);
        let hessian = match hessian_opt {
            Some(hessian) => hessian,
            None => self.finite_differencehessian(x)?,
        };
        hess_out
            .assign_dense(&hessian)
            .map_err(|err| ObjectiveEvalError::fatal(err.to_string()))?;
        Ok(cost)
    }
}
