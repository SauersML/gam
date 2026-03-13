use super::*;

impl<'a> RemlState<'a> {
    pub(super) fn orthonormalize_columns(a: &Array2<f64>, tol: f64) -> Array2<f64> {
        let p = a.nrows();
        let c = a.ncols();
        let mut q = Array2::<f64>::zeros((p, c));
        let mut kept = 0usize;
        for j in 0..c {
            let mut v = a.column(j).to_owned();
            for t in 0..kept {
                let qt = q.column(t);
                let proj = qt.dot(&v);
                v -= &qt.mapv(|x| x * proj);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > tol {
                q.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                kept += 1;
            }
        }
        if kept == c {
            q
        } else {
            q.slice(ndarray::s![.., 0..kept]).to_owned()
        }
    }
}
