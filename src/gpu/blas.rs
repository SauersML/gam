use super::cpu_traits::DeviceBlas;
use super::profile::{OperationKind, record_cpu_fallback};
use ndarray::{Array1, Array2};

#[derive(Clone, Debug, Default)]
pub struct CudaBlas;

impl DeviceBlas for CudaBlas {
    fn gemv(&self, a: &Array2<f64>, x: &Array1<f64>) -> Option<Array1<f64>> {
        record_cpu_fallback(
            "gpu.blas.gemv",
            OperationKind::Gemv,
            a.nrows(),
            a.ncols(),
            1,
            0,
        );
        let _ = x;
        None
    }

    fn gemv_transpose(&self, a: &Array2<f64>, x: &Array1<f64>) -> Option<Array1<f64>> {
        record_cpu_fallback(
            "gpu.blas.gemv_transpose",
            OperationKind::GemvTranspose,
            a.nrows(),
            a.ncols(),
            1,
            0,
        );
        let _ = x;
        None
    }

    fn xt_diag_x_signed(&self, x: &Array2<f64>, w: &Array1<f64>) -> Option<Array2<f64>> {
        record_cpu_fallback(
            "gpu.blas.xt_diag_x_signed",
            OperationKind::XtDiagX,
            x.nrows(),
            x.ncols(),
            x.ncols(),
            0,
        );
        let _ = w;
        None
    }

    fn xt_diag_y_signed(
        &self,
        x: &Array2<f64>,
        w: &Array1<f64>,
        y: &Array2<f64>,
    ) -> Option<Array2<f64>> {
        record_cpu_fallback(
            "gpu.blas.xt_diag_y_signed",
            OperationKind::XtDiagY,
            x.nrows(),
            x.ncols(),
            y.ncols(),
            0,
        );
        let _ = w;
        None
    }
}

pub fn try_fast_av(a: &Array2<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
    CudaBlas.gemv(a, v)
}

pub fn try_fast_atv(a: &Array2<f64>, v: &Array1<f64>) -> Option<Array1<f64>> {
    CudaBlas.gemv_transpose(a, v)
}

pub fn try_fast_xt_diag_x(x: &Array2<f64>, w: &Array1<f64>) -> Option<Array2<f64>> {
    CudaBlas.xt_diag_x_signed(x, w)
}

pub fn try_fast_xt_diag_y(
    x: &Array2<f64>,
    w: &Array1<f64>,
    y: &Array2<f64>,
) -> Option<Array2<f64>> {
    CudaBlas.xt_diag_y_signed(x, w, y)
}
