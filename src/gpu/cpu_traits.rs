use ndarray::{Array1, Array2};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
    Auto,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLocation {
    Host,
    Device,
    Unified,
}

pub trait DeviceBlas {
    fn gemv(&self, a: &Array2<f64>, x: &Array1<f64>) -> Option<Array1<f64>>;
    fn gemv_transpose(&self, a: &Array2<f64>, x: &Array1<f64>) -> Option<Array1<f64>>;
    fn xt_diag_x_signed(&self, x: &Array2<f64>, w: &Array1<f64>) -> Option<Array2<f64>>;
    fn xt_diag_y_signed(
        &self,
        x: &Array2<f64>,
        w: &Array1<f64>,
        y: &Array2<f64>,
    ) -> Option<Array2<f64>>;
}

pub trait DeviceSolver {
    fn potrf_solve(&self, h: &Array2<f64>, rhs: &Array2<f64>) -> Option<Array2<f64>>;
    fn syevd(&self, h: &Array2<f64>) -> Option<(Array1<f64>, Array2<f64>)>;
}

pub trait DeviceDesignOperator {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn location(&self) -> MatrixLocation;
}
