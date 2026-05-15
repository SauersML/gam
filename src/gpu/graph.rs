#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphMode {
    Auto,
    Off,
    Force,
}

#[derive(Clone, Debug, Default)]
pub struct CudaGraphCache {
    pub captured_iterations: usize,
    pub stable_shape: Option<(usize, usize)>,
}

impl CudaGraphCache {
    pub fn should_capture_after_iteration(&self, iter: usize) -> bool {
        iter >= 2 && self.stable_shape.is_some()
    }
}
