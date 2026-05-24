#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamRole {
    Main,
    LinkKernel,
    XtWx,
    Reduction,
}

#[derive(Clone, Debug, Default)]
pub struct GpuStreamPool {
    pub streams: usize,
}

impl GpuStreamPool {
    pub fn new(streams: usize) -> Self {
        Self { streams }
    }
}
