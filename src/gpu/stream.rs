#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuStream {
    id: usize,
}

impl GpuStream {
    #[must_use]
    pub const fn default() -> Self {
        Self { id: 0 }
    }

    #[must_use]
    pub const fn id(self) -> usize {
        self.id
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuEvent;

#[derive(Clone, Debug, Default)]
pub struct StreamPool;

impl StreamPool {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    #[must_use]
    pub const fn next(&self) -> GpuStream {
        GpuStream::default()
    }
}
