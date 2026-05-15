#[derive(Clone, Debug, Default)]
pub struct GpuStream {
    pub ordinal: usize,
}

#[derive(Clone, Debug, Default)]
pub struct GpuEvent {
    pub ordinal: usize,
}

#[derive(Clone, Debug)]
pub struct GpuStreamPool {
    streams: Vec<GpuStream>,
}

impl GpuStreamPool {
    #[must_use]
    pub fn new(size: usize) -> Self {
        let streams = (0..size.max(1))
            .map(|ordinal| GpuStream { ordinal })
            .collect();
        Self { streams }
    }

    #[must_use]
    pub fn get(&self, ordinal: usize) -> &GpuStream {
        &self.streams[ordinal % self.streams.len()]
    }
}
