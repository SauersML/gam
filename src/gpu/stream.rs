#[derive(Clone, Debug, Default)]
pub struct GpuStream {
    pub id: usize,
}

#[derive(Clone, Debug, Default)]
pub struct GpuEvent {
    pub id: usize,
}

#[derive(Clone, Debug, Default)]
pub struct GpuStreamPool {
    streams: Vec<GpuStream>,
}

impl GpuStreamPool {
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            streams: (0..count.max(1)).map(|id| GpuStream { id }).collect(),
        }
    }

    #[must_use]
    pub fn get(&self, index: usize) -> &GpuStream {
        &self.streams[index % self.streams.len()]
    }
}
