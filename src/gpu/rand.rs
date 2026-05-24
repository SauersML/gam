#[derive(Clone, Debug)]
pub struct RademacherGenerator {
    pub seed: u64,
}

impl Default for RademacherGenerator {
    fn default() -> Self {
        Self { seed: 0xCAFE_BABE }
    }
}

impl RademacherGenerator {
    pub fn fill_host(&self, out: &mut [f64]) {
        let mut state = self.seed;
        for value in out {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = if state & 1 == 0 { -1.0 } else { 1.0 };
        }
    }
}
