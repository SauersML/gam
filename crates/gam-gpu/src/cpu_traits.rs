#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLocation {
    Host,
    Device,
    Unified,
}
