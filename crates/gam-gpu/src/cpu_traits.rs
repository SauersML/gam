#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixLocation {
    Host,
    Device,
    Unified,
}
