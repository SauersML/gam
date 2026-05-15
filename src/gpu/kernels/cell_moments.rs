pub const GAUSS_LEGENDRE_NODES: usize = 384;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CellMomentBranch {
    RigidStandardNormal,
    RigidEmpiricalGrid,
    FlexibleQuartic,
    FlexibleSextic,
    TailLeft,
    TailRight,
}
