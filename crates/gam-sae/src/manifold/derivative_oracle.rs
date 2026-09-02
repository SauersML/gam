use super::dual::DualKinkBranchRecord;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MajorizerAnchorMode {
    FrozenAnchor,
    ReanchoredObject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivativeTraceChannel {
    Tt,
    Border,
    Beta,
    Majorizer,
    Prior,
    Other(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PivotBranch {
    Missing,
    Positive,
    NonPositive,
    NonFinite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EigenDerivativeRoute {
    IndividualEigenpairs,
    InvariantSubspaceBlock,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EigenGapCertificate {
    pub min_eigen_gap: f64,
    pub threshold: f64,
    pub scale: f64,
}

pub fn eigen_gap_threshold(eigen_scale: f64, eigen_count: usize) -> f64 {
    f64::EPSILON * (eigen_count.max(1) as f64) * eigen_scale.abs().max(1.0)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BranchCertificate {
    pub anchor_mode: MajorizerAnchorMode,
    pub row_dims: Vec<usize>,
    pub row_offsets: Vec<usize>,
    pub beta_dim: usize,
    pub manifold_mode_fingerprint: u64,
    pub row_hessian_fingerprint: u64,
    pub solver_mode: String,
    pub deflated_rank: usize,
    pub deflated_per_row: Vec<usize>,
    pub spectral_deflated_rows: Vec<bool>,
    pub min_row_pivot_branch: PivotBranch,
    pub min_schur_pivot_branch: PivotBranch,
    pub min_pivot_branch: PivotBranch,
    pub max_pivot_branch: PivotBranch,
    pub min_eigen_gap: f64,
    pub eigen_gap_threshold: f64,
    pub kink_branches: Vec<DualKinkBranchRecord>,
}

impl BranchCertificate {

}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BranchCertificateRefusal {
    BranchChanged,
    UnresolvedInvariantSubspaceBlock,
}

#[derive(Clone, Debug)]
pub struct BranchCertificateMismatch {
    pub refusal: BranchCertificateRefusal,
    pub changed_fields: Vec<String>,
    pub baseline: BranchCertificate,
    pub probe: BranchCertificate,
}

impl std::fmt::Display for BranchCertificateMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.refusal {
            BranchCertificateRefusal::BranchChanged => write!(
                f,
                "derivative oracle branch changed in fields {:?}",
                self.changed_fields
            ),
            BranchCertificateRefusal::UnresolvedInvariantSubspaceBlock => write!(
                f,
                "derivative oracle refuses unresolved invariant-subspace eigen branch: \
                 min_eigen_gap={} threshold={} fields {:?}",
                self.probe.min_eigen_gap, self.probe.eigen_gap_threshold, self.changed_fields
            ),
        }
    }
}

impl std::error::Error for BranchCertificateMismatch {}

#[derive(Clone, Debug)]
pub struct ExactTraceChannel {
    pub channel: DerivativeTraceChannel,
    pub value: f64,
    pub derivative: f64,
    pub certificate: BranchCertificate,
}

#[derive(Clone, Debug)]
pub struct ExactTraceReport {
    pub certificate: BranchCertificate,
    pub channels: Vec<ExactTraceChannel>,
    pub total_value: f64,
    pub total_derivative: f64,
}

