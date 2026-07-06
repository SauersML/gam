use super::{ArrowFactorCache, arrow_factor_max_pivot, arrow_factor_min_pivot};
use super::dual::{Dual, DualKinkBranchRecord};

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

impl PivotBranch {
    fn classify(value: Option<f64>) -> Self {
        match value {
            None => Self::Missing,
            Some(v) if !v.is_finite() => Self::NonFinite,
            Some(v) if v > 0.0 => Self::Positive,
            Some(_) => Self::NonPositive,
        }
    }
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

pub fn eigen_gap_certificate(eigenvalues: &[f64]) -> EigenGapCertificate {
    let mut finite = eigenvalues
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    finite.sort_by(|left, right| left.total_cmp(right));
    let scale = finite
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let mut min_eigen_gap = f64::INFINITY;
    for pair in finite.windows(2) {
        min_eigen_gap = min_eigen_gap.min((pair[1] - pair[0]).abs());
    }
    EigenGapCertificate {
        min_eigen_gap,
        threshold: eigen_gap_threshold(scale, finite.len()),
        scale,
    }
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
    pub cross_row_woodbury_rank: usize,
    pub min_row_pivot_branch: PivotBranch,
    pub min_schur_pivot_branch: PivotBranch,
    pub min_pivot_branch: PivotBranch,
    pub max_pivot_branch: PivotBranch,
    pub min_eigen_gap: f64,
    pub eigen_gap_threshold: f64,
    pub kink_branches: Vec<DualKinkBranchRecord>,
}

impl BranchCertificate {
    pub fn from_arrow_cache(cache: &ArrowFactorCache, anchor_mode: MajorizerAnchorMode) -> Self {
        let min_pivot = arrow_factor_min_pivot(cache);
        let max_pivot = arrow_factor_max_pivot(cache);
        let eigen_gap = Self::eigen_gap_from_arrow_cache(cache);
        Self {
            anchor_mode,
            row_dims: cache.row_dims.to_vec(),
            row_offsets: cache.row_offsets.to_vec(),
            beta_dim: cache.k,
            manifold_mode_fingerprint: cache.manifold_mode_fingerprint,
            row_hessian_fingerprint: cache.row_hessian_fingerprint,
            solver_mode: format!("{:?}", cache.solver_mode),
            deflated_rank: cache.gauge_deflated_directions,
            deflated_per_row: cache
                .deflated_row_directions
                .iter()
                .map(Vec::len)
                .collect(),
            spectral_deflated_rows: cache
                .deflation_row_spectra
                .iter()
                .map(Option::is_some)
                .collect(),
            cross_row_woodbury_rank: cache
                .cross_row_woodbury
                .as_ref()
                .map(|woodbury| woodbury.d.len())
                .unwrap_or(0),
            min_row_pivot_branch: PivotBranch::classify(min_pivot.min_row_pivot),
            min_schur_pivot_branch: PivotBranch::classify(min_pivot.min_schur_pivot),
            min_pivot_branch: PivotBranch::classify(min_pivot.min_pivot),
            max_pivot_branch: PivotBranch::classify(max_pivot),
            min_eigen_gap: eigen_gap.min_eigen_gap,
            eigen_gap_threshold: eigen_gap.threshold,
            kink_branches: Vec::new(),
        }
    }

    fn eigen_gap_from_arrow_cache(cache: &ArrowFactorCache) -> EigenGapCertificate {
        let mut min_eigen_gap = f64::INFINITY;
        let mut max_scale = 0.0_f64;
        let mut max_count = 0_usize;
        for spectrum in cache.deflation_row_spectra.iter().flatten() {
            let raw = spectrum
                .raw_evals
                .as_slice()
                .expect("row deflation spectrum eigenvalues are contiguous");
            let gap = eigen_gap_certificate(raw);
            min_eigen_gap = min_eigen_gap.min(gap.min_eigen_gap);
            max_scale = max_scale.max(gap.scale);
            max_count = max_count.max(raw.len());
        }
        EigenGapCertificate {
            min_eigen_gap,
            threshold: eigen_gap_threshold(max_scale, max_count),
            scale: max_scale,
        }
    }

    pub fn with_eigen_gap(mut self, gap: EigenGapCertificate) -> Self {
        self.min_eigen_gap = gap.min_eigen_gap;
        self.eigen_gap_threshold = gap.threshold;
        self
    }

    pub fn record_kink_branch(&mut self, record: DualKinkBranchRecord) {
        self.kink_branches.push(record);
    }

    /// Route derivatives through individual eigenpairs only when the spectral
    /// separation is resolved above eigensolver round-off. At degeneracy this
    /// module records an unresolved invariant-subspace block and refuses scalar
    /// derivative reporting. It does not implement the confluent
    /// Daleckii-Krein/Sylvester block derivative.
    pub fn eigen_derivative_route(&self) -> EigenDerivativeRoute {
        if self.min_eigen_gap.is_finite() && self.min_eigen_gap < self.eigen_gap_threshold {
            EigenDerivativeRoute::InvariantSubspaceBlock
        } else {
            EigenDerivativeRoute::IndividualEigenpairs
        }
    }

    pub fn assert_derivative_reportable(&self) -> Result<(), BranchCertificateMismatch> {
        match self.eigen_derivative_route() {
            EigenDerivativeRoute::IndividualEigenpairs => Ok(()),
            EigenDerivativeRoute::InvariantSubspaceBlock => Err(BranchCertificateMismatch {
                refusal: BranchCertificateRefusal::UnresolvedInvariantSubspaceBlock,
                changed_fields: vec!["min_eigen_gap".to_string()],
                baseline: self.clone(),
                probe: self.clone(),
            }),
        }
    }

    pub fn assert_same_branch(&self, probe: &Self) -> Result<(), BranchCertificateMismatch> {
        let mut changed_fields = Vec::new();
        if self.anchor_mode != probe.anchor_mode {
            changed_fields.push("majorizer_anchor".to_string());
        }
        if self.row_dims != probe.row_dims {
            changed_fields.push("row_dims".to_string());
        }
        if self.row_offsets != probe.row_offsets {
            changed_fields.push("row_offsets".to_string());
        }
        if self.beta_dim != probe.beta_dim {
            changed_fields.push("beta_dim".to_string());
        }
        if self.manifold_mode_fingerprint != probe.manifold_mode_fingerprint {
            changed_fields.push("manifold_mode_fingerprint".to_string());
        }
        if self.row_hessian_fingerprint != probe.row_hessian_fingerprint {
            changed_fields.push("row_hessian_fingerprint".to_string());
        }
        if self.solver_mode != probe.solver_mode {
            changed_fields.push("solver_mode".to_string());
        }
        if self.deflated_rank != probe.deflated_rank {
            changed_fields.push("deflated_rank".to_string());
        }
        if self.deflated_per_row != probe.deflated_per_row {
            changed_fields.push("deflated_per_row".to_string());
        }
        if self.spectral_deflated_rows != probe.spectral_deflated_rows {
            changed_fields.push("spectral_deflated_rows".to_string());
        }
        if self.cross_row_woodbury_rank != probe.cross_row_woodbury_rank {
            changed_fields.push("cross_row_woodbury_rank".to_string());
        }
        if self.min_row_pivot_branch != probe.min_row_pivot_branch {
            changed_fields.push("min_row_pivot_branch".to_string());
        }
        if self.min_schur_pivot_branch != probe.min_schur_pivot_branch {
            changed_fields.push("min_schur_pivot_branch".to_string());
        }
        if self.min_pivot_branch != probe.min_pivot_branch {
            changed_fields.push("min_pivot_branch".to_string());
        }
        if self.max_pivot_branch != probe.max_pivot_branch {
            changed_fields.push("max_pivot_branch".to_string());
        }
        let baseline_eigen_route = self.eigen_derivative_route();
        let probe_eigen_route = probe.eigen_derivative_route();
        let unresolved_invariant_subspace =
            baseline_eigen_route == EigenDerivativeRoute::InvariantSubspaceBlock
                || probe_eigen_route == EigenDerivativeRoute::InvariantSubspaceBlock;
        if unresolved_invariant_subspace
            || baseline_eigen_route != probe_eigen_route
            || self.min_eigen_gap != probe.min_eigen_gap
            || self.eigen_gap_threshold != probe.eigen_gap_threshold
        {
            changed_fields.push("min_eigen_gap".to_string());
        }
        if self.kink_branches != probe.kink_branches {
            changed_fields.push("kink_branches".to_string());
        }

        if changed_fields.is_empty() {
            Ok(())
        } else {
            Err(BranchCertificateMismatch {
                refusal: if unresolved_invariant_subspace {
                    BranchCertificateRefusal::UnresolvedInvariantSubspaceBlock
                } else {
                    BranchCertificateRefusal::BranchChanged
                },
                changed_fields,
                baseline: self.clone(),
                probe: probe.clone(),
            })
        }
    }
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

impl ExactTraceReport {
    pub fn channel_derivative(&self, channel: DerivativeTraceChannel) -> Option<f64> {
        self.channels
            .iter()
            .find(|entry| entry.channel == channel)
            .map(|entry| entry.derivative)
    }
}

pub fn guarded_exact_trace_report(
    certificate: BranchCertificate,
    channels: Vec<ExactTraceChannel>,
) -> Result<ExactTraceReport, BranchCertificateMismatch> {
    certificate.assert_derivative_reportable()?;
    let mut total_value = 0.0_f64;
    let mut total_derivative = 0.0_f64;
    for channel in &channels {
        certificate.assert_same_branch(&channel.certificate)?;
        channel.certificate.assert_derivative_reportable()?;
        total_value += channel.value;
        total_derivative += channel.derivative;
    }
    Ok(ExactTraceReport {
        certificate,
        channels,
        total_value,
        total_derivative,
    })
}

pub fn dual_spd_logdet(matrix: &[Vec<Dual>]) -> Result<Dual, String> {
    let n = matrix.len();
    if n == 0 {
        return Ok(Dual::constant(0.0));
    }
    for (row, values) in matrix.iter().enumerate() {
        if values.len() != n {
            return Err(format!(
                "dual_spd_logdet: row {row} has width {}, expected {n}",
                values.len()
            ));
        }
    }

    let mut lower = vec![vec![Dual::constant(0.0); n]; n];
    for row in 0..n {
        for col in 0..=row {
            let mut sum = matrix[row][col];
            for inner in 0..col {
                sum = sum - lower[row][inner] * lower[col][inner];
            }
            if row == col {
                if !(sum.re.is_finite() && sum.re > 0.0) {
                    return Err(format!(
                        "dual_spd_logdet: non-positive branch pivot at row {row}: {}",
                        sum.re
                    ));
                }
                lower[row][col] = sum.sqrt();
            } else {
                lower[row][col] = sum / lower[col][col];
            }
        }
    }

    let mut logdet = Dual::constant(0.0);
    for (idx, row) in lower.iter().enumerate() {
        let diag = row[idx];
        if !(diag.re.is_finite() && diag.re > 0.0) {
            return Err(format!(
                "dual_spd_logdet: non-positive Cholesky diagonal at row {idx}: {}",
                diag.re
            ));
        }
        logdet = logdet + diag.ln() * 2.0;
    }
    Ok(logdet)
}

pub fn exact_logdet_channel(
    channel: DerivativeTraceChannel,
    matrix: &[Vec<Dual>],
    certificate: BranchCertificate,
) -> Result<ExactTraceChannel, String> {
    let dual = dual_spd_logdet(matrix)?;
    Ok(ExactTraceChannel {
        channel,
        value: dual.re,
        derivative: dual.eps,
        certificate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn certificate(anchor_mode: MajorizerAnchorMode) -> BranchCertificate {
        BranchCertificate {
            anchor_mode,
            row_dims: vec![2],
            row_offsets: vec![0, 2],
            beta_dim: 1,
            manifold_mode_fingerprint: 11,
            row_hessian_fingerprint: 17,
            solver_mode: "Direct".to_string(),
            deflated_rank: 0,
            deflated_per_row: vec![0],
            spectral_deflated_rows: vec![false],
            cross_row_woodbury_rank: 0,
            min_row_pivot_branch: PivotBranch::Positive,
            min_schur_pivot_branch: PivotBranch::Positive,
            min_pivot_branch: PivotBranch::Positive,
            max_pivot_branch: PivotBranch::Positive,
            min_eigen_gap: f64::INFINITY,
            eigen_gap_threshold: eigen_gap_threshold(1.0, 0),
            kink_branches: Vec::new(),
        }
    }

    #[test]
    fn branch_certificate_refuses_reanchored_majorizer_probe() {
        let baseline = certificate(MajorizerAnchorMode::FrozenAnchor);
        let probe = certificate(MajorizerAnchorMode::ReanchoredObject);
        let err = baseline
            .assert_same_branch(&probe)
            .expect_err("reanchored majorizer differentiates a different object");
        assert_eq!(err.refusal, BranchCertificateRefusal::BranchChanged);
        assert_eq!(err.changed_fields, vec!["majorizer_anchor".to_string()]);
    }

    #[test]
    fn branch_certificate_refuses_deflation_rank_change() {
        let baseline = certificate(MajorizerAnchorMode::FrozenAnchor);
        let mut probe = baseline.clone();
        probe.deflated_rank = 1;
        probe.deflated_per_row = vec![1];
        let err = baseline
            .assert_same_branch(&probe)
            .expect_err("changed deflation branch must refuse derivative report");
        assert!(err.changed_fields.iter().any(|field| field == "deflated_rank"));
        assert!(
            err.changed_fields
                .iter()
                .any(|field| field == "deflated_per_row")
        );
    }

    #[test]
    fn planted_eigen_crossing_routes_to_invariant_subspace_block_and_refuses_report() {
        let near_crossing = eigen_gap_certificate(&[2.0, 2.0]);
        let cert = certificate(MajorizerAnchorMode::FrozenAnchor).with_eigen_gap(near_crossing);
        assert_eq!(
            cert.eigen_derivative_route(),
            EigenDerivativeRoute::InvariantSubspaceBlock
        );
        let channel = ExactTraceChannel {
            channel: DerivativeTraceChannel::Other("crossing"),
            value: 0.0,
            derivative: 1.044,
            certificate: cert.clone(),
        };
        let err = guarded_exact_trace_report(cert, vec![channel])
            .expect_err("individual eigenpair derivative must be refused at a crossing");
        assert_eq!(
            err.refusal,
            BranchCertificateRefusal::UnresolvedInvariantSubspaceBlock
        );
        assert!(err.changed_fields.iter().any(|field| field == "min_eigen_gap"));
    }

    #[test]
    fn well_separated_spectrum_keeps_individual_eigenpair_route() {
        let separated = eigen_gap_certificate(&[1.0, 1.5, 3.0]);
        let cert = certificate(MajorizerAnchorMode::FrozenAnchor).with_eigen_gap(separated);
        assert_eq!(
            cert.eigen_derivative_route(),
            EigenDerivativeRoute::IndividualEigenpairs
        );
        cert.assert_derivative_reportable()
            .expect("well-separated spectrum is smooth for individual eigenpairs");
    }

    #[test]
    fn branch_certificate_refuses_same_near_degenerate_eigen_branch() {
        let near_crossing = eigen_gap_certificate(&[2.0, 2.0]);
        let cert = certificate(MajorizerAnchorMode::FrozenAnchor).with_eigen_gap(near_crossing);
        let err = cert
            .assert_same_branch(&cert)
            .expect_err("same degenerate eigenpair branch still has no scalar derivative");
        assert_eq!(
            err.refusal,
            BranchCertificateRefusal::UnresolvedInvariantSubspaceBlock
        );
        assert!(err.changed_fields.iter().any(|field| field == "min_eigen_gap"));
    }

    #[test]
    fn per_channel_dual_oracle_catches_planted_factor_two_hidden_from_total_fd() {
        let cert = certificate(MajorizerAnchorMode::FrozenAnchor);
        let tt_matrix = vec![
            vec![
                Dual::with_derivative(3.0, 3.0),
                Dual::with_derivative(0.15, 0.0),
            ],
            vec![
                Dual::with_derivative(0.15, 0.0),
                Dual::with_derivative(2.4, 0.0),
            ],
        ];
        let beta_matrix = vec![
            vec![
                Dual::with_derivative(4.0, -4.0),
                Dual::with_derivative(0.05, 0.0),
            ],
            vec![
                Dual::with_derivative(0.05, 0.0),
                Dual::with_derivative(2.1, 0.0),
            ],
        ];
        let tt = exact_logdet_channel(DerivativeTraceChannel::Tt, &tt_matrix, cert.clone())
            .expect("tt channel");
        let beta = exact_logdet_channel(DerivativeTraceChannel::Beta, &beta_matrix, cert.clone())
            .expect("beta channel");
        let report =
            guarded_exact_trace_report(cert, vec![tt, beta]).expect("same branch report");

        let beta_exact = report
            .channel_derivative(DerivativeTraceChannel::Beta)
            .expect("beta channel derivative");
        let planted_beta = 2.0 * beta_exact;
        assert!(
            (planted_beta - beta_exact).abs() > 0.1,
            "planted factor-two beta channel must be visible before total summation"
        );

        fn cancelling_total(x: f64) -> f64 {
            let tt_value = 1.0e17 + 3.0 * x;
            let beta_value = -1.0e17 - 4.0 * x;
            tt_value + beta_value
        }

        let h = 1.0e-6;
        let fd_total = (cancelling_total(h) - cancelling_total(-h)) / (2.0 * h);
        assert_eq!(
            fd_total, 0.0,
            "central FD of the cancelling total has no measurable signal"
        );
        assert!(
            beta_exact.is_finite() && report.total_derivative.is_finite(),
            "dual SPD Cholesky logdet reports exact finite per-channel derivatives"
        );
    }
}
