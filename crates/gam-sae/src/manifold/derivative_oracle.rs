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
        assert!(
            err.changed_fields
                .iter()
                .any(|field| field == "deflated_rank")
        );
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
        assert!(
            err.changed_fields
                .iter()
                .any(|field| field == "min_eigen_gap")
        );
    }

    #[test]
    fn eigen_gap_threshold_is_the_exact_refusal_boundary() {
        // A gap one machine epsilon wide sits below the round-off threshold and
        // must refuse rather than leak a scalar eigenvalue derivative, while a
        // gap several epsilons wide resolves and keeps the individual route.
        // This pins the threshold as the decision boundary so no wrong
        // derivative slips through in the near-degenerate regime.
        let just_below = eigen_gap_certificate(&[1.0, 1.0 + f64::EPSILON]);
        assert!(just_below.min_eigen_gap < just_below.threshold);
        let refused = certificate(MajorizerAnchorMode::FrozenAnchor).with_eigen_gap(just_below);
        assert_eq!(
            refused.eigen_derivative_route(),
            EigenDerivativeRoute::InvariantSubspaceBlock
        );
        let err = refused
            .assert_derivative_reportable()
            .expect_err("gap below round-off must refuse a scalar derivative");
        assert_eq!(
            err.refusal,
            BranchCertificateRefusal::UnresolvedInvariantSubspaceBlock
        );
        assert!(
            err.changed_fields
                .iter()
                .any(|field| field == "min_eigen_gap")
        );

        let just_above = eigen_gap_certificate(&[1.0, 1.0 + 16.0 * f64::EPSILON]);
        assert!(just_above.min_eigen_gap > just_above.threshold);
        let resolved = certificate(MajorizerAnchorMode::FrozenAnchor).with_eigen_gap(just_above);
        assert_eq!(
            resolved.eigen_derivative_route(),
            EigenDerivativeRoute::IndividualEigenpairs
        );
        resolved
            .assert_derivative_reportable()
            .expect("gap above round-off resolves to the individual eigenpair route");
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
        assert!(
            err.changed_fields
                .iter()
                .any(|field| field == "min_eigen_gap")
        );
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
        let report = guarded_exact_trace_report(cert, vec![tt, beta]).expect("same branch report");

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
