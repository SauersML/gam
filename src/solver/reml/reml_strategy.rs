#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GeometryBackendKind {
    DenseSpectral,
    SparseExactSpd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HessianEvalStrategyKind {
    SpectralExact,
    AnalyticFallback,
    DiagnosticNumeric,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct HessianStrategyDecision {
    pub(super) strategy: HessianEvalStrategyKind,
    pub(super) reason: &'static str,
}
