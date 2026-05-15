#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuLinkKernel {
    Logit,
    Probit,
    CLogLog,
    Identity,
    Log,
    Sas,
    BetaLogistic,
}

#[derive(Clone, Debug, Default)]
pub struct LinkKernelOutputs {
    pub emits_mu: bool,
    pub emits_w: bool,
    pub emits_z: bool,
    pub emits_derivative_jets: bool,
    pub emits_deviance_loglik: bool,
}
