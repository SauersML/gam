#![cfg(test)]
//! Execution setup for the gam-predict unit tests moved out of the root quality
//! suite under #2899.

/// Worker stack for the global Rayon pool these tests fan out through: the
/// stack of gam's own process pool (`gam_runtime::parallel::WORKER_STACK_SIZE`).
const RAYON_WORKER_STACK_SIZE: usize = gam_runtime::parallel::WORKER_STACK_SIZE;

/// The root crate's `gam::init_parallelism` without its CUDA GEMM dispatch hook,
/// which lives in the root crate and is out of gam-predict's reach. It registers
/// the deterministic Laplace marginal corrector and the rho-posterior escalator.
/// The tests call engine functions directly rather than through
/// `gam_runtime::parallel::install`, so their parallel work lands on rayon's
/// global pool, which this builds with the process pool's worker stack. Only
/// the first call has effect.
pub(crate) fn init_parallelism() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        drop(
            gam_problem::laplace_sampler_contract::set_laplace_marginal_corrector(Box::new(
                gam_inference::hmc_io::HmcIoLaplaceMarginalCorrector,
            )),
        );
        drop(gam_problem::rho_posterior::set_rho_posterior_escalator(
            Box::new(gam_inference::rho_posterior::HmcIoRhoPosteriorEscalator),
        ));
        drop(
            rayon::ThreadPoolBuilder::new()
                .stack_size(RAYON_WORKER_STACK_SIZE)
                .build_global(),
        );
    });
}
