//! Run the original shape contract with P-IRLS refinement diagnostics enabled.
//! Compile with scripts/compile_warm_probe.py --test against an idle, verified
//! gam library graph, exposing log, ndarray, and gam_runtime. Run the resulting
//! executable without a test-name filter and with --test-threads=1 --nocapture:
//! the first test installs the logger before the unchanged regression runs.

struct RefinementLogger;

impl log::Log for RefinementLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
            || (metadata.level() == log::Level::Debug
                && metadata.target().contains("pirls"))
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: RefinementLogger = RefinementLogger;

#[test]
fn a_install_refinement_logger() {
    assert!(
        log::set_logger(&LOGGER).is_ok(),
        "the diagnostic owns its process logger"
    );
    log::set_max_level(log::LevelFilter::Debug);
    assert_eq!(log::max_level(), log::LevelFilter::Debug);
}

#[path = "../../../tests/regressions/smooths/shape_constrained_fit_survives_its_own_inference_2601.rs"]
mod original_shape_contract;
