//! Progress logging and file digests for the drivers (#2951).

use std::path::Path;
use std::process::Command;

/// A logger writing `log::info!` progress to standard error, for drivers; installed once, later
/// calls are no-ops.
pub fn log_to_stderr() {
    struct Stderr;
    impl log::Log for Stderr {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Info
        }
        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                eprintln!("{}", record.args());
            }
        }
        fn flush(&self) {}
    }
    static LOGGER: Stderr = Stderr;
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Info);
    }
}

/// The lowercase hexadecimal SHA-256 digest of the file at `path`, from the system's `sha256sum`
/// or `shasum`.
pub fn sha256(path: &Path) -> Result<String, String> {
    for (program, args) in [("sha256sum", vec![]), ("shasum", vec!["-a", "256"])] {
        match Command::new(program).args(args).arg("--").arg(path).output() {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
                let hash = text.split_whitespace().next().ok_or("empty hash")?;
                if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) { return Err("invalid SHA256".into()); }
                return Ok(hash.to_ascii_lowercase());
            }
            Ok(output) => return Err(String::from_utf8_lossy(&output.stderr).into_owned()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    Err("SHA256 requires sha256sum or shasum".into())
}
