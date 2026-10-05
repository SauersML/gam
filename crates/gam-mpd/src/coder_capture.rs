//! The SHA-256 digest drivers record for the files they read and write.
use std::process::Command;
use std::path::Path;

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
