use std::process::Command;

#[macro_export]
macro_rules! gam_binary {
    () => {
        option_env!("CARGO_BIN_EXE_gam")
            .map(::std::path::PathBuf::from)
            .unwrap_or_else(|| {
                ::std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/debug/gam")
            })
    };
}

pub fn run_or_panic(mut command: Command, label: &str) {
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn `{label}`: {err}"));
    assert!(
        output.status.success(),
        "`{label}` failed with status {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

pub fn run_capture_or_panic(mut command: Command, label: &str) -> String {
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("failed to spawn `{label}`: {err}"));
    if !output.status.success() {
        panic!(
            "`{label}` failed with status {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    combined
}
