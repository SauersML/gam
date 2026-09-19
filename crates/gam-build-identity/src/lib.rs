//! The git commit a gam engine binary was built from.
//!
//! The package version does not change from commit to commit, so two engines
//! that behave differently can share one version string (gam#3007, gam#3157).
//! `gamfit.build_info()` and `gam --version` report these values so a caller
//! can tell which engine it runs. Both are `None` when the build had no gam git
//! tree to read, as for a build from an unpacked sdist.

const RECORDED_COMMIT: &str = env!("GAM_BUILD_COMMIT");
const RECORDED_DIRTY: &str = env!("GAM_BUILD_DIRTY");

/// The full hash of the commit checked out when this binary was built.
pub const COMMIT: Option<&str> = if RECORDED_COMMIT.is_empty() {
    None
} else {
    Some(RECORDED_COMMIT)
};

/// Whether the tracked files differed from [`COMMIT`] when this binary was
/// built, so that the binary is not exactly that commit's engine.
pub const DIRTY: Option<bool> = match RECORDED_DIRTY.as_bytes() {
    b"1" => Some(true),
    b"0" => Some(false),
    _ => None,
};

/// One line naming the build: the commit, marked `-dirty` when the tree had
/// uncommitted changes, or `unknown` when the build had no gam git tree.
pub fn describe() -> String {
    match (COMMIT, DIRTY) {
        (Some(commit), Some(true)) => format!("{commit}-dirty"),
        (Some(commit), _) => commit.to_owned(),
        (None, _) => "unknown".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn records_the_checked_out_commit_of_the_gam_tree() {
        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("run git rev-parse HEAD");
        assert!(output.status.success(), "the tests run inside a gam checkout");
        let head = String::from_utf8(output.stdout).expect("git prints UTF-8");
        assert_eq!(COMMIT, Some(head.trim()));
        assert!(DIRTY.is_some(), "a build inside a checkout knows its dirty state");
        let described = describe();
        assert!(described.starts_with(head.trim()), "{described}");
    }
}
