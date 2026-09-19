//! Record the git commit this engine is built from, and whether the tracked
//! files differed from it, as `GAM_BUILD_COMMIT` and `GAM_BUILD_DIRTY`.
//!
//! Cargo runs a build script in its package directory, so every git command
//! here reads the gam tree this crate lives in. A build outside a gam checkout
//! (an unpacked sdist, a crate copied out of the tree) records empty values,
//! which the library reads as unknown rather than guessing a commit.

use std::io::Write as _;
use std::path::Path;
use std::process::Command;

/// Run git without taking the optional index lock, so that reading the state
/// never rewrites `.git/index`. `None` when git is missing or the command fails.
fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("--no-optional-locks")
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn main() {
    // The manifest being tracked is what makes the enclosing repository gam's
    // own; an sdist unpacked inside some other repository must not report that
    // repository's commit.
    let in_gam_tree = git(&["ls-files", "--error-unmatch", "Cargo.toml"]).is_some();
    let commit = if in_gam_tree {
        git(&["rev-parse", "HEAD"])
    } else {
        None
    };
    let dirty = commit
        .as_ref()
        .and_then(|_| git(&["status", "--porcelain", "--untracked-files=no"]))
        .map(|changes| !changes.is_empty());

    // Rerun when HEAD moves (a checkout, or a commit to the checked-out branch)
    // and when the engine sources change, which is what can flip the dirty flag.
    let mut watched = vec!["build.rs".to_owned()];
    if in_gam_tree {
        let mut git_files = vec!["HEAD".to_owned(), "packed-refs".to_owned()];
        git_files.extend(git(&["symbolic-ref", "-q", "HEAD"]));
        watched.extend(
            git_files
                .iter()
                .filter_map(|name| git(&["rev-parse", "--git-path", name])),
        );
        watched.extend(
            ["../../Cargo.toml", "../../Cargo.lock", "../../src", "../../crates"].map(String::from),
        );
    }

    let mut stdout = std::io::stdout();
    for path in watched.iter().filter(|path| Path::new(path).exists()) {
        drop(writeln!(stdout, "cargo:rerun-if-changed={path}"));
    }
    let dirty = match dirty {
        Some(true) => "1",
        Some(false) => "0",
        None => "",
    };
    drop(writeln!(
        stdout,
        "cargo:rustc-env=GAM_BUILD_COMMIT={}",
        commit.as_deref().unwrap_or_default()
    ));
    drop(writeln!(stdout, "cargo:rustc-env=GAM_BUILD_DIRTY={dirty}"));
}
