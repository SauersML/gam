//! gam#3007 / gam#3157: `gam --version` names the commit the binary was built
//! from and the saved-model payload version it writes, since the package
//! version is shared by every commit between releases.

use std::process::Command;

#[test]
fn version_names_the_build_commit_and_the_model_payload_version() {
    let output = Command::new(gam_test_support::gam_binary!())
        .arg("--version")
        .output()
        .expect("spawn gam CLI");
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).expect("gam --version prints UTF-8");
    let expected = format!(
        "gam {}\ncommit {}\nmodel payload version {}\n",
        env!("CARGO_PKG_VERSION"),
        gam_build_identity::describe(),
        gam::inference::model::MODEL_PAYLOAD_VERSION
    );
    assert_eq!(stdout, expected);
    let commit = gam_build_identity::COMMIT.expect("the tests build inside a gam checkout");
    assert!(
        commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()),
        "{commit}"
    );
}
