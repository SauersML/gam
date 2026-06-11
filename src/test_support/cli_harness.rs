use std::path::Path;
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
        // SAFETY: test-support helper intentionally panics with command context
        // when the child process cannot even be spawned.
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
        // SAFETY: test-support helper intentionally panics with command context
        // when the child process cannot even be spawned.
        .unwrap_or_else(|err| panic!("failed to spawn `{label}`: {err}"));
    if !output.status.success() {
        // SAFETY: test-support helper intentionally panics with captured child
        // output so failed CLI invocations preserve the relevant diagnostics.
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

pub fn write_predict_csv_rows<const N: usize, I>(path: &Path, header: [&str; N], rows: I)
where
    I: IntoIterator<Item = [String; N]>,
{
    let mut writer = csv::Writer::from_path(path).expect("create predict csv");
    writer.write_record(header).expect("write header");
    for row in rows {
        writer
            .write_record(row.iter().map(String::as_str))
            .expect("write predict row");
    }
    writer.flush().expect("flush predict csv");
}

pub fn read_prediction_means(path: &Path) -> Vec<f64> {
    let mut reader = csv::Reader::from_path(path).expect("open predictions csv");
    let headers = reader.headers().expect("predict csv headers").clone();
    let mean_idx = headers
        .iter()
        .position(|h| h == "mean")
        .or_else(|| headers.iter().position(|h| h == "linear_predictor"))
        .unwrap_or_else(|| {
            // SAFETY: test-support helper intentionally panics with header context
            panic!("predict csv has neither `mean` nor `linear_predictor` column: {headers:?}")
        });
    reader
        .records()
        .map(|rec| {
            let rec = rec.expect("predict csv row");
            rec[mean_idx]
                .parse::<f64>()
                // SAFETY: test-support helper intentionally panics with cell context
                .unwrap_or_else(|_| panic!("non-numeric prediction: {:?}", &rec[mean_idx]))
        })
        .collect()
}

pub fn fit_then_predict_gaussian(
    train_path: &Path,
    formula: &str,
    model_path: &Path,
    predict_path: &Path,
    out_path: &Path,
) -> Vec<f64> {
    let mut fit_cmd = Command::new(crate::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(train_path)
        .arg(formula)
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(model_path);
    run_or_panic(fit_cmd, &format!("gam fit {formula}"));
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    let mut predict_cmd = Command::new(crate::gam_binary!());
    predict_cmd
        .arg("predict")
        .arg(model_path)
        .arg(predict_path)
        .arg("--out")
        .arg(out_path);
    run_or_panic(predict_cmd, "gam predict");

    read_prediction_means(out_path)
}
