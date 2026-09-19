use super::*;

use gam_sae::parameter_decomposition::surface::run_parameter_decomposition;
use ndarray::ArrayD;
use npyz::WriterBuilder;
use std::collections::BTreeMap;
use std::io::BufWriter;

/// The CLI transport of the MPD surface (#2951). It reads the request document and
/// the named NPY arrays, runs the same Rust entry `gamfit.run_parameter_decomposition`
/// runs, and writes `report.json` and one NPY per array the report names. Every
/// validation and refusal belongs to the surface.
pub(crate) fn run_parameter_decomposition_cli(args: ParameterDecompositionArgs) -> CliResult<()> {
    let request = std::fs::read_to_string(&args.request).map_err(|err| {
        CliError::from(format!(
            "read MPD request {}: {err}",
            args.request.display()
        ))
    })?;
    let mut tensors = BTreeMap::new();
    for input in args.tensor {
        let array = read_npy_array(&input.path)?;
        if tensors.insert(input.label.clone(), array).is_some() {
            return Err(CliError::from(format!(
                "input array id {:?} is given more than once",
                input.label
            )));
        }
    }
    let output = run_parameter_decomposition(&request, &tensors)
        .map_err(|err| CliError::from(err.to_string()))?;
    let report = output
        .report_json()
        .map_err(|err| CliError::from(err.to_string()))?;
    create_output_dir(&args.out)?;
    let report_path = args.out.join("report.json");
    std::fs::write(&report_path, format!("{report}\n")).map_err(|err| {
        CliError::FileWriteFailed {
            reason: format!("write MPD report {}: {err}", report_path.display()),
        }
    })?;
    for (id, array) in &output.arrays {
        let path = args.out.join(format!("{id}.npy"));
        if let Some(parent) = path.parent() {
            create_output_dir(parent)?;
        }
        write_npy_array(&path, array)?;
    }
    cli_out!(
        "Wrote MPD report and {} arrays to {}",
        output.arrays.len(),
        args.out.display()
    );
    Ok(())
}

fn create_output_dir(path: &Path) -> CliResult<()> {
    std::fs::create_dir_all(path).map_err(|err| CliError::FileWriteFailed {
        reason: format!("create MPD output directory {}: {err}", path.display()),
    })
}

/// Write one array as a C-order `f64` NPY.
fn write_npy_array(path: &Path, array: &ArrayD<f64>) -> CliResult<()> {
    let failed = |err: std::io::Error| CliError::FileWriteFailed {
        reason: format!("write MPD array {}: {err}", path.display()),
    };
    let shape = array
        .shape()
        .iter()
        .map(|&axis| {
            u64::try_from(axis).map_err(|err| CliError::FileWriteFailed {
                reason: format!(
                    "MPD array {} axis length {axis} does not fit an NPY header: {err}",
                    path.display()
                ),
            })
        })
        .collect::<CliResult<Vec<u64>>>()?;
    let file = std::fs::File::create(path).map_err(failed)?;
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape)
        .writer(BufWriter::new(file))
        .begin_nd()
        .map_err(failed)?;
    writer.extend(array.iter().copied()).map_err(failed)?;
    writer.finish().map_err(failed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam::linalg::roundoff::accumulation_growth;
    use ndarray::array;

    fn write_matrix(path: &Path, matrix: &Array2<f64>) {
        write_npy_array(path, &matrix.clone().into_dyn()).expect("write test NPY");
    }

    fn plane_request(declared_error: f64) -> String {
        format!(
            r#"{{"schema": "gam.mpd-request", "schema_version": 1, "operation": {{"kind": "recover_plane_rotations", "tensor": "w", "declared_error": {declared_error:?}}}}}"#
        )
    }

    /// A declared distance to the orthogonal group covering the true one:
    /// `||W^T W - I||_F` plus `gamma_n` times the Frobenius norm of `|W|^T |W|`.
    fn orthogonality_declaration(matrix: &Array2<f64>) -> f64 {
        let frobenius = |m: &Array2<f64>| m.iter().map(|value| value * value).sum::<f64>().sqrt();
        let columns = matrix.ncols();
        let gram = matrix.t().dot(matrix) - Array2::<f64>::eye(columns);
        let absolute = matrix.mapv(f64::abs);
        frobenius(&gram) + accumulation_growth(columns) * frobenius(&absolute.t().dot(&absolute))
    }

    #[test]
    fn parameter_decomposition_help_exposes_only_transport_arguments() {
        let error = Cli::try_parse_from(["gam", "parameter-decomposition", "--help"])
            .expect_err("--help should return clap's display-help result");
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
        let help = error.to_string();
        let flags: BTreeSet<&str> = help
            .split_whitespace()
            .filter(|token| token.starts_with("--"))
            .collect();
        let expected: BTreeSet<&str> = ["--request", "--tensor", "--out", "--log-level", "--help"]
            .into_iter()
            .collect();
        assert_eq!(
            flags, expected,
            "every scientific choice belongs to the request document:\n{help}"
        );

        let cli = Cli::try_parse_from([
            "gam",
            "parameter-decomposition",
            "--request",
            "request.json",
            "--tensor",
            "w=w.npy",
            "--tensor",
            "b=b.npy",
            "--out",
            "out",
        ])
        .expect("a transport-only command parses");
        let Command::ParameterDecomposition(args) = cli.command else {
            panic!("expected the parameter-decomposition command");
        };
        assert_eq!(args.tensor.len(), 2);
        assert_eq!(args.tensor[1].label, "b");
    }

    #[test]
    fn cli_report_and_arrays_are_the_in_memory_surface_bytes() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (sa, ca) = 0.7_f64.sin_cos();
        let (sb, cb) = 1.9_f64.sin_cos();
        let matrix = array![
            [ca, -sa, 0.0, 0.0, 0.0],
            [sa, ca, 0.0, 0.0, 0.0],
            [0.0, 0.0, cb, -sb, 0.0],
            [0.0, 0.0, sb, cb, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ];
        let tensor_path = dir.path().join("w.npy");
        write_matrix(&tensor_path, &matrix);
        let request = plane_request(orthogonality_declaration(&matrix));
        let request_path = dir.path().join("request.json");
        std::fs::write(&request_path, &request).expect("write request");
        let out = dir.path().join("out");

        run_parameter_decomposition_cli(ParameterDecompositionArgs {
            request: request_path,
            tensor: vec![NamedNpyInput {
                label: "w".to_string(),
                path: tensor_path,
            }],
            out: out.clone(),
        })
        .expect("CLI run");

        let in_memory = run_parameter_decomposition(
            &request,
            &BTreeMap::from([("w".to_string(), matrix.into_dyn())]),
        )
        .expect("in-memory run");
        let report = std::fs::read_to_string(out.join("report.json")).expect("read report");
        assert_eq!(
            report,
            format!("{}\n", in_memory.report_json().expect("report json"))
        );
        // A basis per cluster and at least the two planted planes, so the comparison
        // below is over real arrays.
        assert!(in_memory.arrays.len() >= 3, "{:?}", in_memory.arrays.keys());
        for (id, array) in &in_memory.arrays {
            let written = read_npy_array(&out.join(format!("{id}.npy"))).expect("read written NPY");
            assert_eq!(&written, array, "array {id} changed on the CLI transport");
        }
    }

    #[test]
    fn a_repeated_input_id_is_refused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let tensor_path = dir.path().join("w.npy");
        write_matrix(&tensor_path, &Array2::eye(2));
        let request_path = dir.path().join("request.json");
        std::fs::write(&request_path, plane_request(0.0)).expect("write request");
        let input = NamedNpyInput {
            label: "w".to_string(),
            path: tensor_path,
        };
        let run = |tensor: Vec<NamedNpyInput>| {
            run_parameter_decomposition_cli(ParameterDecompositionArgs {
                request: request_path.clone(),
                tensor,
                out: dir.path().join("out"),
            })
        };
        assert!(run(vec![input.clone()]).is_ok());
        assert!(run(vec![input.clone(), input]).is_err());
    }
}
