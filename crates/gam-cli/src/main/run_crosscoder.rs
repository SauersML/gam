use super::*;

use gam_sae::manifold::{
    NamedCrosscoderTarget, SaeCrosscoderAutoFitOverrides, SaeCrosscoderAutoFitRequest,
    SaeCrosscoderEvaluationConfig, run_auto_sae_crosscoder_fit,
};
use ndarray::ShapeBuilder;
use npyz::{NpyFile, Order};
use std::io::{BufReader, BufWriter, Write};

/// Load one floating-point, two-dimensional NPY without owning any scientific
/// preprocessing. Row alignment, finite values, and crosscoder structure are
/// validated by the GAM-SAE request owner after all matrices are loaded.
fn read_npy_matrix(path: &Path) -> Result<Array2<f64>, String> {
    let file = std::fs::File::open(path)
        .map_err(|err| format!("open activation NPY {}: {err}", path.display()))?;
    let npy = NpyFile::new(BufReader::new(file))
        .map_err(|err| format!("read activation NPY header {}: {err}", path.display()))?;
    let shape = npy.shape();
    let [n_u64, p_u64] = shape else {
        return Err(format!(
            "activation NPY {} must be 2-D; got shape {shape:?}",
            path.display()
        ));
    };
    let n = usize::try_from(*n_u64).map_err(|_| {
        format!(
            "activation NPY {} row count {n_u64} exceeds this platform",
            path.display()
        )
    })?;
    let p = usize::try_from(*p_u64).map_err(|_| {
        format!(
            "activation NPY {} column count {p_u64} exceeds this platform",
            path.display()
        )
    })?;
    n.checked_mul(p).ok_or_else(|| {
        format!(
            "activation NPY {} shape ({n}, {p}) overflows this platform",
            path.display()
        )
    })?;
    let order = npy.order();
    let values = match npy.try_data::<f64>() {
        Ok(reader) => reader
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(|err| format!("read f64 activation NPY {}: {err}", path.display()))?,
        Err(npy) => match npy.try_data::<f32>() {
            Ok(reader) => reader
                .map(|value| value.map(f64::from))
                .collect::<std::io::Result<Vec<_>>>()
                .map_err(|err| format!("read f32 activation NPY {}: {err}", path.display()))?,
            Err(npy) => match npy.try_data::<npyz::half::f16>() {
                Ok(reader) => reader
                    .map(|value| value.map(npyz::half::f16::to_f64))
                    .collect::<std::io::Result<Vec<_>>>()
                    .map_err(|err| format!("read f16 activation NPY {}: {err}", path.display()))?,
                Err(npy) => {
                    return Err(format!(
                        "activation NPY {} must have dtype f16, f32, or f64; got {}",
                        path.display(),
                        npy.dtype().descr()
                    ));
                }
            },
        },
    };
    let result = match order {
        Order::C => Array2::from_shape_vec((n, p), values),
        Order::Fortran => Array2::from_shape_vec((n, p).f(), values),
    };
    result.map_err(|err| {
        format!(
            "activation NPY {} has invalid shape ({n}, {p}): {err}",
            path.display()
        )
    })
}

fn write_wire_report(
    path: &Path,
    report: &gam_sae::manifold::SaeCrosscoderWireReport,
) -> Result<(), CliError> {
    let file = std::fs::File::create(path).map_err(|err| CliError::FileWriteFailed {
        reason: format!("create crosscoder report {}: {err}", path.display()),
    })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, report).map_err(|err| CliError::FileWriteFailed {
        reason: format!("serialize crosscoder report {}: {err}", path.display()),
    })?;
    writer
        .write_all(b"\n")
        .and_then(|()| writer.flush())
        .map_err(|err| CliError::FileWriteFailed {
            reason: format!("write crosscoder report {}: {err}", path.display()),
        })
}

pub(crate) fn run_crosscoder(args: CrosscoderArgs) -> CliResult<()> {
    let anchor = read_npy_matrix(&args.anchor.path).map_err(CliError::from)?;
    let blocks = args
        .block
        .into_iter()
        .map(|input| {
            Ok(NamedCrosscoderTarget {
                label: input.label,
                target: read_npy_matrix(&input.path)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let config = SaeCrosscoderAutoFitOverrides {
        sparsity_strength: args.sparsity_strength,
        smoothness: args.smoothness,
        max_iter: args.max_iter,
        learning_rate: args.learning_rate,
        ridge_ext_coord: args.ridge_ext_coord,
        ridge_beta: args.ridge_beta,
        random_state: args.random_state,
        run_outer_rho_search: args.outer_rho_search,
    }
    .resolve(args.atoms, args.harmonics);
    let fit = run_auto_sae_crosscoder_fit(SaeCrosscoderAutoFitRequest {
        anchor_label: args.anchor.label,
        anchor,
        blocks,
        config,
        cancel: None,
    })
    .map_err(|err| CliError::from(err.to_string()))?;
    let wire = fit
        .wire_report(SaeCrosscoderEvaluationConfig {
            transport_grid_resolution: args.transport_grid_resolution,
            law_gap_tolerance: args.law_gap_tolerance,
        })
        .map_err(CliError::from)?;
    write_wire_report(&args.out, &wire)?;
    cli_out!("Wrote crosscoder report to {}", args.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use npyz::WriterBuilder;

    #[test]
    fn named_npy_input_requires_an_explicit_label() {
        let parsed = "layer=acts=part.npy"
            .parse::<NamedNpyInput>()
            .expect("named input should parse");
        assert_eq!(parsed.label, "layer");
        assert_eq!(parsed.path, PathBuf::from("acts=part.npy"));
        assert!("acts.npy".parse::<NamedNpyInput>().is_err());
        assert!("=acts.npy".parse::<NamedNpyInput>().is_err());
    }

    #[test]
    fn crosscoder_cli_leaves_library_policy_unresolved() {
        let cli = Cli::try_parse_from([
            "gam",
            "crosscoder",
            "--anchor",
            "anchor=anchor.npy",
            "--block",
            "layer-1=layer1.npy",
            "--atoms",
            "4",
            "--harmonics",
            "3",
            "--out",
            "report.json",
        ])
        .expect("minimal crosscoder command should parse");
        let Command::Crosscoder(args) = cli.command else {
            panic!("expected crosscoder command");
        };
        assert_eq!(args.anchor.label, "anchor");
        assert_eq!(args.block.len(), 1);
        assert!(args.sparsity_strength.is_none());
        assert!(args.smoothness.is_none());
        assert!(args.max_iter.is_none());
        assert!(args.learning_rate.is_none());
        assert!(args.ridge_ext_coord.is_none());
        assert!(args.ridge_beta.is_none());
        assert!(args.random_state.is_none());
        assert!(args.outer_rho_search.is_none());
        assert!(args.transport_grid_resolution.is_none());
        assert!(args.law_gap_tolerance.is_none());
    }

    #[test]
    fn transport_tolerance_requires_a_grid() {
        let result = Cli::try_parse_from([
            "gam",
            "crosscoder",
            "--anchor",
            "anchor=anchor.npy",
            "--block",
            "layer-1=layer1.npy",
            "--atoms",
            "4",
            "--harmonics",
            "3",
            "--law-gap-tolerance",
            "0.1",
            "--out",
            "report.json",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn crosscoder_help_exposes_named_npy_and_transport_contracts() {
        let error = Cli::try_parse_from(["gam", "crosscoder", "--help"])
            .expect_err("--help should return clap's display-help result");
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
        let help = error.to_string();
        for required in [
            "--anchor <LABEL=FILE>",
            "--block <LABEL=FILE>",
            "--transport-grid-resolution",
            "--law-gap-tolerance",
            "--out <REPORT.json>",
        ] {
            assert!(help.contains(required), "missing {required:?} in:\n{help}");
        }
    }

    #[test]
    fn npy_reader_accepts_2d_f32_and_preserves_shape() {
        let mut file = tempfile::NamedTempFile::new().expect("temp NPY");
        {
            let mut writer = npyz::WriteOptions::new()
                .default_dtype()
                .shape(&[2, 3])
                .writer(file.as_file_mut())
                .begin_nd()
                .expect("begin NPY");
            writer
                .extend([1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("write NPY values");
            writer.finish().expect("finish NPY");
        }
        let matrix = read_npy_matrix(file.path()).expect("read f32 NPY");
        assert_eq!(matrix.dim(), (2, 3));
        assert_eq!(matrix[[1, 2]], 6.0);
    }

    #[test]
    fn npy_reader_rejects_non_matrix_input() {
        let mut file = tempfile::NamedTempFile::new().expect("temp NPY");
        {
            let mut writer = npyz::WriteOptions::new()
                .default_dtype()
                .shape(&[3])
                .writer(file.as_file_mut())
                .begin_nd()
                .expect("begin NPY");
            writer
                .extend([1.0_f64, 2.0, 3.0])
                .expect("write NPY values");
            writer.finish().expect("finish NPY");
        }
        let error = read_npy_matrix(file.path()).expect_err("1-D NPY must be rejected");
        assert!(error.contains("must be 2-D"), "{error}");
    }
}
