use super::*;

use gam_predict::partial_effect::{
    PartialDependenceGrid, encode_labelled_grid, partial_effect, partial_effect_table,
};

fn read_labelled_grid(path: &Path) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| format!("failed to read grid csv '{}': {e}", path.display()))?;
    let header = reader
        .headers()
        .map_err(|e| format!("failed to read grid csv header '{}': {e}", path.display()))?
        .iter()
        .map(|name| name.trim().to_string())
        .collect();
    let rows = reader
        .records()
        .map(|record| {
            record
                .map(|record| record.iter().map(str::to_string).collect())
                .map_err(|e| format!("failed to read grid csv '{}': {e}", path.display()))
        })
        .collect::<Result<_, _>>()?;
    Ok((header, rows))
}

pub(crate) fn run_partial_effect(args: PartialEffectArgs) -> Result<(), String> {
    reject_multinomial_model(&args.model, "partial-effect")?;
    let model = SavedModel::load_from_path(&args.model)?;
    let grid = match &args.grid {
        None => PartialDependenceGrid::TrainingRange {
            n_points: args.n_points,
        },
        Some(path) => {
            let (header, rows) = read_labelled_grid(path)?;
            // The default grid's table names the term's axes and their levels,
            // which the labelled grid is encoded against.
            let axes = partial_effect_table(
                &model,
                &args.term,
                PartialDependenceGrid::TrainingRange { n_points: 2 },
            )?;
            PartialDependenceGrid::Explicit(encode_labelled_grid(&axes, &header, &rows)?)
        }
    };
    let effect = partial_effect(&model, &args.term, grid, args.level)?;
    let json = || {
        serde_json::to_string_pretty(&effect.record())
            .map_err(|e| format!("failed to serialize partial effect: {e}"))
    };
    match &args.out {
        None => cli_out!("{}", json()?),
        Some(path) => {
            let body = match path.extension().and_then(|ext| ext.to_str()) {
                Some(ext) if ext.eq_ignore_ascii_case("csv") => effect.to_csv(),
                Some(ext) if ext.eq_ignore_ascii_case("json") => json()?,
                _ => {
                    return Err(format!(
                        "partial-effect --out must end in .csv or .json; got '{}'",
                        path.display()
                    ));
                }
            };
            std::fs::write(path, body)
                .map_err(|e| format!("failed to write '{}': {e}", path.display()))?;
        }
    }
    Ok(())
}
