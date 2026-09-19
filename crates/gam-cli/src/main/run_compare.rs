use super::*;
use gam::families::inference::saved_summary::compare_saved_models;

/// Rank saved models on their smoothing-corrected AIC and print the comparison
/// as JSON. The ranking is `compare_saved_models`, the same function behind
/// `gamfit.compare_models`, so the two front ends print one document.
pub(crate) fn run_compare(args: CompareArgs) -> Result<(), String> {
    let names = match args.names {
        Some(names) => {
            if names.len() != args.models.len() {
                return Err(format!(
                    "compare: {} --names given for {} models",
                    names.len(),
                    args.models.len()
                ));
            }
            names
        }
        None => args
            .models
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
    };
    let models = args
        .models
        .iter()
        .map(|path| {
            SavedModel::load_from_path(path)
                .map_err(|err| format!("compare: failed to load {}: {err}", path.display()))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let named = names.into_iter().zip(models.iter()).collect::<Vec<_>>();
    let comparison = compare_saved_models(&named)?;
    let json = serde_json::to_string_pretty(&comparison)
        .map_err(|err| format!("compare: failed to serialize the comparison: {err}"))?;
    cli_out!("{json}");
    Ok(())
}
