use super::*;
use gam::families::inference::saved_residuals::saved_model_residuals;
use gam::families::inference::saved_summary::saved_model_summary;

/// `gam summary MODEL`: the saved model's summary payload as JSON, the same
/// document `gamfit`'s `Model.summary()` reads.
pub(crate) fn run_summary(args: SummaryArgs) -> Result<(), String> {
    reject_multinomial_model(&args.model, "summary")?;
    let model = SavedModel::load_from_path(&args.model)?;
    let summary = saved_model_summary(&model)?;
    let text = serde_json::to_string_pretty(&summary)
        .map_err(|err| format!("failed to serialize summary: {err}"))?;
    cli_out!("{text}");
    Ok(())
}

/// `gam residuals MODEL DATA --type TYPE`: the per-row residuals of the saved
/// model on `DATA` as JSON, the same values `gamfit`'s
/// `Model.residuals(data, type=...)` returns.
pub(crate) fn run_residuals(args: ResidualsArgs) -> Result<(), String> {
    reject_multinomial_model(&args.model, "residuals")?;
    let model = SavedModel::load_from_path(&args.model)?;
    let data = load_datasetwith_model_schema_for_diagnostics(&args.data, &model)?;
    require_dataset_rows("residuals", &args.data, data.values.nrows())?;
    let residuals = saved_model_residuals(&model, &data, args.kind)?;
    let text = serde_json::to_string_pretty(&serde_json::json!({
        "type": args.kind.name(),
        "residuals": residuals.to_vec(),
    }))
    .map_err(|err| format!("failed to serialize residuals: {err}"))?;
    cli_out!("{text}");
    Ok(())
}
