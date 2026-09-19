use super::*;

pub(crate) fn fixed_hazard_multiplier_from_saved_family(
    family: &FittedFamily,
) -> Result<
    (
        f64,
        gam::families::survival::lognormal_kernel::HazardLoading,
    ),
    String,
> {
    let frailty = family.frailty().ok_or_else(|| {
        "saved latent survival/binary model requires a fixed HazardMultiplier frailty specification"
            .to_string()
    })?;
    fixed_latent_hazard_frailty(frailty, "saved latent survival/binary model")
}

pub(crate) fn write_model_json(path: &Path, model: &SavedModel) -> Result<(), String> {
    model.save_to_path(path)?;
    cli_out!("saved model: {}", path.display());
    Ok(())
}

pub(crate) fn write_payload_json(path: &Path, payload: FittedModelPayload) -> Result<(), String> {
    let model = SavedModel::from_payload(payload);
    write_model_json(path, &model)
}

/// Report a fit's notes. Advisories (the model differs from the literal
/// request) go to stderr; informational notes (defaults chosen on the user's
/// behalf) are logged at debug level, shown with `-v`, and stay in the saved
/// model either way.
pub(crate) fn print_inference_summary(advisories: &[String], informational: &[String]) {
    for note in informational {
        log::debug!("fit note: {note}");
    }
    if advisories.is_empty() {
        return;
    }
    cli_err!("Fit notes:");
    for note in advisories {
        cli_err!("  - {}", note);
    }
}
