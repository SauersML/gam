//! Resuming a fit from a saved model's certified outer point (`warm_start_from`).
//!
//! The point is the one a custom-family fit records beside its result
//! (`FitArtifacts::outer_warm_start`): the outer `ρ` and the flat inner mode, in
//! the outer objective's own coordinates. It reaches the new fit as the single
//! final entry of a cache session, the seam the outer optimizer already reads.
//! That entry resumes and recertifies: the search accepts it with zero outer
//! iterations only if it is still stationary on the new fit's data, and runs
//! its ordinary search from it otherwise. It is never a shortcut past the
//! certificate.

use crate::inference::model::FittedModelPayload;
use gam_model_api::RequiredWarmStart;
use gam_runtime::warm_start::{ConfiguredWarmStartStore, Fingerprinter, Session, StoreOptions};
use ndarray::Array1;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A saved model's certified outer point, installed as the one entry of a cache
/// session under a scratch root that the caller owns and removes after the fit.
#[derive(Clone)]
pub struct OuterWarmStart {
    session: Arc<Session>,
    required: RequiredWarmStart,
}

impl std::fmt::Debug for OuterWarmStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OuterWarmStart")
            .field("rho_dim", &self.required.rho_dim)
            .field("beta_dim", &self.required.beta_dim)
            .field("consumed", &self.consumed())
            .finish()
    }
}

impl OuterWarmStart {
    /// The certified point `model` recorded, for a new fit of the same
    /// `formula`. Refused by name when the formulas differ, when the model
    /// records no point (its route does not record one, or it was saved before
    /// the point was recorded), or when the scratch store cannot take it.
    pub fn from_model(
        model: &FittedModelPayload,
        formula: &str,
        scratch_root: PathBuf,
    ) -> Result<Self, String> {
        let spelling = |text: &str| text.split_whitespace().collect::<String>();
        if spelling(&model.formula) != spelling(formula) {
            return Err(format!(
                "warm_start_from: the model was fitted with the formula '{}', and this fit asks \
                 for '{formula}'; a warm start resumes the same model",
                model.formula,
            ));
        }
        let record = model
            .fit_result
            .as_ref()
            .and_then(|fit| fit.artifacts.outer_warm_start.as_ref())
            .ok_or_else(|| {
                "warm_start_from: the model records no certified outer point; the route that \
                 fitted it records none, or it was saved before such points were recorded"
                    .to_string()
            })?;
        let payload = gam_solve::rho_optimizer::encode_outer_warm_start(
            &Array1::from_vec(record.rho.clone()),
            &Array1::from_vec(record.beta.clone()),
        )
        .ok_or_else(|| "warm_start_from: the model's point is not encodable".to_string())?;
        let store = ConfiguredWarmStartStore::new(scratch_root, StoreOptions::default());
        let mut key = Fingerprinter::new();
        key.absorb_str(b"warm-start-from", formula);
        let session = store
            .open_session(key.finalize())
            .ok_or_else(|| "warm_start_from: the scratch store could not be opened".to_string())?;
        if !session.finalize(&payload, None, Some(0)) {
            return Err(
                "warm_start_from: the scratch store refused the certified point".to_string(),
            );
        }
        Ok(Self {
            session,
            required: RequiredWarmStart {
                rho_dim: record.rho.len(),
                beta_dim: record.beta.len(),
                consumed: Arc::new(AtomicBool::new(false)),
            },
        })
    }

    pub(crate) fn session(&self) -> &Arc<Session> {
        &self.session
    }

    pub(crate) fn required(&self) -> &RequiredWarmStart {
        &self.required
    }

    /// Whether a route attached the point to its outer search.
    pub fn consumed(&self) -> bool {
        self.required.consumed.load(Ordering::Relaxed)
    }
}
