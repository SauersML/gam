//! Saved model documents (#2966): one versioned JSON envelope for every model
//! kind, written and read through the same Rust code by the CLI, pyffi and
//! gamfit.
//!
//! A document is `{"kind": ..., "version": ..., "model": ...}`. Each kind's
//! owner names the kind and holds its version constant. Loading probes the kind
//! and the version before it parses the model, and refuses any other kind or
//! version with a typed error; an older payload is never migrated. JSON has no
//! encoding for a non-finite float, so saving refuses one instead of writing
//! `null`. With serde_json's exact float round trip, save → reload reproduces
//! every f64 bit.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Why a saved model could not be written, or was refused.
#[derive(Debug, thiserror::Error)]
pub enum SavedModelError {
    /// The document names another kind of model, or none.
    #[error("the saved model is of kind {found:?}; this reader expects {expected:?}")]
    Kind {
        /// The kind the document names.
        found: Option<String>,
        /// The kind this reader reads.
        expected: &'static str,
    },
    /// The document has another version, or none. No version is migrated: a
    /// payload of another version lacks what this build needs to rebuild its
    /// model, so the remedy is to refit.
    #[error(
        "the saved {kind} model has version {found:?}; this build reads only version {expected}, so refit the model with this build"
    )]
    Version {
        /// The kind of model.
        kind: &'static str,
        /// The version the document names.
        found: Option<u64>,
        /// The version this build reads.
        expected: u64,
    },
    /// The text is not a model document of the expected shape.
    #[error("the saved model is not a model document: {reason}")]
    Malformed {
        /// What the parser refused.
        reason: String,
    },
    /// The model holds a state its owner refuses, or cannot be saved.
    #[error("the saved model holds a state its law cannot: {reason}")]
    Inconsistent {
        /// The owner's refusal.
        reason: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Reading or writing the file failed.
    #[error("{path}: {reason}")]
    Io {
        /// The file.
        path: String,
        /// The operating system's refusal.
        reason: String,
    },
}

#[derive(Serialize)]
struct Envelope<'m, T> {
    kind: &'static str,
    version: u64,
    model: &'m T,
}

/// Distinguishes the temporary files of saves within one process.
static TEMPORARY_FILES: AtomicU64 = AtomicU64::new(0);

/// The saved document of `model`, of `kind` at `version`.
pub fn saved_model_text<T: Serialize>(
    kind: &'static str,
    version: u64,
    model: &T,
) -> Result<String, SavedModelError> {
    gam_problem::ensure_serialized_floats_are_finite(model).map_err(|found| {
        SavedModelError::Inconsistent {
            reason: format!("a saved model cannot hold a non-finite float: {found}").into(),
        }
    })?;
    serde_json::to_string_pretty(&Envelope {
        kind,
        version,
        model,
    })
    .map_err(|error| SavedModelError::Malformed {
        reason: error.to_string(),
    })
}

/// The model in a saved document of `kind` at `version`.
pub fn read_saved_model_text<T: DeserializeOwned>(
    text: &str,
    kind: &'static str,
    version: u64,
) -> Result<T, SavedModelError> {
    let malformed = |error: serde_json::Error| SavedModelError::Malformed {
        reason: error.to_string(),
    };
    let mut document: Value = serde_json::from_str(text).map_err(malformed)?;
    let found_kind = document.get("kind").and_then(Value::as_str);
    if found_kind != Some(kind) {
        return Err(SavedModelError::Kind {
            found: found_kind.map(str::to_string),
            expected: kind,
        });
    }
    let found = document.get("version").and_then(Value::as_u64);
    if found != Some(version) {
        return Err(SavedModelError::Version {
            kind,
            found,
            expected: version,
        });
    }
    let model = document
        .get_mut("model")
        .map(Value::take)
        .ok_or_else(|| SavedModelError::Malformed {
            reason: "the document has no model".to_string(),
        })?;
    serde_json::from_value(model).map_err(malformed)
}

/// Write a saved document atomically: to a sibling temporary file named for
/// this save, renamed over `path`. A failed save removes its temporary file.
pub fn write_saved_model(path: &Path, text: &str) -> Result<(), SavedModelError> {
    let io = |reason: String| SavedModelError::Io {
        path: path.display().to_string(),
        reason,
    };
    let name = path
        .file_name()
        .ok_or_else(|| io("not a file path".to_string()))?;
    let mut temporary = name.to_os_string();
    temporary.push(format!(
        ".tmp{}-{}",
        std::process::id(),
        TEMPORARY_FILES.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary = path.with_file_name(temporary);
    std::fs::write(&temporary, text)
        .and_then(|()| std::fs::rename(&temporary, path))
        .map_err(|error| {
            drop(std::fs::remove_file(&temporary));
            io(error.to_string())
        })
}

/// The text of a saved document.
pub fn read_saved_model_file(path: &Path) -> Result<String, SavedModelError> {
    std::fs::read_to_string(path).map_err(|error| SavedModelError::Io {
        path: path.display().to_string(),
        reason: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch file removed when dropped, so a failing assertion leaves
    /// nothing behind.
    struct Scratch(std::path::PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            drop(std::fs::remove_file(&self.0));
        }
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn documents_round_trip_adversarial_floats_bit_for_bit() {
        let tenth = 0.1_f64.to_bits();
        let values = vec![
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            f64::from_bits(tenth - 1),
            0.1,
            f64::from_bits(tenth + 1),
            f64::MAX,
            -f64::MAX,
            -0.0,
            1.0 / 3.0,
            std::f64::consts::PI,
        ];
        let text = saved_model_text("test", 3, &values).unwrap();
        let reloaded: Vec<f64> = read_saved_model_text(&text, "test", 3).unwrap();
        assert_eq!(bits(&reloaded), bits(&values));
    }

    #[test]
    fn documents_refuse_other_kinds_versions_and_non_finite_floats() {
        let values = vec![0.1_f64, 2.5e300];
        let text = saved_model_text("test", 3, &values).unwrap();
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>(&text, "test", 4),
            Err(SavedModelError::Version { found: Some(3), expected: 4, .. })
        ));
        assert!(text.contains("\"kind\": \"test\""));
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>(
                &text.replace("\"kind\": \"test\"", "\"kind\": \"other\""),
                "test",
                3
            ),
            Err(SavedModelError::Kind { .. })
        ));
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>("{", "test", 3),
            Err(SavedModelError::Malformed { .. })
        ));
        assert!(matches!(
            saved_model_text("test", 3, &vec![f64::NAN]),
            Err(SavedModelError::Inconsistent { .. })
        ));
        let scratch = Scratch(std::env::temp_dir().join(format!(
            "gam-saved-model-{}-{}.json",
            std::process::id(),
            TEMPORARY_FILES.fetch_add(1, Ordering::Relaxed)
        )));
        write_saved_model(&scratch.0, &text).unwrap();
        assert_eq!(read_saved_model_file(&scratch.0).unwrap(), text);
        std::fs::remove_file(&scratch.0).unwrap();
        assert!(matches!(
            read_saved_model_file(&scratch.0),
            Err(SavedModelError::Io { .. })
        ));
    }

    #[test]
    fn documents_refuse_a_missing_model_a_missing_kind_and_a_path_without_a_file_name() {
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>("{\"kind\": \"test\", \"version\": 3}", "test", 3),
            Err(SavedModelError::Malformed { reason }) if reason == "the document has no model"
        ));
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>("{\"version\": 3, \"model\": []}", "test", 3),
            Err(SavedModelError::Kind { found: None, expected: "test" })
        ));
        assert!(matches!(
            write_saved_model(Path::new("/"), "{}"),
            Err(SavedModelError::Io { reason, .. }) if reason == "not a file path"
        ));
    }
}
