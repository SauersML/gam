//! Saved model documents (#2966): one versioned JSON envelope for every model
//! kind, written and read through the same Rust code by the CLI, pyffi and
//! gamfit.
//!
//! A document is `{"kind": ..., "version": ..., "model": ...}`, written as
//! compact JSON with the kind and the version first. Each kind's owner names the
//! kind and holds its version constant. Loading reads the document in one
//! streaming pass: the kind and the version are checked as they are read, before
//! the model, and the model is decoded straight from the bytes, never through an
//! intermediate `serde_json::Value`. Any other kind or version is refused with a
//! typed error; an older payload is never migrated. [`saved_model_header`] reads
//! only the kind and the version and stops, so dispatching on the kind costs
//! the header, not the model. JSON has no encoding for a non-finite float, so
//! saving refuses one instead of writing `null`. With serde_json's exact float
//! round trip, save → reload reproduces every f64 bit.

use serde::Serialize;
use serde::de::{self, DeserializeOwned, DeserializeSeed, IgnoredAny, MapAccess, Visitor};
use std::cell::RefCell;
use std::fmt;
use std::io::Write;
use std::marker::PhantomData;
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
    /// Reading or writing the file failed. The operating system's error keeps
    /// its kind, so a surface raises the class that kind names (a missing
    /// directory is a `FileNotFoundError` in Python).
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: String,
        /// The operating system's refusal.
        source: std::io::Error,
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

/// The saved document of `model`, of `kind` at `version`, as compact JSON.
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
    serde_json::to_string(&Envelope {
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
    read_saved_model_bytes(text.as_bytes(), kind, version)
}

/// The model in a saved document of `kind` at `version`, decoded in one
/// streaming pass over `bytes`.
pub fn read_saved_model_bytes<T: DeserializeOwned>(
    bytes: &[u8],
    kind: &'static str,
    version: u64,
) -> Result<T, SavedModelError> {
    let refusal = RefCell::new(None);
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let decoded = DocumentSeed::<T> {
        kind,
        version,
        refusal: &refusal,
        model: PhantomData,
    }
    .deserialize(&mut deserializer)
    .and_then(|model| deserializer.end().map(|()| model));
    match (decoded, refusal.into_inner()) {
        (_, Some(refusal)) => Err(refusal),
        (Ok(model), None) => Ok(model),
        (Err(error), None) => Err(SavedModelError::Malformed {
            reason: error.to_string(),
        }),
    }
}

/// The header of a saved document: the kind and the version it names, each
/// `None` when the document does not name it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedModelHeader {
    /// The kind of model the document holds.
    pub kind: Option<String>,
    /// The version of that kind's format.
    pub version: Option<u64>,
}

/// The header of a saved document. Reading stops once the kind and the
/// version are both read, which in a document this module wrote is before
/// the model; so the cost is the header's, not the model's. Bytes that are not
/// a JSON object are `Malformed`.
pub fn saved_model_header(bytes: &[u8]) -> Result<SavedModelHeader, SavedModelError> {
    let stopped = RefCell::new(None);
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let scanned =
        de::Deserializer::deserialize_map(&mut deserializer, HeaderVisitor { stopped: &stopped });
    if let Some(header) = stopped.into_inner() {
        return Ok(header);
    }
    scanned.map_err(|error| SavedModelError::Malformed {
        reason: error.to_string(),
    })
}

/// The keys of a saved document.
enum DocumentKey {
    Kind,
    Version,
    Model,
    Other,
}

impl<'de> serde::Deserialize<'de> for DocumentKey {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl Visitor<'_> for KeyVisitor {
            type Value = DocumentKey;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a saved-model document key")
            }
            fn visit_str<E: de::Error>(self, key: &str) -> Result<DocumentKey, E> {
                Ok(match key {
                    "kind" => DocumentKey::Kind,
                    "version" => DocumentKey::Version,
                    "model" => DocumentKey::Model,
                    _ => DocumentKey::Other,
                })
            }
        }
        deserializer.deserialize_identifier(KeyVisitor)
    }
}

/// Decodes a whole document, checking the kind and the version as they are
/// read. A typed refusal is recorded in `refusal` and aborts the parse.
struct DocumentSeed<'r, T> {
    kind: &'static str,
    version: u64,
    refusal: &'r RefCell<Option<SavedModelError>>,
    model: PhantomData<T>,
}

impl<T> DocumentSeed<'_, T> {
    fn refuse<E: de::Error>(&self, refusal: SavedModelError) -> E {
        let message = refusal.to_string();
        *self.refusal.borrow_mut() = Some(refusal);
        E::custom(message)
    }
}

impl<'de, T: DeserializeOwned> DeserializeSeed<'de> for DocumentSeed<'_, T> {
    type Value = T;

    fn deserialize<D: de::Deserializer<'de>>(self, deserializer: D) -> Result<T, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de, T: DeserializeOwned> Visitor<'de> for DocumentSeed<'_, T> {
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "a saved {} model document", self.kind)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<T, A::Error> {
        let mut kind_read = false;
        let mut version_read = false;
        let mut model = None;
        while let Some(key) = map.next_key::<DocumentKey>()? {
            match key {
                DocumentKey::Kind => {
                    let found: Option<String> = map.next_value()?;
                    if found.as_deref() != Some(self.kind) {
                        return Err(self.refuse(SavedModelError::Kind {
                            found,
                            expected: self.kind,
                        }));
                    }
                    kind_read = true;
                }
                DocumentKey::Version => {
                    let found: Option<u64> = map.next_value()?;
                    if found != Some(self.version) {
                        return Err(self.refuse(SavedModelError::Version {
                            kind: self.kind,
                            found,
                            expected: self.version,
                        }));
                    }
                    version_read = true;
                }
                DocumentKey::Model => {
                    // The model is decoded only once its kind and version are
                    // known to be this reader's, so a model of another format
                    // is refused by its header, never by a parse error inside it.
                    if !kind_read {
                        return Err(self.refuse(SavedModelError::Kind {
                            found: None,
                            expected: self.kind,
                        }));
                    }
                    if !version_read {
                        return Err(self.refuse(SavedModelError::Version {
                            kind: self.kind,
                            found: None,
                            expected: self.version,
                        }));
                    }
                    model = Some(map.next_value::<T>()?);
                }
                DocumentKey::Other => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        if !kind_read {
            return Err(self.refuse(SavedModelError::Kind {
                found: None,
                expected: self.kind,
            }));
        }
        model.ok_or_else(|| {
            self.refuse(SavedModelError::Malformed {
                reason: "the document has no model".to_string(),
            })
        })
    }
}

/// Reads a document's kind and version. A document that names both before its
/// end stops the parse there, by recording the header in `stopped` and
/// returning an error; one that does not is read to its end.
struct HeaderVisitor<'c> {
    stopped: &'c RefCell<Option<SavedModelHeader>>,
}

impl<'de> Visitor<'de> for HeaderVisitor<'_> {
    type Value = SavedModelHeader;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a saved-model document")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<SavedModelHeader, A::Error> {
        let mut header = SavedModelHeader {
            kind: None,
            version: None,
        };
        while let Some(key) = map.next_key::<DocumentKey>()? {
            match key {
                // A value of another type names no kind or version.
                DocumentKey::Kind => {
                    header.kind = match map.next_value::<serde_json::Value>()? {
                        serde_json::Value::String(kind) => Some(kind),
                        _ => None,
                    }
                }
                DocumentKey::Version => {
                    header.version = map.next_value::<serde_json::Value>()?.as_u64();
                }
                DocumentKey::Model | DocumentKey::Other => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            if header.kind.is_some() && header.version.is_some() {
                *self.stopped.borrow_mut() = Some(header);
                // Stop here: the rest of the document is the model, which the
                // header does not need. `serde_json` would otherwise require
                // the map to end.
                return Err(de::Error::custom("saved-model header read"));
            }
        }
        Ok(header)
    }
}

/// Write a saved document durably and atomically. The bytes go to a sibling
/// temporary file named for this save, whose data is synced before it is
/// renamed over `path`. A failed write, sync or rename removes the temporary
/// file, so `path` keeps its previous document. On Unix the directory is then
/// synced, so a save that returns `Ok` survives a crash; a directory that
/// cannot be synced fails the save, with the new document in place and its
/// durability unknown. `std` has no directory handle to sync on Windows: there
/// the rename is atomic and its durability is the filesystem's.
pub fn write_saved_model(path: &Path, bytes: &[u8]) -> Result<(), SavedModelError> {
    let io = |source: std::io::Error| SavedModelError::Io {
        path: path.display().to_string(),
        source,
    };
    let name = path.file_name().ok_or_else(|| {
        io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a file path",
        ))
    })?;
    let mut temporary = name.to_os_string();
    temporary.push(format!(
        ".tmp{}-{}",
        std::process::id(),
        TEMPORARY_FILES.fetch_add(1, Ordering::Relaxed)
    ));
    let temporary = path.with_file_name(temporary);
    std::fs::File::create(&temporary)
        .and_then(|mut file| {
            file.write_all(bytes)?;
            file.sync_all()
        })
        .and_then(|()| std::fs::rename(&temporary, path))
        .map_err(|error| {
            drop(std::fs::remove_file(&temporary));
            io(error)
        })?;
    // `std` opens no directory handle off Unix; see above.
    #[cfg(unix)]
    sync_directory_of(path).map_err(|error| {
        io(std::io::Error::new(
            error.kind(),
            format!(
                "the saved model is in place, but its directory could not be synced, so its durability is unknown: {error}"
            ),
        ))
    })?;
    Ok(())
}

/// Sync the directory holding `path`, so the rename that put it there is
/// durable.
#[cfg(unix)]
fn sync_directory_of(path: &Path) -> std::io::Result<()> {
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    std::fs::File::open(directory)?.sync_all()
}

/// The text of a saved document.
pub fn read_saved_model_file(path: &Path) -> Result<String, SavedModelError> {
    std::fs::read_to_string(path).map_err(|source| SavedModelError::Io {
        path: path.display().to_string(),
        source,
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
    fn a_header_is_read_without_the_model_and_names_only_what_the_document_names() {
        let header = |text: &str| saved_model_header(text.as_bytes());
        // The model after a complete header is never parsed, even when it is
        // not JSON at all.
        assert_eq!(
            header("{\"kind\":\"test\",\"version\":3,\"model\": not json").unwrap(),
            SavedModelHeader {
                kind: Some("test".to_string()),
                version: Some(3),
            }
        );
        // A document without a version, and one whose kind is not a string,
        // is read to its end and names what it names.
        assert_eq!(
            header("{\"kind\":\"test\",\"model\":[1,2]}").unwrap(),
            SavedModelHeader {
                kind: Some("test".to_string()),
                version: None,
            }
        );
        assert_eq!(
            header("{\"schema\":\"old\",\"kind\":7,\"version\":2}").unwrap(),
            SavedModelHeader {
                kind: None,
                version: Some(2),
            }
        );
        assert!(matches!(
            header("[1, 2]"),
            Err(SavedModelError::Malformed { .. })
        ));
        assert!(matches!(
            header("{\"kind\":\"test\""),
            Err(SavedModelError::Malformed { .. })
        ));
    }

    #[test]
    fn documents_refuse_other_kinds_versions_and_non_finite_floats() {
        let values = vec![0.1_f64, 2.5e300];
        let text = saved_model_text("test", 3, &values).unwrap();
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>(&text, "test", 4),
            Err(SavedModelError::Version {
                found: Some(3),
                expected: 4,
                ..
            })
        ));
        assert!(text.starts_with("{\"kind\":\"test\",\"version\":3,"));
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>(
                &text.replace("\"kind\":\"test\"", "\"kind\":\"other\""),
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
        write_saved_model(&scratch.0, text.as_bytes()).unwrap();
        assert_eq!(read_saved_model_file(&scratch.0).unwrap(), text);
        std::fs::remove_file(&scratch.0).unwrap();
        assert!(matches!(
            read_saved_model_file(&scratch.0),
            Err(SavedModelError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound
        ));
    }

    /// A scratch directory removed with its contents when dropped.
    struct ScratchDirectory(std::path::PathBuf);

    impl ScratchDirectory {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!(
                "gam-saved-model-dir-{}-{}",
                std::process::id(),
                TEMPORARY_FILES.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&directory).unwrap();
            Self(directory)
        }

        fn entries(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for ScratchDirectory {
        fn drop(&mut self) {
            drop(std::fs::remove_dir_all(&self.0));
        }
    }

    /// gam#3054: a save replaces the previous document whole and leaves no
    /// temporary file; a save that fails after writing its temporary file
    /// removes it and leaves what was at the path untouched; a missing
    /// directory keeps its `NotFound` kind.
    #[test]
    fn a_save_replaces_the_document_whole_or_leaves_the_path_untouched_3054() {
        let directory = ScratchDirectory::new();
        let path = directory.0.join("model.json");
        write_saved_model(&path, b"first").unwrap();
        write_saved_model(&path, b"second").unwrap();
        assert_eq!(read_saved_model_file(&path).unwrap(), "second");
        assert_eq!(directory.entries(), vec!["model.json".to_string()]);

        // The temporary file is written and synced, then the rename over a
        // non-empty directory fails.
        let occupied = directory.0.join("occupied.json");
        std::fs::create_dir(&occupied).unwrap();
        std::fs::write(occupied.join("kept"), b"kept").unwrap();
        assert!(matches!(
            write_saved_model(&occupied, b"third"),
            Err(SavedModelError::Io { path, .. }) if path == occupied.display().to_string()
        ));
        assert_eq!(
            directory.entries(),
            vec!["model.json".to_string(), "occupied.json".to_string()]
        );
        assert_eq!(std::fs::read(occupied.join("kept")).unwrap(), b"kept");

        assert!(matches!(
            write_saved_model(&directory.0.join("missing").join("model.json"), b"fourth"),
            Err(SavedModelError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound
        ));
        assert_eq!(read_saved_model_file(&path).unwrap(), "second");
    }

    #[test]
    fn documents_refuse_a_missing_model_a_missing_kind_and_a_path_without_a_file_name() {
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>("{\"kind\": \"test\", \"version\": 3}", "test", 3),
            Err(SavedModelError::Malformed { reason }) if reason == "the document has no model"
        ));
        assert!(matches!(
            read_saved_model_text::<Vec<f64>>("{\"version\": 3, \"model\": []}", "test", 3),
            Err(SavedModelError::Kind {
                found: None,
                expected: "test"
            })
        ));
        assert!(matches!(
            write_saved_model(Path::new("/"), b"{}"),
            Err(SavedModelError::Io { source, .. })
                if source.kind() == std::io::ErrorKind::InvalidInput
                    && source.to_string() == "not a file path"
        ));
    }
}
