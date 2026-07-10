//! Checkpoint / resume for SAE-manifold fits — the SPEC-sanctioned
//! wall-survival mechanism ("Work survives walls via checkpoint/resume, not via
//! minting partial results").
//!
//! A cluster job killed at its wall must lose at most one checkpoint interval of
//! work and NEVER produce a fit object from the incomplete run. A fresh job
//! resumes the optimization from the last banked incumbent and only the RESUMED
//! run's own convergence mints a fit (`SaeManifoldOuterObjective::into_fitted` is
//! reached only when the outer bridge concludes on its own — an interrupted
//! worker unwinds through the cancel flag and never reaches it).
//!
//! Wiring: the outer objective computes the [`SaeCheckpointFingerprint`] once at
//! construction on the pristine full-`N` target, banks a checkpoint
//! (best-effort, atomic) at every MATERIAL improvement of the outer best cost
//! (`SaeManifoldOuterObjective::bank_checkpoint`), resumes via
//! `try_resume_from_checkpoint` at fit entry, and discards the file when a
//! converged fit is minted (`remove_checkpoint` — wall survival is not
//! cross-fit caching; `persistent_warm_start` owns that).
//!
//! # What is banked
//!
//! The checkpoint holds the *fittable* incumbent state — per-atom decoder
//! coefficients, latent coordinates, log-amplitudes and the curvature-homotopy
//! dial, plus the shared per-row assignment logits — the same mutable state
//! [`crate::manifold::term::SaeManifoldMutableState`] captures for the in-fit
//! keep-best (`best_fit_incumbent`). The transient basis matrices
//! (`basis_values`, `basis_jacobian`, second-jet caches, the intrinsic roughness
//! Gram) are NOT stored: they are a deterministic function of `(coords,
//! evaluator, η)` and are rebuilt on resume by `refresh_basis_from_current_coords`
//! against the freshly constructed term's evaluators. The current ρ (flat outer
//! vector) and the outer termination-ledger counters ride alongside so the
//! resumed outer search opens at the banked coordinate with the accounting
//! intact.
//!
//! # Fingerprint
//!
//! A checkpoint is never resumed against different data. The fingerprint carries
//! the target shape (`n_rows`, `n_cols`), the dictionary size (`k_atoms`) and a
//! SHA-256 content hash of the target matrix; [`SaeFitCheckpoint::verify_compatible`]
//! refuses (typed `Err`) on any mismatch of the schema tag or the fingerprint.
//!
//! # Schema tag
//!
//! [`SAE_FIT_CHECKPOINT_SCHEMA`] follows the `persistent_warm_start` convention:
//! a hand-bumped string tag, deliberately separate from `CARGO_PKG_VERSION`, so a
//! routine library version bump does NOT invalidate an in-flight checkpoint. Bump
//! the trailing version only when the serialized layout changes in a way that
//! makes a prior file unsafe to consume.

use super::term::SaeManifoldTerm;
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// On-disk schema tag. Hand-bumped only on a layout-breaking change (see module
/// docs); [`SaeFitCheckpoint::verify_compatible`] rejects any other value.
pub(crate) const SAE_FIT_CHECKPOINT_SCHEMA: &str = "gam-sae.fit-checkpoint/v1";

/// Data fingerprint: a checkpoint is only ever resumed against the identical
/// fitting problem. Equality of every field (including the SHA-256 content hash
/// of the target) is required by [`SaeFitCheckpoint::verify_compatible`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SaeCheckpointFingerprint {
    pub(crate) n_rows: usize,
    pub(crate) n_cols: usize,
    pub(crate) k_atoms: usize,
    /// Hex SHA-256 of the target matrix (shape + row-major `f64` little-endian
    /// bytes). Distinguishes two same-shape problems with different data.
    pub(crate) content_hash: String,
}

impl SaeCheckpointFingerprint {
    /// Fingerprint the full-data fitting problem: target shape + content hash and
    /// the dictionary size. Computed once when the objective is constructed.
    pub(crate) fn of_target(target: ArrayView2<'_, f64>, k_atoms: usize) -> Self {
        let (n_rows, n_cols) = target.dim();
        let mut hasher = Sha256::new();
        hasher.update((n_rows as u64).to_le_bytes());
        hasher.update((n_cols as u64).to_le_bytes());
        hasher.update((k_atoms as u64).to_le_bytes());
        // Row-major traversal — deterministic regardless of the view's stride.
        for row in target.rows() {
            for &v in row {
                hasher.update(v.to_le_bytes());
            }
        }
        let digest = hasher.finalize();
        let mut content_hash = String::with_capacity(digest.len() * 2);
        for byte in digest {
            content_hash.push_str(&format!("{byte:02x}"));
        }
        Self {
            n_rows,
            n_cols,
            k_atoms,
            content_hash,
        }
    }
}

/// Outer termination-ledger counters carried across a resume. The wall clock is
/// intentionally NOT persisted: a fresh job restarts its own wall measurement
/// (SPEC bans wall-clock budgets), while the evaluation/improvement tallies and
/// best cost continue as checkpoint telemetry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SaeCheckpointLedger {
    pub(crate) evals: u64,
    pub(crate) last_improvement_eval: u64,
    pub(crate) best_cost: Option<f64>,
}

/// Per-atom banked state. Mirrors the recoverable half of
/// [`crate::manifold::term::SaeManifoldAtomSnapshot`] plus the slow
/// log-amplitude gauge; the basis matrices are rebuilt on resume.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SaeCheckpointAtom {
    /// `(M_k, p)` decoder block, nested row-major (numpy `.tolist()` shape).
    pub(crate) decoder_coefficients: Vec<Vec<f64>>,
    /// `(N, d_k)` latent coordinates, nested row-major.
    pub(crate) coords: Vec<Vec<f64>>,
    pub(crate) log_amplitude: f64,
    pub(crate) homotopy_eta: f64,
}

/// A resumable SAE-manifold fit checkpoint. Written atomically at every material
/// improvement of the fit-level incumbent; loaded and installed as the warm
/// start by a fresh job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SaeFitCheckpoint {
    pub(crate) schema: String,
    pub(crate) created_unix_secs: u64,
    pub(crate) fingerprint: SaeCheckpointFingerprint,
    /// The outer coordinate ρ the search had settled on, flat (see
    /// [`crate::manifold::rho::SaeManifoldRho::to_flat`]).
    pub(crate) rho_flat: Vec<f64>,
    pub(crate) ledger: SaeCheckpointLedger,
    /// The banked incumbent's reconstruction EV (telemetry alongside the
    /// objective-keyed ordering; sanitized finite by the writer — serde_json
    /// refuses non-finite floats).
    pub(crate) incumbent_ev: f64,
    pub(crate) atoms: Vec<SaeCheckpointAtom>,
    /// `(N, K)` shared assignment logits, nested row-major.
    pub(crate) logits: Vec<Vec<f64>>,
}

impl SaeFitCheckpoint {
    /// Content-addressed default store path, following the
    /// `persistent_warm_start` convention (`temp_dir()/gam/...`): magic by
    /// default, no user-supplied path. On cluster jobs whose `TMPDIR` points at
    /// persistent project storage (the MSI sbatch contract) this survives the
    /// wall; a re-submitted job on the SAME data finds it by content hash.
    pub(crate) fn default_store_path(fingerprint: &SaeCheckpointFingerprint) -> PathBuf {
        std::env::temp_dir()
            .join("gam")
            .join("sae_fit_checkpoint")
            .join("v1")
            .join(format!("{}.json", &fingerprint.content_hash))
    }

    /// Capture the current fittable incumbent from the term: per-atom decoder /
    /// coords / log-amplitude / η plus the shared logits — the same recoverable
    /// state `SaeManifoldMutableState` snapshots (basis matrices are rebuilt on
    /// resume from `(coords, evaluator, η)`).
    pub(crate) fn capture(
        term: &SaeManifoldTerm,
        fingerprint: &SaeCheckpointFingerprint,
        rho_flat: &[f64],
        ledger: SaeCheckpointLedger,
        incumbent_ev: f64,
    ) -> Self {
        let atoms = term
            .atoms
            .iter()
            .enumerate()
            .map(|(atom_idx, atom)| SaeCheckpointAtom {
                decoder_coefficients: rows_of(&atom.decoder_coefficients),
                coords: rows_of(&term.assignment.coords[atom_idx].as_matrix()),
                log_amplitude: atom.log_amplitude,
                homotopy_eta: atom.homotopy_eta,
            })
            .collect();
        Self {
            schema: SAE_FIT_CHECKPOINT_SCHEMA.to_string(),
            created_unix_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            fingerprint: fingerprint.clone(),
            rho_flat: rho_flat.to_vec(),
            ledger,
            incumbent_ev,
            atoms,
            logits: rows_of(&term.assignment.logits),
        }
    }

    /// Install the banked incumbent into a freshly constructed term: assign the
    /// per-atom decoder / coords / log-amplitude / η and the shared logits in
    /// place, then rebuild the basis caches from the restored coordinates
    /// (deterministic, exactly as `restore_mutable_state` does). Typed `Err` on
    /// any shape mismatch — a checkpoint that fails to install must never
    /// silently degrade into a partial resume. Call only after
    /// [`Self::verify_compatible`] has accepted the fingerprint.
    pub(crate) fn install_into(&self, term: &mut SaeManifoldTerm) -> Result<(), String> {
        if self.atoms.len() != term.atoms.len() {
            return Err(format!(
                "checkpoint install: atom count {} != term {}",
                self.atoms.len(),
                term.atoms.len()
            ));
        }
        for (atom_idx, banked) in self.atoms.iter().enumerate() {
            let decoder = array2_from_rows(&banked.decoder_coefficients)?;
            let atom = &mut term.atoms[atom_idx];
            if decoder.dim() != atom.decoder_coefficients.dim() {
                return Err(format!(
                    "checkpoint install: atom {atom_idx} decoder {:?} != term {:?}",
                    decoder.dim(),
                    atom.decoder_coefficients.dim()
                ));
            }
            atom.decoder_coefficients.assign(&decoder);
            atom.log_amplitude = banked.log_amplitude;
            atom.homotopy_eta = banked.homotopy_eta;
            let coords = array2_from_rows(&banked.coords)?;
            let slot = &mut term.assignment.coords[atom_idx];
            if coords.dim() != (slot.n_obs(), slot.latent_dim()) {
                return Err(format!(
                    "checkpoint install: atom {atom_idx} coords {:?} != term ({}, {})",
                    coords.dim(),
                    slot.n_obs(),
                    slot.latent_dim()
                ));
            }
            let flat: Vec<f64> = coords.iter().copied().collect();
            slot.set_flat(ndarray::ArrayView1::from(&flat));
        }
        let logits = array2_from_rows(&self.logits)?;
        if logits.dim() != term.assignment.logits.dim() {
            return Err(format!(
                "checkpoint install: logits {:?} != term {:?}",
                logits.dim(),
                term.assignment.logits.dim()
            ));
        }
        term.assignment.logits.assign(&logits);
        term.refresh_basis_from_current_coords()
    }

    /// Atomically persist to `path`: encode to JSON, write a sibling temp file,
    /// `fsync` it, then `rename` over the destination. A reader therefore only
    /// ever observes a complete file — a crash mid-write leaves the previous
    /// checkpoint (or nothing) intact, never a torn payload. Best-effort at the
    /// callsite (a checkpoint write must never abort a fit), but the error is
    /// returned so the caller can log it.
    pub(crate) fn save_atomic(&self, path: &Path) -> Result<(), String> {
        use std::io::Write;
        let bytes = serde_json::to_vec(self)
            .map_err(|e| format!("SaeFitCheckpoint::save_atomic: encode: {e}"))?;
        // Sibling temp file so the rename is same-filesystem (hence atomic). A
        // unique suffix keeps concurrent writers to distinct checkpoints from
        // colliding on the temp name.
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(format!(".tmp.{}", std::process::id()));
        let tmp = std::path::PathBuf::from(tmp);
        let write_result = (|| -> std::io::Result<()> {
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            Ok(())
        })();
        if let Err(write_error) = write_result {
            return match remove_file_if_present(&tmp) {
                Ok(()) => Err(format!(
                    "SaeFitCheckpoint::save_atomic: write temp {}: {write_error}",
                    tmp.display()
                )),
                Err(cleanup_error) => Err(format!(
                    "SaeFitCheckpoint::save_atomic: write temp {}: {write_error}; cleanup failed: \
                     {cleanup_error}",
                    tmp.display()
                )),
            };
        }
        match std::fs::rename(&tmp, path) {
            Ok(()) => Ok(()),
            Err(rename_error) => match remove_file_if_present(&tmp) {
                Ok(()) => Err(format!(
                    "SaeFitCheckpoint::save_atomic: rename into {}: {rename_error}",
                    path.display()
                )),
                Err(cleanup_error) => Err(format!(
                    "SaeFitCheckpoint::save_atomic: rename into {}: {rename_error}; cleanup \
                     failed: {cleanup_error}",
                    path.display()
                )),
            },
        }
    }

    /// Load and decode a checkpoint file. Errors on a missing / unreadable file
    /// or a malformed payload — a corrupt checkpoint must never silently degrade
    /// into a partial resume.
    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("SaeFitCheckpoint::load: read {}: {e}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| format!("SaeFitCheckpoint::load: decode {}: {e}", path.display()))
    }

    /// Refuse (typed `Err`) unless the schema tag matches and the data
    /// fingerprint is identical to `expected`. `expected_rho_len` guards the
    /// `from_flat` reconstruction so a shape-mismatched ρ can never panic the
    /// resume.
    pub(crate) fn verify_compatible(
        &self,
        expected: &SaeCheckpointFingerprint,
        expected_rho_len: usize,
    ) -> Result<(), String> {
        if self.schema != SAE_FIT_CHECKPOINT_SCHEMA {
            return Err(format!(
                "SAE checkpoint schema {:?} != expected {:?}; refusing to resume",
                self.schema, SAE_FIT_CHECKPOINT_SCHEMA
            ));
        }
        if &self.fingerprint != expected {
            return Err(format!(
                "SAE checkpoint data fingerprint mismatch (checkpoint {:?} != current {:?}); \
                 refusing to resume a fit against different data",
                self.fingerprint, expected
            ));
        }
        if self.atoms.len() != expected.k_atoms {
            return Err(format!(
                "SAE checkpoint atom count {} != current dictionary size {}; refusing to resume",
                self.atoms.len(),
                expected.k_atoms
            ));
        }
        if self.rho_flat.len() != expected_rho_len {
            return Err(format!(
                "SAE checkpoint ρ length {} != current outer-coordinate length {}; refusing to \
                 resume",
                self.rho_flat.len(),
                expected_rho_len
            ));
        }
        Ok(())
    }
}

/// Remove an atomic-write scratch file when it exists. A failed create leaves
/// no file to remove; every other cleanup failure is surfaced to the caller.
fn remove_file_if_present(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Nested-row `Vec<Vec<f64>>` → `Array2` with a rectangular-shape check. Returns
/// the `(rows, cols)` matrix; an empty outer vec is the `(0, 0)` matrix.
pub(crate) fn array2_from_rows(rows: &[Vec<f64>]) -> Result<Array2<f64>, String> {
    let n = rows.len();
    let cols = rows.first().map(|r| r.len()).unwrap_or(0);
    if rows.iter().any(|r| r.len() != cols) {
        return Err(format!(
            "checkpoint matrix is ragged (expected {cols} columns in every row)"
        ));
    }
    let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Array2::from_shape_vec((n, cols), flat)
        .map_err(|e| format!("checkpoint matrix reshape ({n}×{cols}): {e}"))
}

/// `Array2` → nested-row `Vec<Vec<f64>>` (the on-disk shape).
pub(crate) fn rows_of(matrix: &Array2<f64>) -> Vec<Vec<f64>> {
    matrix.rows().into_iter().map(|r| r.to_vec()).collect()
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use ndarray::array;

    fn sample_checkpoint() -> SaeFitCheckpoint {
        SaeFitCheckpoint {
            schema: SAE_FIT_CHECKPOINT_SCHEMA.to_string(),
            created_unix_secs: 12_345,
            fingerprint: SaeCheckpointFingerprint {
                n_rows: 4,
                n_cols: 3,
                k_atoms: 2,
                content_hash: "deadbeef".to_string(),
            },
            rho_flat: vec![-1.0, 0.5, 0.25],
            ledger: SaeCheckpointLedger {
                evals: 7,
                last_improvement_eval: 5,
                best_cost: Some(-3.5),
            },
            incumbent_ev: 0.42,
            atoms: vec![
                SaeCheckpointAtom {
                    decoder_coefficients: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                    coords: vec![vec![0.1], vec![0.2], vec![0.3], vec![0.4]],
                    log_amplitude: 0.7,
                    homotopy_eta: 1.0,
                },
                SaeCheckpointAtom {
                    decoder_coefficients: vec![vec![-1.0, -2.0, -3.0]],
                    coords: vec![
                        vec![0.9, 0.8],
                        vec![0.7, 0.6],
                        vec![0.5, 0.4],
                        vec![0.3, 0.2],
                    ],
                    log_amplitude: -0.3,
                    homotopy_eta: 0.5,
                },
            ],
            logits: vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![0.5, 0.5],
                vec![0.2, 0.8],
            ],
        }
    }

    /// Save → load reproduces the checkpoint value-for-value (the resume
    /// round-trip contract).
    #[test]
    fn save_load_round_trips() {
        let dir = std::env::temp_dir().join(format!("gam-sae-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.json");
        let ckpt = sample_checkpoint();
        ckpt.save_atomic(&path).expect("save");
        let loaded = SaeFitCheckpoint::load(&path).expect("load");
        assert_eq!(loaded, ckpt, "checkpoint must round-trip through disk");
        std::fs::remove_file(&path).expect("remove round-trip checkpoint");
    }

    /// A data-fingerprint mismatch is a typed refusal, and a matching one passes.
    #[test]
    fn fingerprint_mismatch_refuses() {
        let ckpt = sample_checkpoint();
        let matching = ckpt.fingerprint.clone();
        ckpt.verify_compatible(&matching, ckpt.rho_flat.len())
            .expect("matching fingerprint + rho length must verify");

        let mut mismatched = ckpt.fingerprint.clone();
        mismatched.content_hash = "different".to_string();
        let err = ckpt
            .verify_compatible(&mismatched, ckpt.rho_flat.len())
            .expect_err("content-hash mismatch must refuse");
        assert!(
            err.contains("fingerprint mismatch"),
            "unexpected error: {err}"
        );

        // A ρ-length mismatch (e.g. K changed) also refuses instead of panicking
        // a later `from_flat`.
        let err = ckpt
            .verify_compatible(&matching, ckpt.rho_flat.len() + 1)
            .expect_err("rho-length mismatch must refuse");
        assert!(err.contains("ρ length"), "unexpected error: {err}");
    }

    /// A wrong schema tag refuses (a layout-breaking bump walls off old files).
    #[test]
    fn wrong_schema_refuses() {
        let mut ckpt = sample_checkpoint();
        ckpt.schema = "gam-sae.fit-checkpoint/v0".to_string();
        let fp = ckpt.fingerprint.clone();
        let err = ckpt
            .verify_compatible(&fp, ckpt.rho_flat.len())
            .expect_err("wrong schema must refuse");
        assert!(err.contains("schema"), "unexpected error: {err}");
    }

    /// The atomic write leaves no torn file on a simulated failure: a save into a
    /// non-existent directory returns `Err` and leaves neither the destination
    /// nor a temp sibling behind (the temp file is created in the SAME missing
    /// directory, so `File::create` fails before any bytes are written).
    #[test]
    fn atomic_write_leaves_no_torn_file_on_failure() {
        let missing = std::env::temp_dir()
            .join(format!("gam-sae-ckpt-missing-{}", std::process::id()))
            .join("nested")
            .join("ckpt.json");
        let ckpt = sample_checkpoint();
        let err = ckpt
            .save_atomic(&missing)
            .expect_err("save into missing dir must fail");
        assert!(err.contains("write temp"), "unexpected error: {err}");
        assert!(
            !missing.exists(),
            "destination must not exist after a failed save"
        );
        let mut tmp = missing.as_os_str().to_owned();
        tmp.push(format!(".tmp.{}", std::process::id()));
        assert!(
            !std::path::PathBuf::from(tmp).exists(),
            "no temp sibling may be left behind on failure"
        );
    }

    /// A successful save left no temp sibling behind (the rename consumed it).
    #[test]
    fn successful_save_leaves_no_temp_sibling() {
        let dir = std::env::temp_dir().join(format!("gam-sae-ckpt-clean-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("clean.json");
        sample_checkpoint().save_atomic(&path).expect("save");
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(format!(".tmp.{}", std::process::id()));
        assert!(
            !std::path::PathBuf::from(tmp).exists(),
            "temp sibling must be renamed away after a successful save"
        );
        assert!(
            path.exists(),
            "destination must exist after a successful save"
        );
        std::fs::remove_file(&path).expect("remove successful-save checkpoint");
    }

    #[test]
    fn array2_from_rows_round_trips_and_rejects_ragged() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let rows = rows_of(&m);
        let back = array2_from_rows(&rows).expect("rectangular");
        assert_eq!(back, m);
        let ragged = vec![vec![1.0, 2.0], vec![3.0]];
        assert!(
            array2_from_rows(&ragged).is_err(),
            "ragged input must error"
        );
    }

    /// The fingerprint is content-sensitive: two same-shape targets with
    /// different values hash differently; identical targets hash identically.
    #[test]
    fn fingerprint_is_content_sensitive() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[1.0, 2.0], [3.0, 4.5]];
        let fa = SaeCheckpointFingerprint::of_target(a.view(), 2);
        let fa2 = SaeCheckpointFingerprint::of_target(a.view(), 2);
        let fb = SaeCheckpointFingerprint::of_target(b.view(), 2);
        assert_eq!(fa, fa2, "identical data must fingerprint identically");
        assert_ne!(fa, fb, "different data must fingerprint differently");
    }
}
