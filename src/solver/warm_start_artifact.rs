//! Cross-fit warm-start artifact: a descriptor-indexed, function-space
//! snapshot of a converged fit, designed so a *related* later fit (a
//! leave-one-subject-out fold, a re-fit on a different row population, a
//! different reduced width) can warm-start from it even though the exact
//! response-keyed inner cache (`persistent_warm_start.rs`) misses.
//!
//! The artifact is keyed by *structural identity*, not by data bytes. Two
//! fits of the same term family (same role, same variables, same basis kind
//! and the same STRUCTURAL basis parameters — degree, #centers, nullspace
//! order, …) map to the same [`TermIdentityKey`] even when their realized
//! `centers` / `input_scales` / `length_scale` differ across folds. That is
//! precisely what lets the smoothing parameter ρ transfer survive a fold:
//! "same term, different rows" matches; "3 PCs vs 10 PCs" or "different
//! #centers" deliberately does NOT.
//!
//! Correctness is free. A warm start only sets the *starting iterate*; the
//! outer REML/BFGS loop and the inner constrained Newton solve still run to
//! their KKT certificate, so the converged answer is identical to a cold
//! start within tolerance. Every field that flows back into the solver is
//! finite-guarded at consume time; any anomaly falls back to cold.

use gam_runtime::warm_start::key::{Fingerprint, Fingerprinter};
use serde::{Deserialize, Serialize};

/// On-disk schema version for [`FitArtifact`]. Bump when the serialized
/// layout changes in a way that makes prior payloads unsafe to consume.
pub(crate) const FIT_ARTIFACT_SCHEMA: u32 = 1;

/// Saturation magnitude past which a copied ρ coordinate is considered
/// pinned at the outer optimizer's box and is NOT transferred. Mirrors the
/// persist-side gate in `families/custom_family/persistent_warm_start.rs` and the
/// `[CACHE] hit-clamp` policy in `solver/outer_strategy.rs`.
pub(crate) const RHO_SATURATION: f64 = 9.0;

/// Structural role a term plays in the (possibly multi-channel) model.
///
/// Derived from the block name / channel at capture time. The role is part
/// of the term identity so a "mean" smooth never transfers ρ to a
/// "log-slope" smooth of the same variables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TermRole {
    /// Location / mean channel (the default for a single-channel family).
    Mean,
    /// Log-scale / dispersion / log-slope channel.
    LogSlope,
    /// Any other channel (multinomial categories, frailty, …).
    Generic,
}

impl TermRole {
    /// Stable discriminant byte for hashing.
    fn discriminant(self) -> u8 {
        match self {
            TermRole::Mean => 0,
            TermRole::LogSlope => 1,
            TermRole::Generic => 2,
        }
    }

    /// Heuristic role from a block / channel name. Names are produced by the
    /// family construction layer (e.g. `"<scale>"`, `"logslope"`, `"mean"`);
    /// the classification is structural and deliberately coarse.
    pub fn from_block_name(name: &str) -> TermRole {
        let lower = name.to_ascii_lowercase();
        if lower.contains("logslope")
            || lower.contains("log_slope")
            || lower.contains("scale")
            || lower.contains("sigma")
            || lower.contains("dispersion")
            || lower.contains("disp")
        {
            TermRole::LogSlope
        } else if lower.contains("mean") || lower.contains("loc") || lower.contains("marginal") {
            TermRole::Mean
        } else {
            TermRole::Generic
        }
    }
}

/// Stable structural identity of one term, used to match a parent term to a
/// new-fit term across folds / row populations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TermIdentityKey(pub Fingerprint);

/// Build a term identity at the *block-spec* layer (`fit_custom_family` and
/// friends), where the full `BasisMetadata` / variable names are no longer
/// reachable — the design has already been assembled into a
/// [`crate::families::custom_family::block_spec::ParameterBlockSpec`].
///
/// The block `name` (e.g. `"s(x)"`, `"<scale>"`) is produced by the formula /
/// construction layer and is **fold-invariant**: it encodes the variables and
/// basis kind and does not change when rows are dropped for an LOSO fold. The
/// penalty *structure* (count, precision labels, nullspace dimensions) is also
/// fold-invariant in SHAPE — only the matrix values change across folds, and
/// we hash only the structure, never the values. So this identity matches
/// "same model, different rows" while splitting on a genuine structural change
/// (a different #penalties, a different label set, a different basis size).
///
/// `reduced_width` is the realized per-block coefficient dimension
/// (`spec.design.ncols()`) — the basis column count *after* the
/// identifiability reduction, which is the load-bearing dimension of the
/// block's β. It is fold-invariant within one model (LOSO drops rows, never
/// columns) but DIFFERS across models whose spatial basis collapses to a lower
/// effective support (e.g. a duchon marginal that realizes p=21 on one disease
/// and p=45 on another). Folding it into the identity is what makes a p=37 fit
/// refuse to match a p=85 artifact: without it, two models with the same block
/// name / penalty-label / nullspace SHAPE but different realized β-width hash to
/// the SAME [`TermIdentityKey`] (and hence the same [`FitDescriptor`] key),
/// producing the spurious "cached inner beta has length 85, but blocks require
/// length 37" lookups. With it, only fits whose per-block β actually live in the
/// same-dimension coordinate system match — so the gauge β-projection is always
/// well-posed and same-width folds transfer ρ AND β, while different-width
/// models never collide.
///
/// NOTE (architect-assumption mismatch): the original design routed identity
/// through `SmoothTerm.metadata`, but at this layer that metadata has already
/// been compiled away. The block name + penalty structure + realized reduced
/// width is the honest, fold-invariant identity available here.
pub fn term_identity_from_block(
    role: TermRole,
    block_name: &str,
    precision_labels: &[Option<String>],
    nullspace_dims: &[usize],
    reduced_width: usize,
) -> TermIdentityKey {
    let mut fp = Fingerprinter::new();
    fp.absorb_tag(b"fit-artifact-block-identity-v2");
    fp.absorb_u64(b"role", u64::from(role.discriminant()));
    fp.absorb_str(b"block_name", block_name);
    fp.absorb_u64(b"n_penalties", precision_labels.len() as u64);
    for label in precision_labels {
        match label {
            Some(l) => fp.absorb_str(b"label", l),
            None => fp.absorb_tag(b"label-none"),
        }
    }
    fp.absorb_u64(b"n_nullspace", nullspace_dims.len() as u64);
    for d in nullspace_dims {
        fp.absorb_u64(b"nullspace_dim", *d as u64);
    }
    fp.absorb_u64(b"reduced_width", reduced_width as u64);
    TermIdentityKey(fp.finalize())
}

/// Signature of the response (family + dimensionality) a fit targeted.
/// Carried for diagnostics; deliberately NOT part of the descriptor key so
/// an LOSO fold matches a full-data parent (only the structural term set
/// keys the descriptor).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseSig {
    pub family_kind: String,
    pub n_response_channels: usize,
}

/// Tag describing which rows a fit saw. Carried for diagnostics only; the
/// descriptor key excludes it so different row populations (folds) match.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowPopulationTag {
    pub n_rows: usize,
    /// Optional caller-supplied label (fold id, disease, …).
    pub label: Option<String>,
}

/// Identity descriptor of a whole fit: which family, which structural terms,
/// what response, optionally which rows. The descriptor *key*
/// ([`FitDescriptor::descriptor_key`]) hashes only the family kind and the
/// SORTED term identities — it excludes row population and response bytes —
/// so an LOSO fold of the same model matches a prior full-data artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitDescriptor {
    pub family_kind: String,
    pub term_identities: Vec<TermIdentityKey>,
    pub response_signature: ResponseSig,
    pub row_population: Option<RowPopulationTag>,
}

impl FitDescriptor {
    /// Stable descriptor key = hash(family_kind ⊕ sorted term identities),
    /// EXCLUDING row population and response bytes. This is the keyspace an
    /// LOSO fold and its full-data parent share.
    pub fn descriptor_key(&self) -> Fingerprint {
        let mut fp = Fingerprinter::new();
        fp.absorb_tag(b"fit-artifact-descriptor-v1");
        fp.absorb_str(b"family_kind", &self.family_kind);
        // Sort the term identities so block ORDER does not split the key:
        // the same model assembled in a different block order is the same
        // descriptor.
        let mut keys: Vec<[u8; 32]> = self
            .term_identities
            .iter()
            .map(|k| *k.0.as_bytes())
            .collect();
        keys.sort_unstable();
        fp.absorb_u64(b"n_terms", keys.len() as u64);
        for k in &keys {
            fp.absorb_bytes(b"term", k);
        }
        fp.finalize()
    }
}

/// Per-term captured state. Stores RAW per-term β (lifted from the converged
/// reduced θ via the fit's [`crate::solver::gauge::Gauge`] at capture time —
/// the identifiability transform T is fit-specific and meaningless in another
/// fit, so we persist the gauge-free raw coefficients) plus the term's ρ
/// slice for transfer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TermArtifact {
    pub identity: TermIdentityKey,
    pub role: TermRole,
    /// Serializable structural subset of the term's basis metadata.
    /// `BasisMetadata` itself is not `Serialize` (it carries large
    /// data-derived arrays), so we persist only the fields needed to
    /// re-derive identity and reason about the basis at consume time.
    pub basis_meta: SerializableBasisMeta,
    /// Joint-null absorption rotation captured at fit time, if any. Stored as
    /// a flat row-major matrix so the function-space β projection (Phase 2)
    /// can replay it; `None` when the term carried no rotation.
    pub joint_null_rotation: Option<SerializableMatrix>,
    /// RAW per-term coefficients (post-gauge-lift, pre-identifiability),
    /// concatenated in the term's raw column order.
    pub raw_beta: Vec<f64>,
    /// Converged ρ (log smoothing parameters) for this term's penalties.
    pub rho_for_term: Vec<f64>,
}

impl TermArtifact {
    /// True iff every persisted numeric field is finite (the consume-side
    /// finite-guard precondition).
    pub fn is_finite(&self) -> bool {
        self.raw_beta.iter().all(|v| v.is_finite())
            && self.rho_for_term.iter().all(|v| v.is_finite())
            && self
                .joint_null_rotation
                .as_ref()
                .is_none_or(|m| m.data.iter().all(|v| v.is_finite()))
    }
}

/// A serializable row-major dense matrix snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec<f64>,
}

/// Serializable structural subset of a term's basis metadata. Captures the
/// basis-kind discriminant and the structural parameters used for identity
/// and for diagnostics. Data-derived arrays (centers, basis matrices) are
/// intentionally dropped.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SerializableBasisMeta {
    pub kind: String,
    pub degree: Option<u64>,
    pub num_knots: Option<u64>,
    pub n_centers: Option<u64>,
    pub nullspace_order: Option<u64>,
    pub matern_nu: Option<u64>,
    pub periodic: bool,
}

/// Whole-fit summary numbers carried for selection / logging.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalFitSummary {
    pub outer_objective: f64,
    pub converged: bool,
    pub n_rows: usize,
}

/// Provenance of a per-term transfer, for logging and tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferProvenance {
    /// β was function-projected from the parent (Phase 2).
    Projected,
    /// Only ρ was transferred; β stayed cold (Phase 1).
    RhoOnly,
    /// Nothing transferred; both β and ρ are at their cold defaults.
    Cold,
}

/// The full descriptor-indexed warm-start artifact.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitArtifact {
    pub schema: u32,
    pub created_unix_secs: u64,
    pub descriptor: FitDescriptor,
    pub terms: Vec<TermArtifact>,
    pub global: GlobalFitSummary,
}

impl FitArtifact {
    /// True iff the artifact is structurally usable as warm-start material:
    /// the schema matches, the global summary is finite, and every term's
    /// numeric payload is finite. A failing artifact must be ignored (cold
    /// fallback), never error a fit.
    pub fn is_usable(&self) -> bool {
        self.schema == FIT_ARTIFACT_SCHEMA
            && self.global.outer_objective.is_finite()
            && self.terms.iter().all(TermArtifact::is_finite)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Build a block-layer term identity (the surviving, fold-invariant
    /// identity API). One unlabeled penalty with the given nullspace dim and a
    /// fixed realized reduced width.
    fn block_id(role: TermRole, block_name: &str) -> TermIdentityKey {
        term_identity_from_block(role, block_name, &[None], &[1], 10)
    }

    /// A minimal serializable basis-meta stub, as produced at the block-spec
    /// capture layer.
    fn basis_meta_stub(n_centers: u64) -> SerializableBasisMeta {
        SerializableBasisMeta {
            kind: "block-spec".to_string(),
            degree: None,
            num_knots: None,
            n_centers: Some(n_centers),
            nullspace_order: None,
            matern_nu: None,
            periodic: false,
        }
    }

    #[test]
    fn block_identity_splits_on_block_name() {
        let ka = block_id(TermRole::Mean, "s(x)");
        let kb = block_id(TermRole::Mean, "s(z)");
        assert_ne!(ka, kb, "different block name must split identity");
    }

    #[test]
    fn block_identity_splits_on_role() {
        let mean = block_id(TermRole::Mean, "s(x)");
        let slope = block_id(TermRole::LogSlope, "s(x)");
        assert_ne!(mean, slope, "different role must split identity");
    }

    #[test]
    fn block_identity_splits_on_penalty_structure() {
        let one = term_identity_from_block(TermRole::Mean, "s(x)", &[None], &[1], 10);
        let two = term_identity_from_block(TermRole::Mean, "s(x)", &[None, None], &[1], 10);
        assert_ne!(one, two, "different #penalties must split identity");
    }

    #[test]
    fn block_identity_splits_on_reduced_width() {
        // The biobank LOSO collision: two models with identical block name /
        // penalty / nullspace SHAPE but a different realized per-block β width
        // (p=45 marginal vs the collapsed p=21) MUST hash to distinct
        // identities, so a p=37 fit never matches a p=85 artifact.
        let wide = term_identity_from_block(TermRole::Mean, "s(x)", &[None], &[1], 45);
        let narrow = term_identity_from_block(TermRole::Mean, "s(x)", &[None], &[1], 21);
        assert_ne!(
            wide, narrow,
            "different realized reduced width must split identity"
        );
    }

    #[test]
    fn block_identity_matches_across_folds_at_equal_width() {
        // The marquee LOSO win: same model, same realized width, different rows
        // -> identical identity, so ρ and the gauge β-projection both transfer.
        let fold_a = term_identity_from_block(TermRole::Mean, "s(x)", &[None], &[1], 45);
        let fold_b = term_identity_from_block(TermRole::Mean, "s(x)", &[None], &[1], 45);
        assert_eq!(
            fold_a, fold_b,
            "same model at equal width must share identity across folds"
        );
    }

    #[test]
    fn descriptor_key_excludes_rows_and_response() {
        let id = block_id(TermRole::Mean, "s(x)");
        let full = FitDescriptor {
            family_kind: "gaussian".to_string(),
            term_identities: vec![id],
            response_signature: ResponseSig {
                family_kind: "gaussian".to_string(),
                n_response_channels: 1,
            },
            row_population: Some(RowPopulationTag {
                n_rows: 1000,
                label: Some("full".to_string()),
            }),
        };
        let fold = FitDescriptor {
            family_kind: "gaussian".to_string(),
            term_identities: vec![id],
            response_signature: ResponseSig {
                family_kind: "gaussian".to_string(),
                n_response_channels: 1,
            },
            row_population: Some(RowPopulationTag {
                n_rows: 900, // an LOSO fold dropped 100 rows
                label: Some("fold-3".to_string()),
            }),
        };
        assert_eq!(
            full.descriptor_key(),
            fold.descriptor_key(),
            "LOSO fold must share its full-data parent's descriptor key"
        );
    }

    #[test]
    fn descriptor_key_invariant_to_term_order() {
        let a = block_id(TermRole::Mean, "s(x)");
        let b = block_id(TermRole::Mean, "s(z)");
        let sig = ResponseSig {
            family_kind: "gaussian".to_string(),
            n_response_channels: 1,
        };
        let d1 = FitDescriptor {
            family_kind: "gaussian".to_string(),
            term_identities: vec![a, b],
            response_signature: sig.clone(),
            row_population: None,
        };
        let d2 = FitDescriptor {
            family_kind: "gaussian".to_string(),
            term_identities: vec![b, a],
            response_signature: sig,
            row_population: None,
        };
        assert_eq!(d1.descriptor_key(), d2.descriptor_key());
    }

    #[test]
    fn artifact_usable_guard_rejects_nonfinite() {
        let id = block_id(TermRole::Mean, "s(x)");
        let mut artifact = FitArtifact {
            schema: FIT_ARTIFACT_SCHEMA,
            created_unix_secs: 0,
            descriptor: FitDescriptor {
                family_kind: "gaussian".to_string(),
                term_identities: vec![id],
                response_signature: ResponseSig {
                    family_kind: "gaussian".to_string(),
                    n_response_channels: 1,
                },
                row_population: None,
            },
            terms: vec![TermArtifact {
                identity: id,
                role: TermRole::Mean,
                basis_meta: basis_meta_stub(4),
                joint_null_rotation: None,
                raw_beta: vec![0.1, 0.2, 0.3, 0.4],
                rho_for_term: vec![1.0],
            }],
            global: GlobalFitSummary {
                outer_objective: -123.4,
                converged: true,
                n_rows: 100,
            },
        };
        assert!(artifact.is_usable());
        artifact.terms[0].raw_beta[2] = f64::NAN;
        assert!(
            !artifact.is_usable(),
            "non-finite β must fail the usable guard"
        );

        artifact.terms[0].raw_beta[2] = 0.3;
        artifact.global.outer_objective = f64::INFINITY;
        assert!(
            !artifact.is_usable(),
            "non-finite objective must fail the usable guard"
        );
    }

    #[test]
    fn serializable_basis_meta_roundtrips() {
        let meta = basis_meta_stub(7);
        let bytes = serde_json::to_vec(&meta).expect("serialize");
        let back: SerializableBasisMeta = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(meta, back);
        assert_eq!(back.n_centers, Some(7));
        assert_eq!(back.kind, "block-spec");
    }

    #[test]
    fn serializable_matrix_can_carry_rotation() {
        let q = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let m = SerializableMatrix {
            nrows: q.nrows(),
            ncols: q.ncols(),
            data: q.iter().copied().collect(),
        };
        let bytes = serde_json::to_vec(&m).expect("serialize");
        let back: SerializableMatrix = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(back.nrows, 2);
        assert_eq!(back.data, vec![1.0, 0.0, 0.0, 1.0]);
    }
}
