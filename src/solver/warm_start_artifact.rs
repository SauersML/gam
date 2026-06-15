//! Cross-fit warm-start artifact: a descriptor-indexed, function-space
//! snapshot of a converged fit, designed so a *related* later fit (a
//! leave-one-subject-out fold, a re-fit on a different row population, a
//! different reduced width) can warm-start from it even though the exact
//! response-keyed inner cache (`persistent_cache.rs`) misses.
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

use crate::cache::key::{Fingerprint, Fingerprinter};
use crate::terms::basis::{BasisMetadata, DuchonNullspaceOrder, MaternNu};
use serde::{Deserialize, Serialize};

/// On-disk schema version for [`FitArtifact`]. Bump when the serialized
/// layout changes in a way that makes prior payloads unsafe to consume.
pub(crate) const FIT_ARTIFACT_SCHEMA: u32 = 1;

/// Saturation magnitude past which a copied ρ coordinate is considered
/// pinned at the outer optimizer's box and is NOT transferred. Mirrors the
/// persist-side gate in `families/custom_family/persistent_cache.rs` and the
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

/// Structural discriminant of a basis kind, EXCLUDING all data-derived
/// fields (realized centers, input scales, learned length scale). Hashed
/// into the [`TermIdentityKey`] so a term keeps its identity across folds.
fn absorb_basis_structure(fp: &mut Fingerprinter, basis: &BasisMetadata) {
    match basis {
        BasisMetadata::BSpline1D {
            knots,
            periodic,
            degree,
            ..
        } => {
            fp.absorb_str(b"basis-kind", "bspline1d");
            // Structural: polynomial degree + number of internal knots.
            // The knot *values* are data-derived (quantiles of x), so we
            // hash only the COUNT, not the values.
            fp.absorb_u64(b"degree", degree.map(|d| d as u64).unwrap_or(u64::MAX));
            fp.absorb_u64(b"num_knots", knots.len() as u64);
            fp.absorb_u64(b"periodic", u64::from(periodic.is_some()));
        }
        BasisMetadata::Duchon {
            power,
            nullspace_order,
            periodic,
            length_scale,
            centers,
            ..
        } => {
            fp.absorb_str(b"basis-kind", "duchon");
            fp.absorb_f64(b"power", *power);
            fp.absorb_u64(b"nullspace_order", duchon_order_code(*nullspace_order));
            // n_centers is structural (it is the basis rank / model size);
            // the center *coordinates* are data-derived and excluded.
            fp.absorb_u64(b"n_centers", centers.nrows() as u64);
            fp.absorb_u64(b"periodic", u64::from(periodic.is_some()));
            // κ-fixed? Pure mode (no length_scale) vs scaled mode is structural.
            fp.absorb_u64(b"kappa_fixed", u64::from(length_scale.is_none()));
        }
        BasisMetadata::Matern {
            nu,
            include_intercept,
            periodic,
            centers,
            ..
        } => {
            fp.absorb_str(b"basis-kind", "matern");
            fp.absorb_u64(b"nu", matern_nu_code(*nu));
            fp.absorb_u64(b"include_intercept", u64::from(*include_intercept));
            fp.absorb_u64(b"n_centers", centers.nrows() as u64);
            fp.absorb_u64(b"periodic", u64::from(periodic.is_some()));
        }
        BasisMetadata::ThinPlate {
            periodic, centers, ..
        } => {
            fp.absorb_str(b"basis-kind", "thinplate");
            fp.absorb_u64(b"n_centers", centers.nrows() as u64);
            fp.absorb_u64(b"periodic", u64::from(periodic.is_some()));
        }
        BasisMetadata::Sphere {
            penalty_order,
            method,
            max_degree,
            centers,
            ..
        } => {
            fp.absorb_str(b"basis-kind", "sphere");
            fp.absorb_u64(b"penalty_order", *penalty_order as u64);
            fp.absorb_str(b"method", &format!("{method:?}"));
            fp.absorb_u64(
                b"max_degree",
                max_degree.map(|d| d as u64).unwrap_or(u64::MAX),
            );
            fp.absorb_u64(b"n_centers", centers.nrows() as u64);
        }
        BasisMetadata::ConstantCurvature { centers, .. } => {
            // κ is a learned coordinate; only the model size is structural.
            fp.absorb_str(b"basis-kind", "constant_curvature");
            fp.absorb_u64(b"n_centers", centers.nrows() as u64);
        }
        BasisMetadata::MeasureJet {
            order_s,
            centers,
            eps_band,
            ..
        } => {
            fp.absorb_str(b"basis-kind", "measure_jet");
            // order_s is the mode marker (structural); masses/eps values are
            // data-derived, only the band COUNT is structural.
            fp.absorb_f64(b"order_s", *order_s);
            fp.absorb_u64(b"n_centers", centers.nrows() as u64);
            fp.absorb_u64(b"n_bands", eps_band.len() as u64);
        }
        BasisMetadata::Pca {
            feature_cols,
            centered,
            chunk_size,
            ..
        } => {
            fp.absorb_str(b"basis-kind", "pca");
            fp.absorb_u64(b"n_features", feature_cols.len() as u64);
            fp.absorb_u64(b"centered", u64::from(*centered));
            fp.absorb_u64(b"chunk_size", *chunk_size as u64);
        }
        BasisMetadata::TensorBSpline {
            feature_cols,
            degrees,
            knots,
            periods,
            ..
        } => {
            fp.absorb_str(b"basis-kind", "tensor_bspline");
            fp.absorb_u64(b"n_features", feature_cols.len() as u64);
            for d in degrees {
                fp.absorb_u64(b"degree", *d as u64);
            }
            for k in knots {
                fp.absorb_u64(b"num_knots", k.len() as u64);
            }
            for p in periods {
                fp.absorb_u64(b"periodic", u64::from(p.is_some()));
            }
        }
        BasisMetadata::SphereHarmonics {
            max_degree,
            radians,
        } => {
            fp.absorb_str(b"basis-kind", "sphere_harmonics");
            fp.absorb_u64(b"max_degree", *max_degree as u64);
            fp.absorb_u64(b"radians", u64::from(*radians));
        }
        BasisMetadata::BySmooth {
            inner,
            by_col,
            ordered,
            levels,
        } => {
            fp.absorb_str(b"basis-kind", "by_smooth");
            fp.absorb_u64(b"by_col", *by_col as u64);
            fp.absorb_u64(b"ordered", u64::from(*ordered));
            fp.absorb_u64(
                b"n_levels",
                levels.as_ref().map(|l| l.len() as u64).unwrap_or(0),
            );
            absorb_basis_structure(fp, inner);
        }
        BasisMetadata::FactorSmooth {
            continuous_cols,
            group_col,
            knots,
            degree,
            periodic,
            group_levels,
            flavour,
        } => {
            fp.absorb_str(b"basis-kind", "factor_smooth");
            fp.absorb_u64(b"n_continuous", continuous_cols.len() as u64);
            fp.absorb_u64(b"group_col", *group_col as u64);
            fp.absorb_u64(b"num_knots", knots.len() as u64);
            fp.absorb_u64(b"degree", *degree as u64);
            fp.absorb_u64(b"periodic", u64::from(periodic.is_some()));
            fp.absorb_u64(b"n_groups", group_levels.len() as u64);
            fp.absorb_str(b"flavour", flavour);
        }
    }
}

fn duchon_order_code(order: DuchonNullspaceOrder) -> u64 {
    match order {
        DuchonNullspaceOrder::Zero => 0,
        DuchonNullspaceOrder::Linear => 1,
        DuchonNullspaceOrder::Degree(k) => 2 + k as u64,
    }
}

fn matern_nu_code(nu: MaternNu) -> u64 {
    match nu {
        MaternNu::Half => 1,
        MaternNu::ThreeHalves => 3,
        MaternNu::FiveHalves => 5,
        MaternNu::SevenHalves => 7,
        MaternNu::NineHalves => 9,
    }
}

/// Compute the stable structural identity of a term.
///
/// Hashes: the role discriminant, the sorted variable names, the basis-kind
/// discriminant, and STRUCTURAL basis params only. DELIBERATELY EXCLUDES the
/// realized `centers` coordinates, `input_scales`, and learned
/// `length_scale` — so "same term, different LOSO fold" maps to the SAME
/// identity (this is what lets ρ transfer survive a fold), while "3 PCs vs
/// 10 PCs" or "different #centers" map to DIFFERENT identities.
pub fn term_identity(
    role: TermRole,
    var_names_sorted: &[String],
    basis: &BasisMetadata,
) -> TermIdentityKey {
    let mut fp = Fingerprinter::new();
    fp.absorb_tag(b"fit-artifact-term-identity-v1");
    fp.absorb_u64(b"role", u64::from(role.discriminant()));
    fp.absorb_u64(b"n_vars", var_names_sorted.len() as u64);
    for name in var_names_sorted {
        fp.absorb_str(b"var", name);
    }
    absorb_basis_structure(&mut fp, basis);
    TermIdentityKey(fp.finalize())
}

/// Build a term identity at the *block-spec* layer (`fit_custom_family` and
/// friends), where the full [`BasisMetadata`] / variable names are no longer
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
/// NOTE (architect-assumption mismatch): the original design routed identity
/// through `SmoothTerm.metadata`, but at this layer that metadata has already
/// been compiled away. The block name + penalty structure is the honest,
/// fold-invariant identity available here. The richer
/// [`term_identity`] (basis-metadata-keyed) is retained for callers that DO
/// hold a `BasisMetadata`.
pub fn term_identity_from_block(
    role: TermRole,
    block_name: &str,
    precision_labels: &[Option<String>],
    nullspace_dims: &[usize],
) -> TermIdentityKey {
    let mut fp = Fingerprinter::new();
    fp.absorb_tag(b"fit-artifact-block-identity-v1");
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

/// Serializable structural subset of [`BasisMetadata`]. Captures the
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

impl SerializableBasisMeta {
    /// Extract the serializable structural subset from a full
    /// [`BasisMetadata`].
    pub fn from_metadata(basis: &BasisMetadata) -> SerializableBasisMeta {
        let mut out = SerializableBasisMeta {
            kind: String::new(),
            degree: None,
            num_knots: None,
            n_centers: None,
            nullspace_order: None,
            matern_nu: None,
            periodic: false,
        };
        match basis {
            BasisMetadata::BSpline1D {
                knots,
                periodic,
                degree,
                ..
            } => {
                out.kind = "bspline1d".to_string();
                out.degree = degree.map(|d| d as u64);
                out.num_knots = Some(knots.len() as u64);
                out.periodic = periodic.is_some();
            }
            BasisMetadata::Duchon {
                power: _,
                nullspace_order,
                periodic,
                centers,
                ..
            } => {
                out.kind = "duchon".to_string();
                out.nullspace_order = Some(duchon_order_code(*nullspace_order));
                out.n_centers = Some(centers.nrows() as u64);
                out.periodic = periodic.is_some();
            }
            BasisMetadata::Matern {
                nu,
                periodic,
                centers,
                ..
            } => {
                out.kind = "matern".to_string();
                out.matern_nu = Some(matern_nu_code(*nu));
                out.n_centers = Some(centers.nrows() as u64);
                out.periodic = periodic.is_some();
            }
            BasisMetadata::ThinPlate {
                periodic, centers, ..
            } => {
                out.kind = "thinplate".to_string();
                out.n_centers = Some(centers.nrows() as u64);
                out.periodic = periodic.is_some();
            }
            BasisMetadata::Sphere { centers, .. } => {
                out.kind = "sphere".to_string();
                out.n_centers = Some(centers.nrows() as u64);
            }
            BasisMetadata::ConstantCurvature { centers, .. } => {
                out.kind = "constant_curvature".to_string();
                out.n_centers = Some(centers.nrows() as u64);
            }
            BasisMetadata::MeasureJet { centers, .. } => {
                out.kind = "measure_jet".to_string();
                out.n_centers = Some(centers.nrows() as u64);
            }
            BasisMetadata::Pca { .. } => out.kind = "pca".to_string(),
            BasisMetadata::TensorBSpline { .. } => out.kind = "tensor_bspline".to_string(),
            BasisMetadata::SphereHarmonics { max_degree, .. } => {
                out.kind = "sphere_harmonics".to_string();
                out.degree = Some(*max_degree as u64);
            }
            BasisMetadata::BySmooth { .. } => out.kind = "by_smooth".to_string(),
            BasisMetadata::FactorSmooth { degree, .. } => {
                out.kind = "factor_smooth".to_string();
                out.degree = Some(*degree as u64);
            }
        }
        out
    }
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

    fn duchon_meta(n_centers: usize) -> BasisMetadata {
        BasisMetadata::Duchon {
            centers: Array2::<f64>::zeros((n_centers, 2)),
            length_scale: None,
            periodic: None,
            power: 1.0,
            nullspace_order: DuchonNullspaceOrder::Linear,
            identifiability_transform: None,
            input_scales: None,
            aniso_log_scales: None,
            operator_collocation_points: None,
        }
    }

    #[test]
    fn identity_ignores_centers_coordinates() {
        // Same structure, DIFFERENT realized center coordinates -> SAME id.
        let mut a = duchon_meta(10);
        let b = duchon_meta(10);
        if let BasisMetadata::Duchon { centers, .. } = &mut a {
            centers.fill(3.14); // perturb the data-derived coordinates
        }
        let vars = vec!["x".to_string(), "y".to_string()];
        let ka = term_identity(TermRole::Mean, &vars, &a);
        let kb = term_identity(TermRole::Mean, &vars, &b);
        assert_eq!(
            ka, kb,
            "differing center coordinates must not split identity"
        );
    }

    #[test]
    fn identity_splits_on_n_centers() {
        let vars = vec!["x".to_string()];
        let k10 = term_identity(TermRole::Mean, &vars, &duchon_meta(10));
        let k20 = term_identity(TermRole::Mean, &vars, &duchon_meta(20));
        assert_ne!(k10, k20, "different #centers must split identity");
    }

    #[test]
    fn identity_splits_on_variable_set() {
        let m = duchon_meta(10);
        let ka = term_identity(TermRole::Mean, &["x".to_string()], &m);
        let kb = term_identity(TermRole::Mean, &["z".to_string()], &m);
        assert_ne!(ka, kb, "different variable set must split identity");
    }

    #[test]
    fn identity_splits_on_role() {
        let m = duchon_meta(10);
        let vars = vec!["x".to_string()];
        let mean = term_identity(TermRole::Mean, &vars, &m);
        let slope = term_identity(TermRole::LogSlope, &vars, &m);
        assert_ne!(mean, slope, "different role must split identity");
    }

    #[test]
    fn descriptor_key_excludes_rows_and_response() {
        let vars = vec!["x".to_string()];
        let id = term_identity(TermRole::Mean, &vars, &duchon_meta(10));
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
        let a = term_identity(TermRole::Mean, &["x".to_string()], &duchon_meta(10));
        let b = term_identity(TermRole::Mean, &["z".to_string()], &duchon_meta(10));
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
        let id = term_identity(TermRole::Mean, &["x".to_string()], &duchon_meta(4));
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
                basis_meta: SerializableBasisMeta::from_metadata(&duchon_meta(4)),
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
        let meta = SerializableBasisMeta::from_metadata(&duchon_meta(7));
        let bytes = serde_json::to_vec(&meta).expect("serialize");
        let back: SerializableBasisMeta = serde_json::from_slice(&bytes).expect("deserialize");
        assert_eq!(meta, back);
        assert_eq!(back.n_centers, Some(7));
        assert_eq!(back.kind, "duchon");
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
