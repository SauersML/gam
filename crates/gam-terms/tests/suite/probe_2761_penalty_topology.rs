//! Representer-range invariance of a 1-D measure-jet term's penalty topology
//! (#2761, #3884).
//!
//! The incremental realizer rebuilds a term at every outer ψ trial and refuses
//! when the rebuilt penalty topology differs from the one the term collection
//! cached (`incremental realizer topology changed ... active_penalties=2,
//! cached_penalties=1`). #2761 made that topology ℓ-invariant by construction:
//! the Primary's structural null frame is declared from the constraint
//! transform `z` alone, and the double-penalty ridge's fate is decided in the
//! local chart the same way the collection decides it. These tests sweep the
//! representer range over a ladder and assert that invariant at every rung.

use gam_data::{ColumnKindTag, DataSchema, EncodedDataset as Dataset, SchemaColumn};
use gam_terms::basis::{
    CenterStrategy, MeasureJetBasisSpec, MeasureJetIdentifiability, build_measure_jet_basis,
    measure_jet_quadrature_nodes, realized_measure_jet_length_scale, select_centers_by_strategy,
};
use gam_terms::inference::formula_dsl::parse_formula;
use gam_terms::smooth::{SmoothBasisSpec, build_term_collection_design};
use gam_terms::term_builder::build_termspec;
use ndarray::Array2;

fn hashed_unit(index: u64) -> f64 {
    let mut z = index.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn dataset_1d(n: usize) -> Dataset {
    let headers = ["y", "x"];
    let mut values = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0);
        values[[i, 0]] = (std::f64::consts::TAU * x).sin() + 0.1 * (2.0 * hashed_unit(i as u64) - 1.0);
        values[[i, 1]] = x;
    }
    Dataset {
        headers: headers.iter().map(|h| h.to_string()).collect(),
        values,
        schema: DataSchema {
            columns: headers
                .iter()
                .map(|name| SchemaColumn {
                    name: name.to_string(),
                    kind: ColumnKindTag::Continuous,
                    levels: vec![],
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; headers.len()],
    }
}

/// The identity of a penalty block the incremental realizer aligns on: its
/// term-local original index and its source.
type PenaltyIdentity = (usize, String);

/// Every rung of the representer-range ladder builds the collection with the
/// same ordered penalty topology as the auto-range build.
#[test]
fn term_collection_penalty_topology_is_invariant_to_the_representer_range() {
    let ds = dataset_1d(200);
    let col_map = ds.column_map();
    let parsed = parse_formula("y ~ s(x, bs=\"mjs\")").expect("parse");
    let base = build_termspec(&parsed.terms, &ds, &col_map, &mut Vec::new()).expect("term spec");
    let feature = ds.values.clone();

    // The realized auto range, as the sentinel resolves it.
    let realized = build_term_collection_design(feature.view(), &base).expect("auto design");
    let topology = |penaltyinfo: &[gam_terms::smooth::PenaltyBlockInfo]| -> Vec<PenaltyIdentity> {
        penaltyinfo
            .iter()
            .map(|i| (i.penalty.original_index, format!("{:?}", i.penalty.source)))
            .collect()
    };
    let auto_topology = topology(realized.penaltyinfo.as_slice());
    assert_eq!(realized.penalties.len(), auto_topology.len());

    let auto = {
        let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &base.smooth_terms[0].basis else {
            panic!("expected mjs");
        };
        let seeds = select_centers_by_strategy(feature.view(), &mj.center_strategy).expect("seeds");
        let (nodes, _m) =
            measure_jet_quadrature_nodes(feature.view(), seeds.view()).expect("nodes");
        realized_measure_jet_length_scale(nodes.view(), 0.0).expect("auto")
    };

    let mut mismatches = Vec::new();
    for f in [0.5_f64, 1.0, 2.0, 4.0, 8.0, 16.0] {
        let mut spec = base.clone();
        if let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &mut spec.smooth_terms[0].basis {
            mj.length_scale = auto * f;
        }
        match build_term_collection_design(feature.view(), &spec) {
            Ok(design) => {
                let rung = topology(design.penaltyinfo.as_slice());
                if design.penalties.len() != auto_topology.len() || rung != auto_topology {
                    mismatches.push(format!(
                        "f={f} ell={:.6}: penalties={} topology {rung:?} against auto {auto_topology:?}",
                        auto * f,
                        design.penalties.len()
                    ));
                }
            }
            Err(e) => mismatches.push(format!("f={f} ell={:.6}: build failed: {e}", auto * f)),
        }
    }
    assert!(
        mismatches.is_empty(),
        "the collection's penalty topology changed with the representer range:\n{}",
        mismatches.join("\n")
    );
}

/// The chart the incremental realizer actually rebuilds in is the FROZEN
/// composed transform the collection produced. At every rung its local
/// topology must equal the collection's cached one, which is the realizer's
/// own check.
#[test]
fn frozen_chart_local_penalty_topology_matches_the_cached_one_at_every_range() {
    use gam_terms::basis::{BasisMetadata, MeasureJetFrozenQuadrature};

    let ds = dataset_1d(200);
    let col_map = ds.column_map();
    let parsed = parse_formula("y ~ s(x, bs=\"mjs\")").expect("parse");
    let base = build_termspec(&parsed.terms, &ds, &col_map, &mut Vec::new()).expect("term spec");
    let SmoothBasisSpec::MeasureJet { spec: base_mj, .. } = &base.smooth_terms[0].basis else {
        panic!("expected mjs");
    };
    let feature = ds.values.clone();
    let realized = build_term_collection_design(feature.view(), &base).expect("auto design");
    let identity = |penalties: &[gam_terms::basis::ActivePenalty]| -> Vec<PenaltyIdentity> {
        let mut ids: Vec<PenaltyIdentity> = penalties
            .iter()
            .map(|p| (p.info.original_index, format!("{:?}", p.info.source)))
            .collect();
        ids.sort();
        ids
    };
    let term = &realized.smooth.terms[0];
    let cached = identity(term.active_penalties.as_slice());
    let BasisMetadata::MeasureJet {
        centers,
        length_scale,
        eps_band,
        order_s,
        alpha,
        masses,
        support_means,
        penalty_normalization_scales,
        raw_penalty_normalization_scales,
        fused_penalty_normalization_scale,
        constraint_transform,
        sigma_coord,
        ..
    } = &term.metadata
    else {
        panic!("expected measure-jet metadata");
    };
    // Replay the term that was built: its penalty switches come from the base
    // spec, the rest from the collection's frozen metadata.
    let frozen = MeasureJetBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        order_s: *order_s,
        alpha: *alpha,
        num_scales: eps_band.len(),
        length_scale: length_scale.standardized_value(),
        double_penalty: base_mj.double_penalty,
        learn_length_scale: base_mj.learn_length_scale,
        multiscale: base_mj.multiscale,
        identifiability: MeasureJetIdentifiability::FrozenTransform {
            transform: constraint_transform.clone().expect("fit-time z"),
        },
        frozen_quadrature: Some(MeasureJetFrozenQuadrature {
            masses: masses.clone(),
            eps_band: eps_band.clone(),
            support_means: support_means.clone(),
            penalty_normalization_scales: penalty_normalization_scales.clone(),
            raw_penalty_normalization_scales: raw_penalty_normalization_scales.clone(),
            fused_penalty_normalization_scale: *fused_penalty_normalization_scale,
            sigma_coord: *sigma_coord,
        }),
    };
    // The frozen replay evaluates on the SAME (standardized) coordinates the
    // centers live in, so feed the term its own feature column.
    let feature_col = feature.slice(ndarray::s![.., 1..2]).to_owned();
    let mut mismatches = Vec::new();
    for f in [0.25_f64, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
        let mut spec = frozen.clone();
        spec.length_scale = frozen.length_scale * f;
        match build_measure_jet_basis(feature_col.view(), &spec) {
            Ok(built) => {
                let local = identity(built.active_penalties.as_slice());
                if built.active_penalties.len() != realized.penalties.len() || local != cached {
                    mismatches.push(format!(
                        "f={f} ell={:.6}: local_penalties={} {local:?} against cached {} {cached:?}",
                        spec.length_scale,
                        built.active_penalties.len(),
                        realized.penalties.len()
                    ));
                }
            }
            Err(e) => mismatches.push(format!(
                "f={f} ell={:.6}: build failed: {e}",
                spec.length_scale
            )),
        }
    }
    assert!(
        mismatches.is_empty(),
        "the frozen chart's local penalty topology differs from the cached one:\n{}",
        mismatches.join("\n")
    );
}
