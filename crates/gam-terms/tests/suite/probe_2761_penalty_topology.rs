//! #2761: the penalty topology of a 1-D measure-jet term must not depend on the
//! representer range ℓ.
//!
//! The outer search moves ℓ between trials and the incremental realizer
//! rebuilds the term in the FROZEN composed chart at each one. When that
//! rebuild's local penalty count differed from the collection's cached count,
//! the search aborted with `incremental realizer topology changed ...
//! active_penalties=2, cached_penalties=1`. The repair declares the Primary's
//! structural null frame from the constraint transform `z` alone ("Nothing here
//! reads `ℓ`", `measure_jet_primary_structural_null_frame`), and it decides the
//! double-penalty ridge's fate in the local chart the way the collection
//! decides it. So the topology is ℓ-invariant by construction, and both tests
//! below assert that at every rung of a range ladder (32x for the
//! collection, 128x for the frozen chart). (These started as
//! print-only probes, which could not fail when the abort came back.)

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

#[test]
fn term_collection_topology_is_range_invariant_2761() {
    let ds = dataset_1d(200);
    let col_map = ds.column_map();
    let parsed = parse_formula("y ~ s(x, bs=\"mjs\")").expect("parse");
    let base = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut Vec::new(),
    )
    .expect("term spec");
    let feature = ds.values.clone();

    // The realized auto range, as the sentinel resolves it.
    let realized = build_term_collection_design(feature.view(), &base).expect("auto design");
    println!(
        "[coll] AUTO   penalties={} dropped={:?} p={} info={:?}",
        realized.penalties.len(),
        realized
            .dropped_penaltyinfo
            .iter()
            .map(|i| format!("{:?}/{:?}", i.penalty.source, i.penalty.reason))
            .collect::<Vec<_>>(),
        realized.design.ncols(),
        realized
            .penaltyinfo
            .iter()
            .map(|i| format!(
                "{:?} rank={} frame={:?}",
                i.penalty.source,
                i.penalty.effective_rank,
                i.penalty.structural_null_frame.as_ref().map(|f| f.ncols())
            ))
            .collect::<Vec<_>>()
    );
    let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &base.smooth_terms[0].basis else {
        panic!("expected mjs");
    };
    println!(
        "[coll] spec: double_penalty={} learn_length_scale={} length_scale={} centers={:?}",
        mj.double_penalty,
        mj.learn_length_scale,
        mj.length_scale,
        gam_terms::basis::center_strategy_num_centers(&mj.center_strategy)
    );

    let auto_sources: Vec<String> = realized
        .penaltyinfo
        .iter()
        .map(|i| format!("{:?}", i.penalty.source))
        .collect();
    let mut mismatches = Vec::<String>::new();
    for f in [0.5_f64, 1.0, 2.0, 4.0, 8.0, 16.0] {
        let mut spec = base.clone();
        let auto = {
            let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &spec.smooth_terms[0].basis else {
                unreachable!()
            };
            let strategy = mj.center_strategy.clone();
            let seeds = select_centers_by_strategy(feature.view(), &strategy).expect("seeds");
            let (nodes, _m) =
                measure_jet_quadrature_nodes(feature.view(), seeds.view()).expect("nodes");
            realized_measure_jet_length_scale(nodes.view(), 0.0).expect("auto")
        };
        if let SmoothBasisSpec::MeasureJet { spec: mj, .. } = &mut spec.smooth_terms[0].basis {
            mj.length_scale = auto * f;
        }
        match build_term_collection_design(feature.view(), &spec) {
            Ok(design) => {
                let sources: Vec<String> = design
                    .penaltyinfo
                    .iter()
                    .map(|i| format!("{:?}", i.penalty.source))
                    .collect();
                let dropped: Vec<String> = design
                    .dropped_penaltyinfo
                    .iter()
                    .map(|i| format!("{:?}/{:?}", i.penalty.source, i.penalty.reason))
                    .collect();
                println!(
                    "[coll] f={f:<5} ell={:.6} p={} penalties={} src={sources:?} dropped={dropped:?}",
                    auto * f,
                    design.design.ncols(),
                    design.penalties.len()
                );
                if design.penalties.len() != realized.penalties.len() || sources != auto_sources {
                    mismatches.push(format!(
                        "f={f}: penalties={} src={sources:?} dropped={dropped:?}",
                        design.penalties.len()
                    ));
                }
            }
            Err(e) => {
                println!("[coll] f={f:<5} FAILED: {e}");
                mismatches.push(format!("f={f}: design build failed: {e}"));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "the measure-jet term collection's penalty topology moved with the representer range \
         (auto: penalties={} src={auto_sources:?}):\n  - {}",
        realized.penalties.len(),
        mismatches.join("\n  - ")
    );
}

/// The chart the incremental realizer actually rebuilds in: the FROZEN composed
/// transform the collection produced. Its local topology must equal the
/// collection's cached one, at every range.
#[test]
fn frozen_chart_local_topology_matches_collection_at_every_range_2761() {
    use gam_terms::basis::{BasisMetadata, MeasureJetFrozenQuadrature};

    let ds = dataset_1d(200);
    let col_map = ds.column_map();
    let parsed = parse_formula("y ~ s(x, bs=\"mjs\")").expect("parse");
    let base = build_termspec(
        &parsed.terms,
        &ds,
        &col_map,
        &mut Vec::new(),
    )
    .expect("term spec");
    let feature = ds.values.clone();
    let realized = build_term_collection_design(feature.view(), &base).expect("auto design");
    println!(
        "[frozen] collection cached penalties = {}",
        realized.penalties.len()
    );
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
    } = &realized.smooth.terms[0].metadata
    else {
        panic!("expected measure-jet metadata");
    };
    let SmoothBasisSpec::MeasureJet { spec: base_mj, .. } = &base.smooth_terms[0].basis else {
        panic!("expected mjs");
    };
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
    let mut mismatches = Vec::<String>::new();
    for f in [0.25_f64, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
        let mut spec = frozen.clone();
        spec.length_scale = frozen.length_scale * f;
        match build_measure_jet_basis(feature_col.view(), &spec) {
            Ok(built) => {
                let sources = built
                    .active_penalties
                    .iter()
                    .map(|p| format!("{:?}", p.info.source))
                    .collect::<Vec<_>>();
                println!(
                    "[frozen] f={f:<5} ell={:.6} p={} local_penalties={} sources={sources:?}",
                    spec.length_scale,
                    built.design.ncols(),
                    built.active_penalties.len(),
                );
                // The incremental realizer's own check: the local count must
                // equal the term's cached penalty range, which for this
                // single-smooth, intercept-only model is the collection's count.
                if built.active_penalties.len() != realized.penalties.len() {
                    mismatches.push(format!(
                        "f={f}: local_penalties={} sources={sources:?}",
                        built.active_penalties.len()
                    ));
                }
            }
            Err(e) => {
                println!("[frozen] f={f:<5} BUILD FAILED: {e}");
                mismatches.push(format!("f={f}: frozen rebuild failed: {e}"));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "the frozen composed chart's local penalty topology differs from the collection's \
         cached {} penalties (the `incremental realizer topology changed` abort):\n  - {}",
        realized.penalties.len(),
        mismatches.join("\n  - ")
    );
}
