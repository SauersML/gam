//! TEMPORARY probe: why does the active-penalty count of a 1-D measure-jet
//! TERM COLLECTION change with the representer range? (`incremental realizer
//! topology changed ... active_penalties=2, cached_penalties=1`)

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
fn probe_term_collection_topology_versus_range() {
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
            }
            Err(e) => println!("[coll] f={f:<5} FAILED: {e}"),
        }
    }
}

/// The chart the incremental realizer actually rebuilds in: the FROZEN composed
/// transform the collection produced. Its local topology must equal the
/// collection's cached one, at every range.
#[test]
fn probe_frozen_chart_local_topology_versus_range() {
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
    let frozen = MeasureJetBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        order_s: *order_s,
        alpha: *alpha,
        num_scales: eps_band.len(),
        length_scale: length_scale.standardized_value(),
        double_penalty: true,
        learn_length_scale: true,
        multiscale: false,
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
    for f in [0.25_f64, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
        let mut spec = frozen.clone();
        spec.length_scale = frozen.length_scale * f;
        match build_measure_jet_basis(feature_col.view(), &spec) {
            Ok(built) => println!(
                "[frozen] f={f:<5} ell={:.6} p={} local_penalties={} sources={:?}",
                spec.length_scale,
                built.design.ncols(),
                built.active_penalties.len(),
                built
                    .active_penalties
                    .iter()
                    .map(|p| format!("{:?}", p.info.source))
                    .collect::<Vec<_>>()
            ),
            Err(e) => println!("[frozen] f={f:<5} BUILD FAILED: {e}"),
        }
    }
}
