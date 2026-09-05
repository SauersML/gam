//! Isolate construction and frozen replay for the original #2827 Duchon formula.
//!
//! Usage: spatial_basis_profile_2827 DATA.csv ROWS LOG_KAPPA
//! Loads the numeric corpus once, takes the requested prefix, freezes one cold
//! collection build, then times one local basis rebuild at the supplied coordinate.
//! No likelihood optimization or interpolation tensor is constructed.

use gam_data::load_csvwith_inferred_schema;
use gam_runtime::resource::ResourcePolicy;
use gam_terms::basis::BasisWorkspace;
use gam_terms::inference::formula_dsl::parse_formula;
use gam_terms::smooth::{
    SmoothBasisSpec, build_single_local_smooth_term, build_term_collection_design,
    freeze_term_collection_from_design,
};
use gam_terms::term_builder::build_termspec;
use ndarray::s;
use std::hash::{DefaultHasher, Hasher};
use std::path::Path;
use std::time::Instant;

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: spatial_basis_profile_2827 DATA.csv ROWS LOG_KAPPA".into());
    }
    let rows = args[2]
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let log_kappa = args[3].parse::<f64>().map_err(|error| error.to_string())?;
    let length_scale = (-log_kappa).exp();
    if rows == 0 || !log_kappa.is_finite() || !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(
            "ROWS must be positive and LOG_KAPPA must yield a finite positive length scale".into(),
        );
    }
    let mut data =
        load_csvwith_inferred_schema(Path::new(&args[1])).map_err(|error| error.to_string())?;
    if rows > data.values.nrows() {
        return Err(format!(
            "requested {rows} rows, corpus has {}",
            data.values.nrows()
        ));
    }
    data.values = data.values.slice(s![..rows, ..]).to_owned();
    let formula = "yg ~ duchon(x1,x2,x3,x4,x5,x6,centers=100,length_scale=1)";
    let parsed = parse_formula(formula).map_err(|error| error.to_string())?;
    let spec = build_termspec(
        &parsed.terms,
        &data,
        &data.column_map(),
        &mut Vec::new(),
        &ResourcePolicy::default_library(),
    )
    .map_err(|error| error.to_string())?;
    eprintln!("[2827-basis] rows={rows} cold collection begins");
    let started = Instant::now();
    let collection = build_term_collection_design(data.values.view(), &spec)
        .map_err(|error| error.to_string())?;
    let cold_seconds = started.elapsed().as_secs_f64();
    let mut frozen = freeze_term_collection_from_design(&spec, &collection)
        .map_err(|error| error.to_string())?;
    drop(collection);
    let term = frozen
        .smooth_terms
        .first_mut()
        .ok_or("formula has no smooth")?;
    let SmoothBasisSpec::Duchon { spec, .. } = &mut term.basis else {
        return Err("formula did not resolve to Duchon".into());
    };
    if spec.radial_reparam.is_none() {
        return Err("cold build did not freeze a radial chart".into());
    }
    spec.length_scale = Some(length_scale);
    let mut workspace = BasisWorkspace::default();
    eprintln!(
        "[2827-basis] rows={rows} cold_seconds={cold_seconds:.9} frozen local rebuild begins log_kappa={log_kappa:.9}"
    );
    let started = Instant::now();
    let local = build_single_local_smooth_term(data.values.view(), term, &mut workspace)
        .map_err(|error| error.to_string())?;
    let local_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let design = local
        .design
        .try_to_dense_arc("bounded Duchon basis profile")?;
    let materialize_seconds = started.elapsed().as_secs_f64();
    if design.iter().any(|value| !value.is_finite()) {
        return Err("local rebuild contains non-finite design values".into());
    }
    let mut signature = DefaultHasher::new();
    for value in design.iter() {
        signature.write_u64(value.to_bits());
    }
    println!(
        "rows={rows} columns={} cold_seconds={cold_seconds:.9} local_seconds={local_seconds:.9} materialize_seconds={materialize_seconds:.9} log_kappa={log_kappa:.9} design_bits_signature={:016x}",
        design.ncols(),
        signature.finish()
    );
    Ok(())
}

fn main() -> Result<(), String> {
    std::thread::Builder::new()
        .name("spatial-basis-profile-2827".into())
        .stack_size(64 << 20)
        .spawn(run)
        .map_err(|error| error.to_string())?
        .join()
        .map_err(|_| "basis worker panicked".to_string())?
}
