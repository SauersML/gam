//! Weight-only full-hidden writer proposal diagnostics; no acceptance or layer selection.
//! EXPORT OUT.json ranks=0,32,64,128,256,512 local=2 context=16
#[path = "../src/mlp_tied_writer.rs"]
mod mlp_tied_writer;
use gam_mpd::{
    acceptance::{CostCache, structural_cost},
    artifact::Artifact,
    import::import_language_model,
};
use mlp_tied_writer::*;
use serde_json::json;
use std::{path::Path, time::Instant};
fn norm_squared(m: &ndarray::Array2<f64>) -> f64 {
    m.iter().map(|v| v * v).sum()
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("EXPORT OUT.json ranks=0,32,64,128,256,512 local=2 context=16".into());
    }
    let ranks: Vec<usize> = args[2]
        .strip_prefix("ranks=")
        .ok_or("declare ranks")?
        .split(',')
        .map(|s| s.parse().map_err(|e| format!("{e}")))
        .collect::<Result<_, _>>()?;
    if ranks.is_empty()
        || ranks
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != ranks.len()
    {
        return Err("declare nonempty distinct ranks".into());
    }
    let sequences = args[3]
        .strip_prefix("local=")
        .ok_or("declare local")?
        .parse::<usize>()
        .map_err(|e| e.to_string())?;
    let context = args[4]
        .strip_prefix("context=")
        .ok_or("declare context")?
        .parse::<usize>()
        .map_err(|e| e.to_string())?;
    let output = Path::new(&args[1]);
    if output.exists() || sequences == 0 || context == 0 {
        return Err("fresh output and positive family dimensions required".into());
    }
    let start = Instant::now();
    let imported = import_language_model(Path::new(&args[0]), sequences, context)?;
    let base = Artifact::native(&imported.program)?.f32_literals()?;
    let maps = MlpWriterMap::all(&base.program)?;
    if maps.len()
        != imported.record["config"]["n_layers"]
            .as_u64()
            .ok_or("n_layers")? as usize
    {
        return Err("not all native MLP layers mapped".into());
    }
    let native = base
        .program
        .execute(&imported.contract.family, false)
        .map_err(|e| e.to_string())?;
    let mut cache = CostCache::default();
    let native_cost = structural_cost(&base, &mut cache)?.total();
    let mut layers = Vec::new();
    for map in maps {
        let up = base.program.operators[map.reader].matrix_cow();
        let down = base.program.operators[map.writer].matrix_cow();
        let fit = WriterFit::of(&up, &down)?;
        let residual_svd = WriterSvd::of(&fit.residual)?;
        let native_svd = WriterSvd::of(&down)?;
        let active = &native.values[map.active];
        let target = active.dot(&down.t());
        let native_write_rms = (norm_squared(&target) / target.nrows() as f64).sqrt();
        let tied0 = active.dot(&fit.prediction.t());
        let mut points = Vec::new();
        for &rank in &ranks {
            let residual = residual_svd.operator(
                &base.program.operators[map.writer],
                rank,
                "tied native-weight residual",
            )?;
            let baseline = native_svd.operator(
                &base.program.operators[map.writer],
                rank,
                "native writer SVD",
            )?;
            let proposed = &fit.prediction + &residual.matrix();
            let svd_matrix = baseline.matrix();
            for (family, matrix, candidate) in [
                (
                    "TiedWriterResidual",
                    proposed,
                    tied_candidate(&base, &map, &fit, (rank > 0).then_some(residual))?,
                ),
                (
                    "NativeSvd",
                    svd_matrix,
                    svd_candidate(&base, &map, baseline)?,
                ),
            ] {
                let predicted = active.dot(&matrix.t());
                let error = &predicted - &target;
                let worst = error
                    .rows()
                    .into_iter()
                    .map(|r| r.iter().map(|v| v * v).sum::<f64>().sqrt())
                    .fold(0_f64, f64::max);
                let cost = structural_cost(&candidate, &mut cache)?.total();
                points.push(json!({"family":family,"rank":rank,"C32_bits":cost,"weight_relative_Frobenius":(norm_squared(&(&matrix-&*down))/fit.native_norm_squared).sqrt(),"native_activation_relative_RMS":(norm_squared(&error)/norm_squared(&target)).sqrt(),"native_activation_absolute_worst_L2":worst,"native_activation_worst_over_native_contribution_RMS":worst/native_write_rms,"all_native_places_retained":candidate.places.len()==base.places.len()}));
            }
        }
        layers.push(json!({"native_layer":map.layer,"node_map":{"reader":map.reader,"writer":map.writer,"normed":map.normed,"pre":map.pre,"active":map.active,"skip":map.skip,"output":map.output,"gate":map.gate},"gain_parameters_f32":fit.gains.to_vec(),"shape_up":up.dim(),"shape_down":down.dim(),"gated":map.gate.is_some(),"zero_reader_rows":fit.zero_reader_rows,"gain_range":[fit.gains.iter().copied().fold(f64::INFINITY,f64::min),fit.gains.iter().copied().fold(f64::NEG_INFINITY,f64::max)],"weight_variance_explained_by_tie":1.-norm_squared(&fit.residual)/fit.native_norm_squared,"weight_tie_relative_Frobenius":(norm_squared(&fit.residual)/fit.native_norm_squared).sqrt(),"activation_tie_relative_RMS":(norm_squared(&(&tied0-&target))/norm_squared(&target)).sqrt(),"native_contribution_RMS":native_write_rms,"points":points}));
    }
    let report = json!({"source":imported.record["source"],"config":imported.record["config"],"rank_grid":ranks,"native_C32_bits":native_cost,"native_layers":layers,"sequences":sequences,"context":context,"rows":imported.contract.family.rows,"seconds":start.elapsed().as_secs_f64(),"grammar":"Wdown ~= Wup^T diag(a) + balanced f32 rank-r residual; full hidden activation/nonlinearity/gate/skip/bias untouched; native Wup paid once","fit":"weights only per hidden-column least-squares; f32 gains before residual SVD; no activation fitting, cosine threshold or selected layer","ordering":"active is row batch: (active diag(a)) Wup + active residual^T; a_j pairs Wup row j with Wdown column j","endpoint":"rank0 is tied-only or zero-native-writer baseline; full residual rank reconstructs native writer numerically, not exact arithmetic certificate","acceptance":"none; native-activation diagnostics only, not Local/Run acceptance or autonomous output KL; future exact serialized decoded acceptance required","scope":"exploratory executable candidate grammar, not discovered mechanism; all native MLP layers, explicit ranks"});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{}", output.display());
    Ok(())
}
