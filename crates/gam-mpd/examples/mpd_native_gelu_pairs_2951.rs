//! Exhaustive signed affine-reader geometry, followed by a declared heuristic bank screen.
//! EXPORT EXTRACT.json TOP_K FRESH_OUT. No fit, artifact acceptance, or discovery claim.
use gam_mpd::{
    engine::sha256,
    import::import_language_model,
    operator_program::{Law, Node, OperatorProgram},
    run_check::{layer_nodes, split_sites},
};
use ndarray::Array2;
use serde_json::{Value, json};
use std::{cmp::Ordering, collections::BinaryHeap, path::Path, time::Instant};
#[derive(Clone, Debug)]
struct Pair {
    score: f64,
    i: usize,
    j: usize,
}
impl PartialEq for Pair {
    fn eq(&self, b: &Self) -> bool {
        self.score.to_bits() == b.score.to_bits() && self.i == b.i && self.j == b.j
    }
}
impl Eq for Pair {}
impl PartialOrd for Pair {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Pair {
    fn cmp(&self, b: &Self) -> Ordering {
        self.score
            .total_cmp(&b.score)
            .then(self.i.cmp(&b.i))
            .then(self.j.cmp(&b.j))
    }
}
fn save(p: &Path, v: &Value) -> Result<(), String> {
    std::fs::write(p, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn affine(
    p: &OperatorProgram,
    node: usize,
    source: usize,
) -> Result<(Array2<f64>, Vec<f64>), String> {
    let Node::Affine { terms, bias } = &p.nodes[node] else {
        return Err("native affine required".into());
    };
    if terms.len() != 1 || terms[0].0 != source {
        return Err("single native affine source required".into());
    }
    let matrix = p.operators[terms[0].1].matrix();
    let offset = match bias {
        Some(b) => {
            let m = p.operators[*b].matrix();
            if m.dim() != (matrix.nrows(), 1) {
                return Err("bias dimensions".into());
            }
            m.column(0).to_vec()
        }
        None => vec![0.; matrix.nrows()],
    };
    Ok((matrix, offset))
}
fn panel<'a>(m: &'a Value, name: &str) -> Result<&'a Value, String> {
    let found: Vec<_> = m["panels"]
        .as_array()
        .ok_or("panels")?
        .iter()
        .filter(|p| p["name"] == name)
        .collect();
    if found.len() != 1 {
        return Err("unique panel required".into());
    }
    Ok(found[0])
}
fn load(root: &Path, p: &Value, layer: usize) -> Result<(Array2<f64>, Value), String> {
    let found: Vec<_> = p["values"]
        .as_array()
        .ok_or("values")?
        .iter()
        .filter(|v| v["layer"].as_u64() == Some(layer as u64) && v["role"] == "input")
        .collect();
    if found.len() != 1 {
        return Err("unique input required".into());
    }
    let v = found[0];
    let file = v["file"].as_str().ok_or("file")?;
    if Path::new(file).components().count() != 1
        || !matches!(
            Path::new(file).components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err("sibling input required".into());
    }
    let path = root.join(file);
    let hash = sha256(&path)?;
    if v["sha256"].as_str() != Some(&hash) {
        return Err("array hash mismatch".into());
    }
    let rows = usize::try_from(p["rows"].as_u64().ok_or("rows")?).map_err(|e| e.to_string())?;
    let width = usize::try_from(v["width"].as_u64().ok_or("width")?).map_err(|e| e.to_string())?;
    let b = std::fs::read(path).map_err(|e| e.to_string())?;
    if b.len()
        != rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(8))
            .ok_or("shape overflow")?
    {
        return Err("array size mismatch".into());
    }
    let values: Vec<_> = b
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().expect("eight bytes")))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("finite inputs required".into());
    }
    Ok((
        Array2::from_shape_vec((rows, width), values).map_err(|e| e.to_string())?,
        json!({"descriptor":v,"sha256":hash,"rows":rows,"record":p["record"]}),
    ))
}
fn norm(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(0f64, |a, b| a.hypot(b))
}
// Fixed seed and permutation are declared diagnostics, never proposal selection.
fn writer_direction_null(
    writer: &Array2<f64>,
    norms: &[f64],
    seed: u64,
) -> Result<(Array2<f64>, Vec<usize>), String> {
    let mut permutation: Vec<_> = (0..writer.ncols()).collect();
    let mut state = seed;
    for end in (1..permutation.len()).rev() {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        permutation.swap(end, (z % (end as u64 + 1)) as usize);
    }
    let mut result = Array2::zeros(writer.dim());
    for (j, &donor) in permutation.iter().enumerate() {
        if norms[donor] == 0. && norms[j] != 0. {
            return Err("permuted zero writer direction cannot preserve nonzero norm".into());
        }
        if norms[donor] != 0. {
            for row in 0..writer.nrows() {
                result[(row, j)] = (writer[(row, donor)] / norms[donor]) * norms[j];
            }
        }
    }
    Ok((result, permutation))
}
fn joint_positive(
    z: &Array2<f64>,
    active: &Array2<f64>,
    writer: &Array2<f64>,
    i: usize,
    j: usize,
    native_rms: f64,
    training: bool,
) -> Value {
    let wi = writer.column(i);
    let wj = writer.column(j);
    let contrast_writer = norm(wi.iter().zip(wj).map(|(a, b)| (a - b) * 0.5));
    let common_writer = norm(wi.iter().zip(wj).map(|(a, b)| (a + b) * 0.5));
    let ni = norm(wi.iter().copied());
    let nj = norm(wj.iter().copied());
    let projection = if ni == 0. {
        0.
    } else {
        wi.iter().zip(wj).map(|(a, b)| (a / ni) * b).sum::<f64>()
    };
    let orthogonal = if ni == 0. {
        nj
    } else {
        norm(wj.iter().zip(wi).map(|(b, a)| b - projection * (a / ni)))
    };
    let mut contrast = Vec::with_capacity(z.nrows());
    let mut common = Vec::with_capacity(z.nrows());
    let mut pair = Vec::with_capacity(z.nrows());
    for row in 0..z.nrows() {
        let gi = active[(row, i)];
        let gj = active[(row, j)];
        contrast.push(contrast_writer * (gi - gj).abs());
        common.push(common_writer * (gi + gj).abs());
        pair.push((gi * ni + gj * projection).hypot(gj * orthogonal));
    }
    let summary = |values: &[f64]| {
        let (row, &maximum) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0)))
            .expect("nonempty declared rows");
        let rms = norm(values.iter().copied()) / (values.len() as f64).sqrt();
        json!({"max_row_l2":maximum,"relative_max":maximum/native_rms,"rms_row_l2":rms,"relative_rms":rms/native_rms,"worst_row":row})
    };
    let pair_max = pair.iter().copied().fold(0f64, f64::max);
    let witnesses = |values: &[f64]| -> Vec<Value> {
        if !training {
            return vec![];
        }
        let mut rows: Vec<_> = (0..values.len()).collect();
        rows.sort_by(|a, b| values[*b].total_cmp(&values[*a]).then(a.cmp(b)));
        rows.truncate(8);
        rows.into_iter().map(|row|json!({"row":row,"sequence_ordinal":row/512,"token_position":row%512,"zi":z[(row,i)],"zj":z[(row,j)],"gi":active[(row,i)],"gj":active[(row,j)],"contrast_row_l2":contrast[row],"common_row_l2":common[row],"native_pair_row_l2":pair[row]})).collect()
    };
    json!({"identity":"wi*gi+wj*gj = ((wi-wj)/2)*(gi-gj) + ((wi+wj)/2)*(gi+gj)","contrast_writer_norm":contrast_writer,"common_writer_norm":common_writer,"contrast":summary(&contrast),"common":summary(&common),"pair_removal":summary(&pair),"drop_common_error":summary(&common),"drop_contrast_error":summary(&contrast),"drop_common_over_pair_removal":if pair_max==0.{None}else{Some(common.iter().copied().fold(0f64,f64::max)/pair_max)},"drop_contrast_over_pair_removal":if pair_max==0.{None}else{Some(contrast.iter().copied().fold(0f64,f64::max)/pair_max)},"zero_pair_max":pair_max==0.,"top_contrast_training_rows":witnesses(&contrast),"top_common_training_rows":witnesses(&common),"arithmetic":"fixed F64 native activation values; exact algebraic identity in real arithmetic, separately rounded diagnostics; no whole-forward enclosure"})
}
fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("EXPORT EXTRACT.json TOP_K FRESH_OUT".into());
    }
    let start = Instant::now();
    let k: usize = args[2].parse::<usize>().map_err(|e| e.to_string())?;
    if k == 0 || k > 1024 {
        return Err("explicit top_k in1..1024 required".into());
    }
    let out = Path::new(&args[3]);
    if out.exists() {
        return Err("fresh output required".into());
    }
    std::fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let manifest_path = Path::new(&args[1]);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(manifest_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let imported = import_language_model(Path::new(&args[0]), 2, 512)?;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, 4)?;
    let root = manifest_path.parent().ok_or("manifest parent")?;
    for name in ["train", "eval"] {
        let p = panel(&manifest, name)?;
        if p["record"]["source"]["checkpoint_sha256"]
            != imported.record["source"]["checkpoint_sha256"]
            || p["record"]["config"] != imported.record["config"]
        {
            return Err("archive checkpoint/config mismatch".into());
        }
    }
    let checkpoint = imported.record["source"]["checkpoint_sha256"]
        .as_str()
        .ok_or("explicit checkpoint SHA required")?;
    if checkpoint.len() != 64 {
        return Err("checkpoint SHA shape".into());
    }
    for name in ["train", "eval"] {
        let recorded = panel(&manifest, name)?;
        for (key, value) in imported.record["files"]
            .as_object()
            .ok_or("native files absent")?
        {
            if key.starts_with("blocks.") && (key.contains("mlp.") || key.contains("rms")) {
                let other = &recorded["record"]["files"][key];
                let hash = value["sha256"].as_str().ok_or("weight SHA absent")?;
                if hash.len() != 64
                    || other["sha256"].as_str() != Some(hash)
                    || other["shape"] != value["shape"]
                {
                    return Err(format!(
                        "native archive weight identity mismatch {name}/{key}"
                    ));
                }
            }
        }
    }
    save(
        &out.join("PROTOCOL.json"),
        &json!({"source":imported.record,"extract_sha256":sha256(manifest_path)?,"top_k":k,"selection":"exhaustive unordered signed affine-reader geometry only; writer and train/eval responses excluded from selection; both orientations screened","geometry":"||[r_j,b_j]-sign*[r_i,b_i]|| / sqrt(||[r_i,b_i]||^2+||[r_j,b_j]||^2); f64 Gram estimate; retained scores recomputed directly","panels":["train4096","eval1024"],"export_json_sha256":sha256(&Path::new(&args[0]).join("export.json"))?,"native_literals":"original imported f64, no f32 projection","positive_pair_followup":{"witness_count":8,"witness_selection":"training rows only, independently by contrast/common norms; no heldout selection","writer_null_seed":2951,"null":"within-layer permutation of writer directions rescaled to preserve each unit norm","scope":"same64positivegeometrypairs perlayer, original+null diagnostics"},"scope":"heuristic proposal inventory only; exact single-pair replacement effect on fixed native-parent rows in current CPU arithmetic, no full forward/intervention/C32 claim"}),
    )?;
    let mut reports = Vec::new();
    for (layer, l) in layers.iter().enumerate() {
        let layer_start = Instant::now();
        let (reader, bias) = affine(&native, l.pre, l.normed)?;
        let (writer, writer_bias) = affine(&native, l.mlp, l.active)?;
        let Node::Pointwise { input, laws } = &native.nodes[l.active] else {
            return Err("native pointwise required".into());
        };
        if *input != l.pre
            || laws.is_empty()
            || laws.iter().any(|law| *law != laws[0])
            || !matches!(laws[0], Law::Gelu | Law::GeluTanh)
        {
            return Err("uniform actual native GELU required".into());
        }
        let law = laws[0];
        let u = reader.nrows();
        if reader.ncols() != 768 || u != 3072 || writer.dim() != (768, u) {
            return Err("declared native4L768/3072 required".into());
        }
        let geometry_start = Instant::now();
        let mut augmented = Array2::zeros((u, reader.ncols() + 1));
        for i in 0..u {
            for c in 0..reader.ncols() {
                augmented[(i, c)] = reader[(i, c)]
            }
            augmented[(i, reader.ncols())] = bias[i]
        }
        let gram = gam_linalg::faer_ndarray::fast_abt(&augmented, &augmented);
        let mut selected = Vec::new();
        let mut distributions = Vec::new();
        for sign in [-1., 1.] {
            let mut heap = BinaryHeap::new();
            let mut bins = [0u64; 7];
            let edges = [0.001, 0.01, 0.1, 0.5, 1., 1.5];
            let mut count = 0u64;
            let mut min = f64::INFINITY;
            let mut max = 0f64;
            let mut sum = 0.;
            let mut zero_denominator = 0;
            for i in 0..u {
                for j in i + 1..u {
                    count += 1;
                    let den = gram[(i, i)] + gram[(j, j)];
                    if den == 0. {
                        zero_denominator += 1;
                        continue;
                    }
                    let score = ((den - 2. * sign * gram[(i, j)]).max(0.) / den).sqrt();
                    if !score.is_finite() {
                        return Err("nonfinite geometry".into());
                    }
                    bins[edges
                        .iter()
                        .position(|e| score <= *e)
                        .unwrap_or(edges.len())] += 1;
                    min = min.min(score);
                    max = max.max(score);
                    sum += score;
                    let p = Pair { score, i, j };
                    if heap.len() < k {
                        heap.push(p)
                    } else if heap.peek().is_some_and(|worst| p < *worst) {
                        heap.pop();
                        heap.push(p)
                    }
                }
            }
            distributions.push(json!({"sign":sign,"unordered_pairs":count,"zero_denominator_unresolved":zero_denominator,"min":min,"max":max,"mean":sum/(count-zero_denominator) as f64,"bin_upper_edges":edges,"bins":bins}));
            let mut pairs = heap.into_sorted_vec();
            pairs.sort_by(|a, b| a.cmp(b));
            for p in pairs {
                selected.push((sign, p));
            }
        }
        let geometry_seconds = geometry_start.elapsed().as_secs_f64();
        drop(gram);
        drop(augmented);
        let writer_norms: Vec<_> = (0..u)
            .map(|j| norm(writer.column(j).iter().copied()))
            .collect();
        let null_result = writer_direction_null(&writer, &writer_norms, 2951);
        let (null_writer, null_metadata) = match null_result {
            Ok((w, permutation)) => (
                Some(w),
                json!({"seed":2951,"rng":"SplitMix64 Fisher-Yates modulo selection; deterministic diagnostic, no probabilistic significance claim","permutation":permutation,"norm_preservation":"donor direction divided by donor norm then multiplied by original target writer norm; F64 arithmetic"}),
            ),
            Err(reason) => (None, json!({"seed":2951,"unresolved":reason})),
        };
        let mut results = Vec::new();
        let mut joint_results = Vec::new();
        let mut lineage = Vec::new();
        for name in ["train", "eval"] {
            let panel_start = Instant::now();
            let (x, record) = load(root, panel(&manifest, name)?, layer)?;
            if x.ncols() != reader.ncols() || x.nrows() != if name == "train" { 4096 } else { 1024 }
            {
                return Err("declared panel shape mismatch".into());
            }
            let mut z = gam_linalg::faer_ndarray::fast_abt(&x, &reader);
            for mut row in z.rows_mut() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v += bias[j]
                }
            }
            let active = z.mapv(|v| law.apply(v));
            let mut y = gam_linalg::faer_ndarray::fast_abt(&active, &writer);
            for mut row in y.rows_mut() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v += writer_bias[j]
                }
            }
            let native_rms = norm(y.iter().copied()) / (y.nrows() as f64).sqrt();
            if !native_rms.is_finite() || native_rms == 0. {
                return Err("undefined native RMS".into());
            }
            for (sign, p) in &selected {
                let direct = norm(augmented_difference(&reader, &bias, p.i, p.j, *sign));
                let denominator = norm(
                    reader
                        .row(p.i)
                        .iter()
                        .copied()
                        .chain(std::iter::once(bias[p.i])),
                )
                .hypot(norm(
                    reader
                        .row(p.j)
                        .iter()
                        .copied()
                        .chain(std::iter::once(bias[p.j])),
                ));
                let cosine_den = writer_norms[p.i] * writer_norms[p.j];
                let writer_cos = if cosine_den == 0. {
                    None
                } else {
                    Some(
                        writer
                            .column(p.i)
                            .iter()
                            .zip(writer.column(p.j))
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            / cosine_den,
                    )
                };
                if *sign > 0. {
                    let original =
                        joint_positive(&z, &active, &writer, p.i, p.j, native_rms, name == "train");
                    let null_diagnostic = null_writer
                        .as_ref()
                        .map(|w| joint_positive(&z, &active, w, p.i, p.j, native_rms, false));
                    joint_results.push(json!({"panel":name,"unit_i":p.i,"unit_j":p.j,"geometry_direct_score":direct/denominator,"writer_cosine":writer_cos,"native_rms_row_l2":native_rms,"original":original,"writer_direction_permutation_null":null_diagnostic}));
                }
                for (source, target) in [(p.i, p.j), (p.j, p.i)] {
                    let mut error = 0f64;
                    let mut deletion = 0f64;
                    let mut worst = 0;
                    for r in 0..z.nrows() {
                        let effect = (law.apply(*sign * z[(r, source)]) - active[(r, target)])
                            .abs()
                            * writer_norms[target];
                        if effect > error {
                            error = effect;
                            worst = r
                        }
                        deletion = deletion.max(active[(r, target)].abs() * writer_norms[target])
                    }
                    let ratio = if deletion == 0. {
                        None
                    } else {
                        Some(error / deletion)
                    };
                    let linear = if *sign < 0. {
                        let ni = writer_norms[source];
                        let projection = if ni == 0. {
                            0.
                        } else {
                            writer
                                .column(source)
                                .iter()
                                .zip(writer.column(target))
                                .map(|(a, b)| (a / ni) * b)
                                .sum::<f64>()
                        };
                        let orthogonal = if ni == 0. {
                            writer_norms[target]
                        } else {
                            norm(
                                writer
                                    .column(target)
                                    .iter()
                                    .zip(writer.column(source))
                                    .map(|(b, a)| b - projection * (a / ni)),
                            )
                        };
                        let mut linear_error = 0f64;
                        let mut linear_row = 0;
                        let mut pair_deletion = 0f64;
                        for r in 0..z.nrows() {
                            let a = active[(r, source)];
                            let b = active[(r, target)];
                            let error =
                                ((a - z[(r, source)]) * ni + b * projection).hypot(b * orthogonal);
                            if error > linear_error {
                                linear_error = error;
                                linear_row = r
                            }
                            pair_deletion =
                                pair_deletion.max((a * ni + b * projection).hypot(b * orthogonal));
                        }
                        let mut gain_controls = Vec::new();
                        for writer_sign in [1., -1.] {
                            let mut activity = 0f64;
                            let mut oddness = 0f64;
                            let mut scaling = 0f64;
                            let mut gain_max = [0f64; 4];
                            for r in 0..z.nrows() {
                                let zi = z[(r, source)];
                                let zj = z[(r, target)];
                                let vector_norm = |a: f64, b: f64| {
                                    (a * ni + writer_sign * b * projection).hypot(b * orthogonal)
                                };
                                for (index, gain) in [-1., 0., 1., 2.].into_iter().enumerate() {
                                    gain_max[index] = gain_max[index].max(vector_norm(
                                        law.apply(gain * zi),
                                        law.apply(gain * zj),
                                    ));
                                }
                                activity = activity.max(vector_norm(law.apply(zi), law.apply(zj)));
                                oddness = oddness.max(vector_norm(
                                    law.apply(-zi) + law.apply(zi),
                                    law.apply(-zj) + law.apply(zj),
                                ));
                                scaling = scaling.max(vector_norm(
                                    law.apply(2. * zi) - 2. * law.apply(zi),
                                    law.apply(2. * zj) - 2. * law.apply(zj),
                                ));
                            }
                            gain_controls.push(json!({"target_writer_sign":writer_sign,"common_pre_gains":[-1.,0.,1.,2.],"gain_max_output_l2":gain_max,"pair_activity_max_l2":activity,"oddness_max_l2":oddness,"scaling_defect_max_l2":scaling,"oddness_over_activity":if activity==0.{None}else{Some(oddness/activity)},"scaling_over_activity":if activity==0.{None}else{Some(scaling/activity)},"zero_activity":activity==0.}));
                        }
                        Some(
                            json!({"hypothesis":"w_source*z_source replaces native nonlinear pair","max_row_l2":linear_error,"relative_max":linear_error/native_rms,"worst_row":linear_row,"source_pre":z[(linear_row,source)],"target_pre":z[(linear_row,target)],"source_activation":active[(linear_row,source)],"target_activation":active[(linear_row,target)],"linear_prediction_scalar":z[(linear_row,source)],"pair_deletion_max_row_l2":pair_deletion,"pair_deletion_relative_max":pair_deletion/native_rms,"error_over_pair_deletion":if pair_deletion==0.{None}else{Some(linear_error/pair_deletion)},"common_gain_writer_controls":gain_controls,"arithmetic":"stable two-vector basis with direct orthogonal residual norm; measured f64, not certified enclosure"}),
                        )
                    } else {
                        None
                    };
                    results.push(json!({"panel":name,"sign":sign,"source_unit":source,"replaced_unit":target,"geometry_gram_score":p.score,"geometry_direct_score":direct/denominator,"reader_affine_difference_norm":direct,"writer_cosine":writer_cos,"writer_norm":writer_norms[target],"native_rms_row_l2":native_rms,"replacement_max_row_l2":error,"replacement_relative_max":error/native_rms,"worst_row":worst,"source_pre_at_worst":z[(worst,source)],"target_pre_at_worst":z[(worst,target)],"native_target_activation_at_worst":active[(worst,target)],"replacement_activation_at_worst":law.apply(*sign*z[(worst,source)]),"linear_cancellation":linear,"deletion_max_row_l2":deletion,"deletion_relative_max":deletion/native_rms,"relation_over_deletion":ratio,"zero_deletion":deletion==0.,"cost_note":"deletion removes reader AND writer; relation merge usually only reader. No artifact costs computed"}));
                }
            }
            lineage.push(json!({"panel":name,"record":record,"seconds":panel_start.elapsed().as_secs_f64(),"native_rms_row_l2":native_rms}));
        }
        let mut joint_aggregates = Vec::new();
        for panel in ["train", "eval"] {
            for mode in ["original", "writer_direction_permutation_null"] {
                let records: Vec<_> = joint_results
                    .iter()
                    .filter(|v| v["panel"] == panel && !v[mode].is_null())
                    .collect();
                let mut metrics = serde_json::Map::new();
                for term in ["contrast", "common", "pair_removal"] {
                    let values: Vec<_> = records
                        .iter()
                        .filter_map(|v| v[mode][term]["rms_row_l2"].as_f64())
                        .collect();
                    let maximum = records
                        .iter()
                        .filter_map(|v| v[mode][term]["max_row_l2"].as_f64())
                        .fold(0f64, f64::max);
                    metrics.insert(term.into(),json!({"bank_rms_over_pairs_and_rows":if values.is_empty(){None}else{Some(norm(values.iter().copied())/(values.len() as f64).sqrt())},"max_over_pairs_and_rows":maximum}));
                }
                joint_aggregates.push(json!({"panel":panel,"mode":mode,"pair_count":records.len(),"metrics":metrics,"scope":"same geometry-selected bank; no response ranking or significance claim"}));
            }
        }
        let report = json!({"layer":layer,"law":format!("{law:?}"),"units":u,"geometry_seconds":geometry_seconds,"distributions":distributions,"selected_unordered_pairs":selected.len(),"screened_orientations_per_panel":selected.len()*2,"lineage":lineage,"results":results,"positive_pair_joint":joint_results,"writer_null_metadata":null_metadata,"positive_pair_aggregates":joint_aggregates,"seconds":layer_start.elapsed().as_secs_f64()});
        save(&out.join(format!("layer{layer}.json")), &report)?;
        println!(
            "layer{layer} completed {:.3}s",
            layer_start.elapsed().as_secs_f64()
        );
        reports.push(json!({"layer":layer,"report":format!("layer{layer}.json"),"seconds":layer_start.elapsed().as_secs_f64()}));
    }
    save(
        &out.join("SUMMARY.json"),
        &json!({"layers":reports,"seconds":start.elapsed().as_secs_f64(),"scope":"Complete geometry, heuristic top-k response bank. All retained candidates reported on both panels; no heldout selection, fit, C32, interventions, discovery or acceptance claim."}),
    )
}
fn augmented_difference<'a>(
    reader: &'a Array2<f64>,
    bias: &'a [f64],
    i: usize,
    j: usize,
    sign: f64,
) -> impl Iterator<Item = f64> + 'a {
    reader
        .row(j)
        .to_vec()
        .into_iter()
        .zip(reader.row(i).to_vec())
        .map(move |(b, a)| b - sign * a)
        .chain(std::iter::once(bias[j] - sign * bias[i]))
}
fn main() -> Result<(), String> {
    run()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positive_pair_decomposition_and_drop_errors_are_actual_vector_errors() {
        let writer = ndarray::array![[1., -0.8], [0.5, 0.7], [-0.2, 0.4]];
        let z = ndarray::array![[0.4, 0.5], [-1., -0.9]];
        let active = z.mapv(|v| Law::GeluTanh.apply(v));
        let report = joint_positive(&z, &active, &writer, 0, 1, 1., true);
        let mut max_common = 0f64;
        let mut max_contrast = 0f64;
        for row in 0..z.nrows() {
            let gi = active[(row, 0)];
            let gj = active[(row, 1)];
            let full = writer.column(0).mapv(|w| w * gi) + writer.column(1).mapv(|w| w * gj);
            let contrast = (writer.column(0).to_owned() - writer.column(1)) * ((gi - gj) * 0.5);
            let common = (writer.column(0).to_owned() + writer.column(1)) * ((gi + gj) * 0.5);
            assert!(
                norm(
                    full.iter()
                        .zip(contrast.iter().zip(&common))
                        .map(|(f, (c, m))| f - c - m)
                ) < 1e-15
            );
            max_common = max_common.max(norm(full.iter().zip(&contrast).map(|(a, b)| a - b)));
            max_contrast = max_contrast.max(norm(full.iter().zip(&common).map(|(a, b)| a - b)));
        }
        assert!(
            (report["drop_common_error"]["max_row_l2"]
                .as_f64()
                .expect("common max")
                - max_common)
                .abs()
                < 1e-15
        );
        assert!(
            (report["drop_contrast_error"]["max_row_l2"]
                .as_f64()
                .expect("contrast max")
                - max_contrast)
                .abs()
                < 1e-15
        );
        assert_eq!(
            report["top_contrast_training_rows"]
                .as_array()
                .expect("witnesses")
                .len(),
            2
        );
        assert!(
            joint_positive(&z, &active, &writer, 0, 1, 1., false)["top_common_training_rows"]
                .as_array()
                .expect("eval witnesses")
                .is_empty()
        );
        let norms: Vec<_> = (0..writer.ncols())
            .map(|j| norm(writer.column(j).iter().copied()))
            .collect();
        let (null, permutation) =
            writer_direction_null(&writer, &norms, 2951).expect("null permutation");
        assert_eq!(
            permutation,
            writer_direction_null(&writer, &norms, 2951)
                .expect("same seeded null")
                .1
        );
        for (j, expected) in norms.iter().enumerate() {
            assert!((norm(null.column(j).iter().copied()) - expected).abs() < 1e-15);
        }
    }
    #[test]
    fn duplicate_affine_rows_include_bias_and_both_orientations() {
        let readers = ndarray::array![[1., 2.], [1., 2.]];
        assert_eq!(
            norm(augmented_difference(&readers, &[0.3, 0.3], 0, 1, 1.)),
            0.
        );
        assert!(norm(augmented_difference(&readers, &[0.3, 0.4], 0, 1, 1.)) > 0.);
        for (i, j) in [(0, 1), (1, 0)] {
            let zi = readers.row(i).dot(&ndarray::array![0.5, -0.2]) + 0.3;
            let zj = readers.row(j).dot(&ndarray::array![0.5, -0.2]) + 0.3;
            assert_eq!(Law::GeluTanh.apply(zi), Law::GeluTanh.apply(zj));
        }
    }
    #[test]
    fn opposing_reader_writer_pair_is_linear_but_unpaired_writer_is_not() {
        let w = ndarray::array![0.3, -1.2, 0.7];
        for law in [Law::Gelu, Law::GeluTanh] {
            for t in [-4., -0.3, 0., 0.8, 3.] {
                let pair = w.mapv(|v| v * law.apply(t) - v * law.apply(-t));
                let linear = w.mapv(|v| v * t);
                assert!(norm(pair.iter().zip(&linear).map(|(a, b)| a - b)) < 1e-14);
            }
            let t = 0.8;
            let unmatched = w.mapv(|v| v * law.apply(t) + v * law.apply(-t) - v * t);
            assert!(norm(unmatched.iter().copied()) > 0.1);
        }
    }
}
