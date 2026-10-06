//! The oracle's label table for a library explanation (`library_mdl`, #2951): the measured behaviour
//! of each of its functions, in the schema and file names `bench/oracle/vpd_labels.py` and
//! `vpd_relations.py` write for VPD's subcomponents, so the oracle reads our graph in place of VPD's.
//!
//! A component is one function of the library: an MLP function `h_i u_i`, `h_i = φ(g_i·x̂ + c_i)`
//! (times `w_i·x̂` when gated), or a head `W_O,h z_h`. Its activity at a token is the scalar its
//! write carries: an MLP function's activation `h_i` (the coefficient of its fixed write direction
//! `u_i`, as `v·x` is a subcomponent's), a head's output norm `‖W_O,h z_h‖` (a head writes a vector
//! with no fixed direction, so its size is the one scalar it has). Its native edit is its output
//! scaled by `α` at every position, every later computation run again (`Library::edited`): `α = 1`
//! is the explanation `P` itself, `α = 0` the function's removal.
//!
//! Sites: per layer `l`, `h.{l}.attn.head` (its heads) and `h.{l}.mlp.function` (its MLP functions),
//! files `site_{l}_head.*` and `site_{l}_function.*`; global ids number the functions site by site,
//! the library's function order. Contexts are the export's rows `pool` of `context` tokens.
//!
//! Labels (vpd_labels.py's fields): per function its `top` contexts of largest peak |activity| in
//! the pool and `random` more drawn uniformly from the rest; per context the activity at every
//! position (float16), the peak position p, and for α in (0, 1.5) the KL(clean ‖ edited) of the
//! next-token distribution at p (nats), the change of log p of the actual next token, and the 10
//! tokens whose probability rises most and the 10 that fall most; per function the greedy
//! continuation of `steps` tokens after p in its first context with α = 1.5 beside the clean one.
//!
//! Relations (vpd_relations.py's fields): `continuations` (α = 2, 4, 8 and clean), `downstream` (at
//! each of a function's first `contexts` contexts, at its peak, the exact change of every later
//! function's activity when it is removed; the `k` largest relative to each one's own peak
//! |activity|), and `attribution` (at `positions` random positions of each of the first
//! `attribution_contexts` contexts: X the most probable next token, proposals the `2·proposals`
//! functions whose write at that position has the largest direct effect on X's logit through the
//! final norm, used only to choose what to measure, and each one's exact removal change of
//! log p(X)), `upstream` (at each of a function B's first `contexts` contexts, at its peak, the
//! `proposals` functions writing into what B reads whose write contributes most to B's reads
//! there, `Σ_route ‖R (γ ⊙ w_A(p))‖ / r(p)`, used only to choose what to measure, and each one's
//! exact removal change of B's activity at p, strongest first) and `edges` (at B's strongest
//! context: path patches, `Library::path_patched`, of its two strongest measured upstream
//! neighbours and of its weakest measured proposal: the change of B's activity at p, and of the
//! next-token distribution there). `functions.safetensors` holds the functions' maps
//! (`Library::maps`), an MLP site's also as VPD's `{site}.U` and `{site}.V`.
//!
//! EXPORT SETTINGS.json OUT_DIR [ARTIFACT]
//!
//! `SETTINGS.json`: `{context, pool: [start, end), top, random, steps, contexts, k, proposals,
//! attribution_contexts, positions, batch, numeric_bytes, tile_rows, seed, limit, parts}` (`limit` >
//! 0: only each site's first `limit` functions; `parts`: which of labels, downstream, attribution,
//! upstream and edges to write, the labels with the relations' continuations; the contexts, chosen
//! by `seed`, are the same in every run of one pool).
use gam_gpu::{GpuPolicy, tensor::Device};
use gam_mpd::{
    artifact::Artifact,
    engine::log_to_stderr,
    import::import_language_model,
    library_mdl,
    library_readout::{Activity, Edit, Kind, Library},
    operator_program::{OperatorProgram, SlotValues},
    run_check::{LayerNodes, layer_nodes, split_sites},
};
use ndarray::{Array2, Axis};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Instant};

const ALPHAS: [(&str, f64); 2] = [("ablate", 0.0), ("amplify", 1.5)];
const AMPLIFY: f64 = 1.5;
const RELATION_ALPHAS: [f64; 3] = [2.0, 4.0, 8.0];
const TOP: usize = 10;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    context: usize,
    pool: [usize; 2],
    top: usize,
    random: usize,
    steps: usize,
    contexts: usize,
    k: usize,
    proposals: usize,
    attribution_contexts: usize,
    positions: usize,
    batch: usize,
    numeric_bytes: usize,
    tile_rows: usize,
    seed: u64,
    limit: usize,
    parts: Vec<String>,
}

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// The single-precision device (CUDA in float32, else the Apple GPU), else the host: the labels are
/// measured in float32, as VPD's are; log-probabilities are normalized in float64 on the host.
fn devices() -> Result<(Device, Device), String> {
    let model = Device::single_precision(GpuPolicy::Auto).map_err(error)?.unwrap_or_else(Device::host);
    Ok((model.clone(), model))
}

fn load(export: &Path, sequences: usize, context: usize, artifact: Option<&Path>) -> Result<(OperatorProgram, Vec<LayerNodes>, Vec<u32>, Artifact), String> {
    let imported = import_language_model(export, sequences, context)?;
    let layer_count = imported.record["config"]["n_layers"].as_u64().ok_or("config.n_layers")? as usize;
    let native = split_sites(&imported.program)?;
    let layers = layer_nodes(&native, layer_count)?;
    let SlotValues::Tokens(tokens) = &imported.family.slots[0] else {
        return Err("a token slot".into());
    };
    let tokens = tokens.clone();
    drop(imported);
    let artifact = match artifact {
        Some(path) => Artifact::from_bytes(&std::fs::read(path).map_err(error)?, &native.declarations)?,
        None => library_mdl::explanation(&native, &layers)?.artifact,
    };
    artifact.validate_coverage(&native)?;
    Ok((native, layers, tokens, artifact))
}

// --------------------------------------------------------------------------------- safetensors

/// One array to write: dtype ("F32", "F16", "I32", "I16", "I64"), shape, little-endian bytes.
struct Array {
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn f32s(shape: Vec<usize>, values: impl IntoIterator<Item = f64>) -> Array {
    Array { dtype: "F32", shape, bytes: values.into_iter().flat_map(|v| (v as f32).to_le_bytes()).collect() }
}

fn i32s(shape: Vec<usize>, values: impl IntoIterator<Item = i64>) -> Array {
    Array { dtype: "I32", shape, bytes: values.into_iter().flat_map(|v| (v as i32).to_le_bytes()).collect() }
}

fn i16s(shape: Vec<usize>, values: impl IntoIterator<Item = i64>) -> Array {
    Array { dtype: "I16", shape, bytes: values.into_iter().flat_map(|v| (v as i16).to_le_bytes()).collect() }
}

fn i8s(shape: Vec<usize>, values: impl IntoIterator<Item = i64>) -> Array {
    Array { dtype: "I8", shape, bytes: values.into_iter().map(|v| v as i8 as u8).collect() }
}

fn i64s(shape: Vec<usize>, values: impl IntoIterator<Item = i64>) -> Array {
    Array { dtype: "I64", shape, bytes: values.into_iter().flat_map(i64::to_le_bytes).collect() }
}

/// IEEE half precision of `x`, rounded to nearest even (overflow to infinity).
fn half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mantissa | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut out = m >> shift;
        if rest > half || (rest == half && out & 1 == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    let rest = mantissa & 0x1fff;
    let mut out = ((e as u32) << 10) | (mantissa >> 13);
    if rest > 0x1000 || (rest == 0x1000 && out & 1 == 1) {
        out += 1;
    }
    sign | out as u16
}

fn f16s(shape: Vec<usize>, values: impl IntoIterator<Item = f64>) -> Array {
    Array { dtype: "F16", shape, bytes: values.into_iter().flat_map(|v| half(v as f32).to_le_bytes()).collect() }
}

fn save(path: &Path, arrays: BTreeMap<String, Array>) -> Result<(), String> {
    let mut header = serde_json::Map::new();
    let mut at = 0usize;
    for (name, a) in &arrays {
        let width = match a.dtype {
            "I64" => 8,
            "F32" | "I32" => 4,
            "I8" => 1,
            _ => 2,
        };
        if a.shape.iter().product::<usize>() * width != a.bytes.len() {
            return Err(format!("{name}: {} bytes for the shape {:?}", a.bytes.len(), a.shape));
        }
        header.insert(name.clone(), json!({"dtype": a.dtype, "shape": a.shape, "data_offsets": [at, at + a.bytes.len()]}));
        at += a.bytes.len();
    }
    let mut text = serde_json::to_vec(&Value::Object(header)).map_err(error)?;
    while text.len() % 8 != 0 {
        text.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + text.len() + at);
    out.extend((text.len() as u64).to_le_bytes());
    out.extend(text);
    for a in arrays.values() {
        out.extend(&a.bytes);
    }
    std::fs::write(path, out).map_err(error)
}

// ------------------------------------------------------------------------------------- helpers

/// The indices of the `k` largest values of `score`, descending.
fn top_k(score: impl Iterator<Item = f64>, k: usize) -> Vec<usize> {
    let values: Vec<f64> = score.collect();
    let mut order: Vec<usize> = (0..values.len()).collect();
    let k = k.min(order.len());
    if k == 0 {
        return Vec::new();
    }
    order.select_nth_unstable_by(k - 1, |a, b| values[*b].total_cmp(&values[*a]));
    order.truncate(k);
    order.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
    order
}

/// The clean final streams of the pool's rows `at` ((sequence, position) pairs) on the model's device.
fn clean_rows(model: &Device, clean: &Array2<f32>, t: usize, at: &[(usize, usize)]) -> Result<gam_gpu::tensor::Tensor, String> {
    let rows = Array2::from_shape_fn((at.len(), clean.ncols()), |(r, c)| f64::from(clean[[at[r].0 * t + at[r].1, c]]));
    model.upload(rows.view()).map_err(error)
}

/// Per edit (each reading one row), its next-token change against the clean row: the edits run
/// in batches of `batch`, `next` the token whose log p change is read (none: NaN).
fn effects(library: &Library, model: &Device, pool: &[Vec<u32>], clean: &Array2<f32>, edits: &[Edit], next: &[Option<usize>], top: usize, batch: usize) -> Result<Vec<gam_mpd::library_readout::TokenEffects>, String> {
    let t = pool[0].len();
    let mut out = Vec::with_capacity(edits.len());
    for (chunk, next) in edits.chunks(batch.max(1)).zip(next.chunks(batch.max(1))) {
        let e = library.edited(pool, chunk, &Activity::None, false, false)?;
        let at: Vec<(usize, usize)> = chunk.iter().map(|e| (e.sequence, e.rows[0])).collect();
        out.extend(library.token_effects(&e.last_device, &clean_rows(model, clean, t, &at)?, next, top)?);
    }
    Ok(out)
}

/// Greedy continuations of `steps` tokens: per job (function or none, α, sequence, last prefix
/// position), the tokens after the prefix with the job's edit at every step.
fn greedy(library: &Library, pool: &[Vec<u32>], jobs: &[(Option<usize>, f64, usize, usize)], steps: usize, batch: usize) -> Result<Vec<Vec<u32>>, String> {
    let length = pool[0].len() + steps;
    let mut out = vec![Vec::with_capacity(steps); jobs.len()];
    for (c, chunk) in jobs.chunks(batch.max(1)).enumerate() {
        let mut buffers: Vec<Vec<u32>> = chunk
            .iter()
            .map(|(_, _, s, p)| {
                let mut b = vec![0u32; length];
                b[..=*p].copy_from_slice(&pool[*s][..=*p]);
                b
            })
            .collect();
        for step in 0..steps {
            let edits: Vec<Edit> = chunk
                .iter()
                .enumerate()
                .map(|(k, (f, alpha, _, p))| Edit { sequence: k, scale: f.map(|f| vec![(f, *alpha)]).unwrap_or_default(), add: vec![], rows: vec![p + step] })
                .collect();
            let last = library.edited(&buffers, &edits, &Activity::None, false, false)?.last_device;
            for (k, ((_, _, _, p), next)) in chunk.iter().zip(library.argmax_tokens(&last)?).enumerate() {
                buffers[k][p + step + 1] = next as u32;
                out[c * batch.max(1) + k].push(next as u32);
            }
        }
    }
    Ok(out)
}

fn main() -> Result<(), String> {
    log_to_stderr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (export, settings_path, out, artifact_path) = match args.as_slice() {
        [e, s, o] => (e, s, o, None),
        [e, s, o, a] => (e, s, o, Some(Path::new(a))),
        _ => return Err("EXPORT SETTINGS.json OUT_DIR [ARTIFACT]".into()),
    };
    let settings: Settings = serde_json::from_slice(&std::fs::read(settings_path).map_err(error)?).map_err(error)?;
    let known = ["labels", "downstream", "attribution", "upstream", "edges"];
    if let Some(p) = settings.parts.iter().find(|p| !known.contains(&p.as_str())) {
        return Err(format!("unknown part {p}"));
    }
    let wants = |part: &str| settings.parts.iter().any(|p| p == part);
    let out = Path::new(out);
    std::fs::create_dir_all(out).map_err(error)?;
    let started = Instant::now();
    let (model, wide) = devices()?;
    let t = settings.context;
    let (native, layers, tokens, artifact) = load(Path::new(export), settings.pool[1], t, artifact_path)?;
    let pool: Vec<Vec<u32>> = tokens.chunks(t).skip(settings.pool[0]).take(settings.pool[1] - settings.pool[0]).map(<[u32]>::to_vec).collect();
    if pool.len() != settings.pool[1] - settings.pool[0] || pool.len() <= settings.top + settings.random {
        return Err("the pool holds too few rows for the contexts".into());
    }
    let library = Library::new(&model, &wide, &native, &layers, &artifact, settings.numeric_bytes, settings.tile_rows)?;
    drop(native);
    let functions = library.functions();
    let total = functions.len();
    // Sites in function order: (name, tag, layer, first global id, functions).
    let mut sites: Vec<(String, String, usize, usize, usize)> = Vec::new();
    for (g, f) in functions.iter().enumerate() {
        let (name, tag) = match f.kind {
            Kind::Head => (format!("h.{}.attn.head", f.layer), format!("{}_head", f.layer)),
            Kind::Mlp => (format!("h.{}.mlp.function", f.layer), format!("{}_function", f.layer)),
        };
        match sites.last_mut() {
            Some(s) if s.0 == name => s.4 += 1,
            _ => sites.push((name, tag, f.layer, g, 1)),
        }
    }
    // Measured functions: each site's first `limit` (all without a limit).
    let chosen: Vec<usize> = sites.iter().flat_map(|s| s.3..s.3 + if settings.limit > 0 { settings.limit.min(s.4) } else { s.4 }).collect();
    let later = |a: usize, b: usize| -> bool {
        let (fa, fb) = (&functions[a], &functions[b]);
        fb.layer > fa.layer || (fb.layer == fa.layer && matches!(fa.kind, Kind::Head) && matches!(fb.kind, Kind::Mlp))
    };
    log::info!("{} functions in {} sites, {} measured; pool {} rows of {t}", total, sites.len(), chosen.len(), pool.len());
    save(&out.join("contexts.safetensors"), [("rows".into(), i64s(vec![pool.len()], (settings.pool[0]..settings.pool[1]).map(|r| r as i64))), ("tokens".into(), i32s(vec![pool.len(), t], pool.iter().flatten().map(|v| i64::from(*v))))].into())?;
    let mut maps = BTreeMap::new();
    for (name, shape, values) in library.maps() {
        // An MLP function's read and write vectors also under VPD's names: `{site}.V` (width ×
        // functions, its gate directions as columns) and `{site}.U` (functions × width, its writes).
        if let Some(site) = name.strip_suffix(".gate") {
            let (c, d) = (shape[0], shape[1]);
            maps.insert(format!("{site}.V"), f32s(vec![d, c], (0..d * c).map(|x| values[(x % c) * d + x / c])));
        }
        if let Some(site) = name.strip_suffix(".write") {
            maps.insert(format!("{site}.U"), f32s(shape.clone(), values.iter().copied()));
        }
        maps.insert(name, f32s(shape, values));
    }
    save(&out.join("functions.safetensors"), maps)?;

    // Clean scan: per function and pool row its peak |activity|; the clean final streams.
    let all_rows: Vec<usize> = (0..t).collect();
    let scan_batch = (settings.batch / 4).max(1);
    let mut peaks = Array2::<f32>::zeros((total, pool.len()));
    let mut clean_last = Array2::<f32>::zeros((pool.len() * t, 0));
    let mut lasts = Vec::new();
    for start in (0..pool.len()).step_by(scan_batch) {
        let edits: Vec<Edit> = (start..(start + scan_batch).min(pool.len())).map(|s| Edit { sequence: s, rows: all_rows.clone(), ..Edit::default() }).collect();
        let e = library.edited(&pool, &edits, &Activity::All, false, false)?;
        let a = e.activity.ok_or("activity")?;
        for (k, edit) in edits.iter().enumerate() {
            let rows = a.slice(ndarray::s![k * t..(k + 1) * t, ..]);
            for f in 0..total {
                peaks[[f, edit.sequence]] = rows.column(f).iter().fold(0.0_f64, |m, v| m.max(v.abs())) as f32;
            }
        }
        lasts.push(e.last.mapv(|v| v as f32));
    }
    if !lasts.is_empty() {
        clean_last = ndarray::concatenate(Axis(0), &lasts.iter().map(|a| a.view()).collect::<Vec<_>>()).map_err(error)?;
    }
    drop(lasts);
    log::info!("clean scan in {:.0} s", started.elapsed().as_secs_f64());

    // Contexts per function: the top ones by peak |activity|, then uniform others.
    let kk = settings.top + settings.random;
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let mut contexts = vec![Vec::new(); total];
    for &f in &chosen {
        let order = top_k(peaks.row(f).iter().map(|v| f64::from(*v)), pool.len());
        let mut rest: Vec<usize> = order[settings.top..].to_vec();
        rest.shuffle(&mut rng);
        contexts[f] = order[..settings.top].iter().copied().chain(rest.into_iter().take(settings.random)).collect();
    }
    // Activity traces of the chosen contexts, and the peak positions.
    let mut wanted: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for &f in &chosen {
        for (j, s) in contexts[f].iter().enumerate() {
            wanted.entry(*s).or_default().push((f, j));
        }
    }
    let mut activity = vec![vec![vec![0.0f32; t]; kk]; total];
    let needed: Vec<usize> = wanted.keys().copied().collect();
    for chunk in needed.chunks(scan_batch) {
        let edits: Vec<Edit> = chunk.iter().map(|s| Edit { sequence: *s, rows: all_rows.clone(), ..Edit::default() }).collect();
        let a = library.edited(&pool, &edits, &Activity::All, false, false)?.activity.ok_or("activity")?;
        for (k, s) in chunk.iter().enumerate() {
            for (f, j) in &wanted[s] {
                for p in 0..t {
                    activity[*f][*j][p] = a[[k * t + p, *f]] as f32;
                }
            }
        }
    }
    let position: Vec<Vec<usize>> = (0..total).map(|f| activity[f].iter().map(|row| top_k(row.iter().map(|v| f64::from(v.abs())), 1).first().copied().unwrap_or(0)).collect()).collect();
    // Each function's peak |activity| over the pool (its top context holds it), the scale of its changes.
    let peak_of: Vec<f64> = (0..total).map(|f| peaks.row(f).iter().fold(0.0_f64, |m, v| m.max(f64::from(*v))).max(f64::from(f32::MIN_POSITIVE))).collect();
    log::info!("contexts in {:.0} s", started.elapsed().as_secs_f64());

    let j_count = settings.contexts.min(kk);
    let k = settings.k;
    let steps = settings.steps;
    let pcount = settings.proposals.min(total);
    let (mut u_ids, mut u_delta) = (vec![-1i64; total * j_count * pcount], vec![0.0f64; total * j_count * pcount]);
    if wants("edges") && !wants("upstream") {
        return Err("edges are chosen from the upstream measurements: ask for both".into());
    }
    if wants("labels") {
    // Effects of α = 0 and 1.5 at each (function, context)'s peak.
    let pairs: Vec<(usize, usize)> = chosen.iter().flat_map(|f| (0..kk).map(move |j| (*f, j))).collect();
    let mut effect: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut effect_ids: BTreeMap<String, Vec<i64>> = BTreeMap::new();
    for (alpha_name, alpha) in ALPHAS {
        let edits: Vec<Edit> = pairs.iter().map(|(f, j)| Edit { sequence: contexts[*f][*j], scale: vec![(*f, alpha)], rows: vec![position[*f][*j]], ..Edit::default() }).collect();
        let next_tokens: Vec<Option<usize>> = pairs.iter().map(|(f, j)| (position[*f][*j] + 1 < t).then(|| pool[contexts[*f][*j]][position[*f][*j] + 1] as usize)).collect();
        let measured = effects(&library, &model, &pool, &clean_last, &edits, &next_tokens, TOP, settings.batch)?;
        let (mut kl, mut next) = (Vec::new(), Vec::new());
        let (mut up_ids, mut down_ids, mut up_dp, mut up_dl, mut down_dp, mut down_dl) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for m in &measured {
            kl.push(m.kl);
            next.push(m.next);
            for (ids, dps, dls, list) in [(&mut up_ids, &mut up_dp, &mut up_dl, &m.up), (&mut down_ids, &mut down_dp, &mut down_dl, &m.down)] {
                for &(i, dp, dl) in list {
                    ids.push(i as i64);
                    dps.push(dp);
                    dls.push(dl);
                }
            }
        }
        effect.insert(format!("kl_{alpha_name}"), kl);
        effect.insert(format!("next_{alpha_name}"), next);
        effect_ids.insert(format!("up_ids_{alpha_name}"), up_ids);
        effect_ids.insert(format!("down_ids_{alpha_name}"), down_ids);
        effect.insert(format!("up_dp_{alpha_name}"), up_dp);
        effect.insert(format!("up_dlogp_{alpha_name}"), up_dl);
        effect.insert(format!("down_dp_{alpha_name}"), down_dp);
        effect.insert(format!("down_dlogp_{alpha_name}"), down_dl);
        log::info!("{alpha_name} effects in {:.0} s", started.elapsed().as_secs_f64());
    }

    // Continuations after each function's first context's peak: clean, 1.5, then 2, 4, 8.
    let alphas_all: Vec<f64> = std::iter::once(1.0).chain(std::iter::once(AMPLIFY)).chain(RELATION_ALPHAS).collect();
    let jobs: Vec<(Option<usize>, f64, usize, usize)> = chosen.iter().flat_map(|f| { let (s, p) = (contexts[*f][0], position[*f][0]); alphas_all.iter().map(move |a| (if *a == 1.0 { None } else { Some(*f) }, *a, s, p)) }).collect();
    let continued = greedy(&library, &pool, &jobs, settings.steps, settings.batch)?;
    let continuation = |f_index: usize, a: usize| &continued[f_index * alphas_all.len() + a];
    log::info!("continuations in {:.0} s", started.elapsed().as_secs_f64());

    // Label files, site by site.
    let position_in: BTreeMap<usize, usize> = chosen.iter().enumerate().map(|(i, f)| (*f, i)).collect();
    for (name, tag, layer, first, count) in &sites {
        let ids: Vec<usize> = chosen.iter().copied().filter(|f| (*first..first + count).contains(f)).collect();
        let c = ids.len();
        let rows_of = |f: usize| position_in[&f] * kk..(position_in[&f] + 1) * kk;
        let mut arrays: BTreeMap<String, Array> = BTreeMap::new();
        for (key, values) in &effect {
            let per = if key.starts_with("kl_") || key.starts_with("next_") { 1 } else { TOP };
            let shape = if per == 1 { vec![c, kk] } else { vec![c, kk, TOP] };
            arrays.insert(key.clone(), f32s(shape, ids.iter().flat_map(|f| rows_of(*f).flat_map(|r| values[r * per..(r + 1) * per].to_vec()))));
        }
        for (key, values) in &effect_ids {
            arrays.insert(key.clone(), i32s(vec![c, kk, TOP], ids.iter().flat_map(|f| rows_of(*f).flat_map(|r| values[r * TOP..(r + 1) * TOP].to_vec()))));
        }
        arrays.insert("contexts".into(), i32s(vec![c, kk], ids.iter().flat_map(|f| contexts[*f].iter().map(|s| *s as i64))));
        arrays.insert("activity".into(), f16s(vec![c, kk, t], ids.iter().flat_map(|f| activity[*f].iter().flatten().map(|v| f64::from(*v)))));
        arrays.insert("position".into(), i16s(vec![c, kk], ids.iter().flat_map(|f| position[*f].iter().map(|p| *p as i64))));
        arrays.insert("clean".into(), i32s(vec![c, settings.steps], ids.iter().flat_map(|f| continuation(position_in[f], 0).iter().map(|v| i64::from(*v)))));
        arrays.insert("amplified".into(), i32s(vec![c, settings.steps], ids.iter().flat_map(|f| continuation(position_in[f], 1).iter().map(|v| i64::from(*v)))));
        save(&out.join(format!("site_{tag}.safetensors")), arrays)?;
        let meta = json!({
            "site": name, "layer": layer, "subcomponents": c, "top": settings.top, "random": settings.random, "pool_offset": settings.pool[0], "pool": pool.len(),
            "alphas": {"ablate": 0.0, "amplify": AMPLIFY}, "amplify": AMPLIFY, "continuation_steps": settings.steps, "seconds": started.elapsed().as_secs_f64(), "seed": settings.seed,
            "decomposition": artifact_path.map_or_else(|| "the library's starting point".to_string(), |p| p.display().to_string()), "target": export,
            "component": "a library function: an MLP function h_i u_i (activity h_i) or a head W_O z (activity |W_O z|)",
            "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
        });
        std::fs::write(out.join(format!("site_{tag}.json")), serde_json::to_vec(&meta).map_err(error)?).map_err(error)?;
    }

    // Relations: continuations.
    let mut cont: BTreeMap<String, Array> = BTreeMap::new();
    for (a, key) in [(0usize, "clean".to_string())].into_iter().chain(RELATION_ALPHAS.iter().enumerate().map(|(i, a)| (i + 2, format!("alpha_{a}")))) {
        let mut values = vec![-1i64; total * steps];
        for (i, f) in chosen.iter().enumerate() {
            for (s, v) in continuation(i, a).iter().enumerate() {
                values[f * steps + s] = i64::from(*v);
            }
        }
        cont.insert(key, i32s(vec![total, steps], values));
    }
    save(&out.join("relations_continuations.safetensors"), cont)?;

    }

    if wants("downstream") {
    // Downstream: the exact change of every later function's activity at the peak on removal.
    let down_pairs: Vec<(usize, usize)> = chosen.iter().flat_map(|f| (0..j_count).map(move |j| (*f, j))).collect();
    let removed: Vec<Edit> = down_pairs.iter().map(|(f, j)| Edit { sequence: contexts[*f][*j], scale: vec![(*f, 0.0)], rows: vec![position[*f][*j]], ..Edit::default() }).collect();
    let unedited: Vec<Edit> = down_pairs.iter().map(|(f, j)| Edit { sequence: contexts[*f][*j], rows: vec![position[*f][*j]], ..Edit::default() }).collect();
    let (mut d_ids, mut d_delta, mut d_rel) = (vec![-1i64; total * j_count * k], vec![0.0f64; total * j_count * k], vec![0.0f64; total * j_count * k]);
    let mut strongest = Vec::new();
    // In batches, so the activities held are a batch's (rows × functions), not the whole table's.
    for (chunk, (removed, unedited)) in down_pairs.chunks(settings.batch.max(1)).zip(removed.chunks(settings.batch.max(1)).zip(unedited.chunks(settings.batch.max(1)))) {
    let edited_act = library.edited(&pool, removed, &Activity::All, false, false)?.activity.ok_or("activity")?;
    let clean_act = library.edited(&pool, unedited, &Activity::All, false, false)?.activity.ok_or("activity")?;
    for (r, (f, j)) in chunk.iter().enumerate() {
        let delta: Vec<f64> = (0..total).map(|g| if later(*f, g) { edited_act[[r, g]] - clean_act[[r, g]] } else { 0.0 }).collect();
        let rel: Vec<f64> = delta.iter().zip(&peak_of).map(|(d, p)| d / p).collect();
        let best = top_k(rel.iter().map(|v| v.abs()), k);
        for (i, g) in best.iter().enumerate() {
            let at = (f * j_count + j) * k + i;
            d_ids[at] = *g as i64;
            d_delta[at] = delta[*g];
            d_rel[at] = rel[*g];
        }
        if *j == 0 {
            strongest.push(best.first().map_or(0.0, |g| rel[*g].abs()));
        }
    }
    }
    save(
        &out.join("relations_downstream.safetensors"),
        [("ids".into(), i32s(vec![total, j_count, k], d_ids)), ("delta".into(), f32s(vec![total, j_count, k], d_delta)), ("relative".into(), f32s(vec![total, j_count, k], d_rel))].into(),
    )?;
    strongest.sort_by(f64::total_cmp);
    log::info!("downstream in {:.0} s (median strongest relative change {:.3e})", started.elapsed().as_secs_f64(), strongest.get(strongest.len() / 2).copied().unwrap_or(f64::NAN));

    }

    if wants("attribution") {
    // Attribution: exact removal change of log p(X) for the functions with the largest direct effect.
    let proposals = (2 * settings.proposals).min(total);
    let mut att_rng = StdRng::seed_from_u64(settings.seed);
    let (mut a_ctx, mut a_pos, mut a_tok, mut a_ids, mut a_direct, mut a_delta) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let lower = 16.min(t.saturating_sub(2));
    let mut places: Vec<(usize, usize)> = Vec::new();
    for s in 0..settings.attribution_contexts.min(pool.len()) {
        let mut candidates: Vec<usize> = (lower..t - 1).collect();
        candidates.shuffle(&mut att_rng);
        places.extend(candidates.iter().take(settings.positions).map(|p| (s, *p)));
    }
    for chunk in places.chunks(settings.batch.max(1)) {
        // The clean reads at the chunk's rows: X, every function's activity and each head's read.
        let reads: Vec<Edit> = chunk.iter().map(|(s, p)| Edit { sequence: *s, rows: vec![*p], ..Edit::default() }).collect();
        let clean = library.edited(&pool, &reads, &Activity::All, true, false)?;
        let x_tokens = library.argmax_tokens(&clean.last_device)?;
        let (act, heads) = (clean.activity.ok_or("activity")?, clean.heads.ok_or("head reads")?);
        let mut edits = Vec::new();
        let mut asked = Vec::new();
        for (r, (s, p)) in chunk.iter().enumerate() {
            let head_rows: Vec<_> = heads.iter().map(|h| h.row(r)).collect();
            let direct = library.direct_effects(act.row(r), &head_rows, x_tokens[r], clean.last.row(r));
            let best = top_k(direct.iter().map(|v| v.abs()), proposals);
            edits.extend(best.iter().map(|f| Edit { sequence: *s, scale: vec![(*f, 0.0)], rows: vec![*p], ..Edit::default() }));
            asked.push((*s, *p, x_tokens[r], best, direct));
        }
        let next: Vec<Option<usize>> = asked.iter().flat_map(|(_, _, x, best, _)| best.iter().map(move |_| Some(*x))).collect();
        let measured = effects(&library, &model, &pool, &clean_last, &edits, &next, 0, settings.batch)?;
        let mut at = 0;
        for (s, p, x, best, direct) in asked {
            a_ctx.push(s as i64);
            a_pos.push(p as i64);
            a_tok.push(x as i64);
            for f in &best {
                a_ids.push(*f as i64);
                a_direct.push(direct[*f]);
                a_delta.push(measured[at].next);
                at += 1;
            }
        }
    }
    let n = a_ctx.len();
    save(
        &out.join("relations_attribution.safetensors"),
        [
            ("context".into(), i64s(vec![n], a_ctx)),
            ("position".into(), i64s(vec![n], a_pos)),
            ("token".into(), i64s(vec![n], a_tok)),
            ("ids".into(), i32s(vec![n, proposals], a_ids)),
            ("direct".into(), f32s(vec![n, proposals], a_direct)),
            ("delta".into(), f32s(vec![n, proposals], a_delta)),
        ]
        .into(),
    )?;
    log::info!("attribution in {:.0} s", started.elapsed().as_secs_f64());
    }

    if wants("upstream") {
    // Upstream: at each of B's first contexts, at its peak, proposals among the functions writing
    // into what B reads, ranked by their write's exact contribution to B's reads at that row
    // (Σ over B's read maps of ‖R (γ ⊙ w_A(p))‖ / r(p)), used only to choose what to measure; each
    // proposal's exact removal change of B's activity at p, strongest first.
    let mut by_row: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
    for &b in &chosen {
        for j in 0..j_count {
            by_row.entry((contexts[b][j], position[b][j])).or_default().push((b, j));
        }
    }
    // Readers in chunks: the clean reads at their rows in batches, each reader's couplings once,
    // its proposals at each row, and every measurement in batches.
    let readers: Vec<usize> = chosen.clone();
    for group in readers.chunks((4 * settings.batch.max(1) / j_count.max(1)).max(1)) {
        let rows: Vec<(usize, usize)> = group.iter().flat_map(|b| (0..j_count).map(move |j| (*b, j))).map(|(b, j)| (contexts[b][j], position[b][j])).collect();
        let reads: Vec<Edit> = rows.iter().map(|(s, p)| Edit { sequence: *s, rows: vec![*p], ..Edit::default() }).collect();
        let mut act_parts = Vec::new();
        let mut head_parts: Vec<Vec<Array2<f64>>> = Vec::new();
        let mut inverse_parts = Vec::new();
        for c in reads.chunks(settings.batch.max(1)) {
            let e = library.edited(&pool, c, &Activity::All, true, true)?;
            act_parts.push(e.activity.ok_or("activity")?);
            head_parts.push(e.heads.ok_or("head reads")?);
            inverse_parts.push(e.inverses.ok_or("inverses")?);
        }
        let stack = |parts: &[Array2<f64>]| ndarray::concatenate(Axis(0), &parts.iter().map(|a| a.view()).collect::<Vec<_>>()).map_err(error);
        let act = stack(&act_parts)?;
        let inverses = stack(&inverse_parts)?;
        let heads: Vec<Array2<f64>> = (0..head_parts[0].len()).map(|h| stack(&head_parts.iter().map(|p| p[h].clone()).collect::<Vec<_>>())).collect::<Result<_, String>>()?;
        let mut edits = Vec::new();
        let mut asked = Vec::new();
        for (g, &b) in group.iter().enumerate() {
            let couplings = library.couplings(b)?;
            for j in 0..j_count {
                let r = g * j_count + j;
                let head_rows: Vec<_> = heads.iter().map(|h| h.row(r)).collect();
                let score = couplings.score(act.row(r), &head_rows, inverses.row(r));
                let proposed: Vec<usize> = top_k(score.iter().map(|v| if v.is_nan() { -1.0 } else { *v }), pcount).into_iter().filter(|a| score[*a] >= 0.0 && *a != b).collect();
                edits.extend(proposed.iter().map(|a| Edit { sequence: rows[r].0, scale: vec![(*a, 0.0)], rows: vec![rows[r].1], ..Edit::default() }));
                asked.push((b, j, r, proposed));
            }
        }
        let own: Vec<usize> = asked.iter().flat_map(|(b, _, _, proposed)| proposed.iter().map(move |_| *b)).collect();
        let mut measured = Vec::with_capacity(edits.len());
        for (c, o) in edits.chunks(settings.batch.max(1)).zip(own.chunks(settings.batch.max(1))) {
            measured.extend(library.edited(&pool, c, &Activity::Own(o.to_vec()), false, false)?.activity.ok_or("activity")?.column(0).to_vec());
        }
        let mut at = 0;
        for (b, j, r, proposed) in asked {
            let mut changes: Vec<(usize, f64)> = proposed.iter().enumerate().map(|(i, a)| (*a, measured[at + i] - act[[r, b]])).collect();
            at += proposed.len();
            changes.sort_by(|x, y| y.1.abs().total_cmp(&x.1.abs()));
            for (i, (a, d)) in changes.into_iter().enumerate() {
                u_ids[(b * j_count + j) * pcount + i] = a as i64;
                u_delta[(b * j_count + j) * pcount + i] = d;
            }
        }
    }
    save(&out.join("relations_upstream.safetensors"), [("ids".into(), i32s(vec![total, j_count, pcount], u_ids.clone())), ("delta".into(), f32s(vec![total, j_count, pcount], u_delta.clone()))].into())?;
    log::info!("upstream in {:.0} s", started.elapsed().as_secs_f64());

    }

    if wants("edges") {
    // Edges: at each reader's strongest context, path patches of its two strongest measured upstream
    // neighbours and of its weakest measured proposal (a near-zero edge).
    let e_count = 3;
    let mut e_ids = vec![-1i64; total * e_count];
    let mut e_strong = vec![0i64; total * e_count];
    let (mut e_act, mut e_kl, mut e_next) = (vec![f64::NAN; total * e_count], vec![f64::NAN; total * e_count], vec![f64::NAN; total * e_count]);
    let (mut e_up, mut e_down) = (vec![-1i64; total * e_count * TOP], vec![-1i64; total * e_count * TOP]);
    let (mut e_up_dp, mut e_down_dp) = (vec![0.0f64; total * e_count * TOP], vec![0.0f64; total * e_count * TOP]);
    // Per reader its picks; MLP readers' path patches as activation shifts (one pass per context),
    // measured in batches; heads' path patches directly.
    let mut by_context: BTreeMap<usize, Vec<(usize, usize, usize, i64)>> = BTreeMap::new();
    for &b in &chosen {
        let base = b * j_count * pcount;
        let measured: Vec<(usize, f64)> = (0..pcount).filter(|i| u_ids[base + i] >= 0).map(|i| (u_ids[base + i] as usize, u_delta[base + i])).collect();
        if measured.len() < 2 {
            continue;
        }
        let weakest = measured.iter().min_by(|x, y| x.1.abs().total_cmp(&y.1.abs())).map(|x| x.0).unwrap_or(measured[0].0);
        for (e, (a, strong)) in [(measured[0].0, 1), (measured[1].0, 1), (weakest, 0)].into_iter().enumerate() {
            by_context.entry(contexts[b][0]).or_default().push((b, e, a, strong));
        }
    }
    let mut record = |b: usize, e: usize, a: usize, strong: i64, change: f64, m: &gam_mpd::library_readout::TokenEffects| {
        let at = b * e_count + e;
        e_ids[at] = a as i64;
        e_strong[at] = strong;
        e_act[at] = change;
        e_kl[at] = m.kl;
        e_next[at] = m.next;
        for (i, (v, dp, _)) in m.up.iter().enumerate() {
            e_up[at * TOP + i] = *v as i64;
            e_up_dp[at * TOP + i] = *dp;
        }
        for (i, (v, dp, _)) in m.down.iter().enumerate() {
            e_down[at * TOP + i] = *v as i64;
            e_down_dp[at * TOP + i] = *dp;
        }
    };
    let mut pending: Vec<(Edit, Option<usize>, (usize, usize, usize, i64, f64))> = Vec::new();
    let contexts_list: Vec<usize> = by_context.keys().copied().collect();
    for (ci, s) in contexts_list.iter().enumerate() {
        let picks = &by_context[s];
        let mlp: Vec<&(usize, usize, usize, i64)> = picks.iter().filter(|(b, ..)| matches!(functions[*b].kind, Kind::Mlp)).collect();
        let shifts = library.path_shifts(&pool[*s], &mlp.iter().map(|(b, _, a, _)| (*a, *b)).collect::<Vec<_>>())?;
        for ((b, e, a, strong), shift) in mlp.into_iter().zip(shifts) {
            let p = position[*b][0];
            let change = shift[p];
            let next = (p + 1 < t).then(|| pool[*s][p + 1] as usize);
            pending.push((Edit { sequence: *s, add: vec![(*b, shift)], rows: vec![p], ..Edit::default() }, next, (*b, *e, *a, *strong, change)));
        }
        for (b, e, a, strong) in picks.iter().filter(|(b, ..)| matches!(functions[*b].kind, Kind::Head)) {
            let p = position[*b][0];
            let patched = library.path_patched(&pool[*s..=*s], *a, *b, None)?;
            let rows = model.upload(ndarray::stack(Axis(0), &[patched.last.row(p), patched.base.row(p)]).map_err(error)?.view()).map_err(error)?;
            let (edited_row, clean_row) = (model.rows_of(&rows, 0, 1).map_err(error)?, model.rows_of(&rows, 1, 1).map_err(error)?);
            let next = (p + 1 < t).then(|| pool[*s][p + 1] as usize);
            let m = library.token_effects(&edited_row, &clean_row, &[next], TOP)?.remove(0);
            record(*b, *e, *a, *strong, patched.after[p] - patched.before[p], &m);
        }
        if pending.len() >= settings.batch.max(1) || ci + 1 == contexts_list.len() {
            let edits: Vec<Edit> = pending.iter().map(|x| x.0.clone()).collect();
            let next: Vec<Option<usize>> = pending.iter().map(|x| x.1).collect();
            if !edits.is_empty() {
                let measured = effects(&library, &model, &pool, &clean_last, &edits, &next, TOP, settings.batch)?;
                for ((_, _, (b, e, a, strong, change)), m) in pending.drain(..).zip(measured) {
                    record(b, e, a, strong, change, &m);
                }
            }
        }
    }
    save(
        &out.join("relations_edges.safetensors"),
        [
            ("activity".into(), f32s(vec![total, 1, e_count], e_act)),
            ("kl".into(), f32s(vec![total, 1, e_count], e_kl)),
            ("next".into(), f32s(vec![total, 1, e_count], e_next)),
            ("ids".into(), i32s(vec![total, 1, e_count], e_ids)),
            ("strong".into(), i8s(vec![total, 1, e_count], e_strong)),
            ("up_ids".into(), i32s(vec![total, 1, e_count, TOP], e_up)),
            ("down_ids".into(), i32s(vec![total, 1, e_count, TOP], e_down)),
            ("up_dp".into(), f32s(vec![total, 1, e_count, TOP], e_up_dp)),
            ("down_dp".into(), f32s(vec![total, 1, e_count, TOP], e_down_dp)),
        ]
        .into(),
    )?;
    log::info!("edges in {:.0} s", started.elapsed().as_secs_f64());
    }

    let offsets: serde_json::Map<String, Value> = sites.iter().map(|s| (s.0.clone(), json!(s.3))).collect();
    let meta = json!({
        "alphas": RELATION_ALPHAS, "steps": steps, "contexts": j_count, "k": k, "proposals": settings.proposals, "site_offsets": offsets, "total": total,
        "labels": out.display().to_string(), "parts": settings.parts, "seconds": started.elapsed().as_secs_f64(),
        "source_revision": option_env!("GAM_BUILD_GIT_SHA"),
    });
    std::fs::write(out.join("relations.json"), serde_json::to_vec(&meta).map_err(error)?).map_err(error)?;
    log::info!("done in {:.0} s", started.elapsed().as_secs_f64());
    Ok(())
}
