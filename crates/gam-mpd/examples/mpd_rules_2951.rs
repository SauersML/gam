//! Rules shared across a language model's heads (#2951).
//!
//! `mpd_rules_2951 heads EXPORT_DIR HALF SEQUENCES`
//!
//! Every head's attention on `SEQUENCES` sequences of `HALF` uniformly drawn tokens followed by the
//! same `HALF` again: its induction score (the mean weight a position of the repeat puts on the
//! position after its token's first occurrence), its previous-token score (on the first half) and
//! the weight on the first position; and per layer, every pair of heads compared by what their maps
//! compute (invariant to each head's own coordinates): the output-value circuit `W_O W_V` and, per
//! rotary plane, the query-key form `q̃ k̃ᴴ` (`q̃` the plane's two query rows as one complex reader).
//!
//! `mpd_rules_2951 price EXPORT_DIR LIBRARY_DIR OBSERVATIONS SITES`
//!
//! The structured price (`gam_mpd::describe::Structured`, each site's declared charts) of every
//! subcomponent of the comma-separated `SITES` of a library, timed per site, as a fit prices them.
//!
//! `mpd_rules_2951 library EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS STATISTICS LAYERS`
//!
//! Every site of the first `LAYERS` layers of a library, described in decoding order (per layer:
//! values, outputs, keys, queries, the MLP's input then output) in the charts its interfaces declare
//! (`gam_mpd::describe::declared_charts`, what a fit prices by), and again with the rules' charts
//! beside them, each built from what the decoder already holds (the sites decoded before, the
//! model's norm gains): a residual reader in the frames of every residual writer decoded before it,
//! as written and through the reading norm's gain (`w / g`); a query writer in the frames of the
//! layer's key writers; an output reader in the frames of the layer's value writers and an output
//! writer in the copies of their readers (`(g / g_f) ⊙ r`); an MLP output reader in the frames of
//! the layer's MLP input writers; a query reader in the images of the layer's key readers through
//! every earlier head's decoded output-value circuit. A use pays its chart, its frames and its
//! reals; a body is never sent twice. Reported per site and in all: the bits once and per word
//! (the given sets' runs), which rule charts are taken, and the decode check (every site replaced
//! by its decoded descriptions, on sequence `STATISTICS` under the given sets, against the library).
//!
//! `mpd_rules_2951 rules EXPORT_DIR OUT.json OBSERVATIONS STATISTICS`
//!
//! The attention rules priced on the model's own weights, head by head, library paid once
//! (`gam_mpd::describe::Geometry`, each site's declared charts, statistics on the first
//! `STATISTICS` sequences of 512), every layer's sites in decoding order (values, outputs, keys,
//! queries). A head's block is described alone, and as a rule's prediction scaled plus its residual
//! (`gam_mpd::describe::Geometry::describe_predicted`), each prediction built from what the decoder
//! already holds (the blocks decoded before it, the model's norm gains), the cheaper kept and the
//! choice paid in selector bits on every head:
//!
//! * copy: an output block writes what its value block read through the norm gains,
//!   `W_O ≈ λ diag(g / g_f) W_V⁺` (`g` the layer's norm gain, `g_f` the final one);
//! * match: a query block's content rows (its rotary planes from the slowest down to a plane the
//!   code chooses) read, as the current token, what its key rows read as the previous token through
//!   an earlier head's output-value circuit `M = OV diag(g_l)`: with `Z = K̃ diag(g) M` and
//!   `Z Zᵀ = U Λ Uᵀ`, `Q̃ ≈ λ U_k Λ_k⁻¹ U_kᵀ Z diag(1/g)` (all `k` directions: `(Z Zᵀ)⁺ Z diag(1/g)`,
//!   so `diag(g) Q̃ᵀ K̃ diag(g) M` is `λ` times a projector). The source head, the content planes and
//!   `k` are chosen by the energy the prediction explains in the metric, then priced exactly.
//!
//! Reported per head: its bits alone and with its rule, the rule and binding taken; per site and in
//! all; and the decode check (every attention site replaced by its decoded head blocks, all on, on
//! sequence `STATISTICS`), the KL against the native model of both descriptions.

use gam_mpd::import::import_language_model;
use gam_mpd::operator_program::{FamilyInputs, Node, SequenceLayout, SlotValues};
use ndarray::{Array2, Axis, s};

fn uniform(state: &mut u64, n: u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state % n
}

/// The rows of a head's queries or keys rotated to their positions (rotate-half planes `(i, i + d/2)`,
/// angle `m base^{-2i/d}`).
fn rotated(values: &Array2<f64>, positions: &[u32], base: f64) -> Array2<f64> {
    let d = values.ncols();
    let half = d / 2;
    let mut out = values.clone();
    for (mut row, &m) in out.rows_mut().into_iter().zip(positions) {
        for i in 0..half {
            let angle = f64::from(m) * base.powf(-2.0 * i as f64 / d as f64);
            let (sine, cosine) = angle.sin_cos();
            let (x, y) = (row[i], row[i + half]);
            row[i] = cosine * x - sine * y;
            row[i + half] = sine * x + cosine * y;
        }
    }
    out
}

fn cosine(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    (a * b).sum() / ((a * a).sum() * (b * b).sum()).sqrt()
}

fn heads(export: &std::path::Path, half: usize, sequences: usize) -> Result<(), String> {
    let imported = import_language_model(export, 1, 2)?;
    let model = &imported.program;
    let vocab = 50_000u64;
    let mut state = 0x2951_u64;
    let (mut ids, mut sequence, mut position) = (Vec::new(), Vec::new(), Vec::new());
    for q in 0..sequences {
        let first: Vec<u32> = (0..half).map(|_| 1000 + uniform(&mut state, vocab - 1000) as u32).collect();
        for (p, t) in first.iter().chain(&first).enumerate() {
            ids.push(*t);
            sequence.push(q as u32);
            position.push(p as u32);
        }
    }
    let rows = ids.len();
    let family = FamilyInputs { rows, slots: vec![SlotValues::Tokens(ids)], layout: Some(SequenceLayout { sequence, position: position.clone() }) };
    let trace = model.execute(&family, false).map_err(|e| e.to_string())?;
    let attends: Vec<(usize, usize, f64, usize)> = model
        .nodes
        .iter()
        .filter_map(|n| match n {
            Node::Attend { query, key, rotary: Some(r), .. } => Some((*query, *key, f64::from(r.base), r.dims as usize)),
            _ => None,
        })
        .collect();
    let per_layer = attends.len() / 4;
    let length = 2 * half;
    for (index, (query, key, base, dims)) in attends.iter().enumerate() {
        let (q, k) = (rotated(&trace.values[*query], &position, *base), rotated(&trace.values[*key], &position, *base));
        let (mut induction, mut previous, mut first) = (0.0, 0.0, 0.0);
        for sq in 0..sequences {
            let at = sq * length;
            let qs = q.slice(s![at..at + length, ..]);
            let ks = k.slice(s![at..at + length, ..]);
            let scores = qs.dot(&ks.t()) / (*dims as f64).sqrt();
            for m in 1..length {
                let row = scores.row(m);
                let top = row.slice(s![..=m]).iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = (0..=m).map(|n| (row[n] - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                first += weights[0] / total / (length - 1) as f64;
                if m < half {
                    previous += weights[m - 1] / total / (half - 1) as f64;
                }
                if m > half {
                    induction += weights[m - half + 1] / total / (half - 1) as f64;
                }
            }
        }
        let n = sequences as f64;
        eprintln!(
            "layer {} head {}: induction {:.3}, previous token {:.3}, first position {:.3}",
            index / per_layer,
            index % per_layer,
            induction / n,
            previous / n,
            first / n
        );
    }
    // Per head, how its output-value circuit copies a token: on 1024 evenly spaced tokens `t`, the
    // logits `Eᵀ (g_f ⊙ OV (g ⊙ e_t))` of its direct path, and the rank of `t` among them.
    let operator = |name: String| -> Result<Array2<f64>, String> { model.operators.iter().find(|o| o.name == name).map(|o| o.matrix()).ok_or(format!("no operator {name}")) };
    let embedding = operator("wte".to_string())?;
    let tokens = embedding.ncols();
    let sample: Vec<usize> = (0..1024).map(|i| i * tokens / 1024).collect();
    let final_gain = operator("final_norm.gain".to_string())?.diag().to_owned();
    for layer in 0..4 {
        let g = operator(format!("blocks.{layer}.rms1.gain"))?.diag().to_owned();
        let read = &embedding.select(Axis(1), &sample) * &g.view().insert_axis(Axis(1));
        for h in 0..per_layer {
            let circuit = operator(format!("blocks.{layer}.o{h}"))?.dot(&operator(format!("blocks.{layer}.v{h}"))?);
            let written = &circuit.dot(&read) * &final_gain.view().insert_axis(Axis(1));
            let logits = gam_linalg::faer_ndarray::fast_ab(&embedding.t(), &written);
            let ranks: Vec<usize> = sample.iter().enumerate().map(|(j, t)| logits.column(j).iter().filter(|x| **x > logits[[*t, j]]).count()).collect();
            let mut sorted = ranks.clone();
            sorted.sort_unstable();
            eprintln!(
                "layer {layer} head {h}: copying, the token's own logit first in {:.3} of tokens, in the top 10 in {:.3}, median rank {}",
                ranks.iter().filter(|r| **r == 0).count() as f64 / ranks.len() as f64,
                ranks.iter().filter(|r| **r < 10).count() as f64 / ranks.len() as f64,
                sorted[sorted.len() / 2]
            );
        }
    }
    // Natural text's reads per layer: the attention's normed input, its second moment `C`.
    let natural = import_language_model(export, 4, 512)?;
    let text = model.execute(&natural.contract.family, false).map_err(|e| e.to_string())?;
    let all = gam_mpd::masked::sites(model);
    // On natural text, per head: where the current token occurred before (most recently at `j`, not
    // the previous position), the mean weight on `j + 1` (induction) and on `j` (duplicate).
    let SlotValues::Tokens(words) = &natural.contract.family.slots[0] else { return Err("no tokens".to_string()) };
    let layout = natural.contract.family.layout.as_ref().ok_or("no layout")?;
    for (index, (query, key, base, dims)) in attends.iter().enumerate() {
        let (q, k) = (rotated(&text.values[*query], &layout.position, *base), rotated(&text.values[*key], &layout.position, *base));
        let (mut induction, mut duplicate, mut count) = (0.0, 0.0, 0.0);
        for sq in 0..4 {
            let at = sq * 512;
            let scores = q.slice(s![at..at + 512, ..]).dot(&k.slice(s![at..at + 512, ..]).t()) / (*dims as f64).sqrt();
            let mut last = std::collections::HashMap::new();
            for m in 0..512 {
                let token = words[at + m];
                if let Some(&j) = last.get(&token)
                    && j + 1 < m
                {
                    let row = scores.row(m);
                    let top = row.slice(s![..=m]).iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let total: f64 = (0..=m).map(|n| (row[n] - top).exp()).sum();
                    induction += (row[j + 1] - top).exp() / total;
                    duplicate += (row[j] - top).exp() / total;
                    count += 1.0;
                }
                last.insert(token, m);
            }
        }
        eprintln!("text layer {} head {}: induction {:.3}, duplicate {:.3} over {count} repeats", index / per_layer, index % per_layer, induction / count, duplicate / count);
    }
    // What each pair of a layer's heads compute, compared.
    for layer in 0..4 {
        let site = all.iter().find(|s| s.name == format!("blocks.{layer}.q")).ok_or("no query site")?;
        let x = gam_mpd::masked::read_values(&text, site)?;
        let c = x.t().dot(&x) / x.nrows() as f64;
        let e = gam_mpd::dense::eigh(c.view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None).map_err(|e| format!("{e:?}"))?;
        let mut root = e.vectors.clone();
        for (i, l) in e.values.iter().enumerate() {
            root.column_mut(i).mapv_inplace(|v| v * l.max(0.0).sqrt());
        }
        let root = root.dot(&e.vectors.t());
        // The fraction of `b`'s energy (rows, whitened) in the row space of `a` (whitened).
        let within = |a: &Array2<f64>, b: &Array2<f64>| -> Result<f64, String> {
            let (aw, bw) = (a.dot(&root), b.dot(&root));
            let d = gam_mpd::dense::svd(aw.view(), false).map_err(|e| format!("{e:?}"))?;
            let kept = d.singular_values.iter().filter(|s| **s > d.band).count();
            let basis = d.vt.slice(s![..kept, ..]).to_owned();
            let projected = bw.dot(&basis.t());
            Ok((&projected * &projected).sum() / (&bw * &bw).sum())
        };
        let name = |kind: &str, h: usize| format!("blocks.{layer}.{kind}{h}");
        let op = |kind: &str, h: usize| -> Result<Array2<f64>, String> {
            model.operators.iter().find(|o| o.name == name(kind, h)).map(|o| o.matrix()).ok_or(format!("no operator {}", name(kind, h)))
        };
        let mut ov = Vec::new();
        let mut planes = Vec::new();
        for h in 0..per_layer {
            ov.push(op("o", h)?.dot(&op("v", h)?));
            let (wq, wk) = (op("q", h)?, op("k", h)?);
            let half = wq.nrows() / 2;
            // Per plane, the form's real and imaginary parts: Re(q̃ k̃ᴴ) = a cᵀ + b dᵀ,
            // Im = b cᵀ − a dᵀ with q̃ = a + i b, k̃ = c + i d.
            let forms: Vec<(Array2<f64>, Array2<f64>)> = (0..half)
                .map(|i| {
                    let (a, b) = (wq.row(i).insert_axis(Axis(1)).to_owned(), wq.row(i + half).insert_axis(Axis(1)).to_owned());
                    let (c, d) = (wk.row(i).insert_axis(Axis(0)).to_owned(), wk.row(i + half).insert_axis(Axis(0)).to_owned());
                    (a.dot(&c) + b.dot(&d), b.dot(&c) - a.dot(&d))
                })
                .collect();
            planes.push(forms);
        }
        for h in 0..per_layer {
            for g in h + 1..per_layer {
                // The query-key forms' agreement: Σ_i Re⟨G_i, G'_i⟩ / √(Σ‖G_i‖² Σ‖G'_i‖²) (the forms are
                // invariant to each head's coordinates, so their phases are compared too).
                let (mut inner, mut left, mut right) = (0.0, 0.0, 0.0);
                for ((ra, ia), (rb, ib)) in planes[h].iter().zip(&planes[g]) {
                    let re = (ra * rb).sum() + (ia * ib).sum();
                    inner += re;
                    left += (ra * ra).sum() + (ia * ia).sum();
                    right += (rb * rb).sum() + (ib * ib).sum();
                }
                let (q, k, v) = (within(&op("q", h)?, &op("q", g)?)?, within(&op("k", h)?, &op("k", g)?)?, within(&op("v", h)?, &op("v", g)?)?);
                // The share of head g's query (key) map that is head h's up to one complex scale per
                // rotary plane: Σ_i |⟨q̃ʰ_i, q̃ᵍ_i⟩|² / ‖q̃ʰ_i‖² over Σ_i ‖q̃ᵍ_i‖², whitened.
                let planar = |kind: &str| -> Result<f64, String> {
                    let (a, b) = (op(kind, h)?.dot(&root), op(kind, g)?.dot(&root));
                    let half = a.nrows() / 2;
                    let (mut explained, mut total) = (0.0, 0.0);
                    for i in 0..half {
                        let (ar, ai, br, bi) = (a.row(i), a.row(i + half), b.row(i), b.row(i + half));
                        let re = ar.dot(&br) + ai.dot(&bi);
                        let im = ar.dot(&bi) - ai.dot(&br);
                        let na = ar.dot(&ar) + ai.dot(&ai);
                        explained += (re * re + im * im) / na.max(f64::MIN_POSITIVE);
                        total += br.dot(&br) + bi.dot(&bi);
                    }
                    Ok(explained / total)
                };
                eprintln!(
                    "layer {layer} heads {h},{g}: output-value cosine {:.3} (whitened {:.3}), query-key plane agreement {:.3}; reader overlap q {q:.3} k {k:.3} v {v:.3}; per-plane scale q {:.3} k {:.3}",
                    cosine(&ov[h], &ov[g]),
                    cosine(&ov[h].dot(&root), &ov[g].dot(&root)),
                    inner / (left * right).sqrt(),
                    planar("q")?,
                    planar("k")?
                );
            }
        }
    }
    Ok(())
}

fn read_f64(path: &std::path::Path, cols: usize) -> Result<Array2<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().expect("eight bytes"))).collect();
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
}

/// Per subcomponent of each listed site, the share of positions whose set runs it.
fn frequencies(dir: &std::path::Path) -> Result<std::collections::HashMap<String, Vec<f64>>, String> {
    let read = |name: &str| -> Result<Vec<i64>, String> {
        let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
        Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("eight bytes"))).collect())
    };
    let (indptr, indices) = (read("indptr.i64")?, read("indices.i64")?);
    let listed = std::fs::read_to_string(dir.join("sites.txt")).map_err(|e| e.to_string())?;
    let mut sites = Vec::new();
    let mut offset = 0usize;
    for line in listed.lines() {
        let (name, n) = line.split_once(' ').ok_or(format!("sites.txt: {line}"))?;
        let n: usize = n.parse().map_err(|e| format!("sites.txt: {e}"))?;
        sites.push((name.to_string(), offset, n));
        offset += n;
    }
    let mut counts = vec![0.0; offset];
    for i in &indices {
        counts[*i as usize] += 1.0;
    }
    let positions = (indptr.len() - 1) as f64;
    Ok(sites.into_iter().map(|(name, at, n)| (name, counts[at..at + n].iter().map(|c| c / positions).collect())).collect())
}

/// The frames chart of `columns` (`d × C`), one group a column.
fn frames(name: &str, columns: Array2<f64>) -> Result<gam_mpd::describe::Chart, String> {
    let widths = vec![1; columns.ncols()];
    gam_mpd::describe::Chart::frames(name, columns, &widths)
}

/// Per site of `chosen`, sequence `sequence`'s masks (`context × subcomponents`) from the given sets.
fn given_masks(dir: &std::path::Path, chosen: &[gam_mpd::masked::Site], context: usize, sequence: usize) -> Result<Vec<Array2<f64>>, String> {
    let read = |name: &str| -> Result<Vec<i64>, String> {
        let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
        Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().expect("eight bytes"))).collect())
    };
    let (indptr, indices) = (read("indptr.i64")?, read("indices.i64")?);
    let listed = std::fs::read_to_string(dir.join("sites.txt")).map_err(|e| e.to_string())?;
    let mut offsets = vec![0usize];
    let mut names = Vec::new();
    for line in listed.lines() {
        let (name, n) = line.split_once(' ').ok_or(format!("sites.txt: {line}"))?;
        offsets.push(offsets[offsets.len() - 1] + n.parse::<usize>().map_err(|e| e.to_string())?);
        names.push(name.to_string());
    }
    chosen
        .iter()
        .map(|site| {
            let k = names.iter().position(|n| *n == site.name).ok_or(format!("{}: not in the sets", site.name))?;
            let mut m = Array2::<f64>::zeros((context, offsets[k + 1] - offsets[k]));
            for r in 0..context {
                let position = sequence * context + r;
                for &i in &indices[indptr[position] as usize..indptr[position + 1] as usize] {
                    let i = i as usize;
                    if (offsets[k]..offsets[k + 1]).contains(&i) {
                        m[[r, i - offsets[k]]] = 1.0;
                    }
                }
            }
            Ok(m)
        })
        .collect()
}

fn price(export: &std::path::Path, library: &std::path::Path, observations: f64, names: &str) -> Result<(), String> {
    use gam_mpd::blocks::Describe;
    use gam_mpd::describe::{Geometry, Metric, Structured, declared_charts};
    use rayon::prelude::*;
    let imported = import_language_model(export, 2, 512)?;
    let model = &imported.program;
    let all = gam_mpd::masked::sites(model);
    let chosen: Vec<gam_mpd::masked::Site> =
        names.split(',').map(|n| all.iter().find(|s| s.name == n).cloned().ok_or(format!("no site {n}"))).collect::<Result<_, _>>()?;
    let rows: Vec<usize> = (0..2 * 512).collect();
    let gathered = gam_mpd::site_fit::samples(model, &chosen, [imported.contract.family.select(&rows)], 2, 0x5A4E)?;
    let mut geometries = Vec::new();
    for (site, g) in chosen.iter().zip(&gathered) {
        let w = gam_mpd::masked::matrix(model, site)?;
        let measured = gam_mpd::pieces::Site { mean: ndarray::Array1::zeros(w.ncols()), w, second_moment: g.second_moment.clone(), fisher: g.fisher.clone() };
        let (writers, readers) = declared_charts(model, site)?;
        geometries.push(Geometry::new(Metric::of(&measured, observations), writers, readers)?);
    }
    let structured = Structured::new(geometries);
    for (k, site) in chosen.iter().enumerate() {
        let (d_out, d_in) = (gathered[k].fisher.nrows(), gathered[k].second_moment.nrows());
        let (u, v) = (read_f64(&library.join(format!("{}.u.f64", site.name)), d_out)?, read_f64(&library.join(format!("{}.v.f64", site.name)), d_in)?);
        let started = std::time::Instant::now();
        let bits: Vec<f64> = (0..u.nrows())
            .into_par_iter()
            .map(|c| gam_linalg::faer_ndarray::with_nested_parallel(|| structured.bits_at(k, c, u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]))))
            .collect::<Result<_, String>>()?;
        eprintln!("{}: {} subcomponents priced in {:.1}s, mean {:.1} bits", site.name, bits.len(), started.elapsed().as_secs_f64(), bits.iter().sum::<f64>() / bits.len() as f64);
    }
    Ok(())
}

/// The rows of `x` (`C × d`) as frame columns (`d × C`), each scaled entrywise by `scale` when given.
fn columns_of(x: &[&Array2<f64>], scale: Option<&ndarray::Array1<f64>>) -> Result<Option<Array2<f64>>, String> {
    if x.is_empty() {
        return Ok(None);
    }
    let views: Vec<_> = x.iter().map(|m| m.t()).collect();
    let mut out = ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?;
    if let Some(scale) = scale {
        for (i, mut row) in out.rows_mut().into_iter().enumerate() {
            row *= scale[i];
        }
    }
    Ok(Some(out))
}

/// Decoded descriptions' factors stacked: (writers `C × d_out`, readers `C × d_in`).
fn stacked(descriptions: &[gam_mpd::describe::Description]) -> Result<(Array2<f64>, Array2<f64>), String> {
    let us: Vec<_> = descriptions.iter().map(|d| d.u.view()).collect();
    let vs: Vec<_> = descriptions.iter().map(|d| d.v.view()).collect();
    Ok((ndarray::concatenate(Axis(0), &us).map_err(|e| e.to_string())?, ndarray::concatenate(Axis(0), &vs).map_err(|e| e.to_string())?))
}

fn library(export: &std::path::Path, library: &std::path::Path, sets: &std::path::Path, out: &std::path::Path, observations: f64, statistics: usize, layers: usize) -> Result<(), String> {
    use gam_mpd::blocks::{Blocked, Coded, Generic, measure};
    use gam_mpd::describe::{Description, Geometry, Metric, declared_charts};
    use gam_mpd::masked::{Library, Target, site_statistics, sites};
    use rayon::prelude::*;
    use std::collections::{BTreeMap, HashMap};
    const CONTEXT: usize = 512;
    let imported = import_language_model(export, statistics + 1, CONTEXT)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let all = sites(model);
    let order = ["v", "o", "k", "q", "c_fc", "down_proj"];
    let chosen: Vec<gam_mpd::masked::Site> = (0..layers)
        .flat_map(|l| order.iter().map(move |k| format!("blocks.{l}.{k}")))
        .filter(|n| library.join(format!("{n}.v.f64")).exists())
        .map(|n| all.iter().find(|s| s.name == n).cloned().ok_or(format!("no site {n}")))
        .collect::<Result<_, _>>()?;
    let rows: Vec<usize> = (0..statistics * CONTEXT).collect();
    let started = std::time::Instant::now();
    let measured = site_statistics(model, &chosen, [family.select(&rows)], 4, 0x5EED)?;
    eprintln!("statistics of {} sites on {} words, {:.0}s", chosen.len(), rows.len(), started.elapsed().as_secs_f64());
    let frequency = frequencies(sets)?;
    let op = |name: String| -> Result<Array2<f64>, String> { model.operators.iter().find(|o| o.name == name).map(|o| o.matrix()).ok_or(format!("no operator {name}")) };
    let gain = |name: String| -> Result<ndarray::Array1<f64>, String> { Ok(op(format!("{name}.gain"))?.diag().to_owned()) };
    let gf = gain("final_norm".to_string())?;
    // The rules' decoded sites so far, by name: (writers, readers).
    let mut decoded: HashMap<String, (Array2<f64>, Array2<f64>)> = HashMap::new();
    let mut variants: Vec<(Vec<Description>, Vec<Description>)> = Vec::new();
    let mut libraries = Vec::new();
    let mut report = Vec::new();
    let (mut once, mut per_word) = ([0.0; 2], [0.0; 2]);
    for (k, site) in chosen.iter().enumerate() {
        let mut parts = site.name.split('.');
        let layer: usize = parts.nth(1).and_then(|x| x.parse().ok()).ok_or(format!("{}: no layer", site.name))?;
        let kind = site.name.splitn(3, '.').nth(2).unwrap_or("").to_string();
        let (d_out, d_in) = measured[k].w.dim();
        let u = read_f64(&library.join(format!("{}.u.f64", site.name)), d_out)?;
        let v = read_f64(&library.join(format!("{}.v.f64", site.name)), d_in)?;
        let (writers, readers) = declared_charts(model, &site)?;
        let (mut rule_writers, mut rule_readers) = (Vec::new(), Vec::new());
        let got = |name: String| decoded.get(&name);
        // Residual writers decoded before this site: earlier layers' outputs and MLP outputs, and
        // for the MLP input this layer's attention output.
        let residual = |upto: usize, with_output: bool| -> Vec<&Array2<f64>> {
            let mut w: Vec<&Array2<f64>> = (0..upto).flat_map(|l| [format!("blocks.{l}.o"), format!("blocks.{l}.down_proj")]).filter_map(|n| got(n).map(|x| &x.0)).collect();
            if with_output && let Some(x) = got(format!("blocks.{upto}.o")) {
                w.push(&x.0);
            }
            w
        };
        let norm = if kind == "c_fc" { "rms2" } else { "rms1" };
        if matches!(kind.as_str(), "q" | "k" | "v" | "c_fc") {
            let g = gain(format!("blocks.{layer}.{norm}"))?;
            let w = residual(layer, kind == "c_fc");
            if let Some(c) = columns_of(&w, None)? {
                rule_readers.push(frames("earlier residual writers", c)?);
            }
            if let Some(c) = columns_of(&w, Some(&g.mapv(|x| 1.0 / x)))? {
                rule_readers.push(frames("earlier residual writers through the norm's gain", c)?);
            }
        }
        if kind == "q" {
            if let Some((ku, kv)) = got(format!("blocks.{layer}.k")) {
                rule_writers.push(frames("the layer's key writers", ku.t().to_owned())?);
                // The match rule: key readers through every earlier head's decoded output-value circuit.
                let g = gain(format!("blocks.{layer}.rms1"))?;
                let scaled = kv * &g.view().insert_axis(Axis(0));
                let mut images = Vec::new();
                for l in 0..layer {
                    let (Some((ou, ov)), Some((vu, vv))) = (got(format!("blocks.{l}.o")), got(format!("blocks.{l}.v"))) else { continue };
                    let (w_o, w_v) = (ou.t().dot(ov), vu.t().dot(vv));
                    let gl = gain(format!("blocks.{l}.rms1"))?;
                    for h in 0..6 {
                        let circuit = w_o.slice(s![.., h * 128..(h + 1) * 128]).dot(&w_v.slice(s![h * 128..(h + 1) * 128, ..]));
                        let mut image = circuit.t().dot(&scaled.t());
                        for (i, mut row) in image.rows_mut().into_iter().enumerate() {
                            row *= gl[i] / g[i];
                        }
                        images.push(image);
                    }
                }
                if !images.is_empty() {
                    let views: Vec<_> = images.iter().map(|x| x.view()).collect();
                    rule_readers.push(frames("the layer's key readers through earlier heads", ndarray::concatenate(Axis(1), &views).map_err(|e| e.to_string())?)?);
                }
            }
        }
        if kind == "o"
            && let Some((vu, vv)) = got(format!("blocks.{layer}.v"))
        {
            rule_readers.push(frames("the layer's value writers", vu.t().to_owned())?);
            let g = gain(format!("blocks.{layer}.rms1"))?;
            if let Some(c) = columns_of(&[vv], Some(&(&g / &gf)))? {
                rule_writers.push(frames("copies of the layer's value readers", c)?);
            }
        }
        if kind == "down_proj"
            && let Some((cu, _)) = got(format!("blocks.{layer}.c_fc"))
        {
            rule_readers.push(frames("the layer's hidden writers", cu.t().to_owned())?);
        }
        let metric = Metric::of(&measured[k], observations);
        let plain = Geometry::new(metric.clone(), writers.clone(), readers.clone())?;
        let ruled = !rule_writers.is_empty() || !rule_readers.is_empty();
        let rules = if ruled {
            Some(Geometry::new(metric, writers.into_iter().chain(rule_writers).collect(), readers.into_iter().chain(rule_readers).collect())?)
        } else {
            None
        };
        let started = std::time::Instant::now();
        let both: Vec<(Description, Option<Description>)> = (0..u.nrows())
            .into_par_iter()
            .map(|c| {
                let (uc, vc) = (u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]));
                gam_linalg::faer_ndarray::with_nested_parallel(|| Ok((plain.describe(uc, vc)?, rules.as_ref().map(|r| r.describe(uc, vc)).transpose()?)))
            })
            .collect::<Result<_, String>>()?;
        let (a, b): (Vec<Description>, Vec<Option<Description>>) = both.into_iter().unzip();
        let b: Vec<Description> = b.into_iter().zip(&a).map(|(x, y)| x.unwrap_or_else(|| y.clone())).collect();
        let f = frequency.get(&site.name).ok_or(format!("no sets for {}", site.name))?;
        let total = |d: &[Description]| d.iter().map(Description::total).sum::<f64>();
        let active = |d: &[Description]| d.iter().zip(f).map(|(x, n)| x.total() * n).sum::<f64>();
        let mut taken = BTreeMap::<String, (usize, f64)>::new();
        for (d, n) in b.iter().zip(f) {
            for side in [&d.writer.0, &d.reader.0] {
                let e = taken.entry(side.clone()).or_default();
                e.0 += 1;
                e.1 += n;
            }
        }
        eprintln!(
            "{}: {} subcomponents in {:.0}s; once {:.0} bits declared, {:.0} with the rules ({:+.1}%); per word {:.1} declared, {:.1} with the rules ({:+.1}%)",
            site.name,
            a.len(),
            started.elapsed().as_secs_f64(),
            total(&a),
            total(&b),
            100.0 * (total(&b) - total(&a)) / total(&a),
            active(&a),
            active(&b),
            100.0 * (active(&b) - active(&a)) / active(&a).max(f64::MIN_POSITIVE)
        );
        eprintln!("  charts taken (subcomponents, runs per word): {taken:?}");
        once[0] += total(&a);
        once[1] += total(&b);
        per_word[0] += active(&a);
        per_word[1] += active(&b);
        report.push(serde_json::json!({
            "site": site.name,
            "once_bits": {"declared": total(&a), "rules": total(&b)},
            "per_word_bits": {"declared": active(&a), "rules": active(&b)},
            "charts_taken": taken.iter().map(|(name, (n, runs))| (name.clone(), serde_json::json!({"subcomponents": n, "runs_per_word": runs}))).collect::<serde_json::Map<_, _>>(),
        }));
        decoded.insert(site.name.clone(), stacked(&b)?);
        libraries.push(Library { v, u, mean: ndarray::Array1::zeros(d_in) });
        variants.push((a, b));
    }
    eprintln!(
        "all {} sites: once {:.0} bits declared, {:.0} with the rules ({:+.1}%); per word {:.1} declared, {:.1} with the rules ({:+.1}%)",
        chosen.len(),
        once[0],
        once[1],
        100.0 * (once[1] - once[0]) / once[0],
        per_word[0],
        per_word[1],
        100.0 * (per_word[1] - per_word[0]) / per_word[0]
    );
    // The decode check: every site replaced by its decoded descriptions.
    let check: Vec<usize> = (statistics * CONTEXT..(statistics + 1) * CONTEXT).collect();
    let inputs = family.select(&check);
    let logits = model.execute(&inputs, false).map_err(|e| e.to_string())?.values[model.output].clone();
    let masks = given_masks(sets, &chosen, CONTEXT, statistics)?;
    let given = Blocked::rank_one(libraries, vec![masks]);
    let generic = Generic::new(&measured, observations);
    let coded = Coded { model, sites: chosen.clone(), batches: vec![(inputs, Target::every_row(logits))], observations, samples: 4, describe: &generic };
    let (base, _) = measure(&coded, &given)?;
    let mut checks = Vec::new();
    for (label, pick) in [("declared", 0usize), ("rules", 1)] {
        let mut replaced = given.clone();
        let mut priced = 0.0;
        for (k, (a, b)) in variants.iter().enumerate() {
            let descriptions = if pick == 0 { a } else { b };
            let on: Vec<f64> = given.masks[0][k].sum_axis(Axis(0)).to_vec();
            priced += descriptions.iter().zip(&on).map(|(d, n)| d.kl_bits * n).sum::<f64>();
            let (u, v) = stacked(descriptions)?;
            replaced.libraries[k] = std::sync::Arc::new(Library { u, v, mean: given.libraries[k].mean.clone() });
            replaced.ranks[k] = descriptions.iter().map(|d| d.u.nrows()).collect();
        }
        let (bits, _) = measure(&coded, &replaced)?;
        let words = bits.rows.max(1.0);
        eprintln!(
            "decode check, {label}: measured error {:.1} bits/word (KL {:.5} nats/word against the library's {:.5}), priced {:.1}",
            (bits.kl - base.kl) / words,
            bits.kl_nats / words,
            base.kl_nats / words,
            priced / words
        );
        checks.push(serde_json::json!({"description": label, "measured_error_bits_per_word": (bits.kl - base.kl) / words, "priced_error_bits_per_word": priced / words, "kl_per_word": bits.kl_nats / words, "library_kl_per_word": base.kl_nats / words}));
    }
    let report = serde_json::json!({"layers": layers, "observations": observations, "statistics_sequences": statistics, "sites": report,
        "once_bits": {"declared": once[0], "rules": once[1]}, "per_word_bits": {"declared": per_word[0], "rules": per_word[1]}, "decode_check": checks});
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// Selector rows on a `d`-wide side: row `i` the unit vector at column `rows[i]`.
fn selector(rows: &[usize], d: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((rows.len(), d));
    for (i, j) in rows.iter().enumerate() {
        x[[i, *j]] = 1.0;
    }
    x
}

/// The pseudo-inverse of `x` over its singular values beyond the decomposition's band.
fn pseudo_inverse(x: &Array2<f64>) -> Result<Array2<f64>, String> {
    let d = gam_mpd::dense::svd(x.view(), false).map_err(|e| format!("{e:?}"))?;
    let kept: Vec<usize> = (0..d.singular_values.len()).filter(|i| d.singular_values[*i] > d.band).collect();
    let inverse = ndarray::Array1::from_iter(kept.iter().map(|i| 1.0 / d.singular_values[*i]));
    let scaled = &d.u.select(Axis(1), &kept).t() * &inverse.insert_axis(Axis(1));
    Ok(d.vt.select(Axis(0), &kept).t().dot(&scaled))
}

/// A head's rule as tried: its name and binding, and the prediction's factors on the site.
struct Candidate {
    binding: String,
    pu: Array2<f64>,
    pv: Array2<f64>,
}

fn rules(export: &std::path::Path, out: &std::path::Path, observations: f64, statistics: usize) -> Result<(), String> {
    use gam_mpd::blocks::{Blocked, Coded, Generic, measure};
    use gam_mpd::describe::{Geometry, Metric, declared_charts};
    use gam_mpd::masked::{Library, Target, site_statistics, sites};
    use rayon::prelude::*;
    use std::collections::HashMap;
    const CONTEXT: usize = 512;
    let imported = import_language_model(export, statistics + 1, CONTEXT)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let op = |name: &str| -> Result<Array2<f64>, String> { model.operators.iter().find(|o| o.name == name).map(|o| o.matrix()).ok_or(format!("no operator {name}")) };
    let gain = |name: &str| -> Result<ndarray::Array1<f64>, String> { Ok(op(&format!("{name}.gain"))?.diag().to_owned()) };
    let exists = |name: String| model.operators.iter().any(|o| o.name == name);
    let layers = (0..).take_while(|l| exists(format!("blocks.{l}.q0"))).count();
    let heads = (0..).take_while(|h| exists(format!("blocks.0.q{h}"))).count();
    let width = op("blocks.0.q0")?.nrows();
    let half = width / 2;
    let all = sites(model);
    let order = ["v", "o", "k", "q"];
    let chosen: Vec<gam_mpd::masked::Site> = (0..layers)
        .flat_map(|l| order.iter().map(move |k| format!("blocks.{l}.{k}")))
        .map(|n| all.iter().find(|s| s.name == n).cloned().ok_or(format!("no site {n}")))
        .collect::<Result<_, _>>()?;
    let rows: Vec<usize> = (0..statistics * CONTEXT).collect();
    let started = std::time::Instant::now();
    let measured = site_statistics(model, &chosen, [family.select(&rows)], 4, 0x5EED)?;
    eprintln!("statistics of {} sites on {} words, {:.0}s", chosen.len(), rows.len(), started.elapsed().as_secs_f64());
    let gf = gain("final_norm")?;
    // Per description (alone, with the rules), every head block decoded so far, in its operator's
    // shape (a query, key or value head `width × d`, an output head `d × width`), and per site the
    // decoded blocks' factors.
    let mut decoded: [HashMap<String, Array2<f64>>; 2] = [HashMap::new(), HashMap::new()];
    let mut libraries: [Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>, Vec<usize>)>; 2] = [Vec::new(), Vec::new()];
    let mut report = Vec::new();
    let mut totals = [0.0; 2];
    for (k, site) in chosen.iter().enumerate() {
        let layer: usize = site.name.split('.').nth(1).and_then(|x| x.parse().ok()).ok_or(format!("{}: no layer", site.name))?;
        let kind = site.name.rsplit('.').next().unwrap_or("").to_string();
        let (writers, readers) = declared_charts(model, site)?;
        let geometry = Geometry::new(Metric::of(&measured[k], observations), writers, readers)?;
        let (fisher, moment) = (&geometry.metric.fisher, &geometry.metric.moment);
        let g = gain(&format!("blocks.{layer}.rms1"))?;
        let d = heads * width;
        let rules_decoded = &decoded[1];
        // The selector naming each head's choice: alone, or one rule binding.
        let options = match kind.as_str() {
            "o" => 2.0,
            "q" => 1.0 + (heads * layer * half * width) as f64,
            _ => 1.0,
        };
        let selector_bits = f64::log2(options);
        let started = std::time::Instant::now();
        let per_head: Vec<(gam_mpd::describe::Description, Option<(String, gam_mpd::describe::Predicted)>)> = (0..heads)
            .into_par_iter()
            .map(|h| {
                gam_linalg::faer_ndarray::with_nested_parallel(|| -> Result<_, String> {
                    let own: Vec<usize> = (h * width..(h + 1) * width).collect();
                    let w = op(&format!("blocks.{layer}.{kind}{h}"))?;
                    let (u, v) = if kind == "o" { (w.t().to_owned(), selector(&own, d)) } else { (selector(&own, d), w.clone()) };
                    let alone = geometry.describe(u.view(), v.view())?;
                    let mut candidate: Option<Candidate> = None;
                    if kind == "o"
                        && let Some(value) = rules_decoded.get(&format!("blocks.{layer}.v{h}"))
                    {
                        // Copy: W_O ≈ λ diag(g / g_f) V⁺.
                        let mut p = pseudo_inverse(value)?;
                        for (i, mut row) in p.rows_mut().into_iter().enumerate() {
                            row *= g[i] / gf[i];
                        }
                        candidate = Some(Candidate { binding: "copy".to_string(), pu: p.t().to_owned(), pv: selector(&own, d) });
                    }
                    if kind == "q"
                        && let Some(key) = rules_decoded.get(&format!("blocks.{layer}.k{h}"))
                    {
                        // Match through every earlier head, on the content planes from `first` up: the
                        // source and `first` that explain the most of the block in the metric.
                        // ⟨A, B⟩ = tr(F A C Bᵀ) for blocks on this head's rows.
                        let rows_f = fisher.select(Axis(0), &own).select(Axis(1), &own);
                        let wc = w.dot(moment);
                        let mut best: Option<(f64, String, Array2<f64>, Vec<usize>)> = None;
                        for l in 0..layer {
                            let gl = gain(&format!("blocks.{l}.rms1"))?;
                            for source in 0..heads {
                                let (Some(o), Some(vs)) = (rules_decoded.get(&format!("blocks.{l}.o{source}")), rules_decoded.get(&format!("blocks.{l}.v{source}"))) else { continue };
                                let m = &o.dot(vs) * &gl.view().insert_axis(Axis(0));
                                let z = (key * &g.view().insert_axis(Axis(0))).dot(&m);
                                let zd = &z / &g.view().insert_axis(Axis(0));
                                let (zz, y, x) = (z.dot(&z.t()), zd.dot(moment).dot(&zd.t()), wc.dot(&zd.t()));
                                for first in 0..half {
                                    let planes: Vec<usize> = (first..half).chain(first + half..width).collect();
                                    // Z Zᵀ = U Λ Uᵀ on the planes; the prediction keeps its k leading
                                    // directions, P_k = U_k Λ_k⁻¹ U_kᵀ Z D⁻¹ (k = all is (Z Zᵀ)⁺ Z D⁻¹), and
                                    // ⟨W, P_k⟩, ⟨P_k, P_k⟩ accumulate over k through the precomputed forms.
                                    let e = gam_mpd::dense::eigh(zz.select(Axis(0), &planes).select(Axis(1), &planes).view(), gam_linalg::roundoff::SymmetricAssembly::Mirrored, None)
                                        .map_err(|e| format!("{e:?}"))?;
                                    let mut order: Vec<usize> = (0..planes.len()).filter(|i| e.values[*i] > e.band).collect();
                                    order.sort_by(|a, b| e.values[*b].total_cmp(&e.values[*a]));
                                    let directions = e.vectors.select(Axis(1), &order);
                                    let scales = ndarray::Array1::from_iter(order.iter().map(|i| 1.0 / e.values[*i]));
                                    let (fp, fpp) = (rows_f.select(Axis(1), &planes), rows_f.select(Axis(0), &planes).select(Axis(1), &planes));
                                    let crosses = &(&x.select(Axis(1), &planes).dot(&directions) * &fp.dot(&directions)).sum_axis(Axis(0)) * &scales;
                                    let (uf, uy) = (directions.t().dot(&fpp).dot(&directions), directions.t().dot(&y.select(Axis(0), &planes).select(Axis(1), &planes)).dot(&directions));
                                    let gram = &(&uf * &uy) * &scales.view().insert_axis(Axis(1)) * &scales.view().insert_axis(Axis(0));
                                    let (mut inner, mut norm) = (0.0, 0.0);
                                    for k in 0..order.len() {
                                        inner += crosses[k];
                                        norm += gram[[k, k]] + 2.0 * gram.slice(s![k, ..k]).sum();
                                        let explained = if norm > 0.0 { inner * inner / norm } else { 0.0 };
                                        if best.as_ref().is_none_or(|b| explained > b.0) {
                                            let kept = directions.slice(s![.., ..=k]);
                                            let prediction = (&kept * &scales.slice(s![..=k]).insert_axis(Axis(0))).dot(&kept.t()).dot(&zd.select(Axis(0), &planes));
                                            best = Some((
                                                explained,
                                                format!("match through layer {l} head {source} from plane {first}, {} directions", k + 1),
                                                prediction,
                                                planes.iter().map(|p| h * width + p).collect(),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        if let Some((_, binding, prediction, at)) = best {
                            candidate = Some(Candidate { binding, pu: selector(&at, d), pv: prediction });
                        }
                    }
                    let predicted = match candidate {
                        Some(c) => Some((c.binding, geometry.describe_predicted(u.view(), v.view(), c.pu.view(), c.pv.view())?)),
                        None => None,
                    };
                    Ok((alone, predicted))
                })
            })
            .collect::<Result<_, String>>()?;
        let mut site_report = Vec::new();
        let mut site_totals = [0.0; 2];
        let mut factors: [(Vec<Array2<f64>>, Vec<Array2<f64>>, Vec<usize>); 2] = [(Vec::new(), Vec::new(), Vec::new()), (Vec::new(), Vec::new(), Vec::new())];
        for (h, (alone, predicted)) in per_head.into_iter().enumerate() {
            let taken = predicted.as_ref().filter(|(_, p)| p.total() < alone.total());
            let with_rules = selector_bits + taken.map_or(alone.total(), |(_, p)| p.total());
            site_totals[0] += alone.total();
            site_totals[1] += with_rules;
            eprintln!(
                "  {} head {h}: alone {:.0} bits; {}",
                site.name,
                alone.total(),
                match &predicted {
                    Some((binding, p)) => format!("{binding}, scale {:.4}: {:.0} bits with its residual ({}), {:.0} with the selector", p.scale, p.total(), if taken.is_some() { "taken" } else { "not taken" }, with_rules),
                    None => format!("no rule applies, {with_rules:.0} with the selector"),
                }
            );
            site_report.push(serde_json::json!({
                "head": h,
                "alone_bits": alone.total(),
                "rule": predicted.as_ref().map(|(b, p)| serde_json::json!({"binding": b, "scale": p.scale, "bits": p.total(), "residual_kl_bits": p.residual.kl_bits, "taken": taken.is_some()})),
                "with_rules_bits": with_rules,
            }));
            let pieces = [(alone.u.clone(), alone.v.clone()), match taken {
                Some((_, p)) => p.decoded()?,
                None => (alone.u.clone(), alone.v.clone()),
            }];
            for (variant, (du, dv)) in pieces.into_iter().enumerate() {
                let block = du.t().dot(&dv);
                let shaped = if kind == "o" { block.slice(s![.., h * width..(h + 1) * width]).to_owned() } else { block.slice(s![h * width..(h + 1) * width, ..]).to_owned() };
                decoded[variant].insert(format!("blocks.{layer}.{kind}{h}"), shaped);
                factors[variant].2.push(du.nrows());
                factors[variant].0.push(du);
                factors[variant].1.push(dv);
            }
        }
        eprintln!(
            "{}: {:.0} bits alone, {:.0} with the rules ({:+.2}%), {:.0}s",
            site.name,
            site_totals[0],
            site_totals[1],
            100.0 * (site_totals[1] - site_totals[0]) / site_totals[0],
            started.elapsed().as_secs_f64()
        );
        totals[0] += site_totals[0];
        totals[1] += site_totals[1];
        report.push(serde_json::json!({"site": site.name, "alone_bits": site_totals[0], "with_rules_bits": site_totals[1], "heads": site_report}));
        let [a, b] = factors;
        libraries[0].push(a);
        libraries[1].push(b);
    }
    eprintln!("all attention sites: {:.0} bits alone, {:.0} with the rules ({:+.2}%)", totals[0], totals[1], 100.0 * (totals[1] - totals[0]) / totals[0]);
    // The decode check: every attention site replaced by its decoded head blocks, all on.
    let check: Vec<usize> = (statistics * CONTEXT..(statistics + 1) * CONTEXT).collect();
    let inputs = family.select(&check);
    let logits = model.execute(&inputs, false).map_err(|e| e.to_string())?.values[model.output].clone();
    let generic = Generic::new(&measured, observations);
    let coded = Coded { model, sites: chosen.clone(), batches: vec![(inputs.clone(), Target::every_row(logits))], observations, samples: 4, describe: &generic };
    let mut checks = Vec::new();
    for (variant, label) in ["alone", "with the rules"].iter().enumerate() {
        let mut built = Vec::new();
        let mut ranks = Vec::new();
        let mut masks = Vec::new();
        for (us, vs, r) in &libraries[variant] {
            let u = ndarray::concatenate(Axis(0), &us.iter().map(|x| x.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
            let v = ndarray::concatenate(Axis(0), &vs.iter().map(|x| x.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?;
            masks.push(Array2::ones((inputs.rows, r.len())));
            built.push(Library { mean: ndarray::Array1::zeros(v.ncols()), v, u });
            ranks.push(r.clone());
        }
        let (bits, _) = measure(&coded, &Blocked::new(built, ranks, vec![masks]))?;
        let words = bits.rows.max(1.0);
        eprintln!("decode check, {label}: KL {:.6} nats/word against the native model ({:.1} bits/word at n)", bits.kl_nats / words, bits.kl / words);
        checks.push(serde_json::json!({"description": label, "kl_per_word": bits.kl_nats / words, "kl_bits_per_word": bits.kl / words}));
    }
    let report = serde_json::json!({"observations": observations, "statistics_sequences": statistics, "sites": report, "alone_bits": totals[0], "with_rules_bits": totals[1], "decode_check": checks});
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_rules_2951 heads EXPORT_DIR HALF SEQUENCES | price EXPORT_DIR LIBRARY_DIR OBSERVATIONS SITES | library EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS STATISTICS LAYERS | rules EXPORT_DIR OUT.json OBSERVATIONS STATISTICS";
    match args.get(1).map(String::as_str) {
        Some("heads") if args.len() == 5 => heads(
            std::path::Path::new(&args[2]),
            args[3].parse().map_err(|e| format!("HALF: {e}"))?,
            args[4].parse().map_err(|e| format!("SEQUENCES: {e}"))?,
        ),
        Some("price") if args.len() == 6 => price(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
            args[4].parse().map_err(|e| format!("OBSERVATIONS: {e}"))?,
            &args[5],
        ),
        Some("library") if args.len() == 9 => library(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[4]),
            std::path::Path::new(&args[5]),
            args[6].parse().map_err(|e| format!("OBSERVATIONS: {e}"))?,
            args[7].parse().map_err(|e| format!("STATISTICS: {e}"))?,
            args[8].parse().map_err(|e| format!("LAYERS: {e}"))?,
        ),
        Some("rules") if args.len() == 6 => rules(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
            args[4].parse().map_err(|e| format!("OBSERVATIONS: {e}"))?,
            args[5].parse().map_err(|e| format!("STATISTICS: {e}"))?,
        ),
        _ => Err(usage.to_string()),
    }
}
