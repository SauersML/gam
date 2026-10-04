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
//! `mpd_rules_2951 pairs EXPORT_DIR LIBRARY_DIR SETS_DIR LAYER`
//!
//! The induction rules on a library's subcomponents of layer `LAYER` (`LIBRARY_DIR/{site}.{u,v}.f64`,
//! weighted by how often `SETS_DIR`'s sets run them): copying, an output subcomponent writing what
//! the value subcomponent it reads read (`w ∝ (g / g_f) ⊙ r`, `g` the layer's norm gain and `g_f` the
//! final one); and matching, a query subcomponent reading what a key subcomponent reads through an
//! earlier head's output-value circuit (`r_q ∝ g⁻¹ ⊙ g_l ⊙ OVᵀ (g ⊙ r_k)`), per earlier head.
//!
//! `mpd_rules_2951 price EXPORT_DIR LIBRARY_DIR OBSERVATIONS SITES`
//!
//! The structured price (`gam_mpd::describe::Structured`, each site's declared charts) of every
//! subcomponent of the comma-separated `SITES` of a library, timed per site, as a fit prices them.
//!
//! `mpd_rules_2951 induction EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS STATISTICS LAYER`
//!
//! The two induction rules priced on layer `LAYER`'s attention library, library paid once
//! (`gam_mpd::describe::Geometry`, statistics on the first `STATISTICS` sequences of 512): every
//! subcomponent described in the per-site charts (each side's heads and rotary planes; residual
//! readers in the frames of the earlier layers' output writers; query writers in the frames of the
//! layer's key writers; output readers in the frames of the layer's value writers), and again with
//! the rules' charts beside them: output writers in the copies of the layer's value readers
//! (`(g / g_f) ⊙ r`), query readers in the images of the layer's key readers through every earlier
//! head's output-value circuit as the earlier layers' library decodes it. A rule's body is built
//! from what the decoder already holds (the model's norm gains, the earlier layers' decoded
//! library), so a use pays its chart, its frame and its reals. Reported: the bits once and per word
//! (the given sets' runs), which subcomponents take a rule by head, and the decode check (the
//! layer's four sites replaced by their decoded descriptions, the rest native, on sequence
//! `STATISTICS` under the given sets, against the library itself).

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

fn unit(x: ndarray::ArrayView1<'_, f64>) -> ndarray::Array1<f64> {
    let n = x.dot(&x).sqrt();
    if n > 0.0 { &x / n } else { x.to_owned() }
}

fn pairs(export: &std::path::Path, library: &std::path::Path, sets: &std::path::Path, layer: usize) -> Result<(), String> {
    let imported = import_language_model(export, 1, 2)?;
    let model = &imported.program;
    let op = |name: String| -> Result<Array2<f64>, String> { model.operators.iter().find(|o| o.name == name).map(|o| o.matrix()).ok_or(format!("no operator {name}")) };
    let gain = |name: &str| -> Result<ndarray::Array1<f64>, String> { Ok(op(format!("{name}.gain"))?.diag().to_owned()) };
    let frequency = frequencies(sets)?;
    let load = |site: &str| -> Result<(Array2<f64>, Array2<f64>, Vec<f64>), String> {
        let f = frequency.get(site).ok_or(format!("no sets for {site}"))?.clone();
        Ok((read_f64(&library.join(format!("{site}.u.f64")), 768)?, read_f64(&library.join(format!("{site}.v.f64")), 768)?, f))
    };
    let (g, gf) = (gain(&format!("blocks.{layer}.rms1"))?, gain("final_norm")?);
    // Copying: each output subcomponent against the value subcomponent its path is strongest through.
    let (vu, vv, vf) = load(&format!("blocks.{layer}.v"))?;
    let (ou, ov, of) = load(&format!("blocks.{layer}.o"))?;
    let copy = &g / &gf;
    let alive_v: Vec<usize> = (0..vf.len()).filter(|c| vf[*c] > 0.0).collect();
    let (mut weight, mut partner, mut identity, mut chance, mut best) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for c in (0..of.len()).filter(|c| of[*c] > 0.0) {
        let (reader, writer) = (ov.row(c), ou.row(c));
        let strength = |p: &usize| (reader.dot(&vu.row(*p)) * vv.row(*p).dot(&vv.row(*p)).sqrt()).abs();
        let Some(&p) = alive_v.iter().max_by(|a, b| strength(a).total_cmp(&strength(b))) else { continue };
        let w = unit(writer);
        let image = |r: ndarray::ArrayView1<'_, f64>| unit((&r * &copy).view());
        partner += of[c] * w.dot(&image(vv.row(p))).abs();
        identity += of[c] * w.dot(&unit(vv.row(p))).abs();
        chance += of[c] * w.dot(&image(vv.row(alive_v[(c * 7919) % alive_v.len()]))).abs();
        best += of[c] * alive_v.iter().map(|q| w.dot(&image(vv.row(*q))).abs()).fold(0.0, f64::max);
        weight += of[c];
    }
    eprintln!(
        "copying, layer {layer}: |cos(writer, (g/g_f) ⊙ path partner's reader)| {:.3} (identity map {:.3}, an unrelated value reader {:.3}, best value reader {:.3})",
        partner / weight,
        identity / weight,
        chance / weight,
        best / weight
    );
    // Per output subcomponent, by the head its reader reads most.
    let mut by_head = vec![(0.0, 0.0); 6];
    for c in (0..of.len()).filter(|c| of[*c] > 0.0) {
        let reader = ov.row(c);
        let h = (0..6).max_by(|a, b| {
            let e = |h: &usize| reader.slice(s![h * 128..(h + 1) * 128]).dot(&reader.slice(s![h * 128..(h + 1) * 128]));
            e(a).total_cmp(&e(b))
        }).unwrap_or(0);
        let strength = |p: &usize| (reader.dot(&vu.row(*p)) * vv.row(*p).dot(&vv.row(*p)).sqrt()).abs();
        let Some(&p) = alive_v.iter().max_by(|a, b| strength(a).total_cmp(&strength(b))) else { continue };
        by_head[h].0 += of[c] * unit(ou.row(c)).dot(&unit((&vv.row(p) * &copy).view())).abs();
        by_head[h].1 += of[c];
    }
    for (h, (x, w)) in by_head.iter().enumerate() {
        eprintln!("  output subcomponents reading head {h} most: {:.3} on per position, copy cosine {:.3}", w, x / w.max(f64::MIN_POSITIVE));
    }
    // Matching: each query subcomponent against the best key subcomponent through each earlier head.
    let (_, qv, qf) = load(&format!("blocks.{layer}.q"))?;
    let (_, kv, kf) = load(&format!("blocks.{layer}.k"))?;
    let alive_k: Vec<usize> = (0..kf.len()).filter(|c| kf[*c] > 0.0).collect();
    let mut through: Vec<(String, Array2<f64>)> = vec![("the same reader".to_string(), Array2::eye(768))];
    for l in 0..layer {
        let gl = gain(&format!("blocks.{l}.rms1"))?;
        for h in 0..6 {
            let ov_map = op(format!("blocks.{l}.o{h}"))?.dot(&op(format!("blocks.{l}.v{h}"))?);
            // r_q = g⁻¹ ⊙ g_l ⊙ OVᵀ (g ⊙ r_k), as a matrix on r_k.
            let mut m = ov_map.t().to_owned();
            for (i, mut row) in m.rows_mut().into_iter().enumerate() {
                row *= gl[i] / g[i];
            }
            for (j, mut column) in m.columns_mut().into_iter().enumerate() {
                column *= g[j];
            }
            through.push((format!("layer {l} head {h}"), m));
        }
    }
    for (name, m) in &through {
        let images: Vec<ndarray::Array1<f64>> = alive_k.iter().map(|c| unit(m.dot(&kv.row(*c)).view())).collect();
        let (mut total, mut weight) = (0.0, 0.0);
        for c in (0..qf.len()).filter(|c| qf[*c] > 0.0) {
            let r = unit(qv.row(c));
            total += qf[c] * images.iter().map(|x| r.dot(x).abs()).fold(0.0, f64::max);
            weight += qf[c];
        }
        eprintln!("matching through {name}: mean best |cos(query reader, image of a key reader)| {:.3}", total / weight);
    }
    Ok(())
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

/// The head (of six 128-wide groups) carrying most of `x`'s energy.
fn head_of(x: ndarray::ArrayView1<'_, f64>) -> usize {
    let energy = |h: usize| x.slice(s![h * 128..(h + 1) * 128]).dot(&x.slice(s![h * 128..(h + 1) * 128]));
    (0..6).max_by(|a, b| energy(*a).total_cmp(&energy(*b))).unwrap_or(0)
}

fn induction(export: &std::path::Path, library: &std::path::Path, sets: &std::path::Path, out: &std::path::Path, observations: f64, statistics: usize, layer: usize) -> Result<(), String> {
    use gam_mpd::blocks::{Blocked, Coded, Generic, measure};
    use gam_mpd::describe::{Chart, Description, Geometry, Metric};
    use gam_mpd::masked::{Library, Target, site_statistics, sites};
    use rayon::prelude::*;
    const CONTEXT: usize = 512;
    let imported = import_language_model(export, statistics + 1, CONTEXT)?;
    let model = &imported.program;
    let family = &imported.contract.family;
    let all = sites(model);
    let site = |name: String| all.iter().find(|s| s.name == name).cloned().ok_or(format!("no site {name}"));
    // Decoding order: values, outputs, keys, queries.
    let kinds = ["v", "o", "k", "q"];
    let chosen: Vec<gam_mpd::masked::Site> = kinds.iter().map(|k| site(format!("blocks.{layer}.{k}"))).collect::<Result<_, _>>()?;
    let rows: Vec<usize> = (0..statistics * CONTEXT).collect();
    let started = std::time::Instant::now();
    let measured = site_statistics(model, &chosen, [family.select(&rows)], 4, 0x5EED)?;
    eprintln!("statistics of {} sites on {} words, {:.0}s", chosen.len(), rows.len(), started.elapsed().as_secs_f64());
    let load = |name: &str| -> Result<(Array2<f64>, Array2<f64>), String> {
        Ok((read_f64(&library.join(format!("{name}.u.f64")), 768)?, read_f64(&library.join(format!("{name}.v.f64")), 768)?))
    };
    let frequency = frequencies(sets)?;
    let op = |name: String| -> Result<Array2<f64>, String> { model.operators.iter().find(|o| o.name == name).map(|o| o.matrix()).ok_or(format!("no operator {name}")) };
    let gain = |name: String| -> Result<ndarray::Array1<f64>, String> { Ok(op(format!("{name}.gain"))?.diag().to_owned()) };
    let (g, gf) = (gain(format!("blocks.{layer}.rms1"))?, gain("final_norm".to_string())?);
    let libraries: Vec<(Array2<f64>, Array2<f64>)> = kinds.iter().map(|k| load(&format!("blocks.{layer}.{k}"))).collect::<Result<_, _>>()?;
    let (values, keys) = (&libraries[0], &libraries[2]);
    // The per-site charts.
    let heads = Chart::coordinates("heads", 768, &(0..6).map(|h| (h * 128..(h + 1) * 128).collect()).collect::<Vec<_>>())?;
    let planes: Vec<Vec<usize>> = (0..6).flat_map(|h| (0..64).map(move |i| vec![h * 128 + i, h * 128 + i + 64])).collect();
    let rotary = Chart::coordinates("rotary planes", 768, &planes)?;
    let mut earlier_writers = Vec::new();
    let mut through = Vec::new();
    let scaled_keys = &keys.1 * &g.view().insert_axis(Axis(0));
    for l in 0..layer {
        let (ou, ov) = load(&format!("blocks.{l}.o"))?;
        let (vu, vv) = load(&format!("blocks.{l}.v"))?;
        earlier_writers.push(ou.t().to_owned());
        // Each earlier head's output-value circuit as the library decodes it (every subcomponent on),
        // and the image of every key reader through it: `g⁻¹ ⊙ g_l ⊙ OVᵀ (g ⊙ r_k)`.
        let (w_o, w_v) = (ou.t().dot(&ov), vu.t().dot(&vv));
        let gl = gain(format!("blocks.{l}.rms1"))?;
        for h in 0..6 {
            let ov_map = w_o.slice(s![.., h * 128..(h + 1) * 128]).dot(&w_v.slice(s![h * 128..(h + 1) * 128, ..]));
            let mut image = ov_map.t().dot(&scaled_keys.t());
            for (i, mut row) in image.rows_mut().into_iter().enumerate() {
                row *= gl[i] / g[i];
            }
            through.push(image);
        }
    }
    let earlier = frames("earlier output writers", ndarray::concatenate(Axis(1), &earlier_writers.iter().map(|x| x.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?)?;
    let copies = {
        let mut c = values.1.t().to_owned();
        for (i, mut row) in c.rows_mut().into_iter().enumerate() {
            row *= g[i] / gf[i];
        }
        frames("copies of the layer's value readers", c)?
    };
    let matched = frames("the layer's key readers through earlier heads", ndarray::concatenate(Axis(1), &through.iter().map(|x| x.view()).collect::<Vec<_>>()).map_err(|e| e.to_string())?)?;
    let charts = |kind: &str, rules: bool| -> Result<(Vec<Chart>, Vec<Chart>), String> {
        Ok(match kind {
            "v" => (vec![heads.clone()], vec![earlier.clone()]),
            "o" => {
                let readers = vec![heads.clone(), frames("the layer's value writers", values.0.t().to_owned())?];
                (if rules { vec![copies.clone()] } else { Vec::new() }, readers)
            }
            "k" => (vec![heads.clone(), rotary.clone()], vec![earlier.clone()]),
            _ => {
                let writers = vec![heads.clone(), rotary.clone(), frames("the layer's key writers", keys.0.t().to_owned())?];
                (writers, if rules { vec![earlier.clone(), matched.clone()] } else { vec![earlier.clone()] })
            }
        })
    };
    let mut report = Vec::new();
    let mut decoded: Vec<(Vec<Description>, Vec<Description>)> = Vec::new();
    let (mut once, mut per_word) = ([0.0; 2], [0.0; 2]);
    for (k, kind) in kinds.iter().enumerate() {
        let name = format!("blocks.{layer}.{kind}");
        let metric = Metric::of(&measured[k], observations);
        let ruled = matches!(*kind, "o" | "q");
        let (bw, br) = charts(kind, false)?;
        let plain = Geometry::new(metric.clone(), bw, br)?;
        let rules = if ruled {
            let (rw, rr) = charts(kind, true)?;
            Some(Geometry::new(metric, rw, rr)?)
        } else {
            None
        };
        let (u, v) = &libraries[k];
        let started = std::time::Instant::now();
        let both: Vec<(Description, Option<Description>)> = (0..u.nrows())
            .into_par_iter()
            .map(|c| {
                let (uc, vc) = (u.slice(s![c..c + 1, ..]), v.slice(s![c..c + 1, ..]));
                gam_linalg::faer_ndarray::with_nested_parallel(|| Ok((plain.describe(uc, vc)?, rules.as_ref().map(|r| r.describe(uc, vc)).transpose()?)))
            })
            .collect::<Result<_, String>>()?;
        let f = frequency.get(&name).ok_or(format!("no sets for {name}"))?;
        let (a, b): (Vec<Description>, Vec<Option<Description>>) = both.into_iter().unzip();
        let b: Vec<Description> = b.into_iter().zip(&a).map(|(x, y)| x.unwrap_or_else(|| y.clone())).collect();
        let total = |d: &[Description]| d.iter().map(Description::total).sum::<f64>();
        let active = |d: &[Description]| d.iter().zip(f).map(|(x, n)| x.total() * n).sum::<f64>();
        let rule_chart = |d: &Description| d.writer.0.starts_with("copies") || d.reader.0.ends_with("through earlier heads");
        // Per head (the writer's for queries, the reader's for outputs), how often a rule is taken.
        let mut by_head = vec![(0usize, 0usize, 0.0f64, 0.0f64); 6];
        for (c, d) in b.iter().enumerate() {
            let h = if *kind == "o" { head_of(v.row(c)) } else { head_of(u.row(c)) };
            by_head[h].0 += 1;
            by_head[h].2 += f[c];
            if rule_chart(d) {
                by_head[h].1 += 1;
                by_head[h].3 += f[c];
            }
        }
        let mut sources = std::collections::BTreeMap::<String, usize>::new();
        for d in b.iter().filter(|d| d.reader.0.ends_with("through earlier heads")) {
            if let Some(i) = d.reader.1.first().and_then(|m| m.first()) {
                *sources.entry(format!("layer {} head {}", i / (6 * 512), (i / 512) % 6)).or_default() += 1;
            }
        }
        eprintln!(
            "{name}: {} subcomponents in {:.0}s; once {:.0} bits per-site, {:.0} with the rules ({:+.1}%); per word {:.1} per-site, {:.1} with the rules",
            a.len(),
            started.elapsed().as_secs_f64(),
            total(&a),
            total(&b),
            100.0 * (total(&b) - total(&a)) / total(&a),
            active(&a),
            active(&b)
        );
        for (h, (n, taking, on, on_taking)) in by_head.iter().enumerate() {
            if *n > 0 {
                eprintln!("  head {h}: {taking} of {n} subcomponents take a rule ({on_taking:.2} of {on:.2} runs per word)");
            }
        }
        if !sources.is_empty() {
            eprintln!("  matched through: {sources:?}");
        }
        once[0] += total(&a);
        once[1] += total(&b);
        per_word[0] += active(&a);
        per_word[1] += active(&b);
        report.push(serde_json::json!({
            "site": name,
            "once_bits": {"per_site": total(&a), "rules": total(&b)},
            "per_word_bits": {"per_site": active(&a), "rules": active(&b)},
            "taking_a_rule_by_head": by_head.iter().map(|(n, t, on, ot)| serde_json::json!({"subcomponents": n, "taking": t, "runs": on, "runs_taking": ot})).collect::<Vec<_>>(),
            "matched_through": sources,
        }));
        decoded.push((a, b));
    }
    eprintln!("layer {layer}: once {:.0} bits per-site, {:.0} with the rules; per word {:.1} and {:.1}", once[0], once[1], per_word[0], per_word[1]);
    // The decode check: the layer's sites replaced by their decoded descriptions.
    let check: Vec<usize> = (statistics * CONTEXT..(statistics + 1) * CONTEXT).collect();
    let inputs = family.select(&check);
    let logits = model.execute(&inputs, false).map_err(|e| e.to_string())?.values[model.output].clone();
    let masks = given_masks(sets, &chosen, CONTEXT, statistics)?;
    let given = Blocked::rank_one(libraries.iter().map(|(u, v)| Library { v: v.clone(), u: u.clone(), mean: ndarray::Array1::zeros(v.ncols()) }).collect(), vec![masks]);
    let generic = Generic::new(&measured, observations);
    let coded = Coded { model, sites: chosen.clone(), batches: vec![(inputs, Target::every_row(logits))], observations, samples: 4, describe: &generic, boxed: None };
    let (base, _) = measure(&coded, &given)?;
    let mut checks = Vec::new();
    for (label, pick) in [("per-site", 0usize), ("rules", 1)] {
        let mut replaced = given.clone();
        let mut priced = 0.0;
        for (k, (a, b)) in decoded.iter().enumerate() {
            let descriptions = if pick == 0 { a } else { b };
            let on: Vec<f64> = given.masks[0][k].sum_axis(Axis(0)).to_vec();
            priced += descriptions.iter().zip(&on).map(|(d, n)| d.kl_bits * n).sum::<f64>();
            let us: Vec<_> = descriptions.iter().map(|d| d.u.view()).collect();
            let vs: Vec<_> = descriptions.iter().map(|d| d.v.view()).collect();
            replaced.libraries[k] = std::sync::Arc::new(Library {
                u: ndarray::concatenate(Axis(0), &us).map_err(|e| e.to_string())?,
                v: ndarray::concatenate(Axis(0), &vs).map_err(|e| e.to_string())?,
                mean: given.libraries[k].mean.clone(),
            });
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
    let report = serde_json::json!({"layer": layer, "observations": observations, "statistics_sequences": statistics, "sites": report,
        "once_bits": {"per_site": once[0], "rules": once[1]}, "per_word_bits": {"per_site": per_word[0], "rules": per_word[1]}, "decode_check": checks});
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
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

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_rules_2951 heads EXPORT_DIR HALF SEQUENCES | pairs EXPORT_DIR LIBRARY_DIR SETS_DIR LAYER | price EXPORT_DIR LIBRARY_DIR OBSERVATIONS SITES | induction EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS STATISTICS LAYER";
    match args.get(1).map(String::as_str) {
        Some("heads") if args.len() == 5 => heads(
            std::path::Path::new(&args[2]),
            args[3].parse().map_err(|e| format!("HALF: {e}"))?,
            args[4].parse().map_err(|e| format!("SEQUENCES: {e}"))?,
        ),
        Some("pairs") if args.len() == 6 => pairs(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[4]),
            args[5].parse().map_err(|e| format!("LAYER: {e}"))?,
        ),
        Some("price") if args.len() == 6 => price(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
            args[4].parse().map_err(|e| format!("OBSERVATIONS: {e}"))?,
            &args[5],
        ),
        Some("induction") if args.len() == 9 => induction(
            std::path::Path::new(&args[2]),
            std::path::Path::new(&args[3]),
            std::path::Path::new(&args[4]),
            std::path::Path::new(&args[5]),
            args[6].parse().map_err(|e| format!("OBSERVATIONS: {e}"))?,
            args[7].parse().map_err(|e| format!("STATISTICS: {e}"))?,
            args[8].parse().map_err(|e| format!("LAYER: {e}"))?,
        ),
        _ => Err(usage.to_string()),
    }
}
