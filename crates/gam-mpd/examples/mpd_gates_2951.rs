//! Gate laws for a decomposed language model (#2951): fit `gam_mpd::gates` laws on given per-token
//! sets over the model's own amplitudes, and score them on held-out sequences.
//!
//! `mpd_gates_2951 EXPORT_DIR AMPS_DIR SETS_DIR OUT TRAIN EVAL [CONTEXT]`
//!
//! `EXPORT_DIR` is a language-model export (`gam_mpd::import::import_language_model`); only its
//! program is read, for its sites (`gam_mpd::masked::sites`), which sites are upstream of which
//! (`gam_mpd::gates::upstream`) and their read widths. `AMPS_DIR` holds the amplitudes
//! `a_j = v_jᵀx` of every subcomponent on the clean forward of a run of sequences
//! (`bench/vpd_2951/vpd_gates.py amps`): `sites.txt` (one `name pieces` line per site, in the
//! masked driver's order) and per site `{name}.npy`, float16 `pieces × (sequences · CONTEXT)`.
//! `SETS_DIR` holds the per-token sets of the same sequences (`bench/vpd_2951/vpd_sets_export.py`:
//! `indptr.i64`, `indices.i64`, subcomponents numbered site after site). `TRAIN` and `EVAL` are
//! disjoint sequence ranges `lo:hi`.
//!
//! Every subcomponent gets a law fitted on the training sequences (`gam_mpd::gates::fit`): first
//! the better of its base rate and a law over its own amplitude (pool: its own amplitude, every
//! upstream site's amplitudes at this token and at the previous one, and its own site's at the
//! previous one; a `d`-subset costs `log₂ C(pool, d)`). Then, site by site, rounds of
//! `gam_mpd::gates::screen` over the pool propose one feature more per law, on a stride sample of
//! the training tokens; a law keeps the feature when its refitted total falls, and rounds repeat
//! until no law changes.
//!
//! After the first pass and at the end, `OUT.json` gets the eval sequences' bits per token for
//! sending every subcomponent's on/off: under the laws, under the base rates, and under the masked
//! driver's previous-state context coder (`gam_mpd::masked::Context`, counted on the training
//! sequences); the laws' own bits; the kinds of law; and the multiply–adds per token the laws cost
//! (the amplitudes their features read, `d_in` each, plus the laws themselves) beside every
//! amplitude's. `OUT.laws.json` holds the laws, and `OUT.sets.{indptr,indices}.npy` the sets the
//! laws choose on the eval sequences (int64 CSR over their tokens).

use gam_mpd::gates::{self, Feature, Law};
use gam_mpd::import::import_language_model;
use gam_mpd::masked::{Context, matrix, sites};
use memmap2::Mmap;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde_json::json;
use statrs::function::gamma::ln_gamma;
use std::path::{Path, PathBuf};

/// Per-site float16 amplitude tables, memory-mapped.
struct Amplitudes {
    maps: Vec<Mmap>,
    /// Byte offset of the data in each file, and the tokens per row.
    data: Vec<usize>,
    tokens: usize,
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as f64;
    sign * match exponent {
        0 => mantissa * (-24f64).exp2(),
        31 => {
            if mantissa == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => (1.0 + mantissa / 1024.0) * f64::from(e - 15).exp2(),
    }
}

impl Amplitudes {
    fn open(dir: &Path, names: &[String], pieces: &[usize]) -> Result<Self, String> {
        let mut maps = Vec::new();
        let mut data = Vec::new();
        let mut tokens = None;
        for (name, &c) in names.iter().zip(pieces) {
            let path = dir.join(format!("{name}.npy"));
            let file = std::fs::File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            // SAFETY: the table is read only and no process writes it while this one runs.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))?;
            if map.len() < 10 || &map[..6] != b"\x93NUMPY" {
                return Err(format!("{}: not an npy file", path.display()));
            }
            let (header_len, start) = if map[6] == 1 {
                (u16::from_le_bytes([map[8], map[9]]) as usize, 10)
            } else {
                (u32::from_le_bytes([map[8], map[9], map[10], map[11]]) as usize, 12)
            };
            let header = std::str::from_utf8(&map[start..start + header_len]).map_err(|e| e.to_string())?;
            if !header.contains("'<f2'") || header.contains("'fortran_order': True") {
                return Err(format!("{}: expected a C-ordered float16 table, header {header}", path.display()));
            }
            let offset = start + header_len;
            let n = (map.len() - offset) / 2 / c;
            if n * c * 2 != map.len() - offset || *tokens.get_or_insert(n) != n {
                return Err(format!("{}: {} bytes are not {c} rows of the run's tokens", path.display(), map.len() - offset));
            }
            maps.push(map);
            data.push(offset);
        }
        Ok(Self { maps, data, tokens: tokens.unwrap_or(0) })
    }

    /// One subcomponent's amplitude at token `t`.
    fn value(&self, site: usize, piece: usize, t: usize) -> f64 {
        let at = self.data[site] + 2 * (piece * self.tokens + t);
        let bytes = &self.maps[site];
        f16_to_f64(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
    }
}

fn read_i64(path: &Path) -> Result<Vec<i64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])).collect())
}

/// A one-dimensional little-endian int64 `.npy` file.
fn write_npy_i64(path: &Path, values: &[i64]) -> Result<(), String> {
    let mut header = format!("{{'descr': '<i8', 'fortran_order': False, 'shape': ({},), }}", values.len());
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

fn range(spec: &str) -> Result<(usize, usize), String> {
    let (lo, hi) = spec.split_once(':').ok_or_else(|| format!("{spec}: expected lo:hi"))?;
    let lo: usize = lo.parse().map_err(|e| format!("{spec}: {e}"))?;
    let hi: usize = hi.parse().map_err(|e| format!("{spec}: {e}"))?;
    if hi <= lo {
        return Err(format!("{spec}: empty range"));
    }
    Ok((lo, hi))
}

fn log2_binomial(n: usize, k: usize) -> f64 {
    (ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)) / std::f64::consts::LN_2
}

/// Everything the fit and the scoring share.
struct Run {
    amps: Amplitudes,
    names: Vec<String>,
    pieces: Vec<usize>,
    /// Per site, per subcomponent, its on tokens (ascending, over the whole run).
    on: Vec<Vec<Vec<usize>>>,
    upstream: Vec<Vec<usize>>,
    widths: Vec<usize>,
    context: usize,
}

impl Run {
    fn pool(&self, site: usize) -> usize {
        1 + self.pieces[site] + 2 * self.upstream[site].iter().map(|u| self.pieces[*u]).sum::<usize>()
    }

    fn structure_bits(&self, site: usize, d: usize) -> f64 {
        log2_binomial(self.pool(site), d)
    }

    fn labels(&self, site: usize, piece: usize, lo: usize, hi: usize, stride: usize) -> Vec<bool> {
        let mut y = vec![false; (hi - lo).div_ceil(stride)];
        for &t in &self.on[site][piece] {
            if t >= lo && t < hi && (t - lo) % stride == 0 {
                y[(t - lo) / stride] = true;
            }
        }
        y
    }

    /// A feature's values at tokens `lo..hi` (every `stride`-th); lag 1 reads the previous token,
    /// zero at a sequence's first.
    fn column(&self, f: &Feature, lo: usize, hi: usize, stride: usize) -> Array1<f64> {
        (lo..hi)
            .step_by(stride)
            .map(|t| match f.lag {
                0 => self.amps.value(f.site, f.piece, t),
                _ if t % self.context == 0 => 0.0,
                _ => self.amps.value(f.site, f.piece, t - 1),
            })
            .collect()
    }

    /// Feature rows of `features` at tokens `lo..hi` (every `stride`-th).
    fn rows(&self, features: &[Feature], lo: usize, hi: usize, stride: usize) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros(((hi - lo).div_ceil(stride), features.len()));
        for (k, f) in features.iter().enumerate() {
            x.column_mut(k).assign(&self.column(f, lo, hi, stride));
        }
        x
    }
}

/// The first pass: per subcomponent, the better of its base rate and its own-amplitude law.
fn first_pass(run: &Run, site: usize, train: (usize, usize)) -> Vec<Law> {
    (0..run.pieces[site])
        .into_par_iter()
        .map(|piece| {
            let y = run.labels(site, piece, train.0, train.1, 1);
            let base = gates::base(&y);
            let structure = run.structure_bits(site, 1);
            if base.total_bits() <= gates::least_featured_bits(structure) {
                return base;
            }
            let features = [Feature { site, piece, lag: 0 }];
            let x = run.rows(&features, train.0, train.1, 1);
            let law = gates::fit(x.view(), &y, &features, structure, None);
            if law.total_bits() < base.total_bits() { law } else { base }
        })
        .collect()
}

/// Screening rounds at one site (module note); returns how many features were kept.
fn screening(run: &Run, site: usize, laws: &mut [Law], train: (usize, usize)) -> Result<usize, String> {
    let n_train = train.1 - train.0;
    // The screen only proposes; a sample of the training tokens is enough to rank candidates.
    let stride = n_train.div_ceil(8192).max(1);
    let scale = stride as f64;
    let mut blocks: Vec<(usize, usize)> = run.upstream[site].iter().map(|u| (*u, 0)).collect();
    blocks.extend(run.upstream[site].iter().map(|u| (*u, 1)));
    blocks.push((site, 1));
    let mut kept_total = 0;
    let mut open: Vec<usize> = (0..laws.len()).filter(|j| !laws[*j].features.is_empty()).collect();
    while !open.is_empty() {
        // Residuals and weights of the open laws at the sample.
        let columns: Vec<(Array1<f64>, Array1<f64>)> = open
            .par_iter()
            .map(|&j| {
                let law = &laws[j];
                let x = run.rows(&law.features, train.0, train.1, stride);
                let y = run.labels(site, j, train.0, train.1, stride);
                let mut r = Array1::zeros(y.len());
                let mut w = Array1::zeros(y.len());
                for (t, on) in y.iter().enumerate() {
                    let row: Vec<f64> = x.row(t).to_vec();
                    let p = 1.0 / (1.0 + (-law.logit(&row)).exp());
                    r[t] = if *on { 1.0 } else { 0.0 } - p;
                    w[t] = p * (1.0 - p);
                }
                (r, w)
            })
            .collect();
        let n = columns.first().map_or(0, |c| c.0.len());
        let mut residuals = Array2::<f64>::zeros((n, open.len()));
        let mut weights = Array2::<f64>::zeros((n, open.len()));
        for (k, (r, w)) in columns.into_iter().enumerate() {
            residuals.column_mut(k).assign(&r);
            weights.column_mut(k).assign(&w);
        }
        let mut best: Vec<Option<(f64, Feature)>> = vec![None; open.len()];
        for &(u, lag) in &blocks {
            let mut candidates = Array2::<f64>::zeros((n, run.pieces[u]));
            let rows: Vec<Array1<f64>> = (0..run.pieces[u])
                .into_par_iter()
                .map(|piece| run.column(&Feature { site: u, piece, lag }, train.0, train.1, stride))
                .collect();
            for (piece, column) in rows.into_iter().enumerate() {
                candidates.column_mut(piece).assign(&column);
            }
            let gains = gates::screen(&candidates, &residuals, &weights)?;
            for (k, &j) in open.iter().enumerate() {
                for piece in 0..run.pieces[u] {
                    let feature = Feature { site: u, piece, lag };
                    let gain = gains[[piece, k]] * scale;
                    if laws[j].features.contains(&feature) {
                        continue;
                    }
                    if best[k].as_ref().is_none_or(|b| gain > b.0) {
                        best[k] = Some((gain, feature));
                    }
                }
            }
        }
        // A proposal is refitted when its predicted gain pays for the larger subset alone.
        let refits: Vec<(usize, Option<Law>)> = open
            .par_iter()
            .zip(best.par_iter())
            .map(|(&j, proposal)| {
                let law = &laws[j];
                let d = law.features.len();
                let Some((gain, feature)) = proposal else { return (j, None) };
                if *gain <= run.structure_bits(site, d + 1) - run.structure_bits(site, d) {
                    return (j, None);
                }
                let mut features = law.features.clone();
                features.push(*feature);
                let x = run.rows(&features, train.0, train.1, 1);
                let y = run.labels(site, j, train.0, train.1, 1);
                let refit = gates::fit(x.view(), &y, &features, run.structure_bits(site, d + 1), Some(law));
                (j, (refit.total_bits() < law.total_bits()).then_some(refit))
            })
            .collect();
        open = Vec::new();
        for (j, refit) in refits {
            if let Some(law) = refit {
                laws[j] = law;
                open.push(j);
                kept_total += 1;
            }
        }
        eprintln!("site {}: {} laws took a feature", run.names[site], open.len());
    }
    Ok(kept_total)
}

/// The eval scoring of `laws` (module note), and the laws' sets on the eval sequences.
fn score(run: &Run, laws: &[Vec<Law>], train: (usize, usize), eval: (usize, usize)) -> (serde_json::Value, Vec<i64>, Vec<i64>) {
    let n_eval = eval.1 - eval.0;
    let mut law_bits_eval = 0.0;
    let mut base_bits_eval = 0.0;
    let mut law_bits = 0.0;
    let mut decisions: Vec<Vec<usize>> = vec![Vec::new(); n_eval];
    let mut offset = 0;
    let mut kinds = std::collections::BTreeMap::<String, usize>::new();
    let mut needed = std::collections::BTreeSet::<(usize, usize)>::new();
    let mut law_madds = 0usize;
    for site in 0..run.names.len() {
        let per: Vec<(f64, f64, Vec<usize>)> = (0..run.pieces[site])
            .into_par_iter()
            .map(|piece| {
                let law = &laws[site][piece];
                let y = run.labels(site, piece, eval.0, eval.1, 1);
                let x = run.rows(&law.features, eval.0, eval.1, 1);
                let base = gates::base(&run.labels(site, piece, train.0, train.1, 1));
                let on: Vec<usize> = (0..n_eval).filter(|t| law.on(&x.row(*t).to_vec())).collect();
                (law.label_bits(x.view(), &y), base.label_bits(Array2::<f64>::zeros((n_eval, 0)).view(), &y), on)
            })
            .collect();
        for (piece, (lb, bb, on)) in per.into_iter().enumerate() {
            law_bits_eval += lb;
            base_bits_eval += bb;
            for t in on {
                decisions[t].push(offset + piece);
            }
            let law = &laws[site][piece];
            law_bits += law.law_bits;
            let kind = if law.features.is_empty() {
                if law.beta > 0.0 { "base on".to_string() } else { "base off".to_string() }
            } else {
                format!("{} features, {} units", law.features.len(), law.units.len())
            };
            *kinds.entry(kind).or_default() += 1;
            for f in &law.features {
                needed.insert((f.site, f.piece));
            }
            law_madds += law.multiply_adds();
        }
        offset += run.pieces[site];
    }
    // The previous-state context coder, counted on the training sequences.
    let sequences = |lo: usize, hi: usize| (lo / run.context..hi / run.context);
    let masks_of = |s: usize| -> Vec<Array2<f64>> {
        (0..run.names.len())
            .map(|site| {
                let mut m = Array2::<f64>::zeros((run.context, run.pieces[site]));
                for (piece, tokens) in run.on[site].iter().enumerate() {
                    for &t in tokens {
                        if t / run.context == s {
                            m[[t % run.context, piece]] = 1.0;
                        }
                    }
                }
                m
            })
            .collect()
    };
    let previous: Vec<Option<usize>> = (0..run.context).map(|p| p.checked_sub(1)).collect();
    let mut counts = Context::new(&run.pieces);
    for s in sequences(train.0, train.1) {
        counts.absorb(&masks_of(s), &previous);
    }
    let coder = counts.coder(previous);
    let context_bits: f64 = sequences(eval.0, eval.1).map(|s| coder.bits(&masks_of(s)).sum()).sum();
    let amplitude_madds: usize = needed.iter().map(|(site, _)| run.widths[*site]).sum();
    let every_amplitude: usize = run.pieces.iter().zip(&run.widths).map(|(c, w)| c * w).sum();
    let l0 = decisions.iter().map(Vec::len).sum::<usize>() as f64 / n_eval as f64;
    let given_l0 = (0..run.names.len()).map(|s| run.on[s].iter().map(|t| t.iter().filter(|t| **t >= eval.0 && **t < eval.1).count()).sum::<usize>()).sum::<usize>() as f64
        / n_eval as f64;
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    for set in &decisions {
        indices.extend(set.iter().map(|c| *c as i64));
        indptr.push(indices.len() as i64);
    }
    let report = json!({
        "eval_tokens": n_eval,
        "bits_per_token": {
            "laws": law_bits_eval / n_eval as f64,
            "base_rates": base_bits_eval / n_eval as f64,
            "context_coder": context_bits / n_eval as f64,
        },
        "law_bits_total": law_bits,
        "kinds": kinds,
        "l0_laws": l0,
        "l0_given": given_l0,
        "multiply_adds_per_token": {
            "amplitudes_read": amplitude_madds,
            "laws": law_madds,
            "total": amplitude_madds + law_madds,
            "every_amplitude": every_amplitude,
        },
        "amplitudes_read": needed.len(),
    });
    (report, indptr, indices)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "mpd_gates_2951 EXPORT_DIR AMPS_DIR SETS_DIR OUT TRAIN EVAL [CONTEXT]";
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::parse("auto").ok_or("GPU policy")?);
    let export = PathBuf::from(args.get(1).ok_or(usage)?);
    let amps_dir = PathBuf::from(args.get(2).ok_or(usage)?);
    let sets_dir = PathBuf::from(args.get(3).ok_or(usage)?);
    let out = PathBuf::from(args.get(4).ok_or(usage)?);
    let context: usize = args.get(7).map_or(Ok(512), |v| v.parse()).map_err(|e| format!("CONTEXT: {e}"))?;
    let (train_lo, train_hi) = range(args.get(5).ok_or(usage)?)?;
    let (eval_lo, eval_hi) = range(args.get(6).ok_or(usage)?)?;
    if train_lo < eval_hi && eval_lo < train_hi {
        return Err("TRAIN and EVAL overlap".to_string());
    }
    let train = (train_lo * context, train_hi * context);
    let eval = (eval_lo * context, eval_hi * context);

    let listed = std::fs::read_to_string(amps_dir.join("sites.txt")).map_err(|e| e.to_string())?;
    let mut names = Vec::new();
    let mut pieces = Vec::new();
    for line in listed.lines().filter(|l| !l.trim().is_empty()) {
        let (name, c) = line.split_once(' ').ok_or_else(|| format!("sites.txt: {line}"))?;
        names.push(name.to_string());
        pieces.push(c.trim().parse::<usize>().map_err(|e| e.to_string())?);
    }
    let imported = import_language_model(&export, 1, context)?;
    let program_sites = sites(&imported.program);
    let order: Vec<usize> = names
        .iter()
        .map(|n| program_sites.iter().position(|s| &s.name == n).ok_or_else(|| format!("site {n} is not in the export")))
        .collect::<Result<_, _>>()?;
    let ordered: Vec<_> = order.iter().map(|i| program_sites[*i].clone()).collect();
    let upstream = gates::upstream(&ordered);
    let widths = ordered.iter().map(|s| matrix(&imported.program, s).map(|w| w.ncols())).collect::<Result<Vec<_>, _>>()?;
    drop(imported);
    let amps = Amplitudes::open(&amps_dir, &names, &pieces)?;
    if amps.tokens < train.1.max(eval.1) {
        return Err(format!("the amplitudes cover {} tokens; TRAIN and EVAL need {}", amps.tokens, train.1.max(eval.1)));
    }
    let indptr = read_i64(&sets_dir.join("indptr.i64"))?;
    let indices = read_i64(&sets_dir.join("indices.i64"))?;
    let offsets: Vec<usize> = std::iter::once(0).chain(pieces.iter().scan(0, |acc, c| {
        *acc += c;
        Some(*acc)
    })).collect();
    let mut on: Vec<Vec<Vec<usize>>> = pieces.iter().map(|c| vec![Vec::new(); *c]).collect();
    for t in 0..indptr.len().saturating_sub(1).min(amps.tokens) {
        for &g in &indices[indptr[t] as usize..indptr[t + 1] as usize] {
            let g = g as usize;
            let site = offsets.partition_point(|o| *o <= g) - 1;
            on[site][g - offsets[site]].push(t);
        }
    }
    let run = Run { amps, names, pieces, on, upstream, widths, context };
    eprintln!(
        "{} sites, {} subcomponents; train tokens {}..{}, eval {}..{}",
        run.names.len(),
        offsets[run.names.len()],
        train.0,
        train.1,
        eval.0,
        eval.1
    );

    let write = |stage: &str, laws: &[Vec<Law>], report: &mut serde_json::Value| -> Result<(), String> {
        let (scored, indptr, indices) = score(&run, laws, train, eval);
        report[stage] = scored;
        std::fs::write(out.with_extension("json"), serde_json::to_string_pretty(report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let sites_json: Vec<_> = run.names.iter().zip(laws).map(|(n, l)| json!({"name": n, "laws": l})).collect();
        std::fs::write(out.with_extension("laws.json"), serde_json::to_string(&sites_json).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        write_npy_i64(&out.with_extension("sets.indptr.npy"), &indptr)?;
        write_npy_i64(&out.with_extension("sets.indices.npy"), &indices)?;
        eprintln!("{stage}: {}", report[stage]);
        Ok(())
    };
    let mut report = json!({"train": args[5], "eval": args[6], "sites": run.names});
    let started = std::time::Instant::now();
    let mut laws: Vec<Vec<Law>> = Vec::new();
    for site in 0..run.names.len() {
        laws.push(first_pass(&run, site, train));
        eprintln!("first pass {} ({:.0}s)", run.names[site], started.elapsed().as_secs_f64());
    }
    write("own_amplitude", &laws, &mut report)?;
    for site in 0..run.names.len() {
        let kept = screening(&run, site, &mut laws[site], train)?;
        eprintln!("screening {}: {kept} features kept ({:.0}s)", run.names[site], started.elapsed().as_secs_f64());
    }
    write("screened", &laws, &mut report)?;
    Ok(())
}
