//! Rank-k gated subcomponents chosen by the code (`gam_mpd::blocks`), on a trained toy and on
//! VPD's four-layer model (#2951).
//!
//! `mpd_blocks_2951 modadd EXPORT_DIR OUT.json OBSERVATIONS [SITES]`
//!
//! `EXPORT_DIR` is a `transformer` export (`gam_mpd::import::import`; e.g.
//! `~/mpd-data/engine/p31_s0_generic`, the mod-31 network on all 961 inputs), `SITES` a comma list
//! of site names (default every site). Each site starts as its Fisher-whitened singular
//! subcomponents (`gam_mpd::pieces::fisher_svd`, from moments measured on the family), every one on
//! for every input; the rank-one sets are selected pass after pass under the counts of the pass
//! before until a pass no longer lowers the code (the rank-one point), then the blocks are fitted,
//! merges and splits (`gam_mpd::blocks::fit_blocks`). Each final block's frequency content is
//! reported on whichever side of it the network reads or writes in token coordinates: its output
//! through the unembedding (over the classes), its input through the embedding (over the
//! operands), as the share of its energy at its leading frequency.
//!
//! `mpd_blocks_2951 vpd EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS FIRST SEQUENCES [LIBRARY_ROWS]`
//!
//! `EXPORT_DIR` a language-model export (`gam_mpd::import::import_language_model`, contexts of
//! 512), `LIBRARY_DIR` per site `{site}.v.f64` (subcomponents × d_in) and `{site}.u.f64`
//! (subcomponents × d_out) raw float64 on the uncentred read, `SETS_DIR` per-token sets for that
//! library (`indptr.i64`, `indices.i64`, `sites.txt`, as `mpd_pieces_masked_2951` reads them).
//! Sequences `FIRST..FIRST + SEQUENCES` are coded, one batch each. Three points under the same code
//! (`gam_mpd::blocks::measure`; the library spread over `LIBRARY_ROWS` rows, default the coded
//! rows): the given sets, the sets selected from them (Step A), and the blocks merged from
//! Step A's.
//!
//! Every point reports bits per word (the primary score), KL, active blocks per word and active
//! rank-one equivalents per word (`Σ k_c` over the blocks on), so a rank-k block's one listing is
//! compared with rank-one subcomponents at the rank it actually spends. One gate on a rank-k block
//! scales the block along a line, not the box of k independent masks: any robustness evaluation of
//! these points masks each block's columns together, a weaker claim than per column.

use gam_mpd::blocks::{Bits, Blocked, Coded, block_cosine, fit_blocks, measure, reselect};
use gam_mpd::import::{import, import_language_model};
use gam_mpd::masked::{Library, Site, Target, site_statistics, sites};
use gam_mpd::operator_program::{FamilyInputs, LabelKind, OperatorProgram};
use gam_mpd::pieces::fisher_svd;
use ndarray::{Array1, Array2};
use serde_json::{Value, json};
use std::path::Path;

fn read_raw<T: Copy>(path: &Path, decode: fn([u8; 8]) -> T) -> Result<Vec<T>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes.chunks_exact(8).map(|c| decode(c.try_into().expect("eight bytes"))).collect())
}

fn read_f64(path: &Path, cols: usize) -> Result<Array2<f64>, String> {
    let values = read_raw(path, f64::from_le_bytes)?;
    if values.len() % cols != 0 {
        return Err(format!("{}: {} values in rows of {cols}", path.display(), values.len()));
    }
    Array2::from_shape_vec((values.len() / cols, cols), values).map_err(|e| e.to_string())
}

fn point(name: &str, bits: &Bits) -> Value {
    let (per_word, kl, active, rank) = bits.per_row();
    json!({
        "point": name,
        "bits_per_word": per_word,
        "kl": kl,
        "active_blocks_per_word": active,
        "active_rank_one_equivalents_per_word": rank,
        "library_bits": bits.library,
        "rate_bits": bits.rates,
        "listing_bits": bits.listing,
        "kl_bits": bits.kl,
        "rows": bits.rows,
        "blocks": bits.blocks,
        "subcomponents": bits.pieces,
    })
}

fn say(name: &str, bits: &Bits) {
    let (per_word, kl, active, rank) = bits.per_row();
    eprintln!(
        "{name}: {per_word:.2} bits/word (library {:.0}, rates {:.0}, listing {:.0}, KL {:.0}), KL {kl:.4}, {active:.2} blocks and {rank:.2} rank-one equivalents on per word, {} blocks of {} columns",
        bits.library, bits.rates, bits.listing, bits.kl, bits.blocks, bits.pieces
    );
}

/// The rank histogram of a decomposition, per site.
fn ranks(coded: &Coded<'_>, blocked: &Blocked) -> Value {
    let per_site: Vec<Value> = coded
        .sites
        .iter()
        .zip(&blocked.ranks)
        .map(|(site, r)| {
            let mut histogram = std::collections::BTreeMap::<usize, usize>::new();
            for k in r {
                *histogram.entry(*k).or_default() += 1;
            }
            json!({"site": site.name, "ranks": histogram})
        })
        .collect();
    Value::Array(per_site)
}

/// Each block's firing fraction over the coded rows.
fn firing(blocked: &Blocked, k: usize, b: usize) -> f64 {
    let (on, rows) = blocked.masks.iter().fold((0.0, 0.0), |(on, rows), masks| (on + masks[k].column(b).sum(), rows + masks[k].nrows() as f64));
    on / rows.max(1.0)
}

/// Per frequency `f = 0..=p/2`, the energy of the columns of `y` (`p × k`) at `f`.
fn spectrum(y: &Array2<f64>) -> Vec<f64> {
    let p = y.nrows();
    (0..=p / 2)
        .map(|f| {
            let mut energy = 0.0;
            for col in y.columns() {
                let (mut re, mut im) = (0.0, 0.0);
                for (j, v) in col.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * (f * j) as f64 / p as f64;
                    re += v * angle.cos();
                    im -= v * angle.sin();
                }
                energy += re * re + im * im;
            }
            energy
        })
        .collect()
}

/// `(leading frequency, its share of the energy)` of `y`'s columns.
fn leading(y: &Array2<f64>) -> (usize, f64) {
    let energy = spectrum(y);
    let total: f64 = energy.iter().sum();
    let (f, e) = energy.iter().enumerate().fold((0, 0.0), |best, (f, e)| if *e > best.1 { (f, *e) } else { best });
    (f, if total > 0.0 { e / total } else { 0.0 })
}

fn modadd(dir: &Path, out: &Path, observations: f64, names: Option<Vec<String>>) -> Result<(), String> {
    let imported = import(dir)?;
    let program = imported.program;
    let family = imported.contract.family;
    let logits = program.execute(&family, false).map_err(|e| e.to_string())?.values[program.output].clone();
    let target = Target::every_row(logits);
    let chosen: Vec<Site> = sites(&program).into_iter().filter(|s| names.as_ref().is_none_or(|n| n.contains(&s.name))).collect();
    if chosen.is_empty() {
        return Err(format!("no sites chosen of {:?}", sites(&program).iter().map(|s| s.name.clone()).collect::<Vec<_>>()));
    }
    let statistics = site_statistics(&program, &chosen, [family.clone()], 16, 0x5EED)?;
    let mut libraries = Vec::new();
    for (site, measured) in chosen.iter().zip(&statistics) {
        let library = fisher_svd(measured)?;
        eprintln!("{}: {}×{}, {} rank-one subcomponents", site.name, measured.w.nrows(), measured.w.ncols(), library.u.nrows());
        libraries.push(Library { v: library.v.t().to_owned(), u: library.u, mean: measured.mean.clone() });
    }
    let masks = vec![libraries.iter().map(|l| Array2::<f64>::ones((family.rows, l.v.nrows()))).collect()];
    let coded = Coded {
        model: &program,
        sites: chosen.clone(),
        batches: vec![(family.clone(), target)],
        observations,
        samples: 16,
        bits_per_real: 32.0,
        library_rows: family.rows as f64,
    };
    // The rank-one point: selection passes until one no longer lowers the code.
    let mut rank_one = Blocked::rank_one(libraries, masks);
    let (mut bits, _) = measure(&coded, &rank_one)?;
    say("all on", &bits);
    loop {
        let next = reselect(&coded, &rank_one)?;
        let (next_bits, _) = measure(&coded, &next)?;
        say("rank-one pass", &next_bits);
        if next_bits.total() >= bits.total() {
            break;
        }
        (rank_one, bits) = (next, next_bits);
    }
    let rank_one_bits = bits;
    let (blocked, block_bits) = fit_blocks(&coded, rank_one.clone(), true)?;
    say("rank one", &rank_one_bits);
    say("blocks", &block_bits);
    let readout = token_operator(&program, true);
    let embedding = token_operator(&program, false);
    let mut described = Vec::new();
    for (k, site) in chosen.iter().enumerate() {
        let library = &blocked.libraries[k];
        let mut start = 0;
        for (b, rank) in blocked.ranks[k].iter().enumerate() {
            let (u, v) = (library.u.slice(ndarray::s![start..start + rank, ..]).t().to_owned(), library.v.slice(ndarray::s![start..start + rank, ..]).t().to_owned());
            start += rank;
            let side = |op: &Option<Array2<f64>>, m: &Array2<f64>| op.as_ref().filter(|w| w.ncols() == m.nrows()).map(|w| leading(&w.dot(m)));
            described.push(json!({
                "site": site.name,
                "rank": rank,
                "firing": firing(&blocked, k, b),
                "output_frequency": side(&readout, &u),
                "input_frequency": side(&embedding, &v),
            }));
        }
    }
    let report = json!({
        "observations": observations,
        "points": [point("rank one", &rank_one_bits), point("blocks", &block_bits)],
        "rank_one_ranks": ranks(&coded, &rank_one),
        "block_ranks": ranks(&coded, &blocked),
        "blocks": described,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// The unembedding (`classes × d`, `readout`) or the embedding's operand rows (`tokens × d`, the
/// `=` token's row left out when the vocabulary is one wider than the classes) as a matrix whose
/// rows are token coordinates.
fn token_operator(program: &OperatorProgram, readout: bool) -> Option<Array2<f64>> {
    let tokens = |i: &gam_mpd::operator_program::Interface| i.groups().iter().any(|g| g.label.kind == LabelKind::Token);
    let op = program.operators.iter().find(|op| if readout { tokens(&op.rows) && !tokens(&op.cols) } else { tokens(&op.cols) && !tokens(&op.rows) })?;
    let m = op.matrix();
    if readout {
        Some(m)
    } else {
        // Columns are tokens: the operands only (the last token, `=`, left out).
        let e = m.t().to_owned();
        let operands = e.nrows().saturating_sub(1);
        Some(e.slice(ndarray::s![..operands, ..]).to_owned())
    }
}

/// Per batch, every site's masks from the given CSR sets.
fn given_sets(dir: &Path, chosen: &[(String, usize)], rows: usize, first: usize, sequences: usize) -> Result<Vec<Vec<Array2<f64>>>, String> {
    let listed = std::fs::read_to_string(dir.join("sites.txt")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let expected: Vec<String> = chosen.iter().map(|(name, pieces)| format!("{name} {pieces}")).collect();
    if listed.lines().collect::<Vec<_>>() != expected {
        return Err(format!("{}: its sites are not the library's ({expected:?})", dir.display()));
    }
    let mut offsets = vec![0usize];
    for (_, pieces) in chosen {
        offsets.push(offsets[offsets.len() - 1] + pieces);
    }
    let indptr = read_raw(&dir.join("indptr.i64"), i64::from_le_bytes)?;
    let indices = read_raw(&dir.join("indices.i64"), i64::from_le_bytes)?;
    if (indptr.len().saturating_sub(1)) < (first + sequences) * rows {
        return Err(format!("{}: fewer than {} sequences of sets", dir.display(), first + sequences));
    }
    let mut out = Vec::new();
    for s in first..first + sequences {
        let mut masks: Vec<Array2<f64>> = chosen.iter().map(|(_, p)| Array2::zeros((rows, *p))).collect();
        for r in 0..rows {
            let position = s * rows + r;
            for &i in &indices[indptr[position] as usize..indptr[position + 1] as usize] {
                let i = i as usize;
                let k = offsets.partition_point(|o| *o <= i) - 1;
                masks[k][[r, i - offsets[k]]] = 1.0;
            }
        }
        out.push(masks);
    }
    Ok(out)
}

/// The merged blocks of `blocked` (rank ≥ 2): site, rank, firing, and the largest `|cos|` between
/// two of its columns.
fn merged(coded: &Coded<'_>, blocked: &Blocked) -> Value {
    let mut out = Vec::new();
    for (k, site) in coded.sites.iter().enumerate() {
        let mut start = 0;
        for (b, rank) in blocked.ranks[k].iter().enumerate() {
            if *rank >= 2 {
                let library = &blocked.libraries[k];
                let mut cos: f64 = 0.0;
                for i in start..start + rank {
                    for j in i + 1..start + rank {
                        cos = cos.max(block_cosine(library, (i, 1), (j, 1)).abs());
                    }
                }
                out.push(json!({"site": site.name, "rank": rank, "firing": firing(blocked, k, b), "largest_column_cos": cos}));
            }
            start += rank;
        }
    }
    Value::Array(out)
}

#[allow(clippy::too_many_arguments)]
fn vpd(
    dir: &Path,
    library_dir: &Path,
    sets_dir: &Path,
    out: &Path,
    observations: f64,
    first: usize,
    sequences: usize,
    library_rows: Option<f64>,
) -> Result<(), String> {
    const CONTEXT: usize = 512;
    let imported = import_language_model(dir, first + sequences, CONTEXT)?;
    let program = imported.program;
    let family = imported.contract.family;
    let mut chosen = Vec::new();
    let mut libraries = Vec::new();
    for site in sites(&program) {
        let v_path = library_dir.join(format!("{}.v.f64", site.name));
        if !v_path.exists() {
            continue;
        }
        let w = gam_mpd::masked::matrix(&program, &site)?;
        let (d_out, d_in) = w.dim();
        let v = read_f64(&v_path, d_in)?;
        let u = read_f64(&library_dir.join(format!("{}.u.f64", site.name)), d_out)?;
        if u.nrows() != v.nrows() {
            return Err(format!("{}: {} v and {} u subcomponents", site.name, v.nrows(), u.nrows()));
        }
        chosen.push(site);
        libraries.push(Library { v, u, mean: Array1::zeros(d_in) });
    }
    let named: Vec<(String, usize)> = chosen.iter().zip(&libraries).map(|(s, l)| (s.name.clone(), l.v.nrows())).collect();
    let masks = given_sets(sets_dir, &named, CONTEXT, first, sequences)?;
    let mut batches = Vec::new();
    for s in first..first + sequences {
        let rows: Vec<usize> = (s * CONTEXT..(s + 1) * CONTEXT).collect();
        let inputs: FamilyInputs = family.select(&rows);
        let logits = program.execute(&inputs, false).map_err(|e| e.to_string())?.values[program.output].clone();
        batches.push((inputs, Target::every_row(logits)));
    }
    let coded_rows = (sequences * CONTEXT) as f64;
    let coded = Coded {
        model: &program,
        sites: chosen,
        batches,
        observations,
        samples: 4,
        bits_per_real: 32.0,
        library_rows: library_rows.unwrap_or(coded_rows),
    };
    let given = Blocked::rank_one(libraries, masks);
    let (given_bits, _) = measure(&coded, &given)?;
    say("given sets", &given_bits);
    let step_a = reselect(&coded, &given)?;
    let (step_a_bits, _) = measure(&coded, &step_a)?;
    say("Step A", &step_a_bits);
    let (blocked, block_bits) = fit_blocks(&coded, step_a.clone(), false)?;
    say("blocks", &block_bits);
    let report = json!({
        "observations": observations,
        "sequences": [first, first + sequences],
        "points": [point("given sets", &given_bits), point("Step A", &step_a_bits), point("blocks", &block_bits)],
        "block_ranks": ranks(&coded, &blocked),
        "merged": merged(&coded, &blocked),
    });
    std::fs::write(out, serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    gam_mpd::engine::log_to_stderr();
    let args: Vec<String> = std::env::args().collect();
    let number = |i: usize, what: &str| -> Result<f64, String> { args.get(i).ok_or(format!("missing {what}"))?.parse::<f64>().map_err(|e| format!("{what}: {e}")) };
    match args.get(1).map(String::as_str) {
        Some("modadd") if args.len() >= 5 => {
            let names = args.get(5).map(|s| s.split(',').map(str::to_string).collect());
            modadd(Path::new(&args[2]), Path::new(&args[3]), number(4, "OBSERVATIONS")?, names)
        }
        Some("vpd") if args.len() >= 9 => vpd(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
            number(6, "OBSERVATIONS")?,
            number(7, "FIRST")? as usize,
            number(8, "SEQUENCES")? as usize,
            args.get(9).map(|s| s.parse::<f64>().map_err(|e| e.to_string())).transpose()?,
        ),
        _ => Err("mpd_blocks_2951 modadd EXPORT_DIR OUT.json OBSERVATIONS [SITES] | vpd EXPORT_DIR LIBRARY_DIR SETS_DIR OUT.json OBSERVATIONS FIRST SEQUENCES [LIBRARY_ROWS]".to_string()),
    }
}
