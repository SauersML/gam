//! Behaviour discovery on the VPD 4L Pile target (#2951): the family's rows partitioned into
//! behaviours by `behaviors::discover`, each behaviour explained by its own table program.
//!
//! Usage: `mpd_behaviors_vpd_2951 TOKENIZER_JSON HARVEST_DIR [ROWS...]`: one discovery per prefix
//! of `ROWS` rows (whole documents first), and the n-frontier of the code against the data size.
//!
//! # The family and the reference
//!
//! A row is a (document, position `t ≥ 2`) of Pile validation text; the model's next-token
//! distribution `p_r` there is the behaviour to explain. The harvest (a forward pass, outside the
//! crate) writes per row the tokens `x_t, x_{t−1}, x_{t−2}`, the 64 most probable next tokens with
//! their probabilities and the mass outside them, the entropy `H(p_r)`, the cross-entropy against
//! `u` (the family's mean next-token distribution) and each layer-head's argmax key offset.
//!
//! # The programs
//!
//! The target is not yet an operator program here, so a group's subprogram is a table of the
//! model's behaviour on the group: `q = softmax(log û + δ_c)`, reading one declared feature (none,
//! `x_t` or `x_{t−1}`) whose value selects the cell `c`, with `δ_c` supported on a set `J_c` of
//! next tokens. It is the operator program `Readout(Affine(Feature, D) + log û)` with `D`'s present
//! blocks the cells' supports. Its message: the feature read (a fixed index of three); per cell
//! (the values of the feature on the group's rows, which the decoder knows), `J_c` in the
//! enumerative subset code over the vocabulary, the lattice's precision and each `δ_j`'s lattice
//! index in the signed prefix code. `log û`, the base, is one operator on its own lattice, shared
//! by every group through the library.
//!
//! A cell's fit: `δ_j = log(q̄_j/u_j)` on the lattice, `q̄` the cell's mean of the stored
//! probabilities, and `J_c` the prefix of the tokens ordered by `q̄_j log(q̄_j/u_j)` whose total
//! (support, precision and reals against the rows' data bits) is shortest, over every prefix and
//! every precision. Row `r`'s data bits are `KL(p_r ‖ q)/ln 2 = KL(p_r ‖ u)/ln 2 − (Σ_{j∈J} p_rj δ_j
//! − ln Z_c)/ln 2`, with `p_rj` the stored probability; a token of `J` outside the row's stored 64
//! is counted at zero, so the screened figure is never below the true one, and the certificate
//! bounds the difference by the row's tail mass.
//!
//! # Seeds, recognizer, report
//!
//! The recognizer reads `x_t`, `x_{t−1}`, `x_{t−2}`. The seeds are every `feature == value` atom,
//! ordered by the number of the 24 heads whose argmax key offset is one value on every row of the
//! atom, then by size. Per behaviour the report prints its rule, rows, the feature its program
//! reads, its heaviest cells' supports decoded to text, algorithm and data bits, the certified
//! band, argmax agreement and the fraction of its rows on which each head attends to `t − 1`. Per
//! data size it prints the model's bits against the family's data bits.

use gam_mpd::behaviors::{
    BehaviorError, GroupFitter, ProgramParts, RowFeature, Seed, Within, discover,
};
use gam_mpd::codec::{fixed_index_len_bits, signed_prefix_integer_len_bits, subset_code_len_bits};
use statrs::function::gamma::ln_gamma;
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::LN_2;
use std::path::Path;
use std::rc::Rc;

/// The reads a table may make: none, `x_t`, `x_{t−1}`.
const READS: [&str; 3] = ["nothing", "x_t", "x_{t-1}"];
/// The lattice precisions a cell's reals are tried at, `2^−b`.
const PRECISIONS: std::ops::RangeInclusive<i32> = -2..=12;
/// Reals of the 24 sites of the target at 2 bits each, from the VPD bits table.
const TARGET_SITE_BITS_AT_2: f64 = 148_192_280.0;

struct Harvest {
    rows: usize,
    top: usize,
    vocab: usize,
    heads: usize,
    feat: Vec<[u32; 3]>,
    top_ids: Vec<u32>,
    top_p: Vec<f32>,
    tail: Vec<f64>,
    /// `KL(p_r ‖ u)/ln 2`.
    base_bits: Vec<f64>,
    head_offsets: Vec<u8>,
    base: Vec<f64>,
}

/// The first `count` records of `width` bytes of `name.bin`, which must hold at least that many.
fn read_bytes(dir: &Path, name: &str, count: usize, width: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let path = dir.join(format!("{name}.bin"));
    let mut bytes = Vec::with_capacity(count * width);
    std::fs::File::open(&path)
        .and_then(|file| file.take((count * width) as u64).read_to_end(&mut bytes))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != count * width {
        return Err(format!("{}: {} bytes, expected {}", path.display(), bytes.len(), count * width));
    }
    Ok(bytes)
}

fn read_f64(dir: &Path, name: &str, count: usize) -> Result<Vec<f64>, String> {
    Ok(read_bytes(dir, name, count, 8)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect())
}

fn read_f32(dir: &Path, name: &str, count: usize) -> Result<Vec<f32>, String> {
    Ok(read_bytes(dir, name, count, 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4]))).collect())
}

fn read_u32(dir: &Path, name: &str, count: usize) -> Result<Vec<u32>, String> {
    Ok(read_bytes(dir, name, count, 4)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap_or([0; 4]))).collect())
}

fn meta_field(meta: &serde_json::Value, key: &str) -> Result<usize, String> {
    meta.get(key).and_then(serde_json::Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("meta.json lacks {key}"))
}

impl Harvest {
    /// The harvest's rows, or its first `limit` (a prefix of whole documents when a multiple of
    /// the rows per document).
    fn load(dir: &Path, limit: Option<usize>) -> Result<Self, String> {
        let meta: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("meta.json")).map_err(|e| format!("{}: {e}", dir.display()))?,
        )
        .map_err(|e| e.to_string())?;
        let (all, top, vocab, heads) =
            (meta_field(&meta, "rows")?, meta_field(&meta, "top")?, meta_field(&meta, "vocab")?, meta_field(&meta, "heads")?);
        let rows = limit.map_or(all, |n| n.min(all));
        let flat = read_u32(dir, "feat", rows * 3)?;
        let feat = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let ent = read_f64(dir, "ent", rows)?;
        let xent = read_f64(dir, "xent", rows)?;
        let base_bits = xent.iter().zip(&ent).map(|(x, h)| (x - h).max(0.0) / LN_2).collect();
        Ok(Self {
            rows,
            top,
            vocab,
            heads,
            feat,
            top_ids: read_u32(dir, "top_ids", rows * top)?,
            top_p: read_f32(dir, "top_p", rows * top)?,
            tail: read_f64(dir, "tail", rows)?,
            base_bits,
            head_offsets: read_bytes(dir, "heads", rows * heads, 1)?,
            base: read_f64(dir, "base", vocab)?,
        })
    }

    fn top_of(&self, row: usize) -> (&[u32], &[f32]) {
        let range = row * self.top..(row + 1) * self.top;
        (&self.top_ids[range.clone()], &self.top_p[range])
    }

    /// The cell a row falls in under `reads`.
    fn cell(&self, reads: usize, row: usize) -> u32 {
        match reads {
            0 => 0,
            other => self.feat[row][other - 1],
        }
    }
}

// ------------------------------------------------------------------------------------------ cells

/// One fitted cell: its support, reals and code.
struct CellCode {
    deltas: HashMap<u32, f64>,
    log_z: f64,
    bits: u64,
    /// `Σ_rows (Σ_J p δ − ln Z)/ln 2` over the cell's rows.
    gain_bits: f64,
    /// The cell's argmax class.
    argmax: u32,
    largest_delta: f64,
}

/// A cell's sufficient statistics and fit.
struct Cell {
    n: usize,
    /// Summed stored probabilities per class.
    sums: HashMap<u32, f64>,
    code: CellCode,
}

struct Base {
    log_u: Vec<f64>,
    argmax: u32,
    bits: u64,
    /// The whole family's extra data bits from rounding `log u` to its lattice.
    rounding_bits: f64,
}

fn screened_subset_bits(universe: usize, size: usize) -> f64 {
    let log_binomial = ln_gamma(universe as f64 + 1.0) - ln_gamma(size as f64 + 1.0) - ln_gamma((universe - size) as f64 + 1.0);
    // L_int(k + 1) is at most 2 log2(k + 2) + 1 bits; the screen ranks, the exact code is reported.
    2.0 * ((size + 2) as f64).log2() + 1.0 + log_binomial / LN_2
}

fn lattice(value: f64, precision: i32) -> i64 {
    (value * 2f64.powi(precision)).round() as i64
}

fn fit_cell(n: usize, sums: &HashMap<u32, f64>, base: &Base, vocab: usize) -> Result<CellCode, String> {
    let nf = n as f64;
    let mut candidates: Vec<(f64, u32, f64, f64)> = sums
        .iter()
        .filter_map(|(&j, &s)| {
            let q = s / nf;
            let log_ratio = q.ln() - base.log_u[j as usize];
            (log_ratio > 0.0).then(|| (q * log_ratio, j, q, log_ratio))
        })
        .collect();
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let empty = subset_code_len_bits(vocab, 0).map_err(|e| format!("{e:?}"))? as f64;
    // (total, m, precision)
    let mut best: (f64, usize, i32) = (empty, 0, 0);
    for precision in PRECISIONS {
        let step = 2f64.powi(-precision);
        let field = signed_prefix_integer_len_bits(i64::from(precision)).map_err(|e| format!("{e:?}"))? as f64;
        let (mut reals, mut q_delta, mut u_sum, mut w_sum) = (0.0, 0.0, 0.0, 0.0);
        for (m, &(_, j, q, log_ratio)) in candidates.iter().enumerate() {
            let index = lattice(log_ratio, precision);
            let delta = index as f64 * step;
            reals += signed_prefix_integer_len_bits(index).map_err(|e| format!("{e:?}"))? as f64;
            let u = base.log_u[j as usize].exp();
            q_delta += q * delta;
            u_sum += u;
            w_sum += u * delta.exp();
            let log_z = (1.0 - u_sum + w_sum).ln();
            let gain = nf * (q_delta - log_z) / LN_2;
            let total = screened_subset_bits(vocab, m + 1) + field + reals - gain;
            if total < best.0 {
                best = (total, m + 1, precision);
            }
        }
    }
    let (_, m, precision) = best;
    let step = 2f64.powi(-precision);
    let mut deltas = HashMap::with_capacity(m);
    let mut bits = subset_code_len_bits(vocab, m).map_err(|e| format!("{e:?}"))?;
    if m > 0 {
        bits += signed_prefix_integer_len_bits(i64::from(precision)).map_err(|e| format!("{e:?}"))?;
    }
    let (mut q_delta, mut u_sum, mut w_sum, mut largest) = (0.0, 0.0, 0.0, 0.0_f64);
    for &(_, j, q, log_ratio) in &candidates[..m] {
        let index = lattice(log_ratio, precision);
        bits += signed_prefix_integer_len_bits(index).map_err(|e| format!("{e:?}"))?;
        let delta = index as f64 * step;
        deltas.insert(j, delta);
        let u = base.log_u[j as usize].exp();
        q_delta += q * delta;
        u_sum += u;
        w_sum += u * delta.exp();
        largest = largest.max(delta.abs());
    }
    let log_z = (1.0 - u_sum + w_sum).ln();
    let mut argmax = (base.argmax, base.log_u[base.argmax as usize]);
    for (&j, &delta) in &deltas {
        let score = base.log_u[j as usize] + delta;
        if score > argmax.1 {
            argmax = (j, score);
        }
    }
    Ok(CellCode { deltas, log_z, bits, gain_bits: nf * (q_delta - log_z) / LN_2, argmax: argmax.0, largest_delta: largest })
}

// ---------------------------------------------------------------------------------------- tables

/// A group's statistics under one read.
#[derive(Clone)]
struct ReadTable {
    cells: BTreeMap<u32, Rc<Cell>>,
    /// Program bits of the cells.
    bits: u64,
    gain_bits: f64,
}

/// A group's statistics under every read.
struct Stats {
    tables: Vec<ReadTable>,
    /// `Σ_rows KL(p_r ‖ u)/ln 2`.
    base_bits: f64,
}

#[derive(Clone)]
struct Table {
    reads: usize,
    stats: Rc<Stats>,
    id: usize,
}

struct TableFitter<'a> {
    harvest: &'a Harvest,
    base: &'a Base,
    next_id: usize,
}

impl TableFitter<'_> {
    fn read_bits() -> u64 {
        u64::from(fixed_index_len_bits(READS.len()).unwrap_or(2))
    }

    fn table(&self, cells: BTreeMap<u32, Rc<Cell>>) -> ReadTable {
        let bits = cells.values().map(|c| c.code.bits).sum();
        let gain_bits = cells.values().map(|c| c.code.gain_bits).sum();
        ReadTable { cells, bits, gain_bits }
    }

    fn accumulate(&self, reads: usize, rows: &[usize], sign: f64, into: &mut HashMap<u32, (i64, HashMap<u32, f64>)>) {
        for &row in rows {
            let entry = into.entry(self.harvest.cell(reads, row)).or_default();
            entry.0 += sign as i64;
            let (ids, ps) = self.harvest.top_of(row);
            for (&j, &p) in ids.iter().zip(ps) {
                *entry.1.entry(j).or_insert(0.0) += sign * f64::from(p);
            }
        }
    }

    fn choose(&self, stats: &Stats, prefer: Option<usize>) -> usize {
        let cost = |r: usize| stats.tables[r].bits as f64 - stats.tables[r].gain_bits;
        let mut order: Vec<usize> = (0..READS.len()).collect();
        if let Some(p) = prefer {
            order.retain(|r| *r != p);
            order.insert(0, p);
        }
        let mut best = order[0];
        for &r in &order {
            if cost(r) < cost(best) {
                best = r;
            }
        }
        best
    }

    fn data_bits(stats: &Stats, reads: usize) -> f64 {
        stats.base_bits - stats.tables[reads].gain_bits
    }
}

impl GroupFitter for TableFitter<'_> {
    type Program = Table;
    type Certificate = TableCertificate;

    fn rows(&self) -> usize {
        self.harvest.rows
    }

    fn fit(&mut self, rows: &[usize], within: Option<Within<'_, Table>>) -> Result<(Table, f64), BehaviorError> {
        let (stats, prefer) = match within {
            Some(parent) => {
                let mut tables = Vec::with_capacity(READS.len());
                for reads in 0..READS.len() {
                    let mut delta: HashMap<u32, (i64, HashMap<u32, f64>)> = HashMap::new();
                    self.accumulate(reads, parent.removed, -1.0, &mut delta);
                    let mut cells = parent.program.stats.tables[reads].cells.clone();
                    for (key, (dn, dsums)) in delta {
                        let Some(old) = cells.get(&key) else {
                            return Err(BehaviorError::Fit(format!("removed rows fall in cell {key} the group does not have")));
                        };
                        let n = (old.n as i64 + dn) as usize;
                        if n == 0 {
                            cells.remove(&key);
                            continue;
                        }
                        let mut sums = old.sums.clone();
                        for (j, s) in dsums {
                            *sums.entry(j).or_insert(0.0) += s;
                        }
                        sums.retain(|_, s| *s > 0.0);
                        let code = fit_cell(n, &sums, self.base, self.harvest.vocab).map_err(BehaviorError::Fit)?;
                        cells.insert(key, Rc::new(Cell { n, sums, code }));
                    }
                    tables.push(self.table(cells));
                }
                let removed: f64 = parent.removed.iter().map(|r| self.harvest.base_bits[*r]).sum();
                (Stats { tables, base_bits: parent.program.stats.base_bits - removed }, Some(parent.program.reads))
            }
            None => {
                let mut tables = Vec::with_capacity(READS.len());
                for reads in 0..READS.len() {
                    let mut acc: HashMap<u32, (i64, HashMap<u32, f64>)> = HashMap::new();
                    self.accumulate(reads, rows, 1.0, &mut acc);
                    let mut cells = BTreeMap::new();
                    for (key, (n, sums)) in acc {
                        let code = fit_cell(n as usize, &sums, self.base, self.harvest.vocab).map_err(BehaviorError::Fit)?;
                        cells.insert(key, Rc::new(Cell { n: n as usize, sums, code }));
                    }
                    tables.push(self.table(cells));
                }
                (Stats { tables, base_bits: rows.iter().map(|r| self.harvest.base_bits[*r]).sum() }, None)
            }
        };
        let reads = self.choose(&stats, prefer);
        let data = TableFitter::data_bits(&stats, reads);
        self.next_id += 1;
        Ok((Table { reads, stats: Rc::new(stats), id: self.next_id }, data))
    }

    fn row_bits(&mut self, program: &Table) -> Result<Vec<f64>, BehaviorError> {
        let cells = &program.stats.tables[program.reads].cells;
        Ok((0..self.harvest.rows)
            .map(|row| {
                let Some(cell) = cells.get(&self.harvest.cell(program.reads, row)) else {
                    return self.harvest.base_bits[row];
                };
                let (ids, ps) = self.harvest.top_of(row);
                let dot: f64 = ids.iter().zip(ps).filter_map(|(j, p)| cell.code.deltas.get(j).map(|d| f64::from(*p) * d)).sum();
                (self.harvest.base_bits[row] - (dot - cell.code.log_z) / LN_2).max(0.0)
            })
            .collect())
    }

    fn parts(&self, program: &Table) -> Result<ProgramParts, BehaviorError> {
        let own = TableFitter::read_bits() + program.stats.tables[program.reads].bits;
        Ok(ProgramParts {
            bits: self.base.bits + own,
            operators: vec![("base".to_string(), self.base.bits), (format!("table {}", program.id), own)],
        })
    }

    fn certify(&mut self, program: &Table, rows: &[usize]) -> Result<(TableCertificate, f64), BehaviorError> {
        let cells = &program.stats.tables[program.reads].cells;
        let (mut data, mut band, mut agree) = (0.0, 0.0, 0usize);
        let mut attends_previous = vec![0usize; self.harvest.heads];
        for &row in rows {
            let (ids, ps) = self.harvest.top_of(row);
            let (bits, row_band, argmax) = match cells.get(&self.harvest.cell(program.reads, row)) {
                None => (self.harvest.base_bits[row], 0.0, self.base.argmax),
                Some(cell) => {
                    let dot: f64 = ids.iter().zip(ps).filter_map(|(j, p)| cell.code.deltas.get(j).map(|d| f64::from(*p) * d)).sum();
                    let stored = cell.code.deltas.keys().filter(|j| ids.contains(j)).count();
                    let missing = cell.code.deltas.len() - stored;
                    // A token of J outside the stored 64 has probability at most the 64th and at
                    // most the tail mass in all.
                    let smallest = ps.last().copied().map_or(0.0, f64::from);
                    let bound = (missing as f64 * smallest).min(self.harvest.tail[row]) * cell.code.largest_delta / LN_2;
                    (self.harvest.base_bits[row] - (dot - cell.code.log_z) / LN_2, bound, cell.code.argmax)
                }
            };
            data += bits;
            band += row_band;
            agree += usize::from(argmax == ids[0]);
            for (h, count) in attends_previous.iter_mut().enumerate() {
                *count += usize::from(self.harvest.head_offsets[row * self.harvest.heads + h] == 1);
            }
        }
        Ok((
            TableCertificate {
                rows: rows.len(),
                band_bits: band,
                argmax_agreement: agree as f64 / rows.len() as f64,
                attends_previous: attends_previous.iter().map(|c| *c as f64 / rows.len() as f64).collect(),
            },
            data,
        ))
    }
}

/// A behaviour's certified fidelity.
#[derive(Clone, Debug)]
struct TableCertificate {
    rows: usize,
    /// The data bits are exact up to this many bits (tokens of a support outside a row's stored 64).
    band_bits: f64,
    argmax_agreement: f64,
    /// Per head, the fraction of the rows on which it attends to `t − 1`.
    attends_previous: Vec<f64>,
}

// ------------------------------------------------------------------------------------------ base

fn fit_base(harvest: &Harvest) -> Result<Base, String> {
    let n = harvest.rows as f64;
    let log_u: Vec<f64> = harvest.base.iter().map(|u| u.max(f64::MIN_POSITIVE).ln()).collect();
    let mut best: Option<(f64, i32)> = None;
    for precision in PRECISIONS {
        let step = 2f64.powi(-precision);
        let mut reals = signed_prefix_integer_len_bits(i64::from(precision)).map_err(|e| format!("{e:?}"))? as f64;
        let (mut cross, mut z) = (0.0, 0.0);
        for (u, lu) in harvest.base.iter().zip(&log_u) {
            let index = lattice(*lu, precision);
            reals += signed_prefix_integer_len_bits(index).map_err(|e| format!("{e:?}"))? as f64;
            let rounded = index as f64 * step;
            cross += u * (lu - rounded);
            z += rounded.exp();
        }
        let total = reals + n * (cross + z.ln()) / LN_2;
        if best.is_none_or(|(b, _)| total < b) {
            best = Some((total, precision));
        }
    }
    let precision = best.map_or(0, |(_, p)| p);
    let step = 2f64.powi(-precision);
    let mut bits = signed_prefix_integer_len_bits(i64::from(precision)).map_err(|e| format!("{e:?}"))?;
    let (mut cross, mut z) = (0.0, 0.0);
    let mut rounded_log = Vec::with_capacity(log_u.len());
    for (u, lu) in harvest.base.iter().zip(&log_u) {
        let index = lattice(*lu, precision);
        bits += signed_prefix_integer_len_bits(index).map_err(|e| format!("{e:?}"))?;
        let rounded = index as f64 * step;
        cross += u * (lu - rounded);
        z += rounded.exp();
        rounded_log.push(rounded);
    }
    let log_z = z.ln();
    let argmax = (0..harvest.vocab).max_by(|a, b| log_u[*a].total_cmp(&log_u[*b])).unwrap_or(0) as u32;
    // The tables are fitted against the exact `u` the harvest's cross-entropies use; the rounding's
    // data bits are the same for every partition and reported once.
    Ok(Base { log_u, argmax, bits, rounding_bits: n * (cross + log_z) / LN_2 })
}

// ---------------------------------------------------------------------------------------- tokens

fn load_vocabulary(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let vocab = json.pointer("/model/vocab").and_then(serde_json::Value::as_object).ok_or("tokenizer.json lacks model.vocab")?;
    let size = vocab.values().filter_map(serde_json::Value::as_u64).max().unwrap_or(0) as usize + 1;
    let mut out = vec![String::new(); size];
    for (token, id) in vocab {
        if let Some(id) = id.as_u64() {
            out[id as usize] = token.replace('Ġ', " ").replace('Ċ', "\\n");
        }
    }
    Ok(out)
}

fn token(vocabulary: &[String], id: u32) -> String {
    vocabulary.get(id as usize).map_or_else(|| format!("#{id}"), |t| format!("{t:?}"))
}

// ------------------------------------------------------------------------------------------- run

fn seeds(harvest: &Harvest, features: &[RowFeature]) -> Vec<Seed> {
    let mut out: Vec<(usize, Seed)> = Vec::new();
    for feature in features {
        let mut by_value: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (row, value) in feature.values.iter().enumerate() {
            by_value.entry(*value).or_default().push(row);
        }
        for (value, rows) in by_value {
            if rows.len() < 2 || rows.len() == harvest.rows {
                continue;
            }
            let fixed = (0..harvest.heads)
                .filter(|h| {
                    let first = harvest.head_offsets[rows[0] * harvest.heads + h];
                    rows.iter().all(|r| harvest.head_offsets[r * harvest.heads + h] == first)
                })
                .count();
            out.push((fixed, Seed { rows, description: format!("{} = {value}", feature.name) }));
        }
    }
    out.sort_by(|(fa, a), (fb, b)| fb.cmp(fa).then(b.rows.len().cmp(&a.rows.len())).then(a.description.cmp(&b.description)));
    out.into_iter().map(|(_, seed)| seed).collect()
}

/// One point of the n-frontier.
struct Point {
    rows: usize,
    family_bits: f64,
    one_group: f64,
    discovered: f64,
    behaviours: usize,
}

fn run(dir: &Path, limit: Option<usize>, vocabulary: &[String]) -> Result<Point, String> {
    let harvest = Harvest::load(dir, limit)?;
    let base = fit_base(&harvest)?;
    let features: Vec<RowFeature> = ["x_t", "x_{t-1}", "x_{t-2}"]
        .iter()
        .enumerate()
        .map(|(i, name)| RowFeature {
            name: name.to_string(),
            alphabet: harvest.vocab,
            values: harvest.feat.iter().map(|f| f[i]).collect(),
            definition_bits: None,
        })
        .collect();
    let seeds = seeds(&harvest, &features);
    let family_bits: f64 = harvest.base_bits.iter().sum();
    println!("== {} : {} rows, {} seeds", dir.display(), harvest.rows, seeds.len());
    println!(
        "   data: Σ KL(p‖u) = {:.0} bits ({:.3} per row); the target's 24 sites at 2 bits/real = {:.0} bits, {:.1}× the data",
        family_bits,
        family_bits / harvest.rows as f64,
        TARGET_SITE_BITS_AT_2,
        TARGET_SITE_BITS_AT_2 / family_bits
    );
    println!("   base log u: {} bits; its rounding costs {:.1} data bits on every partition", base.bits, base.rounding_bits);
    let mut fitter = TableFitter { harvest: &harvest, base: &base, next_id: 0 };
    let discovery = discover(&mut fitter, &features, &seeds).map_err(|e| e.to_string())?;
    println!(
        "   one behaviour: {:.0} bits; discovered: {:.0} bits in {} behaviours ({} fits, {} seeds evaluated)",
        discovery.one_group.total(),
        discovery.code.total(),
        discovery.behaviours.len(),
        discovery.fits,
        discovery.evaluated_seeds
    );
    println!(
        "   code: count {} + library {} + programs {} + partition {:.0} + data {:.0}",
        discovery.code.count_bits,
        discovery.code.library_bits,
        discovery.code.program_bits.iter().sum::<u64>(),
        discovery.code.partition_bits,
        discovery.code.data_bits.iter().sum::<f64>()
    );
    let mut order: Vec<usize> = (0..discovery.behaviours.len()).collect();
    order.sort_by(|a, b| discovery.behaviours[*b].rows.len().cmp(&discovery.behaviours[*a].rows.len()));
    for k in order {
        let b = &discovery.behaviours[k];
        let rule: Vec<String> = b
            .rule
            .iter()
            .map(|path| {
                let tests: Vec<String> = path
                    .tests
                    .iter()
                    .map(|t| format!("{} {} {}", t.feature, if t.equal { "=" } else { "≠" }, token(vocabulary, t.value)))
                    .collect();
                format!("[{}] ({}/{} rows)", if tests.is_empty() { "always".to_string() } else { tests.join(" ∧ ") }, path.own_rows, path.leaf_rows)
            })
            .collect();
        let table = &b.program.stats.tables[b.program.reads];
        let mut heavy: Vec<(&u32, &Rc<Cell>)> = table.cells.iter().filter(|(_, c)| !c.code.deltas.is_empty()).collect();
        heavy.sort_by(|a, b| b.1.code.gain_bits.total_cmp(&a.1.code.gain_bits));
        let prev_heads: Vec<String> = b
            .certificate
            .attends_previous
            .iter()
            .enumerate()
            .filter(|(_, f)| **f >= 0.5)
            .map(|(h, f)| format!("L{}H{} {:.2}", h / 6, h % 6, f))
            .collect();
        println!(
            "-- behaviour {k}{}: {} rows; rule {}; reads {}; program {} bits ({} own), data {:.0} ± {:.1} bits ({:.3}/row); argmax agreement {:.3}",
            if b.general { " (general)" } else { "" },
            b.certificate.rows,
            rule.join(" ∨ "),
            READS[b.program.reads],
            b.program_bits,
            b.own_program_bits,
            b.data_bits,
            b.certificate.band_bits,
            b.data_bits / b.certificate.rows as f64,
            b.certificate.argmax_agreement
        );
        println!("   heads attending t-1 on ≥ half its rows: {}", if prev_heads.is_empty() { "none".to_string() } else { prev_heads.join(", ") });
        for (value, cell) in heavy.iter().take(6) {
            let mut support: Vec<(&u32, &f64)> = cell.code.deltas.iter().collect();
            support.sort_by(|a, b| b.1.total_cmp(a.1));
            let shown: Vec<String> = support.iter().take(8).map(|(j, d)| format!("{} {:+.2}", token(vocabulary, **j), d)).collect();
            println!(
                "   cell {} = {} ({} rows, {} bits, saves {:.0} bits): {}{}",
                READS[b.program.reads],
                if b.program.reads == 0 { "·".to_string() } else { token(vocabulary, **value) },
                cell.n,
                cell.code.bits,
                cell.code.gain_bits,
                shown.join(", "),
                if support.len() > 8 { format!(" … {} tokens", support.len()) } else { String::new() }
            );
        }
    }
    for line in &discovery.moves {
        println!("   move: {line}");
    }
    Ok(Point {
        rows: harvest.rows,
        family_bits,
        one_group: discovery.one_group.total(),
        discovered: discovery.code.total(),
        behaviours: discovery.behaviours.len(),
    })
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [tokenizer, dir, sizes @ ..] = args.as_slice() else {
        return Err("usage: mpd_behaviors_vpd_2951 TOKENIZER_JSON HARVEST_DIR [ROWS...]".to_string());
    };
    let vocabulary = load_vocabulary(Path::new(tokenizer))?;
    let mut limits: Vec<Option<usize>> =
        sizes.iter().map(|n| n.parse().map(Some).map_err(|e| format!("{n}: {e}"))).collect::<Result<_, _>>()?;
    if limits.is_empty() {
        limits.push(None);
    }
    let mut frontier = Vec::new();
    for limit in limits {
        frontier.push(run(Path::new(dir), limit, &vocabulary)?);
    }
    println!("== n-frontier (bits; the target's reals alone {TARGET_SITE_BITS_AT_2:.3e})");
    println!("   {:>9} {:>12} {:>12} {:>12} {:>3} {:>9} {:>11}", "rows", "Σ KL(p‖u)", "one group", "discovered", "K", "bits/row", "target/data");
    for p in &frontier {
        println!(
            "   {:>9} {:>12.0} {:>12.0} {:>12.0} {:>3} {:>9.4} {:>11.1}",
            p.rows,
            p.family_bits,
            p.one_group,
            p.discovered,
            p.behaviours,
            p.discovered / p.rows as f64,
            TARGET_SITE_BITS_AT_2 / p.family_bits
        );
    }
    Ok(())
}
