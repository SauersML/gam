//! Causal-state machines of a model's behaviour (#2951).
//!
//! # The object
//!
//! A behaviour is a readout over token positions: after each prefix `x_1..x_t` of a
//! sequence the model emits a categorical distribution `p_t` over `R` readout classes
//! (the next token restricted to the behaviour's classes, or the whole next-token
//! distribution of a small vocabulary). Two prefixes are *causally equivalent* when every
//! continuation yields the same future readouts. The classes of that relation are the
//! causal states of computational mechanics (Crutchfield and Shalizi), and the minimal
//! deterministic machine that carries them forward is the behaviour's ε-machine. This
//! module extracts that machine from a model as an executable, decodable program and
//! selects it by its two-part code.
//!
//! # The machine language
//!
//! A [`Machine`] reads tokens through a [`SymbolMap`] (tokens not listed map to symbol
//! `0`) and carries a state of three kinds. Every kind competes under one code, so none
//! of them is assumed:
//!
//! * a finite class `q ∈ {0..K}` with an arbitrary table `q ← δ(q, s)`: any finite
//!   deterministic machine;
//! * counters `x_c ← x_c + d_c(s)`, integer increments per symbol, unbounded or modulo a
//!   period `p_c` (a rotation on a cycle of `p_c` points);
//! * retrieve registers: register `r` over counter `c` with write symbols `W_r` holds
//!   the symbol of the most recent position that read a symbol of `W_r` and left
//!   counter `c` at the value it has now. This is a keyed memory (the top of a stack
//!   indexed by depth), which a transformer realizes with one attention law whose query
//!   and key are the counter.
//!
//! The readout is a table of logit vectors over cells `(q, clamp_c(x_c), register
//! values)`. An unbounded counter enters the readout clamped to `[lo_c, hi_c]`, a
//! periodic one by its residue. Only cells that occur carry a vector; any other cell
//! reads the uniform distribution.
//!
//! # The code
//!
//! A machine is one self-delimiting message through `codec`'s integer codes
//! ([`Machine::encode`], [`Machine::decode`]). The decoder knows the vocabulary size and
//! the readout width `R`, which are declarations of the harvest. The message holds:
//!
//! 1. the symbol count `S` in the prefix code, then the listed tokens as one subset of
//!    the vocabulary and each listed token's symbol as a fixed index;
//! 2. `K` in the prefix code, the initial class and the `K·S` table as fixed indices;
//! 3. the counter count plus one, each counter's period plus one (`1` for none), and its
//!    increments: a fixed index modulo the period, or a signed prefix integer;
//! 4. the register count plus one, each register's counter as a fixed index and its
//!    write symbols as a subset;
//! 5. the readout: each unbounded counter's `lo` (signed) and `hi − lo + 1`, the present
//!    cells as a subset of the cell universe, the lattice's fraction bits (signed), and per
//!    present cell its reference class as a fixed index and every other class's logit
//!    relative to the reference as a lattice index `k ≤ 0`, sent as `1 − k`.
//!
//! The reference is the cell's most probable class, so every relative logit is at most
//! zero and needs no sign.
//!
//! # The score
//!
//! As in the engine's contract, a machine `M` is scored by
//!
//! ```text
//! L_total(M) = L(M) + n Σ_rows KL(p_row ‖ q_M(row)) / ln 2,
//! ```
//!
//! the decoded message length plus the expected excess code length of `n` observations
//! of every row coded with the machine's distribution. There is no tolerance and no
//! declared precision: the lattice step of the readout, the clamps and the machine's
//! size are all chosen by minimizing `L_total`. For a fixed cell assignment, the
//! distribution that minimizes `Σ KL(p_row ‖ q)` over a cell's rows is their mean, so each
//! cell's vector is the log of its mean before rounding to the lattice.
//!
//! # Search
//!
//! [`search`] is greedy best-improvement over a fixed family of structural moves, each
//! accepted only when it strictly lowers `L_total` of the refitted machine: split a class
//! by a partition of its incoming edges (a Bregman 2-means over the edges' mean
//! readouts, and each edge alone; the split separates exactly the class whose incoming
//! histories disagree), redirect one edge, merge two classes, merge two
//! symbols, add a counter (every increment vector over the symbols in `{−1, 0, 1}`, each
//! unbounded and at every period up to the longest sequence), change one increment,
//! remove a counter, and add or remove a register. A
//! [`Language`] restricts the kinds, so the best machine of every kind is found by the
//! same search and compared on one code. [`unroll`] turns any register-free machine into
//! the finite machine of its reachable states on the harvest, which is the pure finite
//! competitor of the same behaviour at the same readout.
//!
//! # Quotient consistency and refinement
//!
//! A harvest samples prefixes; the causal-state relation is about every continuation.
//! [`consistency`] measures how far the model's readout is from factoring through a
//! machine's cells on any weighted harvest, in particular one extended by [`Branch`]es,
//! one-symbol continuations the harvest never sampled ([`Harvest::with_branches`] appends
//! each as its prefix at weight `0` and its continuation at weight `1`). [`refine`] is
//! counterexample-guided abstraction refinement over a pool of branches: a branch is
//! admitted when a readout cell of its own would pay for itself under the code, and the
//! search resumes on the harvest with every admitted branch, until the machine is
//! certified on the whole pool.
//!
//! # Interventions
//!
//! A machine's state is a causal claim only if setting it moves the model the way the
//! machine says. [`StateIntervention`] compiles `do(state := s')` into one global edit of a
//! stored writer into the residual stream through the edit compiler
//! (`compile::linear::compile_linear_site`), or returns its infeasibility witness.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Range;

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use super::codec::{
    BitReader, BitString, CodecError, decode_fixed_index, decode_prefix_integer,
    decode_signed_prefix_integer, decode_subset, encode_fixed_index, encode_prefix_integer,
    encode_signed_prefix_integer, encode_subset,
};
use super::compile::linear::{
    EditMetric, LinearSiteProblem, LinearSiteReport, OffTargetInputs, Requirement, ResponseClass, compile_linear_site,
};
use super::lift::{TensorId, TensorRegistry, UseSiteId};
use super::precision::{DeclaredPrecision, LatticeCode};
use gam_runtime::resource::MemoryGovernor;

/// Why a harvest, a machine or a search was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum CausalError {
    /// The harvest's arrays disagree or hold invalid values.
    Harvest(String),
    /// A machine violates its own contract.
    Machine(String),
    /// A code could not be written or read.
    Codec(CodecError),
    /// A lattice code was refused.
    Precision(String),
    /// The edit compiler refused an intervention.
    Compile(String),
}

impl fmt::Display for CausalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Harvest(reason) => write!(formatter, "harvest: {reason}"),
            Self::Machine(reason) => write!(formatter, "machine: {reason}"),
            Self::Codec(error) => write!(formatter, "codec: {error}"),
            Self::Precision(reason) => write!(formatter, "precision: {reason}"),
            Self::Compile(reason) => write!(formatter, "compile: {reason}"),
        }
    }
}

impl std::error::Error for CausalError {}

impl From<CodecError> for CausalError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

/// The model's behaviour on a finite family of token sequences.
#[derive(Clone, Debug)]
pub struct Harvest {
    vocabulary: usize,
    observations: u64,
    tokens: Vec<u32>,
    starts: Vec<usize>,
    probabilities: Array2<f64>,
    /// Per row, the weight of its observations in the code (`1` for a sampled row, `0` for
    /// a prefix row that only carries the state to a branch).
    weights: Vec<f64>,
    /// `Σ_rows w Σ_j p_j ln p_j`, the rows' weighted negative entropy in nats.
    negentropy: f64,
    longest: usize,
}

impl Harvest {
    /// A harvest of `starts.len() − 1` sequences: sequence `i` is
    /// `tokens[starts[i]..starts[i + 1]]`, and row `r` of `probabilities` (`rows × R`) is the
    /// model's readout after the prefix ending at `tokens[r]`. Each row is divided by its sum;
    /// a row with a negative or non-finite entry, or a zero sum, is refused. `observations` is
    /// the number `n` of observations of each row the code explains.
    pub fn new(
        vocabulary: usize,
        observations: u64,
        tokens: Vec<u32>,
        starts: Vec<usize>,
        probabilities: Array2<f64>,
    ) -> Result<Self, CausalError> {
        let weights = vec![1.0; tokens.len()];
        Self::weighted(vocabulary, observations, tokens, starts, probabilities, weights)
    }

    /// [`Harvest::new`] with a nonnegative finite weight per row: row `r` enters the code as
    /// `w_r · n` observations.
    pub fn weighted(
        vocabulary: usize,
        observations: u64,
        tokens: Vec<u32>,
        starts: Vec<usize>,
        probabilities: Array2<f64>,
        weights: Vec<f64>,
    ) -> Result<Self, CausalError> {
        let rows = tokens.len();
        if weights.len() != rows || weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0) {
            return Err(CausalError::Harvest("one nonnegative finite weight per row".to_string()));
        }
        if probabilities.nrows() != rows {
            return Err(CausalError::Harvest(format!(
                "{} probability rows for {rows} tokens",
                probabilities.nrows()
            )));
        }
        if probabilities.ncols() < 2 {
            return Err(CausalError::Harvest("a readout needs at least two classes".to_string()));
        }
        if observations == 0 {
            return Err(CausalError::Harvest("zero observations per row".to_string()));
        }
        if starts.first() != Some(&0) || starts.last() != Some(&rows) || starts.len() < 2 {
            return Err(CausalError::Harvest("sequence starts must run from 0 to the row count".to_string()));
        }
        if starts.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(CausalError::Harvest("every sequence must be nonempty".to_string()));
        }
        if let Some(token) = tokens.iter().find(|&&token| token as usize >= vocabulary) {
            return Err(CausalError::Harvest(format!("token {token} outside a vocabulary of {vocabulary}")));
        }
        let mut probabilities = probabilities;
        let mut negentropy = 0.0_f64;
        for (row, mut values) in probabilities.rows_mut().into_iter().enumerate() {
            if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
                return Err(CausalError::Harvest(format!("row {row} holds a negative or non-finite probability")));
            }
            let total: f64 = values.sum();
            if total <= 0.0 {
                return Err(CausalError::Harvest(format!("row {row} sums to zero")));
            }
            values.mapv_inplace(|value| value / total);
            negentropy += weights[row]
                * values.iter().filter(|value| **value > 0.0).map(|value| value * value.ln()).sum::<f64>();
        }
        let longest = starts.windows(2).map(|pair| pair[1] - pair[0]).max().unwrap_or(0);
        Ok(Self { vocabulary, observations, tokens, starts, probabilities, weights, negentropy, longest })
    }

    /// Vocabulary size.
    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }

    /// Readout width `R`.
    pub fn readout(&self) -> usize {
        self.probabilities.ncols()
    }

    /// Row count.
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    /// Sequence count.
    pub fn sequences(&self) -> usize {
        self.starts.len() - 1
    }

    /// The rows of sequence `index`.
    pub fn sequence(&self, index: usize) -> Range<usize> {
        self.starts[index]..self.starts[index + 1]
    }

    /// Every token, row-aligned.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The model's readout rows.
    pub fn probabilities(&self) -> ArrayView2<'_, f64> {
        self.probabilities.view()
    }

    /// The longest sequence.
    pub fn longest(&self) -> usize {
        self.longest
    }

    /// The observations per row.
    pub fn observations(&self) -> u64 {
        self.observations
    }

    /// The per-row weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// This harvest with every branch appended as one more sequence: the branch's prefix
    /// rows at weight `0` (they only carry the state) and its continuation row at weight `1`.
    pub fn with_branches(&self, branches: &[Branch]) -> Result<Self, CausalError> {
        let readout = self.readout();
        let mut tokens = self.tokens.clone();
        let mut starts = self.starts.clone();
        let mut weights = self.weights.clone();
        let mut rows: Vec<f64> = self.probabilities.iter().copied().collect();
        for branch in branches {
            if branch.row >= self.rows() || branch.probabilities.len() != readout {
                return Err(CausalError::Harvest(format!("branch at row {} of {}", branch.row, self.rows())));
            }
            let sequence = self.starts.partition_point(|&start| start <= branch.row) - 1;
            for row in self.starts[sequence]..=branch.row {
                tokens.push(self.tokens[row]);
                weights.push(0.0);
                rows.extend(self.probabilities.row(row).iter());
            }
            tokens.push(branch.token);
            weights.push(1.0);
            rows.extend(&branch.probabilities);
            starts.push(tokens.len());
        }
        let probabilities = Array2::from_shape_vec((tokens.len(), readout), rows)
            .map_err(|error| CausalError::Harvest(error.to_string()))?;
        Self::weighted(self.vocabulary, self.observations, tokens, starts, probabilities, weights)
    }

    /// The distinct tokens of the harvest, ascending, with their row counts.
    fn token_counts(&self) -> BTreeMap<u32, usize> {
        let mut counts = BTreeMap::new();
        for (&token, &weight) in self.tokens.iter().zip(&self.weights) {
            if weight > 0.0 {
                *counts.entry(token).or_insert(0) += 1;
            }
        }
        counts
    }
}

/// Tokens to symbols. Symbol `0` is every token not listed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolMap {
    symbols: usize,
    /// Listed tokens, ascending, with their symbols in `1..symbols`.
    members: Vec<(u32, usize)>,
}

impl SymbolMap {
    /// A map of `symbols ≥ 1` symbols listing `members` (token, symbol ≥ 1). Every symbol
    /// above `0` must be used.
    pub fn new(symbols: usize, members: Vec<(u32, usize)>) -> Result<Self, CausalError> {
        if symbols == 0 {
            return Err(CausalError::Machine("a symbol map needs one symbol".to_string()));
        }
        let mut members = members;
        members.sort_unstable();
        if members.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(CausalError::Machine("a token is listed twice".to_string()));
        }
        let mut used = vec![false; symbols];
        for &(token, symbol) in &members {
            if symbol == 0 || symbol >= symbols {
                return Err(CausalError::Machine(format!("token {token} maps to symbol {symbol} of {symbols}")));
            }
            used[symbol] = true;
        }
        if used.iter().skip(1).any(|flag| !flag) {
            return Err(CausalError::Machine("a listed symbol has no token".to_string()));
        }
        Ok(Self { symbols, members })
    }

    /// Each of `tokens` its own symbol, in order from `1`; everything else is symbol `0`.
    pub fn distinct(tokens: &[u32]) -> Result<Self, CausalError> {
        Self::new(tokens.len() + 1, tokens.iter().enumerate().map(|(index, &token)| (token, index + 1)).collect())
    }

    /// Symbol count.
    pub fn symbols(&self) -> usize {
        self.symbols
    }

    /// The listed tokens and their symbols.
    pub fn members(&self) -> &[(u32, usize)] {
        &self.members
    }

    /// The symbol of `token`.
    pub fn symbol(&self, token: u32) -> usize {
        match self.members.binary_search_by_key(&token, |&(listed, _)| listed) {
            Ok(position) => self.members[position].1,
            Err(_) => 0,
        }
    }

    /// The map after folding symbol `gone` into `kept`, with the symbols above `gone` shifted
    /// down by one; returns the map and the old-to-new symbol index.
    fn merge(&self, kept: usize, gone: usize) -> Result<(Self, Vec<usize>), CausalError> {
        let renumber: Vec<usize> = (0..self.symbols)
            .map(|symbol| {
                let target = if symbol == gone { kept } else { symbol };
                if target > gone { target - 1 } else { target }
            })
            .collect();
        let symbols = self.symbols - 1;
        let default = renumber[0];
        // Symbol 0 must stay the unlisted class: when the default folds into another
        // symbol, that symbol becomes the default and its tokens are unlisted.
        let relabel = |symbol: usize| -> usize {
            if default == 0 {
                symbol
            } else if symbol == default {
                0
            } else if symbol == 0 {
                default
            } else {
                symbol
            }
        };
        let final_index: Vec<usize> = renumber.iter().map(|&symbol| relabel(symbol)).collect();
        let members = self
            .members
            .iter()
            .filter_map(|&(token, symbol)| {
                let target = final_index[symbol];
                (target != 0).then_some((token, target))
            })
            .collect();
        Ok((Self::new(symbols, members)?, final_index))
    }
}

/// A counter `x ← x + d(s)`, unbounded or modulo a period.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Counter {
    /// One increment per symbol; in `0..period` for a periodic counter.
    pub increments: Vec<i64>,
    /// The period `p ≥ 2`, or none.
    pub period: Option<u64>,
}

impl Counter {
    fn step(&self, value: i64, symbol: usize) -> i64 {
        let next = value + self.increments[symbol];
        match self.period {
            Some(period) => next.rem_euclid(period as i64),
            None => next,
        }
    }
}

/// A retrieve register: the symbol of the most recent write position whose counter value
/// equals the counter's current value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Register {
    /// The key counter.
    pub counter: usize,
    /// The write symbols, ascending. The register's value is `0` (nothing written at the
    /// current key) or `1 + k` for `writes[k]`.
    pub writes: Vec<usize>,
}

/// Everything of a machine except its readout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Structure {
    pub symbols: SymbolMap,
    pub classes: usize,
    pub initial: usize,
    /// `classes × symbols`, row-major.
    pub table: Vec<usize>,
    pub counters: Vec<Counter>,
    pub registers: Vec<Register>,
}

/// The inclusive readout range of an unbounded counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Clamp {
    pub lo: i64,
    pub hi: i64,
}

/// The readout table.
#[derive(Clone, Debug, PartialEq)]
pub struct Readout {
    /// One entry per counter: the clamp of an unbounded counter, none for a periodic one.
    pub clamps: Vec<Option<Clamp>>,
    /// The cells that carry a vector, ascending.
    pub present: Vec<usize>,
    /// Per present cell, its reference (most probable) class.
    pub references: Vec<usize>,
    /// Per present cell, the `R − 1` non-reference relative logits as lattice indices `≤ 0`,
    /// in class order.
    pub logits: LatticeCode,
}

/// A decodable machine.
#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    pub structure: Structure,
    pub readout: Readout,
}

/// A machine's two-part code on a harvest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    /// The decoded message length.
    pub machine_bits: u64,
    /// `n Σ KL / ln 2`.
    pub data_bits: f64,
}

impl Score {
    /// `machine_bits + data_bits`.
    pub fn total(&self) -> f64 {
        self.machine_bits as f64 + self.data_bits
    }
}

/// Which state kinds a search may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Language {
    pub counters: bool,
    pub registers: bool,
}

impl Language {
    /// Arbitrary finite machines only.
    pub const FINITE: Self = Self { counters: false, registers: false };
    /// Finite classes and counters.
    pub const COUNTERS: Self = Self { counters: true, registers: false };
    /// Every kind.
    pub const FULL: Self = Self { counters: true, registers: true };
}

impl Structure {
    /// The one-class machine with no counters or registers over `symbols`.
    pub fn memoryless(symbols: SymbolMap) -> Self {
        let width = symbols.symbols();
        Self { symbols, classes: 1, initial: 0, table: vec![0; width], counters: Vec::new(), registers: Vec::new() }
    }

    fn validate(&self) -> Result<(), CausalError> {
        let width = self.symbols.symbols();
        if self.classes == 0 || self.initial >= self.classes {
            return Err(CausalError::Machine("no classes, or an initial class out of range".to_string()));
        }
        if self.table.len() != self.classes * width || self.table.iter().any(|&next| next >= self.classes) {
            return Err(CausalError::Machine("the class table has the wrong size or an entry out of range".to_string()));
        }
        for counter in &self.counters {
            if counter.increments.len() != width {
                return Err(CausalError::Machine("a counter needs one increment per symbol".to_string()));
            }
            if let Some(period) = counter.period {
                if period < 2 || counter.increments.iter().any(|&step| step < 0 || step >= period as i64) {
                    return Err(CausalError::Machine("a periodic counter needs a period of 2 or more and residues".to_string()));
                }
            }
        }
        for register in &self.registers {
            if register.counter >= self.counters.len()
                || register.writes.is_empty()
                || register.writes.windows(2).any(|pair| pair[0] >= pair[1])
                || register.writes.iter().any(|&symbol| symbol >= width)
            {
                return Err(CausalError::Machine("a register needs a counter and ascending write symbols".to_string()));
            }
        }
        Ok(())
    }

    /// The state fields of every row: `[class, counters.., registers..]`, row-major.
    pub fn trace(&self, harvest: &Harvest) -> Result<Vec<i64>, CausalError> {
        self.validate()?;
        let width = self.symbols.symbols();
        let fields = 1 + self.counters.len() + self.registers.len();
        let mut out = vec![0_i64; harvest.rows() * fields];
        let mut values = vec![0_i64; self.counters.len()];
        let mut memories: Vec<HashMap<i64, usize>> = vec![HashMap::new(); self.registers.len()];
        for sequence in 0..harvest.sequences() {
            let mut class = self.initial;
            values.iter_mut().for_each(|value| *value = 0);
            memories.iter_mut().for_each(HashMap::clear);
            for row in harvest.sequence(sequence) {
                let symbol = self.symbols.symbol(harvest.tokens[row]);
                class = self.table[class * width + symbol];
                for (value, counter) in values.iter_mut().zip(&self.counters) {
                    *value = counter.step(*value, symbol);
                }
                let slot = &mut out[row * fields..(row + 1) * fields];
                slot[0] = class as i64;
                slot[1..1 + values.len()].copy_from_slice(&values);
                for (index, (register, memory)) in self.registers.iter().zip(memories.iter_mut()).enumerate() {
                    let key = values[register.counter];
                    if let Ok(position) = register.writes.binary_search(&symbol) {
                        memory.insert(key, position + 1);
                    }
                    slot[1 + values.len() + index] = memory.get(&key).copied().unwrap_or(0) as i64;
                }
            }
        }
        Ok(out)
    }

    fn encode(&self, out: &mut BitString, vocabulary: usize) -> Result<(), CausalError> {
        self.validate()?;
        let width = self.symbols.symbols();
        encode_prefix_integer(out, width as u64)?;
        if width > 1 {
            let tokens: Vec<usize> = self.symbols.members.iter().map(|&(token, _)| token as usize).collect();
            encode_subset(out, vocabulary, &tokens)?;
            for &(_, symbol) in &self.symbols.members {
                encode_fixed_index(out, symbol - 1, width - 1)?;
            }
        }
        encode_prefix_integer(out, self.classes as u64)?;
        encode_fixed_index(out, self.initial, self.classes)?;
        for &next in &self.table {
            encode_fixed_index(out, next, self.classes)?;
        }
        encode_prefix_integer(out, self.counters.len() as u64 + 1)?;
        for counter in &self.counters {
            encode_prefix_integer(out, counter.period.unwrap_or(0) + 1)?;
            for &step in &counter.increments {
                match counter.period {
                    Some(period) => encode_fixed_index(out, step as usize, period as usize)?,
                    None => encode_signed_prefix_integer(out, step)?,
                }
            }
        }
        encode_prefix_integer(out, self.registers.len() as u64 + 1)?;
        for register in &self.registers {
            encode_fixed_index(out, register.counter, self.counters.len())?;
            encode_subset(out, width, &register.writes)?;
        }
        Ok(())
    }

    fn decode(reader: &mut BitReader<'_>, vocabulary: usize) -> Result<Self, CausalError> {
        let width = usize::try_from(decode_prefix_integer(reader)?)
            .map_err(|error| CausalError::Machine(error.to_string()))?;
        let mut members = Vec::new();
        if width > 1 {
            let tokens = decode_subset(reader, vocabulary)?;
            for token in tokens {
                members.push((token as u32, decode_fixed_index(reader, width - 1)? + 1));
            }
        }
        let symbols = SymbolMap::new(width, members)?;
        let classes = decode_prefix_integer(reader)? as usize;
        let initial = decode_fixed_index(reader, classes)?;
        let mut table = Vec::with_capacity(classes * width);
        for _entry in 0..classes * width {
            table.push(decode_fixed_index(reader, classes)?);
        }
        let counter_count = decode_prefix_integer(reader)? as usize - 1;
        let mut counters = Vec::with_capacity(counter_count);
        for _counter in 0..counter_count {
            let period = match decode_prefix_integer(reader)? - 1 {
                0 => None,
                period => Some(period),
            };
            let mut increments = Vec::with_capacity(width);
            for _symbol in 0..width {
                increments.push(match period {
                    Some(period) => decode_fixed_index(reader, period as usize)? as i64,
                    None => decode_signed_prefix_integer(reader)?,
                });
            }
            counters.push(Counter { increments, period });
        }
        let register_count = decode_prefix_integer(reader)? as usize - 1;
        let mut registers = Vec::with_capacity(register_count);
        for _register in 0..register_count {
            let counter = decode_fixed_index(reader, counter_count)?;
            registers.push(Register { counter, writes: decode_subset(reader, width)? });
        }
        let structure = Self { symbols, classes, initial, table, counters, registers };
        structure.validate()?;
        Ok(structure)
    }
}

/// The cell universe of a structure under clamps: the mixed radix of
/// `(class, counter cells.., register values..)`.
fn cell_radix(structure: &Structure, clamps: &[Option<Clamp>]) -> Result<Vec<usize>, CausalError> {
    let mut radix = vec![structure.classes];
    for (counter, clamp) in structure.counters.iter().zip(clamps) {
        radix.push(match (counter.period, clamp) {
            (Some(period), None) => period as usize,
            (None, Some(clamp)) if clamp.hi >= clamp.lo => (clamp.hi - clamp.lo + 1) as usize,
            _ => return Err(CausalError::Machine("a clamp must match its counter's kind".to_string())),
        });
    }
    for register in &structure.registers {
        radix.push(register.writes.len() + 1);
    }
    let mut total = 1_usize;
    for &width in &radix {
        total = total
            .checked_mul(width)
            .ok_or_else(|| CausalError::Machine("the cell universe overflows".to_string()))?;
    }
    Ok(radix)
}

fn cell_of(fields: &[i64], structure: &Structure, clamps: &[Option<Clamp>], radix: &[usize]) -> usize {
    let mut index = fields[0] as usize;
    let counters = structure.counters.len();
    for (position, clamp) in clamps.iter().enumerate() {
        let value = fields[1 + position];
        let digit = match clamp {
            Some(clamp) => (value.clamp(clamp.lo, clamp.hi) - clamp.lo) as usize,
            None => value as usize,
        };
        index = index * radix[1 + position] + digit;
    }
    for position in 0..structure.registers.len() {
        index = index * radix[1 + counters + position] + fields[1 + counters + position] as usize;
    }
    index
}

/// Per distinct state tuple, its row count and summed readout rows.
#[derive(Clone, Debug)]
struct StateStats {
    fields: usize,
    keys: Vec<i64>,
    counts: Vec<f64>,
    sums: Vec<f64>,
    readout: usize,
}

impl StateStats {
    fn collect(trace: &[i64], fields: usize, probabilities: ArrayView2<'_, f64>, weights: &[f64]) -> Self {
        let readout = probabilities.ncols();
        let mut index: HashMap<&[i64], usize> = HashMap::new();
        let mut stats = Self { fields, keys: Vec::new(), counts: Vec::new(), sums: Vec::new(), readout };
        for (row, key) in trace.chunks_exact(fields).enumerate() {
            let weight = weights[row];
            if weight == 0.0 {
                continue;
            }
            let slot = match index.get(key) {
                Some(&slot) => slot,
                None => {
                    let slot = stats.counts.len();
                    index.insert(key, slot);
                    stats.keys.extend_from_slice(key);
                    stats.counts.push(0.0);
                    stats.sums.extend(std::iter::repeat_n(0.0, readout));
                    slot
                }
            };
            stats.counts[slot] += weight;
            for (sum, value) in stats.sums[slot * readout..(slot + 1) * readout].iter_mut().zip(probabilities.row(row)) {
                *sum += weight * value;
            }
        }
        stats
    }

    fn len(&self) -> usize {
        self.counts.len()
    }

    fn key(&self, slot: usize) -> &[i64] {
        &self.keys[slot * self.fields..(slot + 1) * self.fields]
    }

    /// The statistics with field `field` replaced by `map` of it, merging tuples that meet.
    fn remap(&self, field: usize, map: impl Fn(i64) -> i64) -> Self {
        let mut index: HashMap<Vec<i64>, usize> = HashMap::new();
        let mut out = Self { fields: self.fields, keys: Vec::new(), counts: Vec::new(), sums: Vec::new(), readout: self.readout };
        for slot in 0..self.len() {
            let mut key = self.key(slot).to_vec();
            key[field] = map(key[field]);
            let target = match index.get(&key) {
                Some(&target) => target,
                None => {
                    let target = out.counts.len();
                    out.keys.extend_from_slice(&key);
                    index.insert(key, target);
                    out.counts.push(0.0);
                    out.sums.extend(std::iter::repeat_n(0.0, self.readout));
                    target
                }
            };
            out.counts[target] += self.counts[slot];
            for position in 0..self.readout {
                out.sums[target * self.readout + position] += self.sums[slot * self.readout + position];
            }
        }
        out
    }

    /// The observed range of field `field`.
    fn range(&self, field: usize) -> (i64, i64) {
        let mut low = i64::MAX;
        let mut high = i64::MIN;
        for slot in 0..self.len() {
            let value = self.key(slot)[field];
            low = low.min(value);
            high = high.max(value);
        }
        (low, high)
    }
}

/// Aggregated statistics of the present cells.
struct CellStats {
    cells: Vec<usize>,
    sums: Vec<f64>,
}

fn aggregate(stats: &StateStats, structure: &Structure, clamps: &[Option<Clamp>]) -> Result<CellStats, CausalError> {
    let radix = cell_radix(structure, clamps)?;
    let readout = stats.readout;
    let mut by_cell: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for slot in 0..stats.len() {
        let cell = cell_of(stats.key(slot), structure, clamps, &radix);
        let entry = by_cell.entry(cell).or_insert_with(|| vec![0.0; readout]);
        for (sum, value) in entry.iter_mut().zip(&stats.sums[slot * readout..(slot + 1) * readout]) {
            *sum += value;
        }
    }
    let mut cells = Vec::with_capacity(by_cell.len());
    let mut sums = Vec::with_capacity(by_cell.len() * readout);
    for (cell, values) in by_cell {
        cells.push(cell);
        sums.extend(values);
    }
    Ok(CellStats { cells, sums })
}

/// The most negative relative logit a cell with a zero mean class takes: the log of the
/// smallest positive normal `f64`.
fn floor_logit() -> f64 {
    f64::MIN_POSITIVE.ln()
}

/// Relative logits `ln(m_j / m_ref)` of one cell's summed rows and its reference class.
fn relative_logits(sums: &[f64]) -> (usize, Vec<f64>) {
    let mut reference = 0;
    for (class, &value) in sums.iter().enumerate() {
        if value > sums[reference] {
            reference = class;
        }
    }
    let top = sums[reference];
    let logits = sums
        .iter()
        .enumerate()
        .filter(|&(class, _)| class != reference)
        .map(|(_, &value)| if value > 0.0 { (value / top).ln().max(floor_logit()) } else { floor_logit() })
        .collect();
    (reference, logits)
}

/// `softmax(logits)`.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let normalizer: f64 = logits.iter().map(|logit| (logit - top).exp()).sum();
    logits.iter().map(|logit| (logit - top).exp() / normalizer).collect()
}

/// `KL(p ‖ q)` in nats.
fn divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p.ln() - q.ln())).sum::<f64>().max(0.0)
}

/// `−Σ_j S_j ln q_j` of a cell under the decoded relative logits.
fn cell_cross_entropy(sums: &[f64], reference: usize, decoded: &[f64]) -> f64 {
    let mut logits = Vec::with_capacity(sums.len());
    let mut others = decoded.iter();
    for class in 0..sums.len() {
        logits.push(if class == reference { 0.0 } else { *others.next().unwrap_or(&0.0) });
    }
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let normalizer = top + logits.iter().map(|logit| (logit - top).exp()).sum::<f64>().ln();
    sums.iter().zip(&logits).filter(|(sum, _)| **sum > 0.0).map(|(sum, logit)| sum * (normalizer - logit)).sum()
}

/// The fraction bits the readout lattice is searched over: every step from `2^8` down to
/// `2^-40`, which spans the `f64` logits a readout can carry.
const FRACTION_BITS: std::ops::RangeInclusive<i32> = -8..=40;

/// The readout of the present cells at the fraction bits that minimize the total code,
/// with the readout's own code length and the data bits.
fn fit_readout(
    cells: &CellStats,
    readout: usize,
    universe: usize,
    harvest: &Harvest,
) -> Result<(Vec<usize>, LatticeCode, u64, f64), CausalError> {
    let targets: Vec<(usize, Vec<f64>)> =
        cells.sums.chunks_exact(readout).map(relative_logits).collect();
    let presence_bits = super::codec::subset_code_len_bits(universe, cells.cells.len())?;
    let reference_bits = u64::from(super::codec::fixed_index_len_bits(readout)?) * cells.cells.len() as u64;
    let scale = harvest.observations as f64 / std::f64::consts::LN_2;
    let mut best: Option<(f64, LatticeCode, u64, f64)> = None;
    for fraction_bits in FRACTION_BITS {
        let precision = DeclaredPrecision::new(fraction_bits).map_err(CausalError::Precision)?;
        let flat: Vec<f64> = targets.iter().flat_map(|(_, logits)| logits.iter().copied()).collect();
        let code = match LatticeCode::encode(&flat, precision) {
            Ok(code) => code,
            Err(_) => continue,
        };
        let step = precision.step();
        let mut bits = presence_bits
            + reference_bits
            + super::codec::signed_prefix_integer_len_bits(i64::from(fraction_bits))?;
        for &index in code.indices() {
            bits += super::codec::prefix_integer_len_bits((1 - index.min(0)) as u64)?;
        }
        let mut cross = 0.0;
        for (position, (reference, _)) in targets.iter().enumerate() {
            let decoded: Vec<f64> = code.indices()[position * (readout - 1)..(position + 1) * (readout - 1)]
                .iter()
                .map(|&index| index.min(0) as f64 * step)
                .collect();
            cross += cell_cross_entropy(&cells.sums[position * readout..(position + 1) * readout], *reference, &decoded);
        }
        let data = scale * (harvest.negentropy + cross).max(0.0);
        let total = bits as f64 + data;
        if best.as_ref().is_none_or(|(incumbent, ..)| total < *incumbent) {
            best = Some((total, code, bits, data));
        }
    }
    let (_, code, bits, data) =
        best.ok_or_else(|| CausalError::Precision("no lattice step encodes the readout".to_string()))?;
    let indices: Vec<i64> = code.indices().iter().map(|&index| index.min(0)).collect();
    let code = LatticeCode::from_indices(code.precision(), indices).map_err(CausalError::Precision)?;
    Ok((targets.into_iter().map(|(reference, _)| reference).collect(), code, bits, data))
}

/// A fitted machine and its score.
#[derive(Clone, Debug)]
pub struct Fitted {
    pub machine: Machine,
    pub score: Score,
}

/// Fit the readout of `structure` under `clamps` and score the machine.
fn fit_with_clamps(
    structure: &Structure,
    stats: &StateStats,
    clamps: &[Option<Clamp>],
    harvest: &Harvest,
) -> Result<Fitted, CausalError> {
    let radix = cell_radix(structure, clamps)?;
    let universe = radix.iter().product();
    let cells = aggregate(stats, structure, clamps)?;
    let (references, logits, readout_bits, data_bits) = fit_readout(&cells, harvest.readout(), universe, harvest)?;
    let mut header = BitString::new();
    structure.encode(&mut header, harvest.vocabulary)?;
    let mut clamp_bits = 0_u64;
    for clamp in clamps.iter().flatten() {
        clamp_bits += super::codec::signed_prefix_integer_len_bits(clamp.lo)?
            + super::codec::prefix_integer_len_bits((clamp.hi - clamp.lo + 1) as u64)?;
    }
    let machine = Machine {
        structure: structure.clone(),
        readout: Readout { clamps: clamps.to_vec(), present: cells.cells, references, logits },
    };
    Ok(Fitted { machine, score: Score { machine_bits: header.len_bits() + clamp_bits + readout_bits, data_bits } })
}

/// The clamps of every unbounded counter chosen by coordinate descent on the total code,
/// starting from the observed range; `start` seeds the clamps of existing counters.
fn fit_clamps(
    structure: &Structure,
    stats: &StateStats,
    start: Option<&[Option<Clamp>]>,
    harvest: &Harvest,
) -> Result<Fitted, CausalError> {
    let mut clamps: Vec<Option<Clamp>> = structure
        .counters
        .iter()
        .enumerate()
        .map(|(position, counter)| {
            counter.period.map_or_else(
                || {
                    let (low, high) = stats.range(1 + position);
                    let seeded = start.and_then(|start| start.get(position).copied().flatten());
                    Some(match seeded {
                        Some(clamp) => {
                            let lo = clamp.lo.clamp(low, high);
                            Clamp { lo, hi: clamp.hi.clamp(lo, high) }
                        }
                        None => Clamp { lo: low, hi: high },
                    })
                },
                |_| None,
            )
        })
        .collect();
    let mut best = fit_with_clamps(structure, stats, &clamps, harvest)?;
    loop {
        let mut improved = false;
        for position in 0..clamps.len() {
            let Some(current) = clamps[position] else { continue };
            let (low, high) = stats.range(1 + position);
            let mut candidates = Vec::new();
            for lo in low..=current.hi {
                candidates.push(Clamp { lo, hi: current.hi });
            }
            for hi in current.lo..=high {
                candidates.push(Clamp { lo: current.lo, hi });
            }
            for candidate in candidates {
                let mut trial = clamps.clone();
                trial[position] = Some(candidate);
                let fitted = fit_with_clamps(structure, stats, &trial, harvest)?;
                if fitted.score.total() < best.score.total() {
                    best = fitted;
                    clamps = trial;
                    improved = true;
                }
            }
        }
        if !improved {
            return Ok(best);
        }
    }
}

/// Fit `structure`'s readout and clamps on `harvest`: the machine of that structure with the
/// shortest total code.
pub fn fit(structure: &Structure, harvest: &Harvest) -> Result<Fitted, CausalError> {
    fit_seeded(structure, harvest, None)
}

fn fit_seeded(structure: &Structure, harvest: &Harvest, start: Option<&[Option<Clamp>]>) -> Result<Fitted, CausalError> {
    let trace = structure.trace(harvest)?;
    let fields = 1 + structure.counters.len() + structure.registers.len();
    let stats = StateStats::collect(&trace, fields, harvest.probabilities(), &harvest.weights);
    fit_clamps(structure, &stats, start, harvest)
}

impl Machine {
    /// The machine's message.
    pub fn encode(&self, vocabulary: usize, readout_width: usize) -> Result<BitString, CausalError> {
        let mut out = BitString::new();
        self.structure.encode(&mut out, vocabulary)?;
        let readout = &self.readout;
        if readout.clamps.len() != self.structure.counters.len() {
            return Err(CausalError::Machine("one clamp entry per counter".to_string()));
        }
        for clamp in readout.clamps.iter().flatten() {
            encode_signed_prefix_integer(&mut out, clamp.lo)?;
            encode_prefix_integer(&mut out, (clamp.hi - clamp.lo + 1) as u64)?;
        }
        let radix = cell_radix(&self.structure, &readout.clamps)?;
        let universe: usize = radix.iter().product();
        encode_subset(&mut out, universe, &readout.present)?;
        encode_signed_prefix_integer(&mut out, i64::from(readout.logits.precision().fraction_bits()))?;
        let width = readout_width - 1;
        if readout.references.len() != readout.present.len() || readout.logits.indices().len() != readout.present.len() * width {
            return Err(CausalError::Machine("one reference and R - 1 logits per present cell".to_string()));
        }
        for (cell, &reference) in readout.references.iter().enumerate() {
            encode_fixed_index(&mut out, reference, readout_width)?;
            for &index in &readout.logits.indices()[cell * width..(cell + 1) * width] {
                if index > 0 {
                    return Err(CausalError::Machine("a relative logit above its reference".to_string()));
                }
                encode_prefix_integer(&mut out, (1 - index) as u64)?;
            }
        }
        Ok(out)
    }

    /// Read a machine back given the vocabulary size and the readout width.
    pub fn decode(message: &BitString, vocabulary: usize, readout: usize) -> Result<Self, CausalError> {
        let mut reader = message.reader();
        let structure = Structure::decode(&mut reader, vocabulary)?;
        let mut clamps = Vec::with_capacity(structure.counters.len());
        for counter in &structure.counters {
            clamps.push(match counter.period {
                Some(_) => None,
                None => {
                    let lo = decode_signed_prefix_integer(&mut reader)?;
                    let span = decode_prefix_integer(&mut reader)? as i64;
                    Some(Clamp { lo, hi: lo + span - 1 })
                }
            });
        }
        let radix = cell_radix(&structure, &clamps)?;
        let universe: usize = radix.iter().product();
        let present = decode_subset(&mut reader, universe)?;
        let fraction_bits = i32::try_from(decode_signed_prefix_integer(&mut reader)?)
            .map_err(|error| CausalError::Precision(error.to_string()))?;
        let precision = DeclaredPrecision::new(fraction_bits).map_err(CausalError::Precision)?;
        let mut references = Vec::with_capacity(present.len());
        let mut indices = Vec::with_capacity(present.len() * (readout - 1));
        for _cell in 0..present.len() {
            references.push(decode_fixed_index(&mut reader, readout)?);
            for _class in 1..readout {
                indices.push(1 - decode_prefix_integer(&mut reader)? as i64);
            }
        }
        reader.finish()?;
        let logits = LatticeCode::from_indices(precision, indices).map_err(CausalError::Precision)?;
        Ok(Self { structure, readout: Readout { clamps, present, references, logits } })
    }

    /// The readout cell of every row of `harvest`.
    pub fn cells(&self, harvest: &Harvest) -> Result<Vec<usize>, CausalError> {
        let trace = self.structure.trace(harvest)?;
        let fields = 1 + self.structure.counters.len() + self.structure.registers.len();
        let radix = cell_radix(&self.structure, &self.readout.clamps)?;
        Ok(trace.chunks_exact(fields).map(|key| cell_of(key, &self.structure, &self.readout.clamps, &radix)).collect())
    }

    /// The decoded distribution of `cell` over `readout` classes: uniform for a cell the
    /// readout does not carry.
    pub fn distribution(&self, cell: usize, readout: usize) -> Vec<f64> {
        let Ok(position) = self.readout.present.binary_search(&cell) else {
            return vec![1.0 / readout as f64; readout];
        };
        let step = self.readout.logits.precision().step();
        let width = readout - 1;
        let reference = self.readout.references[position];
        let mut others = self.readout.logits.indices()[position * width..(position + 1) * width].iter();
        let logits: Vec<f64> = (0..readout)
            .map(|class| if class == reference { 0.0 } else { *others.next().unwrap_or(&0) as f64 * step })
            .collect();
        softmax(&logits)
    }

    /// The machine's readout distribution on every row of `harvest`, `rows × R`.
    pub fn predict(&self, harvest: &Harvest) -> Result<Array2<f64>, CausalError> {
        let readout = harvest.readout();
        let mut out = Array2::<f64>::zeros((harvest.rows(), readout));
        let mut memo: HashMap<usize, Vec<f64>> = HashMap::new();
        for (row, cell) in self.cells(harvest)?.into_iter().enumerate() {
            let distribution = memo.entry(cell).or_insert_with(|| self.distribution(cell, readout));
            out.row_mut(row).iter_mut().zip(distribution.iter()).for_each(|(target, value)| *target = *value);
        }
        Ok(out)
    }

    /// Encode, decode and score the decoded machine on `harvest`: its message length and
    /// `n Σ w KL(p ‖ q) / ln 2`, with the unweighted per-row KL in nats.
    pub fn score(&self, harvest: &Harvest) -> Result<(Score, Vec<f64>), CausalError> {
        let message = self.encode(harvest.vocabulary, harvest.readout())?;
        let decoded = Self::decode(&message, harvest.vocabulary, harvest.readout())?;
        let predicted = decoded.predict(harvest)?;
        let divergences: Vec<f64> = harvest
            .probabilities
            .rows()
            .into_iter()
            .zip(predicted.rows())
            .map(|(model, machine)| divergence(&model.to_vec(), &machine.to_vec()))
            .collect();
        let weighted: f64 = divergences.iter().zip(&harvest.weights).map(|(kl, weight)| weight * kl).sum();
        let data_bits = harvest.observations as f64 * weighted / std::f64::consts::LN_2;
        Ok((Score { machine_bits: message.len_bits(), data_bits }, divergences))
    }

    /// Which state kinds the machine uses.
    pub fn kind(&self) -> &'static str {
        match (
            self.structure.classes > 1,
            !self.structure.counters.is_empty(),
            !self.structure.registers.is_empty(),
        ) {
            (false, false, false) => "memoryless",
            (true, false, false) => "finite",
            (_, true, false) => "counter",
            (_, _, true) => "counter+register",
        }
    }
}

/// The finite machine of `machine`'s reachable `(class, counters)` states on `harvest`: one
/// class per state the harvest visits (and the initial state), its table read off the
/// visited transitions, every unvisited transition a self-loop. On the harvest it moves
/// through exactly the states of `machine`. Refused for a machine with registers, whose
/// memory is not a finite state.
pub fn unroll(machine: &Machine, harvest: &Harvest) -> Result<Structure, CausalError> {
    let structure = &machine.structure;
    if !structure.registers.is_empty() {
        return Err(CausalError::Machine("a register's memory does not unroll to finite states".to_string()));
    }
    let width = structure.symbols.symbols();
    let mut states: HashMap<Vec<i64>, usize> = HashMap::new();
    let mut transitions: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let start: Vec<i64> = std::iter::once(structure.initial as i64).chain(structure.counters.iter().map(|_| 0)).collect();
    states.insert(start.clone(), 0);
    for sequence in 0..harvest.sequences() {
        let mut state = start.clone();
        for row in harvest.sequence(sequence) {
            let symbol = structure.symbols.symbol(harvest.tokens[row]);
            let mut next = Vec::with_capacity(state.len());
            next.push(structure.table[state[0] as usize * width + symbol] as i64);
            for (position, counter) in structure.counters.iter().enumerate() {
                next.push(counter.step(state[1 + position], symbol));
            }
            let from = states[&state];
            let count = states.len();
            let to = *states.entry(next.clone()).or_insert(count);
            transitions.insert((from, symbol), to);
            state = next;
        }
    }
    let classes = states.len();
    let mut table: Vec<usize> = (0..classes * width).map(|entry| entry / width).collect();
    for ((from, symbol), to) in transitions {
        table[from * width + symbol] = to;
    }
    Ok(Structure {
        symbols: structure.symbols.clone(),
        classes,
        initial: 0,
        table,
        counters: Vec::new(),
        registers: Vec::new(),
    })
}

/// One accepted move of a search.
#[derive(Clone, Debug)]
pub struct Step {
    pub description: String,
    pub score: Score,
}

/// A search's result.
#[derive(Clone, Debug)]
pub struct SearchReport {
    pub fitted: Fitted,
    pub steps: Vec<Step>,
}

/// Every candidate the moves propose from one machine.
fn proposals(current: &Fitted, harvest: &Harvest, language: Language) -> Result<Vec<(String, Structure)>, CausalError> {
    let structure = &current.machine.structure;
    let width = structure.symbols.symbols();
    let classes = structure.classes;
    let mut out = Vec::new();
    // Split a class by a partition of its incoming edges.
    let trace = structure.trace(harvest)?;
    let fields = 1 + structure.counters.len() + structure.registers.len();
    let readout = harvest.readout();
    let mut edges: BTreeMap<(usize, (usize, usize)), (f64, Vec<f64>)> = BTreeMap::new();
    for sequence in 0..harvest.sequences() {
        let mut previous = structure.initial;
        for row in harvest.sequence(sequence) {
            let class = trace[row * fields] as usize;
            let symbol = structure.symbols.symbol(harvest.tokens[row]);
            let weight = harvest.weights[row];
            let entry = edges.entry((class, (previous, symbol))).or_insert_with(|| (0.0, vec![0.0; readout]));
            entry.0 += weight;
            for (sum, value) in entry.1.iter_mut().zip(harvest.probabilities.row(row)) {
                *sum += weight * value;
            }
            previous = class;
        }
    }
    for class in 0..classes {
        let incoming: Vec<((usize, usize), f64, Vec<f64>)> = edges
            .range((class, (0, 0))..(class + 1, (0, 0)))
            .filter(|(_, (count, _))| *count > 0.0)
            .map(|(&(_, edge), (count, sums))| (edge, *count, sums.clone()))
            .collect();
        if incoming.len() < 2 {
            continue;
        }
        let mut groups: Vec<Vec<(usize, usize)>> = incoming.iter().map(|(edge, ..)| vec![*edge]).collect();
        groups.push(bregman_bisection(&incoming));
        for group in groups {
            if group.is_empty() || group.len() == incoming.len() {
                continue;
            }
            let mut table = structure.table.clone();
            for &(from, symbol) in &group {
                table[from * width + symbol] = classes;
            }
            let copied: Vec<usize> = table[class * width..(class + 1) * width].to_vec();
            table.extend(copied);
            out.push((
                format!("split class {class} by edges {group:?}"),
                Structure { classes: classes + 1, table, ..structure.clone() },
            ));
        }
    }
    // Redirect one edge.
    for from in 0..classes {
        for symbol in 0..width {
            for target in 0..classes {
                if target != structure.table[from * width + symbol] {
                    let mut table = structure.table.clone();
                    table[from * width + symbol] = target;
                    out.push((format!("redirect ({from}, {symbol}) -> {target}"), Structure { table, ..structure.clone() }));
                }
            }
        }
    }
    // Merge two classes.
    for kept in 0..classes {
        for gone in kept + 1..classes {
            out.push((format!("merge classes {kept} and {gone}"), merge_classes(structure, kept, gone)));
        }
    }
    // Merge two symbols, keeping either one's transitions.
    for first in 0..width {
        for second in first + 1..width {
            for (kept, gone) in [(first, second), (second, first)] {
                out.push((format!("merge symbol {gone} into {kept}"), merge_symbols(structure, kept, gone)?));
            }
        }
    }
    if language.counters {
        counter_proposals(structure, &mut out);
    }
    if language.registers {
        for (position, _) in structure.counters.iter().enumerate() {
            for mask in 1_u64..(1_u64 << width.min(12)) {
                let writes: Vec<usize> = (0..width).filter(|bit| mask >> bit & 1 == 1).collect();
                let register = Register { counter: position, writes };
                if structure.registers.contains(&register) {
                    continue;
                }
                let mut registers = structure.registers.clone();
                registers.push(register.clone());
                out.push((format!("add register {register:?}"), Structure { registers, ..structure.clone() }));
            }
        }
        for position in 0..structure.registers.len() {
            let mut registers = structure.registers.clone();
            registers.remove(position);
            out.push((format!("remove register {position}"), Structure { registers, ..structure.clone() }));
        }
    }
    Ok(out)
}

/// Every increment vector over the symbols in `{−1, 0, 1}` whose first nonzero entry is
/// positive (a counter and its negation carry the same state).
fn increment_family(width: usize) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    if width > 8 {
        return out;
    }
    for code in 1..3_usize.pow(width as u32) {
        let increments: Vec<i64> =
            (0..width).map(|symbol| (code / 3_usize.pow(symbol as u32) % 3) as i64 - 1).collect();
        if increments.iter().find(|&&step| step != 0).is_some_and(|&step| step > 0) {
            out.push(increments);
        }
    }
    out
}

/// The best machine that adds one counter with `increments` to `structure`: unbounded, or
/// at any period from 2 to the longest sequence (a longer period never wraps on the
/// harvest). One trace serves every period, since a residue is a function of the raw value.
fn fit_counter_family(
    structure: &Structure,
    increments: &[i64],
    harvest: &Harvest,
    clamps: &[Option<Clamp>],
) -> Result<Fitted, CausalError> {
    let mut unbounded = structure.clone();
    unbounded.counters.push(Counter { increments: increments.to_vec(), period: None });
    let trace = unbounded.trace(harvest)?;
    let fields = 1 + unbounded.counters.len() + unbounded.registers.len();
    let stats = StateStats::collect(&trace, fields, harvest.probabilities(), &harvest.weights);
    let mut seeded = clamps.to_vec();
    seeded.truncate(structure.counters.len());
    let mut best = fit_clamps(&unbounded, &stats, Some(&seeded), harvest)?;
    let field = structure.counters.len() + 1;
    for period in 2..=harvest.longest() as u64 {
        let mut periodic = structure.clone();
        periodic.counters.push(Counter {
            increments: increments.iter().map(|step| step.rem_euclid(period as i64)).collect(),
            period: Some(period),
        });
        let residues = stats.remap(field, |value| value.rem_euclid(period as i64));
        let fitted = fit_clamps(&periodic, &residues, Some(&seeded), harvest)?;
        if fitted.score.total() < best.score.total() {
            best = fitted;
        }
    }
    Ok(best)
}

fn counter_proposals(structure: &Structure, out: &mut Vec<(String, Structure)>) {
    let width = structure.symbols.symbols();
    for (position, counter) in structure.counters.iter().enumerate() {
        let mut counters = structure.counters.clone();
        counters.remove(position);
        let registers = structure
            .registers
            .iter()
            .filter(|register| register.counter != position)
            .map(|register| Register {
                counter: if register.counter > position { register.counter - 1 } else { register.counter },
                writes: register.writes.clone(),
            })
            .collect();
        out.push((format!("remove counter {position}"), Structure { counters, registers, ..structure.clone() }));
        for symbol in 0..width {
            for delta in [-1_i64, 1] {
                let mut changed = counter.clone();
                changed.increments[symbol] = match counter.period {
                    Some(period) => (changed.increments[symbol] + delta).rem_euclid(period as i64),
                    None => changed.increments[symbol] + delta,
                };
                let mut counters = structure.counters.clone();
                counters[position] = changed;
                out.push((
                    format!("counter {position} increment of symbol {symbol} {delta:+}"),
                    Structure { counters, ..structure.clone() },
                ));
            }
        }
    }
}

/// A two-group partition of incoming edges by Bregman 2-means under `KL(p ‖ centroid)`, the
/// data term's own divergence. The seeds are the heaviest edge and the edge whose rows cost
/// most under the heaviest edge's mean; returns the second group.
fn bregman_bisection(incoming: &[((usize, usize), f64, Vec<f64>)]) -> Vec<(usize, usize)> {
    let cost = |sums: &[f64], centroid: &[f64]| -> f64 {
        let total: f64 = centroid.iter().sum();
        sums.iter()
            .zip(centroid)
            .filter(|(sum, _)| **sum > 0.0)
            .map(|(sum, c)| if *c > 0.0 { -sum * (c / total).ln() } else { f64::INFINITY })
            .sum()
    };
    let heaviest = (0..incoming.len())
        .max_by(|&a, &b| incoming[a].1.total_cmp(&incoming[b].1))
        .unwrap_or(0);
    let farthest = (0..incoming.len())
        .filter(|&edge| edge != heaviest)
        .max_by(|&a, &b| {
            let excess = |edge: usize| cost(&incoming[edge].2, &incoming[heaviest].2) - cost(&incoming[edge].2, &incoming[edge].2);
            excess(a).total_cmp(&excess(b))
        })
        .unwrap_or(0);
    let mut centroids = [incoming[heaviest].2.clone(), incoming[farthest].2.clone()];
    let mut assignment = vec![0_usize; incoming.len()];
    for _round in 0..=incoming.len() {
        let next: Vec<usize> = incoming
            .iter()
            .map(|(_, _, sums)| usize::from(cost(sums, &centroids[1]) < cost(sums, &centroids[0])))
            .collect();
        if next == assignment {
            break;
        }
        assignment = next;
        for (group, centroid) in centroids.iter_mut().enumerate() {
            let mut total = vec![0.0; centroid.len()];
            for (edge, (_, _, sums)) in incoming.iter().enumerate() {
                if assignment[edge] == group {
                    for (target, value) in total.iter_mut().zip(sums) {
                        *target += value;
                    }
                }
            }
            if total.iter().any(|value| *value > 0.0) {
                *centroid = total;
            }
        }
    }
    incoming.iter().zip(&assignment).filter(|(_, group)| **group == 1).map(|((edge, ..), _)| *edge).collect()
}

fn merge_classes(structure: &Structure, kept: usize, gone: usize) -> Structure {
    let width = structure.symbols.symbols();
    let renumber = |class: usize| -> usize {
        let class = if class == gone { kept } else { class };
        if class > gone { class - 1 } else { class }
    };
    let mut table = Vec::with_capacity((structure.classes - 1) * width);
    for class in (0..structure.classes).filter(|&class| class != gone) {
        table.extend(structure.table[class * width..(class + 1) * width].iter().map(|&next| renumber(next)));
    }
    Structure { classes: structure.classes - 1, initial: renumber(structure.initial), table, ..structure.clone() }
}

fn merge_symbols(structure: &Structure, kept: usize, gone: usize) -> Result<Structure, CausalError> {
    let width = structure.symbols.symbols();
    let (symbols, index) = structure.symbols.merge(kept, gone)?;
    let new_width = symbols.symbols();
    let mut sources = vec![usize::MAX; new_width];
    for old in 0..width {
        if old != gone {
            sources[index[old]] = old;
        }
    }
    let mut table = Vec::with_capacity(structure.classes * new_width);
    for class in 0..structure.classes {
        table.extend(sources.iter().map(|&old| structure.table[class * width + old]));
    }
    let counters = structure
        .counters
        .iter()
        .map(|counter| Counter {
            increments: sources.iter().map(|&old| counter.increments[old]).collect(),
            period: counter.period,
        })
        .collect();
    let registers = structure
        .registers
        .iter()
        .map(|register| {
            let mut writes: Vec<usize> = register.writes.iter().map(|&old| index[old]).collect();
            writes.sort_unstable();
            writes.dedup();
            Register { counter: register.counter, writes }
        })
        .collect();
    Ok(Structure { symbols, table, counters, registers, ..structure.clone() })
}

/// Keep only the classes reachable from the initial class through the table.
fn prune(structure: &Structure) -> Structure {
    let width = structure.symbols.symbols();
    let mut reachable = vec![false; structure.classes];
    let mut stack = vec![structure.initial];
    reachable[structure.initial] = true;
    while let Some(class) = stack.pop() {
        for &next in &structure.table[class * width..(class + 1) * width] {
            if !reachable[next] {
                reachable[next] = true;
                stack.push(next);
            }
        }
    }
    let mut renumber = vec![0; structure.classes];
    let mut count = 0;
    for class in 0..structure.classes {
        renumber[class] = count;
        count += usize::from(reachable[class]);
    }
    let mut table = Vec::with_capacity(count * width);
    for class in (0..structure.classes).filter(|&class| reachable[class]) {
        table.extend(structure.table[class * width..(class + 1) * width].iter().map(|&next| renumber[next]));
    }
    Structure { classes: count, initial: renumber[structure.initial], table, ..structure.clone() }
}

/// Greedy best-improvement search from `start` over the moves `language` allows. Every
/// accepted move strictly lowers the total code of the refitted machine.
pub fn search(start: &Structure, harvest: &Harvest, language: Language) -> Result<SearchReport, CausalError> {
    let mut current = fit(start, harvest)?;
    let mut steps = vec![Step { description: "start".to_string(), score: current.score }];
    loop {
        let mut candidates: Vec<(String, Option<Structure>, Vec<i64>)> = proposals(&current, harvest, language)?
            .into_iter()
            .map(|(description, structure)| (description, Some(structure), Vec::new()))
            .collect();
        if language.counters {
            for increments in increment_family(current.machine.structure.symbols.symbols()) {
                candidates.push((format!("add counter with increments {increments:?}"), None, increments));
            }
        }
        let clamps = current.machine.readout.clamps.clone();
        let incumbent = current.score.total();
        let evaluated: Vec<Result<Option<(f64, usize, Fitted)>, CausalError>> = candidates
            .par_iter()
            .enumerate()
            .map(|(position, (_, structure, increments))| {
                let fitted = match structure {
                    Some(structure) => {
                        let structure = prune(structure);
                        // Seed the clamps of the counters that kept their place.
                        let seeded: Vec<Option<Clamp>> = (0..structure.counters.len())
                            .map(|index| clamps.get(index).copied().flatten())
                            .collect();
                        fit_seeded(&structure, harvest, Some(&seeded))?
                    }
                    None => fit_counter_family(&current.machine.structure, increments, harvest, &clamps)?,
                };
                Ok((fitted.score.total() < incumbent).then(|| (fitted.score.total(), position, fitted)))
            })
            .collect();
        let mut best: Option<(f64, usize, Fitted)> = None;
        for result in evaluated {
            if let Some(candidate) = result? {
                if best.as_ref().is_none_or(|(total, position, _)| (candidate.0, candidate.1) < (*total, *position)) {
                    best = Some(candidate);
                }
            }
        }
        match best {
            Some((_, position, fitted)) => {
                steps.push(Step { description: candidates[position].0.clone(), score: fitted.score });
                current = fitted;
            }
            None => return Ok(SearchReport { fitted: current, steps }),
        }
    }
}

/// The distinct-symbol start for `harvest`: every token that occurs is its own symbol, except
/// the most frequent, which is the unlisted symbol `0`.
pub fn distinct_start(harvest: &Harvest) -> Result<Structure, CausalError> {
    let counts = harvest.token_counts();
    let most = counts.iter().max_by_key(|&(token, count)| (*count, std::cmp::Reverse(*token))).map(|(token, _)| *token);
    let listed: Vec<u32> = counts.keys().copied().filter(|token| Some(*token) != most).collect();
    Ok(Structure::memoryless(SymbolMap::distinct(&listed)?))
}

/// One continuation of a harvested prefix: the model's readout after the prefix ending at
/// harvest row `row` followed by `token`.
#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    pub row: usize,
    pub token: u32,
    pub probabilities: Vec<f64>,
}

/// How far the model's behaviour is from factoring through a machine's cells.
#[derive(Clone, Debug, PartialEq)]
pub struct Consistency {
    /// `n Σ_rows w KL(p_row ‖ p̄_cell) / ln 2`, `p̄_cell` the weighted mean of the cell's rows:
    /// the data bits that no readout of these cells removes.
    pub defect_bits: f64,
    /// The cells the weighted rows occupy.
    pub cells: usize,
    /// The row farthest from its cell's mean, its cell and its KL in nats.
    pub worst: Option<(usize, usize, f64)>,
}

/// Quotient consistency of `machine` on `harvest`.
///
/// A machine's state after a prefix is a function of the state before it and the symbol
/// read, by construction. So the cells are a quotient of the behaviour on the harvested
/// prefixes (every prefix's readout a function of its cell, and the successor state a
/// function of the state and the symbol) exactly when every row's readout equals its
/// cell's mean, that is, when the defect is zero. On `harvest.with_branches(..)` of every
/// one-symbol continuation this is the one-step congruence the causal-state relation
/// demands, checked on continuations the harvest never sampled. The mean is the
/// minimizer of the cell's summed divergence, so the defect is also the least data term
/// any readout of the machine's cells pays; the fitted readout pays it plus its lattice
/// rounding.
pub fn consistency(machine: &Machine, harvest: &Harvest) -> Result<Consistency, CausalError> {
    let readout = harvest.readout();
    let cells = machine.cells(harvest)?;
    let mut means: HashMap<usize, (f64, Vec<f64>)> = HashMap::new();
    for (row, &cell) in cells.iter().enumerate() {
        let weight = harvest.weights[row];
        if weight == 0.0 {
            continue;
        }
        let entry = means.entry(cell).or_insert_with(|| (0.0, vec![0.0; readout]));
        entry.0 += weight;
        for (sum, value) in entry.1.iter_mut().zip(harvest.probabilities.row(row)) {
            *sum += weight * value;
        }
    }
    for (total, sums) in means.values_mut() {
        sums.iter_mut().for_each(|sum| *sum /= *total);
    }
    let mut defect = 0.0;
    let mut worst: Option<(usize, usize, f64)> = None;
    for (row, &cell) in cells.iter().enumerate() {
        let weight = harvest.weights[row];
        if weight == 0.0 {
            continue;
        }
        let model: Vec<f64> = harvest.probabilities.row(row).to_vec();
        let kl = divergence(&model, &means[&cell].1);
        defect += weight * kl;
        if worst.is_none_or(|(_, _, incumbent)| kl > incumbent) {
            worst = Some((row, cell, kl));
        }
    }
    Ok(Consistency {
        defect_bits: harvest.observations as f64 * defect / std::f64::consts::LN_2,
        cells: means.len(),
        worst,
    })
}

/// The code length of one more readout cell carrying `probabilities` at the fitted lattice
/// of `fitted`, and that cell's decoded distribution.
fn fresh_cell(fitted: &Fitted, probabilities: &[f64], universe: usize) -> Result<(u64, Vec<f64>), CausalError> {
    let readout = probabilities.len();
    let step = fitted.machine.readout.logits.precision().step();
    let (reference, targets) = relative_logits(probabilities);
    let mut bits = u64::from(super::codec::fixed_index_len_bits(readout)?);
    let present = fitted.machine.readout.present.len();
    if present < universe {
        bits += super::codec::subset_code_len_bits(universe, present + 1)?
            .saturating_sub(super::codec::subset_code_len_bits(universe, present)?);
    }
    let mut logits = Vec::with_capacity(readout);
    let mut others = targets.iter();
    for class in 0..readout {
        if class == reference {
            logits.push(0.0);
            continue;
        }
        let index = ((*others.next().unwrap_or(&0.0)) / step).round().min(0.0);
        bits += super::codec::prefix_integer_len_bits((1.0 - index) as u64)?;
        logits.push(index * step);
    }
    Ok((bits, softmax(&logits)))
}

/// The rows of one machine cell that a cell of their own would explain better: a Bregman
/// 2-means under `KL(p ‖ ·)` with one centroid pinned at the cell's decoded distribution
/// `current` and the other seeded at the row farthest from it. Returns the members of the
/// free centroid and their mean.
fn disagreeing(rows: &[Vec<f64>], current: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let Some(seed) = (0..rows.len()).max_by(|&a, &b| divergence(&rows[a], current).total_cmp(&divergence(&rows[b], current)))
    else {
        return (Vec::new(), Vec::new());
    };
    let mut centroid = rows[seed].clone();
    let mut members: Vec<usize> = Vec::new();
    for _round in 0..=rows.len() {
        let next: Vec<usize> =
            (0..rows.len()).filter(|&row| divergence(&rows[row], &centroid) < divergence(&rows[row], current)).collect();
        if next.is_empty() || next == members {
            break;
        }
        members = next;
        centroid = vec![0.0; current.len()];
        for &row in &members {
            centroid.iter_mut().zip(&rows[row]).for_each(|(sum, value)| *sum += value / members.len() as f64);
        }
    }
    (members, centroid)
}

/// One round of [`refine`].
#[derive(Clone, Debug)]
pub struct Round {
    /// Branches admitted as counterexamples before the round's search.
    pub counterexamples: usize,
    /// The round's machine and its code on the harvest with every admitted branch.
    pub fitted: Fitted,
}

/// What [`refine`] found.
#[derive(Clone, Debug)]
pub struct Refinement {
    /// The last search.
    pub report: SearchReport,
    /// Every round, the first being the search on the harvest alone.
    pub rounds: Vec<Round>,
    /// The indices into the pool of the admitted branches.
    pub admitted: Vec<usize>,
    /// The harvest with every admitted branch.
    pub harvest: Harvest,
}

/// Counterexample-guided refinement over a pool of branches.
///
/// Search on the harvest; then check the machine on every branch of the pool not yet
/// admitted. Per machine cell, the branches landing in it that a cell of their own would
/// explain better are split off by a Bregman 2-means against the cell's distribution;
/// they are counterexamples when that cell pays for itself: the bits they save,
/// `n Σ (KL(p ‖ q_M) − KL(p ‖ q_fresh)) / ln 2`, exceed the bits of one more cell at the
/// machine's lattice. The fresh cell's bits leave out whatever structure would separate
/// the rows, so the test admits every group a refinement could explain and lets the next
/// search decide. The
/// counterexamples join the harvest as weighted branches and the search resumes from the
/// current machine. It stops when no branch is a counterexample (the machine is certified
/// on the whole pool at its own code) or every branch is admitted.
pub fn refine(start: &Structure, harvest: &Harvest, pool: &[Branch], language: Language) -> Result<Refinement, CausalError> {
    let mut report = search(start, harvest, language)?;
    let mut rounds = vec![Round { counterexamples: 0, fitted: report.fitted.clone() }];
    let mut admitted: Vec<usize> = Vec::new();
    let mut taken = vec![false; pool.len()];
    let mut current = harvest.clone();
    let scale = harvest.observations as f64 / std::f64::consts::LN_2;
    loop {
        let open: Vec<usize> = (0..pool.len()).filter(|&index| !taken[index]).collect();
        if open.is_empty() {
            break;
        }
        let branches: Vec<Branch> = open.iter().map(|&index| pool[index].clone()).collect();
        let probe = harvest.with_branches(&branches)?;
        let machine = &report.fitted.machine;
        let cells = machine.cells(&probe)?;
        let radix = cell_radix(&machine.structure, &machine.readout.clamps)?;
        let universe: usize = radix.iter().product();
        let readout = harvest.readout();
        // The branch rows are the weight-one rows after the harvest's own.
        let rows: Vec<usize> = (harvest.rows()..probe.rows()).filter(|&row| probe.weights[row] > 0.0).collect();
        let mut groups: BTreeMap<usize, Vec<(usize, Vec<f64>)>> = BTreeMap::new();
        for (&index, &row) in open.iter().zip(&rows) {
            groups.entry(cells[row]).or_default().push((index, probe.probabilities.row(row).to_vec()));
        }
        let mut found = Vec::new();
        for (cell, members) in groups {
            let current = machine.distribution(cell, readout);
            let distributions: Vec<Vec<f64>> = members.iter().map(|(_, probabilities)| probabilities.clone()).collect();
            let (chosen, centroid) = disagreeing(&distributions, &current);
            if chosen.is_empty() {
                continue;
            }
            let (bits, fresh) = fresh_cell(&report.fitted, &centroid, universe)?;
            let gain: f64 = chosen
                .iter()
                .map(|&member| divergence(&distributions[member], &current) - divergence(&distributions[member], &fresh))
                .sum();
            if scale * gain > bits as f64 {
                found.extend(chosen.iter().map(|&member| members[member].0));
            }
        }
        if found.is_empty() {
            break;
        }
        for &index in &found {
            taken[index] = true;
        }
        admitted.extend(&found);
        let branches: Vec<Branch> = admitted.iter().map(|&index| pool[index].clone()).collect();
        current = harvest.with_branches(&branches)?;
        report = search(&report.fitted.machine.structure, &current, language)?;
        rounds.push(Round { counterexamples: found.len(), fitted: report.fitted.clone() });
    }
    Ok(Refinement { report, rounds, admitted, harvest: current })
}

/// A do-intervention on a machine's state, set in the model through one writer into the
/// residual stream.
///
/// The writer is a stored matrix `W` (`width × inputs`) whose use `y = W a` adds into the
/// residual. Every row carries its writer input `a`, the residual `h` the write lands in
/// and a machine state label. Setting state `from` to state `to` is the set-type
/// requirement that, at the mean input of every state, the writer writes its native value
/// plus `h̄_to − h̄_from` for state `from` and its native value for every other state; the
/// edit compiler (`compile::linear`) returns the minimum-norm global edit of `W` that meets
/// it, or a witness that none does, with the damage on every row of the other states.
#[derive(Clone, Debug)]
pub struct StateIntervention<'a> {
    pub registry: &'a TensorRegistry,
    pub storage: TensorId,
    pub site: UseSiteId,
    /// The stored `W`, `width × inputs`.
    pub native: ArrayView2<'a, f64>,
    /// Per row, the writer's input, `rows × inputs`.
    pub inputs: ArrayView2<'a, f64>,
    /// Per row, the residual the write lands in, `rows × width`.
    pub residual: ArrayView2<'a, f64>,
    /// Per row, its state label.
    pub labels: &'a [usize],
}

impl StateIntervention<'_> {
    /// Compile `do(state := to)` on the rows of state `from`.
    pub fn compile(&self, from: usize, to: usize, governor: &MemoryGovernor) -> Result<LinearSiteReport, CausalError> {
        let rows = self.inputs.nrows();
        if self.residual.nrows() != rows || self.labels.len() != rows || self.residual.ncols() != self.native.nrows() {
            return Err(CausalError::Harvest("inputs, residual and labels must share their rows".to_string()));
        }
        let states = self.labels.iter().copied().max().map_or(0, |largest| largest + 1);
        let mut counts = vec![0.0_f64; states];
        let mut inputs = Array2::<f64>::zeros((states, self.inputs.ncols()));
        let mut residual = Array2::<f64>::zeros((states, self.residual.ncols()));
        for (row, &label) in self.labels.iter().enumerate() {
            counts[label] += 1.0;
            inputs.row_mut(label).scaled_add(1.0, &self.inputs.row(row));
            residual.row_mut(label).scaled_add(1.0, &self.residual.row(row));
        }
        if counts.get(from).is_none_or(|count| *count == 0.0) || counts.get(to).is_none_or(|count| *count == 0.0) {
            return Err(CausalError::Harvest(format!("state {from} or {to} has no row")));
        }
        let occupied: Vec<usize> = (0..states).filter(|&state| counts[state] > 0.0).collect();
        for &state in &occupied {
            inputs.row_mut(state).mapv_inplace(|value| value / counts[state]);
            residual.row_mut(state).mapv_inplace(|value| value / counts[state]);
        }
        let shift = &residual.row(to) - &residual.row(from);
        let mut means = Array2::<f64>::zeros((occupied.len(), self.inputs.ncols()));
        let mut targets = Array2::<f64>::zeros((occupied.len(), self.native.nrows()));
        for (position, &state) in occupied.iter().enumerate() {
            means.row_mut(position).assign(&inputs.row(state));
            let mut target = self.native.dot(&inputs.row(state));
            if state == from {
                target += &shift;
            }
            targets.row_mut(position).assign(&target);
        }
        let off: Vec<usize> = (0..rows).filter(|&row| self.labels[row] != from).collect();
        let off_inputs = self.inputs.select(ndarray::Axis(0), &off);
        let problem = LinearSiteProblem {
            registry: self.registry,
            storage: self.storage.clone(),
            native: self.native,
            requirements: vec![Requirement::Linear {
                site: self.site.clone(),
                inputs: means.view(),
                targets: targets.view(),
                target_radius: 0.0,
                class: ResponseClass::Sample,
            }],
            metric: EditMetric::frobenius(),
            ties: Vec::new(),
            off_target: vec![OffTargetInputs { site: self.site.clone(), inputs: off_inputs.view() }],
        };
        compile_linear_site(&problem, &format!("state {from} := {to}"), governor).map_err(|error| CausalError::Compile(error.to_string()))
    }
}

#[cfg(test)]
#[path = "causal_states_tests.rs"]
mod tests;
