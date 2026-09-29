//! Baseline driver (#2951): greedy MDL selection of an executable program for the one-layer
//! modular-addition transformer, an independent reference for the general decomposition engine
//! (`examples/mpd_engine_modadd_2951.rs`). It reads the float64 export of
//! `bench/mpd_engine_export_2951.py` and prints the selected program's total bits, reals, clean
//! max KL and argmax as JSON. It is research code for one toy, so it lives here, not in the
//! library; it calls the library owners (`codec`, `precision`, `fit::decide_proposal`,
//! `bounds::kl_over_logit_boxes`, `cyclic_action`, `verify`).
//!
//! Greedy MDL selection of an executable program for a one-layer transformer on a cyclic token
//! group (#2951).
//!
//! # The model and its exact rewrite `R`
//!
//! [`CyclicTransformer`] is a one-layer attention + ReLU MLP transformer that reads two operand
//! tokens `a, b ∈ Z_p` and a query token `=` and emits logits over `Z_p` at `=` (no LayerNorm;
//! the `#2951` modular-addition benchmark). Only the `=` position is read, and its query is
//! input-independent. With the characters `D(ω_k x) = (cos ω_k x, sin ω_k x)`, `ω_k = 2πk/p`,
//! `k = 1..(p−1)/2`, and the closed-form planes of `cyclic_action::cyclic_planes`
//! (`e(x) = c₀ + Σ_k U_k D(ω_k x)` for the embedding, `W_U[c] = u₀ + Σ_k V_k D(ω_k c)`), every
//! stage is exact over all planes ([`ExactRewrite`]):
//!
//! ```text
//! scores   s_hj = σ_hj + Σ_k g_hk · D(ω_k x_j)                  (j = 0, 1; s_h2 = σ_h2)
//! routing  α_h  = softmax(s_h0, s_h1, s_h2)
//! moved    ζ_hk = α_h0 D(ω_k a) + α_h1 D(ω_k b)
//! MLP      pre_n = β_n + Σ_hj α_hj κ_nhj + Σ_hk w_nhk · ζ_hk,       act_n = relu(pre_n)
//! readout  y_k  = y0_k + Σ_n ρ_nk act_n + Σ_hj dk_khj α_hj + Σ_hk' dw_khk' · ζ_hk'
//! logits   ℓ(c) = Σ_k D(ω_k c) · y_k                               (up to a c-constant)
//! ```
//!
//! # Programs
//!
//! A program is `R` under a [`ProgramStructure`]: the embedding planes `S_E` it reads (scores and
//! values), the readout planes `S_U`, each head's score planes and direct-path planes (subsets of
//! `S_E`), each neuron's read planes (a subset of `S_E`), which heads route by their law (the mean
//! routing `ᾱ_h` of `R` over all `p²` inputs, a function of the weights) and which heads' `κ` are
//! folded into `β` at `ᾱ_h`. A neuron with no read and every `κ` folded is constant and folds into
//! `y0`. Every real is sent on the dyadic lattice of one declared precision per real group
//! (`precision::LatticeCode`), and the program's code is one message ([`ProgramCode`]): the
//! architecture integers in the prefix code, the plane sets and every per-head and per-neuron plane
//! set in the enumerative subset code, the head flags as fixed indices, and every real group as one
//! lattice message (`codec`, `precision`).
//!
//! # The counterfactual contract
//!
//! A decoded program is checked against `R` (equal to the model up to roundoff, which
//! [`ExactRewrite::forward_gap`] reports) by the exhaustive maximum of `KL(R ‖ P)` per row over
//! three declared families: every input (`clean`); for every plane `k`, every input with operand
//! `a`'s plane `k` turned by every shift `s = 1..p−1` (`rotate_k`, `cyclic_action::frequency_edit`
//! at the pos0 use site, applied to `R` and `P` alike); for every plane `k`, every input with plane
//! `k`'s moved content `ζ_·k` interchanged from a declared donor permutation (`swap_k`). Each row's
//! divergence is `bounds::kl_over_logit_boxes` over both executables' forward-error radii
//! ([`Executable::radius`]). Two certified bounds avoid re-evaluating rows (see [`Contract`]); every
//! row a bound cannot place within the tolerance is evaluated exactly.
//!
//! # Search
//!
//! [`search`] is coarse-to-fine batched greedy MDL: each batch of structural edits is accepted iff
//! `fit::decide_proposal` accepts it (strictly shorter decoded code, fidelity proven within the
//! tolerance) on one exhaustive contract evaluation. The tolerance is the only declared input.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};

use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_runtime::resource::{MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{Array1, Array2, Array3, Array4, ArrayD, ArrayView1, ArrayView2, ArrayView3, Axis, IxDyn};
use rayon::prelude::*;

use gam_sae::parameter_decomposition::bounds::{BoundError, kl_over_logit_boxes};
use gam_sae::parameter_decomposition::codec::{
    BitString, CodecError, encode_fixed_index, encode_prefix_integer, encode_subset, fixed_index_len_bits,
    prefix_integer_len_bits, signed_prefix_integer_len_bits, subset_code_len_bits,
};
use gam_sae::parameter_decomposition::cyclic_action::{CyclicActionError, RowCycle, cyclic_planes};
use gam_sae::parameter_decomposition::fit::{ProposalKind, ProposalRejection, decide_proposal};
use gam_sae::parameter_decomposition::precision::{DecodableArtifact, DecodedFidelity, DeclaredPrecision, LatticeCode, decode_then_evaluate};
use gam_sae::parameter_decomposition::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis};

/// Why a cyclic-program operation was refused.
#[derive(Debug)]
pub enum CyclicProgramError {
    /// A weight has the wrong shape, or the model's dimensions are inconsistent.
    Shape(String),
    /// A declared input (tolerance, donor permutation, structure, thread count) is invalid.
    Declaration(String),
    CyclicAction(CyclicActionError),
    Codec(CodecError),
    Lattice(String),
    Bound(BoundError),
    Evidence(EvidenceStatusError),
    Memory(MemoryReservationError),
    /// The start program misses the tolerance at every precision the derivation allows.
    StartMissesTolerance { max_kl: f64, tolerance: f64 },
    /// `fit::decide_proposal` refused the loop itself (the reference misses the tolerance).
    ReferenceMissesTolerance(String),
    /// The message written for a program is not the length its accounting states.
    CodeMismatch { accounted: u64, written: u64 },
    Threads(String),
}

impl fmt::Display for CyclicProgramError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(reason) => write!(formatter, "cyclic program: shape: {reason}"),
            Self::Declaration(reason) => write!(formatter, "cyclic program: declaration: {reason}"),
            Self::CyclicAction(error) => write!(formatter, "cyclic program: {error}"),
            Self::Codec(error) => write!(formatter, "cyclic program: code: {error}"),
            Self::Lattice(reason) => write!(formatter, "cyclic program: lattice: {reason}"),
            Self::Bound(error) => write!(formatter, "cyclic program: {error}"),
            Self::Evidence(error) => write!(formatter, "cyclic program: {error}"),
            Self::Memory(error) => write!(formatter, "cyclic program: {error}"),
            Self::StartMissesTolerance { max_kl, tolerance } => write!(
                formatter,
                "cyclic program: the start program's max KL {max_kl} misses the tolerance {tolerance}"
            ),
            Self::ReferenceMissesTolerance(reason) => {
                write!(formatter, "cyclic program: decide_proposal refused the reference: {reason}")
            }
            Self::CodeMismatch { accounted, written } => write!(
                formatter,
                "cyclic program: the accounted code length {accounted} differs from the written message's {written}"
            ),
            Self::Threads(reason) => write!(formatter, "cyclic program: thread pool: {reason}"),
        }
    }
}

impl std::error::Error for CyclicProgramError {}

impl From<CodecError> for CyclicProgramError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

impl From<BoundError> for CyclicProgramError {
    fn from(error: BoundError) -> Self {
        Self::Bound(error)
    }
}

impl From<EvidenceStatusError> for CyclicProgramError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

impl From<MemoryReservationError> for CyclicProgramError {
    fn from(error: MemoryReservationError) -> Self {
        Self::Memory(error)
    }
}

impl From<CyclicActionError> for CyclicProgramError {
    fn from(error: CyclicActionError) -> Self {
        Self::CyclicAction(error)
    }
}

type Result<T> = std::result::Result<T, CyclicProgramError>;

fn shape(what: &str, found: &[usize], expected: &[usize]) -> Result<()> {
    if found == expected {
        Ok(())
    } else {
        Err(CyclicProgramError::Shape(format!("{what}: expected {expected:?}, found {found:?}")))
    }
}

// ------------------------------------------------------------------------------------------ model

/// The weights of a one-layer transformer on `Z_p` read at the `=` position.
#[derive(Clone, Copy, Debug)]
pub struct CyclicTransformer<'a> {
    /// `W_E`, `(p + 1) × d`; row `p` is the `=` token.
    pub embed: ArrayView2<'a, f64>,
    /// `W_pos`, `3 × d`.
    pub position: ArrayView2<'a, f64>,
    /// `W_Q`, `W_K`, `W_V`, each `heads × d_head × d`.
    pub query: ArrayView3<'a, f64>,
    pub key: ArrayView3<'a, f64>,
    pub value: ArrayView3<'a, f64>,
    /// `W_O`, `d × (heads · d_head)`.
    pub output: ArrayView2<'a, f64>,
    /// `W_in`, `n × d`, and `b_in`, `n`.
    pub read_in: ArrayView2<'a, f64>,
    pub read_in_bias: ArrayView1<'a, f64>,
    /// `W_out`, `d × n`, and `b_out`, `d`.
    pub write_out: ArrayView2<'a, f64>,
    pub write_out_bias: ArrayView1<'a, f64>,
    /// `W_U`, `p × d`.
    pub unembed: ArrayView2<'a, f64>,
}

impl CyclicTransformer<'_> {
    /// `(p, d, heads, d_head, neurons)` after checking every shape against them.
    pub fn dimensions(&self) -> Result<(usize, usize, usize, usize, usize)> {
        let (rows, d) = self.embed.dim();
        if rows < 4 || rows % 2 != 0 {
            return Err(CyclicProgramError::Shape(format!(
                "W_E has {rows} rows; a cyclic group of odd order p ≥ 3 plus the = token needs an even count ≥ 4"
            )));
        }
        let p = rows - 1;
        let (heads, d_head, width) = self.query.dim();
        let n = self.read_in.nrows();
        shape("W_pos", self.position.shape(), &[3, d])?;
        shape("W_Q", &[width], &[d])?;
        shape("W_K", self.key.shape(), &[heads, d_head, d])?;
        shape("W_V", self.value.shape(), &[heads, d_head, d])?;
        shape("W_O", self.output.shape(), &[d, heads * d_head])?;
        shape("W_in", self.read_in.shape(), &[n, d])?;
        shape("b_in", self.read_in_bias.shape(), &[n])?;
        shape("W_out", self.write_out.shape(), &[d, n])?;
        shape("b_out", self.write_out_bias.shape(), &[d])?;
        shape("W_U", self.unembed.shape(), &[p, d])?;
        if heads == 0 || d_head == 0 || n == 0 {
            return Err(CyclicProgramError::Shape("heads, d_head and neurons must be positive".to_string()));
        }
        Ok((p, d, heads, d_head, n))
    }

    /// The model's logits at `=` for operands `(a, b)`; `pos0` is added to operand `a`'s embedding
    /// (an edit of `W_E` at the pos0 use site).
    pub fn logits(&self, a: usize, b: usize, pos0: Option<ArrayView1<'_, f64>>) -> Array1<f64> {
        let p = self.unembed.nrows();
        let (heads, d_head, _) = self.query.dim();
        let mut x0 = &self.embed.row(a) + &self.position.row(0);
        if let Some(delta) = pos0 {
            x0 += &delta;
        }
        let x1 = &self.embed.row(b) + &self.position.row(1);
        let x2 = &self.embed.row(p) + &self.position.row(2);
        let tokens = [x0, x1, x2.clone()];
        let scale = 1.0 / (d_head as f64).sqrt();
        let mut concatenated = Array1::<f64>::zeros(heads * d_head);
        for h in 0..heads {
            let q = self.query.index_axis(Axis(0), h).dot(&x2);
            let scores: Vec<f64> =
                tokens.iter().map(|x| self.key.index_axis(Axis(0), h).dot(x).dot(&q) * scale).collect();
            let weights = softmax3([scores[0], scores[1], scores[2]]);
            for (j, x) in tokens.iter().enumerate() {
                let v = self.value.index_axis(Axis(0), h).dot(x);
                concatenated.slice_mut(ndarray::s![h * d_head..(h + 1) * d_head]).scaled_add(weights[j], &v);
            }
        }
        let mid = &x2 + &self.output.dot(&concatenated);
        let pre = self.read_in.dot(&mid) + self.read_in_bias;
        let act = pre.mapv(|value| value.max(0.0));
        let fin = &mid + &self.write_out.dot(&act) + self.write_out_bias;
        self.unembed.dot(&fin)
    }
}

fn softmax3(scores: [f64; 3]) -> [f64; 3] {
    let top = scores[0].max(scores[1]).max(scores[2]);
    let e = [(scores[0] - top).exp(), (scores[1] - top).exp(), (scores[2] - top).exp()];
    let total = e[0] + e[1] + e[2];
    [e[0] / total, e[1] / total, e[2] / total]
}

/// The token cycle `x → x + 1 (mod p)` over `W_E`'s first `p` rows; row `p` (`=`) is fixed.
fn token_cycle(p: usize, rows: usize) -> Result<RowCycle> {
    let successor: Vec<usize> = (0..rows).map(|x| if x < p { (x + 1) % p } else { x }).collect();
    Ok(RowCycle::from_successor(&successor)?)
}

/// `D(ω_k x)` for every token `x` and plane `k = 1..K`: `p × K × 2`.
fn characters(p: usize, planes: usize) -> Array3<f64> {
    Array3::from_shape_fn((p, planes, 2), |(x, k, t)| {
        let angle = 2.0 * PI * (((k + 1) * x) % p) as f64 / p as f64;
        if t == 0 { angle.cos() } else { angle.sin() }
    })
}

// ------------------------------------------------------------------------------- exact rewrite R

/// The real groups of a program, in message order.
pub const GROUPS: [&str; 10] = ["sig", "g", "abar", "beta", "kap", "w", "rho", "y0", "dk", "dw"];
const SIG: usize = 0;
const G: usize = 1;
const ABAR: usize = 2;
const BETA: usize = 3;
const KAP: usize = 4;
const W: usize = 5;
const RHO: usize = 6;
const Y0: usize = 7;
const DK: usize = 8;
const DW: usize = 9;

/// Every coefficient of `R` over all planes, from the weights alone.
#[derive(Clone, Debug)]
pub struct ExactRewrite {
    pub p: usize,
    pub planes: usize,
    pub heads: usize,
    pub neurons: usize,
    /// `σ` (`heads × 3`), `g` (`heads × K × 2`).
    pub sig: Array2<f64>,
    pub g: Array3<f64>,
    /// `β` (`n`), `κ` (`n × heads × 3`), `w` (`n × heads × K × 2`).
    pub beta: Array1<f64>,
    pub kap: Array3<f64>,
    pub w: Array4<f64>,
    /// `ρ` (`n × K × 2`), `y0` (`K × 2`), `dk` (`K × 2 × heads × 3`), `dw` (`K × 2 × heads × K × 2`).
    pub rho: Array3<f64>,
    pub y0: Array2<f64>,
    pub dk: Array4<f64>,
    pub dw: ArrayD<f64>,
    /// `ᾱ_h`, the mean of `R`'s routing over all `p²` inputs (`heads × 3`).
    pub abar: Array2<f64>,
    /// `D(ω_k x)`, `p × K × 2`.
    pub chars: Array3<f64>,
    /// The largest `|R(a, b, c) − model(a, b, c)|` over all inputs after centring each row over `c`.
    pub forward_gap: f64,
}

impl ExactRewrite {
    pub fn new(model: &CyclicTransformer<'_>) -> Result<Self> {
        let (p, d, heads, d_head, n) = model.dimensions()?;
        let planes = (p - 1) / 2;
        let embed = cyclic_planes(model.embed, &token_cycle(p, p + 1)?)?;
        let unembed = cyclic_planes(model.unembed, &token_cycle(p, p)?)?;
        let u = |k: usize, t: usize| embed.planes.column(2 * k + t);
        let v = |k: usize, t: usize| unembed.planes.column(2 * k + t);
        let x2 = &model.embed.row(p) + &model.position.row(2);
        let base = [&embed.mean + &model.position.row(0), &embed.mean + &model.position.row(1), x2.clone()];
        let scale = 1.0 / (d_head as f64).sqrt();
        let mut qk = Array2::<f64>::zeros((heads, d));
        let mut ov = Vec::with_capacity(heads);
        for h in 0..heads {
            let q = model.query.index_axis(Axis(0), h).dot(&x2);
            qk.row_mut(h).assign(&(model.key.index_axis(Axis(0), h).t().dot(&q) * scale));
            let block = model.output.slice(ndarray::s![.., h * d_head..(h + 1) * d_head]);
            ov.push(block.dot(&model.value.index_axis(Axis(0), h)));
        }
        let mut sig = Array2::<f64>::zeros((heads, 3));
        let mut g = Array3::<f64>::zeros((heads, planes, 2));
        let mut ou = Array4::<f64>::zeros((heads, planes, 2, d));
        let mut ob = Array3::<f64>::zeros((heads, 3, d));
        for h in 0..heads {
            for j in 0..3 {
                sig[[h, j]] = qk.row(h).dot(&base[j]);
                ob.slice_mut(ndarray::s![h, j, ..]).assign(&ov[h].dot(&base[j]));
            }
            for k in 0..planes {
                for t in 0..2 {
                    g[[h, k, t]] = qk.row(h).dot(&u(k, t));
                    ou.slice_mut(ndarray::s![h, k, t, ..]).assign(&ov[h].dot(&u(k, t)));
                }
            }
        }
        let beta = model.read_in.dot(&x2) + model.read_in_bias;
        let mut kap = Array3::<f64>::zeros((n, heads, 3));
        let mut w = Array4::<f64>::zeros((n, heads, planes, 2));
        for (neuron, row) in model.read_in.outer_iter().enumerate() {
            for h in 0..heads {
                for j in 0..3 {
                    kap[[neuron, h, j]] = row.dot(&ob.slice(ndarray::s![h, j, ..]));
                }
                for k in 0..planes {
                    for t in 0..2 {
                        w[[neuron, h, k, t]] = row.dot(&ou.slice(ndarray::s![h, k, t, ..]));
                    }
                }
            }
        }
        let mut rho = Array3::<f64>::zeros((n, planes, 2));
        let mut y0 = Array2::<f64>::zeros((planes, 2));
        let mut dk = Array4::<f64>::zeros((planes, 2, heads, 3));
        let mut dw = ArrayD::<f64>::zeros(IxDyn(&[planes, 2, heads, planes, 2]));
        let bias = &x2 + &model.write_out_bias;
        for k in 0..planes {
            for t in 0..2 {
                let vk = v(k, t);
                let written = model.write_out.t().dot(&vk);
                for neuron in 0..n {
                    rho[[neuron, k, t]] = written[neuron];
                }
                y0[[k, t]] = vk.dot(&bias);
                for h in 0..heads {
                    for j in 0..3 {
                        dk[[k, t, h, j]] = vk.dot(&ob.slice(ndarray::s![h, j, ..]));
                    }
                    for k2 in 0..planes {
                        for s in 0..2 {
                            dw[[k, t, h, k2, s]] = vk.dot(&ou.slice(ndarray::s![h, k2, s, ..]));
                        }
                    }
                }
            }
        }
        let chars = characters(p, planes);
        let mut rewrite = Self {
            p,
            planes,
            heads,
            neurons: n,
            sig,
            g,
            beta,
            kap,
            w,
            rho,
            y0,
            dk,
            dw,
            abar: Array2::zeros((heads, 3)),
            chars,
            forward_gap: 0.0,
        };
        rewrite.abar = rewrite.mean_routing();
        rewrite.forward_gap = rewrite.gap_to(model);
        Ok(rewrite)
    }

    /// `R`'s routing at `(a, b)`.
    fn routing(&self, a: usize, b: usize) -> Array2<f64> {
        let mut alpha = Array2::<f64>::zeros((self.heads, 3));
        for h in 0..self.heads {
            let mut s = [self.sig[[h, 0]], self.sig[[h, 1]], self.sig[[h, 2]]];
            for k in 0..self.planes {
                for t in 0..2 {
                    s[0] += self.g[[h, k, t]] * self.chars[[a, k, t]];
                    s[1] += self.g[[h, k, t]] * self.chars[[b, k, t]];
                }
            }
            let weights = softmax3(s);
            for j in 0..3 {
                alpha[[h, j]] = weights[j];
            }
        }
        alpha
    }

    fn mean_routing(&self) -> Array2<f64> {
        let mut total = Array2::<f64>::zeros((self.heads, 3));
        for a in 0..self.p {
            for b in 0..self.p {
                total += &self.routing(a, b);
            }
        }
        total / (self.p * self.p) as f64
    }

    /// `R`'s logits at `(a, b)`, straight from the coefficients.
    pub fn logits(&self, a: usize, b: usize) -> Array1<f64> {
        let alpha = self.routing(a, b);
        let mut y = self.y0.clone();
        let mut act = self.beta.clone();
        for neuron in 0..self.neurons {
            for h in 0..self.heads {
                for j in 0..3 {
                    act[neuron] += alpha[[h, j]] * self.kap[[neuron, h, j]];
                }
                for k in 0..self.planes {
                    for t in 0..2 {
                        let zeta = alpha[[h, 0]] * self.chars[[a, k, t]] + alpha[[h, 1]] * self.chars[[b, k, t]];
                        act[neuron] += self.w[[neuron, h, k, t]] * zeta;
                    }
                }
            }
        }
        act.mapv_inplace(|value| value.max(0.0));
        for k in 0..self.planes {
            for t in 0..2 {
                y[[k, t]] += (0..self.neurons).map(|neuron| self.rho[[neuron, k, t]] * act[neuron]).sum::<f64>();
                for h in 0..self.heads {
                    for j in 0..3 {
                        y[[k, t]] += self.dk[[k, t, h, j]] * alpha[[h, j]];
                    }
                    for k2 in 0..self.planes {
                        for s in 0..2 {
                            let zeta = alpha[[h, 0]] * self.chars[[a, k2, s]] + alpha[[h, 1]] * self.chars[[b, k2, s]];
                            y[[k, t]] += self.dw[[k, t, h, k2, s]] * zeta;
                        }
                    }
                }
            }
        }
        Array1::from_shape_fn(self.p, |c| {
            (0..self.planes).map(|k| y[[k, 0]] * self.chars[[c, k, 0]] + y[[k, 1]] * self.chars[[c, k, 1]]).sum()
        })
    }

    fn gap_to(&self, model: &CyclicTransformer<'_>) -> f64 {
        let p = self.p;
        (0..p * p)
            .into_par_iter()
            .map(|row| {
                let (a, b) = (row / p, row % p);
                let r = self.logits(a, b);
                let m = model.logits(a, b, None);
                let diff = &r - &m;
                let mean = diff.sum() / p as f64;
                diff.iter().fold(0.0_f64, |worst, value| worst.max((value - mean).abs()))
            })
            .reduce(|| 0.0, f64::max)
    }

    /// The values of each real group before coding, for a structure.
    fn group_values(&self, structure: &ProgramStructure, fold_at: ArrayView2<'_, f64>) -> Vec<ArrayD<f64>> {
        let fold: Vec<bool> = (0..self.heads).map(|h| structure.folded[h] || structure.law[h]).collect();
        let vary = structure.varying(self);
        let mut beta = self.beta.clone();
        for neuron in 0..self.neurons {
            for h in (0..self.heads).filter(|&h| fold[h]) {
                for j in 0..3 {
                    beta[neuron] += self.kap[[neuron, h, j]] * fold_at[[h, j]];
                }
            }
        }
        let mut y0 = self.y0.clone();
        for k in 0..self.planes {
            for t in 0..2 {
                for neuron in (0..self.neurons).filter(|&neuron| !vary[neuron]) {
                    y0[[k, t]] += beta[neuron].max(0.0) * self.rho[[neuron, k, t]];
                }
                for h in (0..self.heads).filter(|&h| structure.law[h]) {
                    for j in 0..3 {
                        y0[[k, t]] += self.dk[[k, t, h, j]] * fold_at[[h, j]];
                    }
                }
            }
        }
        vec![
            self.sig.clone().into_dyn(),
            self.g.clone().into_dyn(),
            fold_at.to_owned().into_dyn(),
            beta.into_dyn(),
            self.kap.clone().into_dyn(),
            self.w.clone().into_dyn(),
            self.rho.clone().into_dyn(),
            y0.into_dyn(),
            self.dk.clone().into_dyn(),
            self.dw.clone(),
        ]
    }
}

// ------------------------------------------------------------------------------------- structure

/// Which parts of `R` a program keeps (module documentation, *Programs*).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramStructure {
    /// `S_E` and `S_U` over the `K` planes.
    pub embed: Vec<bool>,
    pub readout: Vec<bool>,
    /// Per head, its score planes and direct-path planes (`heads × K`, read within `S_E`).
    pub score: Array2<bool>,
    pub direct: Array2<bool>,
    /// Per neuron, its read planes (`n × K`, read within `S_E`).
    pub reads: Array2<bool>,
    /// Heads routing by their law `ᾱ_h`, and heads whose `κ` is folded into `β` at `ᾱ_h`.
    pub law: Vec<bool>,
    pub folded: Vec<bool>,
    /// The declared precision of each real group in `GROUPS` order; `None` sends exact reals (a
    /// reference program, never coded).
    pub fraction_bits: Option<[i32; 10]>,
}

impl ProgramStructure {
    /// `R` itself: every plane, read, score plane and direct path, nothing folded.
    pub fn full(rewrite: &ExactRewrite, fraction_bits: Option<[i32; 10]>) -> Self {
        let (k, h, n) = (rewrite.planes, rewrite.heads, rewrite.neurons);
        Self {
            embed: vec![true; k],
            readout: vec![true; k],
            score: Array2::from_elem((h, k), true),
            direct: Array2::from_elem((h, k), true),
            reads: Array2::from_elem((n, k), true),
            law: vec![false; h],
            folded: vec![false; h],
            fraction_bits,
        }
    }

    fn planes_in(&self, row: ArrayView1<'_, bool>) -> Vec<usize> {
        row.iter().enumerate().filter(|&(k, &on)| on && self.embed[k]).map(|(k, _)| k).collect()
    }

    /// Neurons whose pre-activation depends on the input.
    fn varying(&self, rewrite: &ExactRewrite) -> Vec<bool> {
        let unfolded = (0..rewrite.heads).any(|h| !(self.folded[h] || self.law[h]));
        (0..rewrite.neurons).map(|neuron| unfolded || !self.planes_in(self.reads.row(neuron)).is_empty()).collect()
    }

    /// The presence mask of each real group.
    fn masks(&self, rewrite: &ExactRewrite) -> Vec<ArrayD<bool>> {
        let (k, heads, n) = (rewrite.planes, rewrite.heads, rewrite.neurons);
        let vary = self.varying(rewrite);
        let fold: Vec<bool> = (0..heads).map(|h| self.folded[h] || self.law[h]).collect();
        let live = |h: usize| !self.law[h];
        let e = &self.embed;
        let u = &self.readout;
        vec![
            ArrayD::from_shape_fn(IxDyn(&[heads, 3]), |i| live(i[0])),
            ArrayD::from_shape_fn(IxDyn(&[heads, k, 2]), |i| live(i[0]) && self.score[[i[0], i[1]]] && e[i[1]]),
            ArrayD::from_shape_fn(IxDyn(&[heads, 3]), |i| self.law[i[0]]),
            ArrayD::from_shape_fn(IxDyn(&[n]), |i| vary[i[0]]),
            ArrayD::from_shape_fn(IxDyn(&[n, heads, 3]), |i| vary[i[0]] && !fold[i[1]]),
            ArrayD::from_shape_fn(IxDyn(&[n, heads, k, 2]), |i| self.reads[[i[0], i[2]]] && e[i[2]]),
            ArrayD::from_shape_fn(IxDyn(&[n, k, 2]), |i| vary[i[0]] && u[i[1]]),
            ArrayD::from_shape_fn(IxDyn(&[k, 2]), |i| u[i[0]]),
            ArrayD::from_shape_fn(IxDyn(&[k, 2, heads, 3]), |i| u[i[0]] && live(i[2])),
            ArrayD::from_shape_fn(IxDyn(&[k, 2, heads, k, 2]), |i| u[i[0]] && self.direct[[i[2], i[3]]] && e[i[3]]),
        ]
    }
}

/// One structural edit of a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Proposal {
    /// One bit coarser declared precision for a real group (by index in `GROUPS`).
    Coarsen { group: usize },
    DropEmbedPlane { plane: usize },
    DropReadoutPlane { plane: usize },
    /// Head routes by its law `ᾱ_h`.
    RouteLaw { head: usize },
    /// Head's `κ` folds into `β` at `ᾱ_h`.
    FoldKap { head: usize },
    DropScorePlane { head: usize, plane: usize },
    DropDirectPlane { head: usize, plane: usize },
    DropRead { neuron: usize, plane: usize },
}

impl Proposal {
    fn kind_rank(&self) -> usize {
        match self {
            Self::Coarsen { .. } => 0,
            Self::DropEmbedPlane { .. } => 1,
            Self::DropReadoutPlane { .. } => 2,
            Self::RouteLaw { .. } => 3,
            Self::FoldKap { .. } => 4,
            Self::DropScorePlane { .. } => 5,
            Self::DropDirectPlane { .. } => 6,
            Self::DropRead { .. } => 7,
        }
    }

    /// The edited structure, or `None` when the edit no longer applies.
    pub fn apply(&self, structure: &ProgramStructure) -> Option<ProgramStructure> {
        let mut next = structure.clone();
        match *self {
            Self::Coarsen { group } => {
                let bits = next.fraction_bits.as_mut()?;
                bits[group] -= 1;
            }
            Self::DropEmbedPlane { plane } if structure.embed[plane] => next.embed[plane] = false,
            Self::DropReadoutPlane { plane } if structure.readout[plane] => next.readout[plane] = false,
            Self::RouteLaw { head } if !structure.law[head] => next.law[head] = true,
            Self::FoldKap { head } if !structure.law[head] && !structure.folded[head] => next.folded[head] = true,
            Self::DropScorePlane { head, plane } if structure.score[[head, plane]] && structure.embed[plane] => {
                next.score[[head, plane]] = false;
            }
            Self::DropDirectPlane { head, plane } if structure.direct[[head, plane]] && structure.embed[plane] => {
                next.direct[[head, plane]] = false;
            }
            Self::DropRead { neuron, plane } if structure.reads[[neuron, plane]] && structure.embed[plane] => {
                next.reads[[neuron, plane]] = false;
            }
            _ => return None,
        }
        Some(next)
    }

    /// Its identity for the refused set: a coarsening is named with the precision it proposes.
    fn key(&self, structure: &ProgramStructure) -> (Proposal, i32) {
        match (self, structure.fraction_bits) {
            (Self::Coarsen { group }, Some(bits)) => (*self, bits[*group] - 1),
            _ => (*self, 0),
        }
    }
}

/// Every single edit of `structure` at `level`.
fn proposals(structure: &ProgramStructure, realized: &RealizedProgram, level: usize) -> Vec<Proposal> {
    let mut out = Vec::new();
    let (heads, planes) = structure.score.dim();
    match level {
        0 => {
            out.extend((0..GROUPS.len()).filter(|&g| realized.counts[g] > 0).map(|group| Proposal::Coarsen { group }));
            out.extend((0..planes).filter(|&k| structure.embed[k]).map(|plane| Proposal::DropEmbedPlane { plane }));
            out.extend((0..planes).filter(|&k| structure.readout[k]).map(|plane| Proposal::DropReadoutPlane { plane }));
            for head in (0..heads).filter(|&h| !structure.law[h]) {
                out.push(Proposal::RouteLaw { head });
                if !structure.folded[head] {
                    out.push(Proposal::FoldKap { head });
                }
            }
        }
        1 => {
            for head in 0..heads {
                for plane in (0..planes).filter(|&k| structure.embed[k]) {
                    if !structure.law[head] && structure.score[[head, plane]] {
                        out.push(Proposal::DropScorePlane { head, plane });
                    }
                    if structure.direct[[head, plane]] {
                        out.push(Proposal::DropDirectPlane { head, plane });
                    }
                }
            }
        }
        _ => {
            for ((neuron, plane), &on) in structure.reads.indexed_iter() {
                if on && structure.embed[plane] {
                    out.push(Proposal::DropRead { neuron, plane });
                }
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------ realize and code

/// A program's reals as its decoder rebuilds them, and its exact code length.
#[derive(Clone, Debug)]
pub struct RealizedProgram {
    pub structure: ProgramStructure,
    /// Decoded values of each group in `GROUPS` order, full shape, zero where absent.
    pub decoded: Vec<ArrayD<f64>>,
    pub present: Vec<ArrayD<bool>>,
    /// Present reals per group.
    pub counts: [usize; 10],
    /// The lattice indices of each group's present reals, row-major (empty for exact reals).
    pub indices: Vec<Vec<i64>>,
    /// The index codewords' length per group.
    pub index_bits: [u64; 10],
    /// The whole message's length (`None` for exact reals).
    pub bits: Option<u64>,
    pub vary: Vec<bool>,
}

/// Lattice decodings memoized per group and precision (the values of every group but `β` and `y0`
/// are fixed by the weights).
#[derive(Default)]
struct LatticeCache {
    groups: HashMap<(usize, i32), (Vec<f64>, Vec<i64>, Vec<u64>)>,
}

impl LatticeCache {
    /// The decoded value, index and index codeword length of every entry of `values`.
    fn code(&mut self, group: usize, values: &ArrayD<f64>, bits: i32) -> Result<(Vec<f64>, Vec<i64>, Vec<u64>)> {
        let fixed = !matches!(group, BETA | Y0);
        if fixed && let Some(hit) = self.groups.get(&(group, bits)) {
            return Ok(hit.clone());
        }
        let precision = DeclaredPrecision::new(bits).map_err(CyclicProgramError::Lattice)?;
        let flat: Vec<f64> = values.iter().copied().collect();
        let code = LatticeCode::encode(&flat, precision).map_err(CyclicProgramError::Lattice)?;
        let decoded = code.decode().map_err(CyclicProgramError::Lattice)?;
        let indices = code.indices().to_vec();
        let lengths = indices.iter().map(|&index| signed_prefix_integer_len_bits(index)).collect::<std::result::Result<Vec<_>, _>>()?;
        let coded = (decoded, indices, lengths);
        if fixed {
            self.groups.insert((group, bits), coded.clone());
        }
        Ok(coded)
    }
}

fn realize(
    rewrite: &ExactRewrite,
    structure: &ProgramStructure,
    fold_at: ArrayView2<'_, f64>,
    cache: &mut LatticeCache,
) -> Result<RealizedProgram> {
    let values = rewrite.group_values(structure, fold_at);
    let present = structure.masks(rewrite);
    let mut decoded = Vec::with_capacity(GROUPS.len());
    let mut indices = Vec::with_capacity(GROUPS.len());
    let mut counts = [0usize; 10];
    let mut index_bits = [0u64; 10];
    for group in 0..GROUPS.len() {
        let mask = &present[group];
        counts[group] = mask.iter().filter(|&&on| on).count();
        match structure.fraction_bits {
            None => {
                decoded.push(ndarray::Zip::from(&values[group]).and(mask).map_collect(|&v, &on| if on { v } else { 0.0 }));
                indices.push(Vec::new());
            }
            Some(bits) => {
                let (dec, idx, lengths) = cache.code(group, &values[group], bits[group])?;
                let mut out = ArrayD::<f64>::zeros(mask.raw_dim());
                let mut kept = Vec::with_capacity(counts[group]);
                for (position, (slot, &on)) in out.iter_mut().zip(mask.iter()).enumerate() {
                    if on {
                        *slot = dec[position];
                        kept.push(idx[position]);
                        index_bits[group] += lengths[position];
                    }
                }
                decoded.push(out);
                indices.push(kept);
            }
        }
    }
    let mut realized = RealizedProgram {
        structure: structure.clone(),
        decoded,
        present,
        counts,
        indices,
        index_bits,
        bits: None,
        vary: structure.varying(rewrite),
    };
    if let Some(bits) = structure.fraction_bits {
        realized.bits = Some(code_length(rewrite, &realized, bits)?);
    }
    Ok(realized)
}

/// The per-neuron and per-head plane subsets the message sends, each within `S_E`.
fn subsets(structure: &ProgramStructure) -> (Vec<usize>, Vec<Option<Vec<usize>>>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let se: Vec<usize> = (0..structure.embed.len()).filter(|&k| structure.embed[k]).collect();
    let local = |row: ArrayView1<'_, bool>| -> Vec<usize> {
        se.iter().enumerate().filter(|&(_, &k)| row[k]).map(|(position, _)| position).collect()
    };
    let score = (0..structure.law.len())
        .map(|h| if structure.law[h] { None } else { Some(local(structure.score.row(h))) })
        .collect();
    let direct = (0..structure.law.len()).map(|h| local(structure.direct.row(h))).collect();
    let reads = structure.reads.outer_iter().map(local).collect();
    (se, score, direct, reads)
}

/// The exact length of the program's message (module documentation, *Programs*), accounted item
/// by item with `codec`'s length functions.
fn code_length(rewrite: &ExactRewrite, realized: &RealizedProgram, bits: [i32; 10]) -> Result<u64> {
    let structure = &realized.structure;
    let (se, score, direct, reads) = subsets(structure);
    let m = se.len();
    let u = structure.readout.iter().filter(|&&on| on).count();
    let flag = u64::from(fixed_index_len_bits(2)?);
    let mut total = 0u64;
    for value in [rewrite.p, rewrite.planes, rewrite.heads, rewrite.neurons] {
        total += prefix_integer_len_bits(value as u64)?;
    }
    total += subset_code_len_bits(rewrite.planes, m)? + subset_code_len_bits(rewrite.planes, u)?;
    for h in 0..rewrite.heads {
        total += flag;
        if let Some(planes) = &score[h] {
            total += flag + subset_code_len_bits(m, planes.len())?;
        }
        total += subset_code_len_bits(m, direct[h].len())?;
    }
    for planes in &reads {
        total += subset_code_len_bits(m, planes.len())?;
    }
    for group in 0..GROUPS.len() {
        total += prefix_integer_len_bits(realized.counts[group] as u64 + 1)?
            + signed_prefix_integer_len_bits(i64::from(bits[group]))?
            + realized.index_bits[group];
    }
    Ok(total)
}

/// The program's message, written: the length [`code_length`] accounts for must be its length.
fn write_message(rewrite: &ExactRewrite, realized: &RealizedProgram) -> Result<u64> {
    let Some(bits) = realized.structure.fraction_bits else {
        return Err(CyclicProgramError::Declaration("a program of exact reals has no message".to_string()));
    };
    let structure = &realized.structure;
    let (se, score, direct, reads) = subsets(structure);
    let readout: Vec<usize> = (0..rewrite.planes).filter(|&k| structure.readout[k]).collect();
    let mut out = BitString::new();
    for value in [rewrite.p, rewrite.planes, rewrite.heads, rewrite.neurons] {
        encode_prefix_integer(&mut out, value as u64)?;
    }
    encode_subset(&mut out, rewrite.planes, &se)?;
    encode_subset(&mut out, rewrite.planes, &readout)?;
    for h in 0..rewrite.heads {
        encode_fixed_index(&mut out, usize::from(structure.law[h]), 2)?;
        if let Some(planes) = &score[h] {
            encode_fixed_index(&mut out, usize::from(structure.folded[h]), 2)?;
            encode_subset(&mut out, se.len(), planes)?;
        }
        encode_subset(&mut out, se.len(), &direct[h])?;
    }
    for planes in &reads {
        encode_subset(&mut out, se.len(), planes)?;
    }
    for group in 0..GROUPS.len() {
        let precision = DeclaredPrecision::new(bits[group]).map_err(CyclicProgramError::Lattice)?;
        LatticeCode::from_indices(precision, realized.indices[group].clone())
            .and_then(|code| code.write(&mut out))
            .map_err(CyclicProgramError::Lattice)?;
    }
    let written = out.len_bits();
    match realized.bits {
        Some(accounted) if accounted == written => Ok(written),
        Some(accounted) => Err(CyclicProgramError::CodeMismatch { accounted, written }),
        None => Err(CyclicProgramError::Declaration("the program's length was not accounted".to_string())),
    }
}

// ------------------------------------------------------------------------------------ executable

/// A declared family of the contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    /// Every input `(a, b)`.
    Clean,
    /// Every input with plane `k`'s moved content interchanged from the donor permutation.
    Swap(usize),
    /// Every input and shift `s = 1..p−1` with operand `a`'s plane `k` turned by `s`.
    Rotate(usize),
}

impl Family {
    pub fn rows(&self, p: usize) -> usize {
        match self {
            Self::Rotate(_) => (p - 1) * p * p,
            _ => p * p,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Self::Clean => "clean".to_string(),
            Self::Swap(k) => format!("swap_k{}", k + 1),
            Self::Rotate(k) => format!("rotate_k{}", k + 1),
        }
    }

    /// Row `r`'s operands `(a, b)` and its edited operand: the turned `a` of a rotation or the
    /// donor row of a swap.
    fn row(&self, p: usize, r: usize, donor: &[usize]) -> (usize, usize, usize) {
        match self {
            Self::Clean => (r / p, r % p, 0),
            Self::Swap(_) => (r / p, r % p, donor[r]),
            Self::Rotate(_) => {
                let (shift, input) = (r / (p * p) + 1, r % (p * p));
                let a = input / p;
                (a, input % p, (a + shift) % p)
            }
        }
    }
}

/// A decoded program restricted to what it reads, its sums regrouped by operand token (exact
/// rearrangements of the stage sums of the module documentation), with a forward-error radius.
#[derive(Clone, Debug)]
pub struct Executable {
    p: usize,
    heads: usize,
    /// Plane `k`'s position in `S_E`, if read.
    local: Vec<Option<usize>>,
    law: Vec<bool>,
    sig: Array2<f64>,
    abar: Array2<f64>,
    /// `g_h · D(ω_k x)` per token, `S_E` plane and head, and summed over `S_E`: `p × m × H`, `p × H`.
    g_plane: Array3<f64>,
    g_sum: Array2<f64>,
    /// The varying neurons (indices into all neurons).
    pub neurons: Vec<usize>,
    beta: Array1<f64>,
    /// `κ` as `3H × n_v`.
    kap: Array2<f64>,
    /// `w_nh · D(ω_k x)`: `p × m × H × n_v`, and summed over `S_E`: `p × H × n_v`.
    w_plane: Array4<f64>,
    w_sum: Array3<f64>,
    /// `r_n(c) = Σ_{k ∈ S_U} ρ_nk · D(ω_k c)`: `n_v × p`.
    readout: Array2<f64>,
    /// `Σ_{k∈S_U} dk_khj · D(ω_k c)`: `3H × p`; `Σ_{k∈S_U} y0_k · D(ω_k c)`: `p`.
    dk: Array2<f64>,
    y0: Array1<f64>,
    /// `Σ_{k∈S_U} D(ω_k c) · dw_hk' D(ω_k' x)`: `p × m × H × p`, and summed over `S_E`: `p × H × p`.
    e_plane: Array4<f64>,
    e_sum: Array3<f64>,
    /// `max_c r_n(c) − min_c r_n(c)` per varying neuron.
    pub range: Array1<f64>,
    /// A bound on every computed pre-activation's error against its exact value, per varying neuron.
    pub pre_error: Array1<f64>,
    /// A bound on every computed logit's error against its exact value.
    pub radius: f64,
}

impl Executable {
    pub fn new(realized: &RealizedProgram, chars: &Array3<f64>) -> Result<Self> {
        let structure = &realized.structure;
        let (p, planes, _) = chars.dim();
        let heads = structure.law.len();
        let se: Vec<usize> = (0..planes).filter(|&k| structure.embed[k]).collect();
        let su: Vec<usize> = (0..planes).filter(|&k| structure.readout[k]).collect();
        let mut local = vec![None; planes];
        for (position, &k) in se.iter().enumerate() {
            local[k] = Some(position);
        }
        let neurons: Vec<usize> = (0..realized.vary.len()).filter(|&n| realized.vary[n]).collect();
        let (m, nv) = (se.len(), neurons.len());
        let d = &realized.decoded;
        let matrix = |group: usize| -> Result<Array2<f64>> {
            d[group].clone().into_dimensionality().map_err(|error| CyclicProgramError::Shape(format!("{}: {error}", GROUPS[group])))
        };
        let (sig, abar) = (matrix(SIG)?, matrix(ABAR)?);
        let dot2 = |x: usize, k: usize, v0: f64, v1: f64| v0 * chars[[x, k, 0]] + v1 * chars[[x, k, 1]];
        let g_plane = Array3::from_shape_fn((p, m, heads), |(x, j, h)| {
            dot2(x, se[j], d[G][[h, se[j], 0]], d[G][[h, se[j], 1]])
        });
        let g_sum = g_plane.sum_axis(Axis(1));
        let beta = Array1::from_shape_fn(nv, |v| d[BETA][[neurons[v]]]);
        let kap = Array2::from_shape_fn((3 * heads, nv), |(hj, v)| d[KAP][[neurons[v], hj / 3, hj % 3]]);
        let w_plane = Array4::from_shape_fn((p, m, heads, nv), |(x, j, h, v)| {
            dot2(x, se[j], d[W][[neurons[v], h, se[j], 0]], d[W][[neurons[v], h, se[j], 1]])
        });
        let w_sum = w_plane.sum_axis(Axis(1));
        let at_c = |c: usize, f: &dyn Fn(usize, usize) -> f64| -> f64 {
            su.iter().map(|&k| f(k, 0) * chars[[c, k, 0]] + f(k, 1) * chars[[c, k, 1]]).sum()
        };
        let readout = Array2::from_shape_fn((nv, p), |(v, c)| at_c(c, &|k, t| d[RHO][[neurons[v], k, t]]));
        let dk = Array2::from_shape_fn((3 * heads, p), |(hj, c)| at_c(c, &|k, t| d[DK][[k, t, hj / 3, hj % 3]]));
        let y0 = Array1::from_shape_fn(p, |c| at_c(c, &|k, t| d[Y0][[k, t]]));
        let dwc = Array4::from_shape_fn((heads, m, 2, p), |(h, j, s, c)| at_c(c, &|k, t| d[DW][[k, t, h, se[j], s]]));
        let e_plane = Array4::from_shape_fn((p, m, heads, p), |(x, j, h, c)| {
            dwc[[h, j, 0, c]] * chars[[x, se[j], 0]] + dwc[[h, j, 1, c]] * chars[[x, se[j], 1]]
        });
        let e_sum = e_plane.sum_axis(Axis(1));
        let range = Array1::from_shape_fn(nv, |v| {
            let row = readout.row(v);
            row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - row.fold(f64::INFINITY, |a, &b| a.min(b))
        });
        let mut executable = Self {
            p,
            heads,
            local,
            law: structure.law.clone(),
            sig,
            abar,
            g_plane,
            g_sum,
            neurons,
            beta,
            kap,
            w_plane,
            w_sum,
            readout,
            dk,
            y0,
            e_plane,
            e_sum,
            range,
            pre_error: Array1::zeros(nv),
            radius: 0.0,
        };
        executable.bound_errors(realized, &se, &su);
        Ok(executable)
    }

    /// The forward-error bounds. Every stage is a short sum of products of tables (each itself a
    /// sum of products with the characters, whose evaluation errs by at most a few ulps) and routing
    /// weights. A computed sum of `t` terms errs by at most `γ_t Σ|terms|`, and each term's inputs
    /// carry their own errors. Routing: a score errs by at most `e_s = γ(|σ| + Σ|g|)`, and the
    /// softmax moves each weight by at most `e_s` (its total variation is `tanh(osc/4) ≤ e_s/2`) plus
    /// `6u` for its own evaluation. A ReLU is 1-Lipschitz. The bounds are row-independent: every
    /// routing weight is in `[0, 1]`, `α_h0 + α_h1 ≤ 1`, and every character entry is in `[−1, 1]`.
    fn bound_errors(&mut self, realized: &RealizedProgram, se: &[usize], su: &[usize]) {
        let d = &realized.decoded;
        let (heads, m, nv) = (self.heads, se.len(), self.neurons.len());
        let u = UNIT_ROUNDOFF;
        let l1 = |a: f64, b: f64| a.abs() + b.abs();
        let gamma_table = accumulation_growth(2 * m.max(su.len()) * 2 + 8);
        // Scores and routing.
        let mut routing_error = vec![0.0_f64; heads];
        for h in 0..heads {
            if self.law[h] {
                continue;
            }
            let g_abs: f64 = se.iter().map(|&k| l1(d[G][[h, k, 0]], d[G][[h, k, 1]])).sum();
            let g_max = se.iter().map(|&k| l1(d[G][[h, k, 0]], d[G][[h, k, 1]])).fold(0.0, f64::max);
            let sig_max = (0..3).map(|j| d[SIG][[h, j]].abs()).fold(0.0, f64::max);
            let score_error = accumulation_growth(2 * m + 8) * (sig_max + g_abs + 2.0 * g_max);
            routing_error[h] = score_error + 6.0 * u;
        }
        // Pre-activations.
        let gamma_pre = accumulation_growth(9 * heads + 2);
        let mut act_bound = vec![0.0_f64; nv];
        for v in 0..nv {
            let neuron = self.neurons[v];
            let (mut magnitude, mut propagated) = (d[BETA][[neuron]].abs(), 0.0_f64);
            for h in 0..heads {
                let kap_max = (0..3).map(|j| d[KAP][[neuron, h, j]].abs()).fold(0.0, f64::max);
                let plane_abs: Vec<f64> = se.iter().map(|&k| l1(d[W][[neuron, h, k, 0]], d[W][[neuron, h, k, 1]])).collect();
                let reads = plane_abs.iter().sum::<f64>() + 4.0 * plane_abs.iter().copied().fold(0.0, f64::max);
                magnitude += kap_max + reads;
                propagated += routing_error[h] * (3.0 * kap_max + 2.0 * reads) + gamma_table * reads;
            }
            self.pre_error[v] = gamma_pre * magnitude + propagated;
            act_bound[v] = magnitude + self.pre_error[v];
        }
        // Logits.
        let readout_abs: Vec<f64> = (0..nv)
            .map(|v| su.iter().map(|&k| l1(d[RHO][[self.neurons[v], k, 0]], d[RHO][[self.neurons[v], k, 1]])).sum())
            .collect();
        let mut magnitude: f64 = su.iter().map(|&k| l1(d[Y0][[k, 0]], d[Y0][[k, 1]])).sum();
        let mut propagated = gamma_table * magnitude;
        for v in 0..nv {
            magnitude += act_bound[v] * readout_abs[v];
            propagated += self.pre_error[v] * readout_abs[v] + act_bound[v] * gamma_table * readout_abs[v];
        }
        for h in 0..heads {
            let dk_max = (0..3)
                .map(|j| su.iter().map(|&k| l1(d[DK][[k, 0, h, j]], d[DK][[k, 1, h, j]])).sum::<f64>())
                .fold(0.0, f64::max);
            let plane_abs: Vec<f64> = se
                .iter()
                .map(|&k2| su.iter().map(|&k| (0..2).map(|t| l1(d[DW][[k, t, h, k2, 0]], d[DW][[k, t, h, k2, 1]])).sum::<f64>()).sum())
                .collect();
            let direct = plane_abs.iter().sum::<f64>() + 4.0 * plane_abs.iter().copied().fold(0.0, f64::max);
            magnitude += 3.0 * dk_max + 2.0 * direct;
            propagated += routing_error[h] * (3.0 * dk_max + 2.0 * direct) + gamma_table * (3.0 * dk_max + 2.0 * direct);
        }
        let gamma_logit = accumulation_growth(nv + 9 * heads + 2);
        self.radius = ((gamma_logit * magnitude + propagated) * (1.0 + 8.0 * u)).next_up();
    }

    /// Routing of one row: `(x0, x1)` with operand `x0`'s plane `local` read at `turned` when
    /// given.
    fn routing(&self, x0: usize, x1: usize, turn: Option<(usize, usize)>) -> Vec<[f64; 3]> {
        let mut alpha = vec![[0.0; 3]; self.heads];
        for h in 0..self.heads {
            if self.law[h] {
                alpha[h] = [self.abar[[h, 0]], self.abar[[h, 1]], self.abar[[h, 2]]];
                continue;
            }
            let mut s0 = self.sig[[h, 0]] + self.g_sum[[x0, h]];
            if let Some((j, turned)) = turn {
                s0 += self.g_plane[[turned, j, h]] - self.g_plane[[x0, j, h]];
            }
            alpha[h] = softmax3([s0, self.sig[[h, 1]] + self.g_sum[[x1, h]], self.sig[[h, 2]]]);
        }
        alpha
    }

    /// The pre-activations of one row, for the varying neurons in `columns` (all when `None`).
    fn pre_row(&self, family: Family, a: usize, b: usize, edited: usize, alpha: &[[f64; 3]], donor_alpha: &[[f64; 3]], out: &mut [f64], columns: Option<&[usize]>) {
        let plane = match family {
            Family::Clean => None,
            Family::Swap(k) | Family::Rotate(k) => self.local[k],
        };
        let count = columns.map_or(self.neurons.len(), <[usize]>::len);
        for position in 0..count {
            let v = columns.map_or(position, |c| c[position]);
            let mut pre = self.beta[v];
            for h in 0..self.heads {
                for j in 0..3 {
                    pre += alpha[h][j] * self.kap[[3 * h + j, v]];
                }
                pre += alpha[h][0] * self.w_sum[[a, h, v]] + alpha[h][1] * self.w_sum[[b, h, v]];
                if let Some(j) = plane {
                    match family {
                        Family::Rotate(_) => pre += alpha[h][0] * (self.w_plane[[edited, j, h, v]] - self.w_plane[[a, j, h, v]]),
                        Family::Swap(_) => {
                            let (da, db) = (edited / self.p, edited % self.p);
                            pre += donor_alpha[h][0] * self.w_plane[[da, j, h, v]] + donor_alpha[h][1] * self.w_plane[[db, j, h, v]]
                                - alpha[h][0] * self.w_plane[[a, j, h, v]]
                                - alpha[h][1] * self.w_plane[[b, j, h, v]];
                        }
                        Family::Clean => {}
                    }
                }
            }
            out[position] = pre;
        }
    }

    fn row_alphas(&self, family: Family, a: usize, b: usize, edited: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        match family {
            Family::Rotate(k) => (self.routing(a, b, self.local[k].map(|j| (j, edited))), Vec::new()),
            Family::Swap(k) if self.local[k].is_some() => {
                (self.routing(a, b, None), self.routing(edited / self.p, edited % self.p, None))
            }
            _ => (self.routing(a, b, None), Vec::new()),
        }
    }

    /// The logits of rows `rows` of `family` (`rows × p`).
    pub fn logits(&self, family: Family, rows: std::ops::Range<usize>, donor: &[usize]) -> Array2<f64> {
        let (p, nv, count) = (self.p, self.neurons.len(), rows.len());
        let mut act = Array2::<f64>::zeros((count, nv));
        let mut extra = Array2::<f64>::zeros((count, p));
        let plane = match family {
            Family::Clean => None,
            Family::Swap(k) | Family::Rotate(k) => self.local[k],
        };
        for (i, r) in rows.enumerate() {
            let (a, b, edited) = family.row(p, r, donor);
            let (alpha, donor_alpha) = self.row_alphas(family, a, b, edited);
            if let Some(slice) = act.row_mut(i).as_slice_mut() {
                self.pre_row(family, a, b, edited, &alpha, &donor_alpha, slice, None);
                for value in slice.iter_mut() {
                    *value = value.max(0.0);
                }
            }
            let mut row = extra.row_mut(i);
            row.assign(&self.y0);
            for h in 0..self.heads {
                for j in 0..3 {
                    row.scaled_add(alpha[h][j], &self.dk.row(3 * h + j));
                }
                row.scaled_add(alpha[h][0], &self.e_sum.slice(ndarray::s![a, h, ..]));
                row.scaled_add(alpha[h][1], &self.e_sum.slice(ndarray::s![b, h, ..]));
                if let Some(j) = plane {
                    match family {
                        Family::Rotate(_) => {
                            row.scaled_add(alpha[h][0], &self.e_plane.slice(ndarray::s![edited, j, h, ..]));
                            row.scaled_add(-alpha[h][0], &self.e_plane.slice(ndarray::s![a, j, h, ..]));
                        }
                        Family::Swap(_) => {
                            let (da, db) = (edited / p, edited % p);
                            row.scaled_add(donor_alpha[h][0], &self.e_plane.slice(ndarray::s![da, j, h, ..]));
                            row.scaled_add(donor_alpha[h][1], &self.e_plane.slice(ndarray::s![db, j, h, ..]));
                            row.scaled_add(-alpha[h][0], &self.e_plane.slice(ndarray::s![a, j, h, ..]));
                            row.scaled_add(-alpha[h][1], &self.e_plane.slice(ndarray::s![b, j, h, ..]));
                        }
                        Family::Clean => {}
                    }
                }
            }
        }
        act.dot(&self.readout) + extra
    }

    /// Whether the program reads plane `k` (an edit of plane `k` moves its outputs).
    pub fn reads(&self, k: usize) -> bool {
        self.local[k].is_some()
    }
}

// -------------------------------------------------------------------------------------- contract

/// A certified upper bound on `KL(m ‖ q′)` from `U ≥ KL(m ‖ q)` when `q′`'s logits are `q`'s moved
/// by `Δ` with `osc(Δ) ≤ O`:
///
/// ```text
/// KL(m‖q′) − KL(m‖q) = (E_q − E_m)[Δ] + (log E_q e^Δ − E_q Δ) ≤ TV(m, q)·O + O²/8,
/// ```
///
/// Hoeffding's lemma for the second term, and `TV(m, q) ≤ √(KL(m‖q)/2) ≤ √(U/2)` (Pinsker).
pub fn kl_moved(upper: f64, oscillation: f64) -> f64 {
    let bound = upper + (upper / 2.0).sqrt() * oscillation + oscillation * oscillation / 8.0;
    (bound * (1.0 + 8.0 * UNIT_ROUNDOFF)).next_up()
}

/// The largest oscillation `O` with `kl_moved(U, O) ≤ ε` (zero when `U ≥ ε`).
pub fn kl_budget(upper: f64, tolerance: f64) -> f64 {
    if upper >= tolerance {
        return 0.0;
    }
    let t = (upper / 2.0).sqrt();
    (4.0 * (-t + (t * t + (tolerance - upper) / 2.0).sqrt()) * (1.0 - 8.0 * UNIT_ROUNDOFF)).next_down().max(0.0)
}

fn log_softmax(row: ArrayView1<'_, f64>) -> (Array1<f64>, f64) {
    let top = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let lse = top + row.iter().map(|&z| (z - top).exp()).sum::<f64>().ln();
    (row.mapv(|z| z - lse), lse.abs() + row.fold(0.0_f64, |a, &b| a.max(b.abs())))
}

/// One row's divergence `KL(softmax r ‖ softmax c)` over both logit boxes: `(value, error)`, the
/// error infinite when the owner leaves the row unresolved.
fn row_kl(reference: ArrayView1<'_, f64>, reference_radius: ArrayView1<'_, f64>, candidate: ArrayView1<'_, f64>, candidate_radius: ArrayView1<'_, f64>) -> Result<(f64, f64)> {
    Ok(match kl_over_logit_boxes(reference, reference_radius, candidate, candidate_radius)? {
        EvidenceStatus::Exact { value, numerical_error, .. } => (value, numerical_error),
        other => (other.lower_bound().unwrap_or(0.0).max(0.0), f64::INFINITY),
    })
}

/// One family's statistics in a full evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct FamilyStats {
    pub family: Family,
    /// The largest certified upper end over the family's rows.
    pub upper: f64,
    /// Rows evaluated exactly; the rest (if any) are within a certified bound.
    pub exact_rows: u64,
    pub rows: u64,
    /// Mean exact KL and argmax agreement over the exactly evaluated rows.
    pub mean_kl: f64,
    pub argmax_agreement: f64,
}

/// What an evaluation of the contract proved.
#[derive(Clone, Debug)]
pub struct Evaluation {
    pub complete: bool,
    /// The largest certified upper end over every row.
    pub upper: f64,
    /// The largest exact row value and evaluation error.
    pub value: f64,
    pub error: f64,
    pub rows: u64,
    /// Some row is placed by a certified bound instead of its exact value.
    pub bounded: bool,
    /// Some row's divergence is unresolved over its logit boxes.
    pub unresolved: bool,
    /// A row whose proven lower end exceeds the tolerance: family, row, value, error.
    pub counterexample: Option<(Family, usize, f64, f64)>,
    /// Per rotate plane read by the program, every row's certified upper end.
    pub row_bounds: HashMap<usize, Vec<f64>>,
    pub families: Vec<FamilyStats>,
}

impl Evaluation {
    fn empty() -> Self {
        Self {
            complete: false,
            upper: 0.0,
            value: 0.0,
            error: 0.0,
            rows: 0,
            bounded: false,
            unresolved: false,
            counterexample: None,
            row_bounds: HashMap::new(),
            families: Vec::new(),
        }
    }

    /// The evaluation as an evidence status at `tolerance`.
    pub fn status(&self, tolerance: f64) -> Result<EvidenceStatus<String, String>> {
        let domain = "clean, swap_k and rotate_k for every plane k, every row".to_string();
        Ok(if let Some((family, row, value, error)) = self.counterexample {
            EvidenceStatus::counterexample(value, error, tolerance, format!("{} row {row}", family.name()))?
        } else if self.unresolved || !self.complete {
            EvidenceStatus::unresolved(self.value, f64::INFINITY, gam_sae::parameter_decomposition::supports::Extremum::Supremum, None, domain)?
        } else if self.bounded {
            EvidenceStatus::uniform_bound(self.upper, 0.0, domain)?
        } else {
            EvidenceStatus::exact(self.value, self.error, ExactBasis::Exhaustive { cardinality: self.rows }, None, domain)?
        })
    }
}

/// Accumulates rows of one family.
#[derive(Clone, Copy, Debug)]
struct Tally {
    upper: f64,
    value: f64,
    error: f64,
    exact_rows: u64,
    rows: u64,
    kl_sum: f64,
    agree: u64,
    unresolved: bool,
    /// The row with the largest proven lower end above the tolerance.
    worst: Option<(usize, f64, f64)>,
}

impl Tally {
    fn new() -> Self {
        Self { upper: 0.0, value: 0.0, error: 0.0, exact_rows: 0, rows: 0, kl_sum: 0.0, agree: 0, unresolved: false, worst: None }
    }

    fn exact(&mut self, row: usize, value: f64, error: f64, agree: bool, tolerance: f64) {
        self.rows += 1;
        self.exact_rows += 1;
        self.upper = self.upper.max((value + error).next_up());
        self.value = self.value.max(value);
        self.error = self.error.max(error);
        self.kl_sum += value;
        self.agree += u64::from(agree);
        self.unresolved |= !error.is_finite();
        let lower = if error.is_finite() { (value - error).next_down() } else { value };
        if lower > tolerance && self.worst.is_none_or(|(_, best, _)| value > best) {
            self.worst = Some((row, value, error));
        }
    }

    fn bounded(&mut self, bound: f64) {
        self.rows += 1;
        self.upper = self.upper.max(bound);
    }

    fn merge(mut self, other: Self) -> Self {
        self.upper = self.upper.max(other.upper);
        self.value = self.value.max(other.value);
        self.error = self.error.max(other.error);
        self.exact_rows += other.exact_rows;
        self.rows += other.rows;
        self.kl_sum += other.kl_sum;
        self.agree += other.agree;
        self.unresolved |= other.unresolved;
        self.worst = match (self.worst, other.worst) {
            (Some(x), Some(y)) => Some(if y.1 > x.1 { y } else { x }),
            (x, y) => x.or(y),
        };
        self
    }
}

/// The columns a read-drop candidate changes, with its reference.
pub struct ReadDrop<'e> {
    pub reference: &'e Executable,
    /// Positions of the changed neurons among the (shared) varying neurons.
    pub columns: Vec<usize>,
}

/// The declared contract (module documentation, *The counterfactual contract*), with its
/// precomputed bounds. For a rotate row,
/// `KL(R_s ‖ P_s) = KL(R_s ‖ R_0) + Σ_c R_s(c) (log R_0(c) − log P_s(c)) ≤ Kb + Σ_c Mb(c) (log R_0(c) − log P_s(c))⁺`
/// with `Kb(a, b) = max_s KL(R_s ‖ R_0)` and `Mb(a, b, c) = max_s R_s(c)` over every shift: a
/// certified bound that needs only `P` (`P_s = P_0` when `P` does not read the plane).
pub struct Contract {
    pub p: usize,
    pub planes: usize,
    pub tolerance: f64,
    reference: Executable,
    donor: Vec<usize>,
    /// `R`'s clean logits and log-softmax (`p² × p`).
    reference_clean: Array2<f64>,
    reference_clean_log: Array2<f64>,
    reference_clean_scale: Array1<f64>,
    /// `Kb` (`K × p²`, each an upper end) and `Mb` (`K` of `p² × p`, rounded up).
    kb: Array2<f64>,
    mb: Vec<Array2<f32>>,
    /// Planes whose bound once exceeded the tolerance; evaluated exactly first.
    hot: Vec<bool>,
    /// Early-exit order: the family that last refuted comes first.
    order: Vec<Family>,
    reservation: Option<MemoryReservation>,
}

fn blocks(rows: usize, p: usize) -> usize {
    rows.div_ceil(p)
}

impl Contract {
    pub fn new(rewrite: &ExactRewrite, donor: Vec<usize>, tolerance: f64, governor: &MemoryGovernor) -> Result<Self> {
        let p = rewrite.p;
        if !(tolerance.is_finite() && tolerance > 0.0) {
            return Err(CyclicProgramError::Declaration(format!("tolerance {tolerance} must be finite and positive")));
        }
        let mut seen = vec![false; p * p];
        if donor.len() != p * p || donor.iter().any(|&r| r >= p * p || std::mem::replace(&mut seen[r], true)) {
            return Err(CyclicProgramError::Declaration("the donor map must be a permutation of the p² inputs".to_string()));
        }
        let reservation = governor.try_reserve_dense_f64_copies(p * p, p * rewrite.planes, 1, "cyclic program contract bounds")?;
        let full = realize(rewrite, &ProgramStructure::full(rewrite, None), rewrite.abar.view(), &mut LatticeCache::default())?;
        let reference = Executable::new(&full, &rewrite.chars)?;
        let reference_clean = reference.logits(Family::Clean, 0..p * p, &donor);
        let mut reference_clean_log = Array2::<f64>::zeros((p * p, p));
        let mut reference_clean_scale = Array1::<f64>::zeros(p * p);
        for (i, row) in reference_clean.outer_iter().enumerate() {
            let (log, scale) = log_softmax(row);
            reference_clean_log.row_mut(i).assign(&log);
            reference_clean_scale[i] = scale;
        }
        let planes = rewrite.planes;
        let mut kb = Array2::<f64>::zeros((planes, p * p));
        let mut mb = Vec::with_capacity(planes);
        let radius = Array1::from_elem(p, reference.radius);
        for k in 0..planes {
            let family = Family::Rotate(k);
            let per_input: Vec<(usize, Vec<f64>, f64)> = (0..blocks(family.rows(p), p))
                .into_par_iter()
                .map(|block| -> Result<Vec<(usize, Vec<f64>, f64)>> {
                    let rows = block * p..((block + 1) * p).min(family.rows(p));
                    let logits = reference.logits(family, rows.clone(), &donor);
                    let mut out = Vec::with_capacity(rows.len());
                    for (i, r) in rows.enumerate() {
                        let input = r % (p * p);
                        let (value, error) = row_kl(logits.row(i), radius.view(), reference_clean.row(input), radius.view())?;
                        let (log, _) = log_softmax(logits.row(i));
                        out.push((input, log.iter().map(|&l| l.exp()).collect(), (value + error).next_up()));
                    }
                    Ok(out)
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect();
            let mut most = Array2::<f32>::zeros((p * p, p));
            for (input, probabilities, kl) in per_input {
                kb[[k, input]] = kb[[k, input]].max(kl);
                for (c, &probability) in probabilities.iter().enumerate() {
                    let up = ((probability * (1.0 + 16.0 * UNIT_ROUNDOFF)) as f32).next_up();
                    most[[input, c]] = most[[input, c]].max(up.min(1.0));
                }
            }
            mb.push(most);
        }
        Ok(Self {
            p,
            planes,
            tolerance,
            reference,
            donor,
            reference_clean,
            reference_clean_log,
            reference_clean_scale,
            kb,
            mb,
            hot: vec![false; planes],
            order: Vec::new(),
            reservation: Some(reservation),
        })
    }

    pub fn reference(&self) -> &Executable {
        &self.reference
    }

    /// The certified bound of the rows of a rotate family from the candidate's log-softmax rows
    /// `log` at inputs `inputs` (log-probability scales `scale`, radius `radius`).
    fn rotate_bound(&self, k: usize, input: usize, log: ArrayView1<'_, f64>, scale: f64, radius: f64) -> f64 {
        let p = self.p;
        let margin = 2.0 * (self.reference.radius + radius)
            + accumulation_growth(p + 4) * (scale + self.reference_clean_scale[input]);
        let mut total = 0.0_f64;
        for c in 0..p {
            let gap = self.reference_clean_log[[input, c]] - log[c] + margin;
            if gap > 0.0 {
                total += f64::from(self.mb[k][[input, c]]) * gap;
            }
        }
        ((self.kb[[k, input]] + total) * (1.0 + accumulation_growth(p + 2))).next_up()
    }

    /// Evaluates the exact rows `rows` of `family` for `candidate` (its clean logits `clean` serve
    /// the families it does not read).
    fn exact_rows(&self, candidate: &Executable, clean: &Array2<f64>, family: Family, rows: std::ops::Range<usize>, stop: &AtomicBool, stats: bool) -> Result<(Tally, Vec<f64>)> {
        let p = self.p;
        let (rr, rc) = (Array1::from_elem(p, self.reference.radius), Array1::from_elem(p, candidate.radius));
        let start = rows.start;
        let parts: Vec<(Tally, Vec<f64>)> = (0..blocks(rows.len(), p))
            .into_par_iter()
            .map(|block| -> Result<(Tally, Vec<f64>)> {
                let mut tally = Tally::new();
                if stop.load(Ordering::Relaxed) && !stats {
                    return Ok((tally, Vec::new()));
                }
                let block_rows = start + block * p..(start + (block + 1) * p).min(rows.end);
                let reference = match family {
                    Family::Clean => self.reference_clean.slice(ndarray::s![block_rows.clone(), ..]).to_owned(),
                    _ => self.reference.logits(family, block_rows.clone(), &self.donor),
                };
                let reads = match family {
                    Family::Clean => true,
                    Family::Swap(k) | Family::Rotate(k) => candidate.reads(k),
                };
                let own = if reads { Some(candidate.logits(family, block_rows.clone(), &self.donor)) } else { None };
                let mut uppers = Vec::with_capacity(block_rows.len());
                for (i, r) in block_rows.enumerate() {
                    let row = match &own {
                        Some(logits) => logits.row(i),
                        None => clean.row(r % (p * p)),
                    };
                    let (value, error) = row_kl(reference.row(i), rr.view(), row, rc.view())?;
                    let agree = gam_sae::parameter_decomposition::verify::computed_argmax(reference.row(i)) == gam_sae::parameter_decomposition::verify::computed_argmax(row);
                    tally.exact(r, value, error, agree, self.tolerance);
                    uppers.push((value + error).next_up());
                }
                if tally.worst.is_some() {
                    stop.store(true, Ordering::Relaxed);
                }
                Ok((tally, uppers))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tally = Tally::new();
        let mut uppers = Vec::with_capacity(rows.len());
        for (part, values) in parts {
            tally = tally.merge(part);
            uppers.extend(values);
        }
        Ok((tally, uppers))
    }

    /// Every row's certified upper end of a rotate family `P` reads, from the bound; `None` when
    /// some row's bound exceeds the tolerance.
    fn bounded_rotate(&self, candidate: &Executable, k: usize) -> Option<Vec<f64>> {
        let p = self.p;
        let family = Family::Rotate(k);
        let over = AtomicBool::new(false);
        let parts: Vec<Vec<f64>> = (0..blocks(family.rows(p), p))
            .into_par_iter()
            .map(|block| {
                if over.load(Ordering::Relaxed) {
                    return Vec::new();
                }
                let rows = block * p..((block + 1) * p).min(family.rows(p));
                let logits = candidate.logits(family, rows.clone(), &self.donor);
                let bounds: Vec<f64> = rows
                    .enumerate()
                    .map(|(i, r)| {
                        let (log, scale) = log_softmax(logits.row(i));
                        self.rotate_bound(k, r % (p * p), log.view(), scale, candidate.radius)
                    })
                    .collect();
                if bounds.iter().any(|&bound| bound > self.tolerance) {
                    over.store(true, Ordering::Relaxed);
                }
                bounds
            })
            .collect();
        if over.load(Ordering::Relaxed) { None } else { Some(parts.into_iter().flatten().collect()) }
    }

    /// Every row's certified upper end of a rotate family for a read-drop candidate, moved from
    /// the reference's row bounds `previous` by `kl_moved`; blocks the moved bound cannot place
    /// within the tolerance are evaluated exactly. `None` with the refuting tally when an exact
    /// row refutes.
    fn moved_rotate(&self, candidate: &Executable, clean: &Array2<f64>, drop: &ReadDrop<'_>, k: usize, previous: &[f64], stop: &AtomicBool) -> Result<(Tally, Vec<f64>)> {
        let p = self.p;
        let family = Family::Rotate(k);
        let reference = drop.reference;
        let columns = &drop.columns;
        let parts: Vec<(Tally, Vec<f64>)> = (0..blocks(family.rows(p), p))
            .into_par_iter()
            .map(|block| -> Result<(Tally, Vec<f64>)> {
                let mut tally = Tally::new();
                if stop.load(Ordering::Relaxed) {
                    return Ok((tally, Vec::new()));
                }
                let rows = block * p..((block + 1) * p).min(family.rows(p));
                let (mut before, mut after) = (vec![0.0; columns.len()], vec![0.0; columns.len()]);
                let numerical: f64 = columns
                    .iter()
                    .map(|&v| (reference.pre_error[v] + candidate.pre_error[v]) * reference.range[v])
                    .sum();
                let mut moved = Vec::with_capacity(rows.len());
                for r in rows.clone() {
                    let (a, b, edited) = family.row(p, r, &self.donor);
                    let (alpha, donor_alpha) = reference.row_alphas(family, a, b, edited);
                    reference.pre_row(family, a, b, edited, &alpha, &donor_alpha, &mut before, Some(columns));
                    candidate.pre_row(family, a, b, edited, &alpha, &donor_alpha, &mut after, Some(columns));
                    let oscillation: f64 = columns
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (after[i].max(0.0) - before[i].max(0.0)).abs() * reference.range[v])
                        .sum::<f64>()
                        + numerical;
                    moved.push(kl_moved(previous[r], oscillation * (1.0 + accumulation_growth(columns.len() + 4))));
                }
                if moved.iter().all(|&bound| bound <= self.tolerance) {
                    for &bound in &moved {
                        tally.bounded(bound);
                    }
                    return Ok((tally, moved));
                }
                let (exact, uppers) = self.exact_rows(candidate, clean, family, rows, stop, false)?;
                Ok((exact, uppers))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut tally = Tally::new();
        let mut uppers = Vec::with_capacity(family.rows(p));
        for (part, values) in parts {
            tally = tally.merge(part);
            uppers.extend(values);
        }
        Ok((tally, uppers))
    }

    /// The contract for `candidate`: early exit at the first refuting row unless `stats`.
    /// `exact_planes` are evaluated exactly, never by a bound. With `drop` and the reference's
    /// row bounds `previous`, rotate families of read planes move the reference's bounds.
    pub fn evaluate(&mut self, candidate: &Executable, stats: bool, exact_planes: &[usize], drop: Option<&ReadDrop<'_>>, previous: Option<&HashMap<usize, Vec<f64>>>) -> Result<Evaluation> {
        let p = self.p;
        let clean = candidate.logits(Family::Clean, 0..p * p, &self.donor);
        let mut units: Vec<Family> = vec![Family::Clean];
        units.extend((0..self.planes).map(Family::Swap));
        units.extend((0..self.planes).map(Family::Rotate));
        let front: Vec<Family> = self.order.iter().copied().filter(|f| units.contains(f)).collect();
        units.retain(|f| !front.contains(f));
        let units: Vec<Family> = front.into_iter().chain(units).collect();
        let mut evaluation = Evaluation::empty();
        let stop = AtomicBool::new(false);
        for family in units {
            let (tally, row_bounds) = match family {
                Family::Rotate(k) if !exact_planes.contains(&k) && !self.hot[k] => {
                    if candidate.reads(k) {
                        match (drop, previous.and_then(|bounds| bounds.get(&k))) {
                            (Some(drop), Some(previous)) => {
                                let (tally, uppers) = self.moved_rotate(candidate, &clean, drop, k, previous, &stop)?;
                                (tally, Some(uppers))
                            }
                            _ => match self.bounded_rotate(candidate, k) {
                                Some(bounds) => {
                                    let mut tally = Tally::new();
                                    for &bound in &bounds {
                                        tally.bounded(bound);
                                    }
                                    (tally, Some(bounds))
                                }
                                None => {
                                    self.hot[k] = true;
                                    let (tally, uppers) = self.exact_rows(candidate, &clean, family, 0..family.rows(p), &stop, stats)?;
                                    (tally, Some(uppers))
                                }
                            },
                        }
                    } else {
                        let mut tally = Tally::new();
                        let mut over = false;
                        for input in 0..p * p {
                            let (log, scale) = log_softmax(clean.row(input));
                            let bound = self.rotate_bound(k, input, log.view(), scale, candidate.radius);
                            over |= bound > self.tolerance;
                            tally.bounded(bound);
                        }
                        tally.rows = family.rows(p) as u64;
                        if over {
                            self.hot[k] = true;
                            let (exact, uppers) = self.exact_rows(candidate, &clean, family, 0..family.rows(p), &stop, stats)?;
                            (exact, Some(uppers))
                        } else {
                            (tally, None)
                        }
                    }
                }
                _ => {
                    let (tally, uppers) = self.exact_rows(candidate, &clean, family, 0..family.rows(p), &stop, stats)?;
                    (tally, if matches!(family, Family::Rotate(_)) { Some(uppers) } else { None })
                }
            };
            evaluation.upper = evaluation.upper.max(tally.upper);
            evaluation.value = evaluation.value.max(tally.value);
            evaluation.error = evaluation.error.max(tally.error);
            evaluation.rows += tally.rows;
            evaluation.bounded |= tally.exact_rows < tally.rows;
            evaluation.unresolved |= tally.unresolved;
            if stats {
                evaluation.families.push(FamilyStats {
                    family,
                    upper: tally.upper,
                    exact_rows: tally.exact_rows,
                    rows: tally.rows,
                    mean_kl: if tally.exact_rows > 0 { tally.kl_sum / tally.exact_rows as f64 } else { 0.0 },
                    argmax_agreement: if tally.exact_rows > 0 { tally.agree as f64 / tally.exact_rows as f64 } else { 0.0 },
                });
            }
            if let (Family::Rotate(k), Some(bounds)) = (family, row_bounds)
                && candidate.reads(k)
                && bounds.len() == family.rows(p)
            {
                evaluation.row_bounds.insert(k, bounds);
            }
            if let Some((row, value, error)) = tally.worst {
                evaluation.counterexample = Some((family, row, value, error));
                if !stats {
                    self.order.retain(|f| *f != family);
                    self.order.insert(0, family);
                    return Ok(evaluation);
                }
            }
        }
        evaluation.complete = true;
        Ok(evaluation)
    }

    /// Releases the bounds' memory reservation (the bounds stay; the reservation guards them
    /// while the search runs).
    pub fn release(&mut self) {
        drop(self.reservation.take());
    }
}

// -------------------------------------------------------------------------------------- precision

/// `S_G = max over clean inputs and classes c of Σ_{reals of G} |∂ℓ_c/∂real|` for every group of
/// `R` (first order, hand-derived through the stages; `ᾱ` scored as if every head routed by its
/// law). A real on the lattice `2^-p` moves by at most `2^-(p+1)`, so group `G` moves logit `c` by at
/// most `2^-(p_G+1) S_G`, and the logit change's range over `c` is at most `2 Σ_G 2^-(p_G+1) S_G`.
/// Hoeffding's lemma bounds `KL(softmax z ‖ softmax(z + δ))` by `range(δ)²/8`, so
/// `Σ_G 2^-(p_G+1) S_G ≤ √(2ε)` meets `ε` to first order ([`start_precision`]).
pub fn sensitivities(rewrite: &ExactRewrite, reference: &Executable) -> [f64; 10] {
    let (p, heads, n, planes) = (rewrite.p, rewrite.heads, rewrite.neurons, rewrite.planes);
    let chars = &rewrite.chars;
    let class_abs: Vec<f64> = (0..p).map(|c| (0..planes).map(|k| chars[[c, k, 0]].abs() + chars[[c, k, 1]].abs()).sum()).collect();
    let rows: Vec<[f64; 10]> = (0..p * p)
        .into_par_iter()
        .map(|row| {
            let (a, b) = (row / p, row % p);
            let alpha = reference.routing(a, b, None);
            let mut pre = vec![0.0; n];
            reference.pre_row(Family::Clean, a, b, 0, &alpha, &[], &mut pre, None);
            let zeta_abs: f64 = (0..heads)
                .map(|h| {
                    (0..planes)
                        .map(|k| (0..2).map(|t| (alpha[h][0] * chars[[a, k, t]] + alpha[h][1] * chars[[b, k, t]]).abs()).sum::<f64>())
                        .sum::<f64>()
                })
                .sum();
            let act_sum: f64 = pre.iter().map(|&x| x.max(0.0)).sum();
            let mut best = [0.0_f64; 10];
            let tokens = [a, b];
            for c in 0..p {
                let mut beta = 0.0;
                let mut galpha = vec![[0.0_f64; 3]; heads];
                for v in 0..n {
                    if pre[v] <= 0.0 {
                        continue;
                    }
                    let gradient = reference.readout[[v, c]];
                    beta += gradient.abs();
                    for h in 0..heads {
                        for j in 0..3 {
                            let mut q = reference.kap[[3 * h + j, v]];
                            if j < 2 {
                                q += reference.w_sum[[tokens[j], h, v]];
                            }
                            galpha[h][j] += gradient * q;
                        }
                    }
                }
                let mut sig = 0.0;
                let mut g = 0.0;
                let mut abar = 0.0;
                for h in 0..heads {
                    for j in 0..3 {
                        galpha[h][j] += reference.dk[[3 * h + j, c]];
                        if j < 2 {
                            galpha[h][j] += reference.e_sum[[tokens[j], h, c]];
                        }
                        abar += galpha[h][j].abs();
                    }
                    let mean: f64 = (0..3).map(|j| alpha[h][j] * galpha[h][j]).sum();
                    let gs: Vec<f64> = (0..3).map(|j| alpha[h][j] * (galpha[h][j] - mean)).collect();
                    sig += gs.iter().map(|x| x.abs()).sum::<f64>();
                    for k in 0..planes {
                        for t in 0..2 {
                            g += (gs[0] * chars[[a, k, t]] + gs[1] * chars[[b, k, t]]).abs();
                        }
                    }
                }
                let cand = [
                    sig,
                    g,
                    abar,
                    beta,
                    beta * heads as f64,
                    beta * zeta_abs,
                    act_sum * class_abs[c],
                    class_abs[c],
                    heads as f64 * class_abs[c],
                    zeta_abs * class_abs[c],
                ];
                for (slot, value) in best.iter_mut().zip(cand) {
                    *slot = slot.max(value);
                }
            }
            best
        })
        .collect();
    let mut out = [0.0_f64; 10];
    for row in rows {
        for (slot, value) in out.iter_mut().zip(row) {
            *slot = slot.max(value);
        }
    }
    out
}

/// The start precision of every group from the tolerance: `p_G = ⌈log₂(10 S_G / √(2ε))⌉ − 1`, so
/// each of the ten groups takes an equal share `√(2ε)/10` of the first-order budget.
pub fn start_precision(sensitivity: &[f64; 10], tolerance: f64) -> [i32; 10] {
    let mut bits = [0i32; 10];
    for (slot, &s) in bits.iter_mut().zip(sensitivity) {
        let ratio = GROUPS.len() as f64 * s.max(f64::MIN_POSITIVE) / (2.0 * tolerance).sqrt();
        *slot = ratio.log2().ceil() as i32 - 1;
    }
    bits
}

// ------------------------------------------------------------------------------------------ search

/// A realized program with the characters its decoder evaluates on: decoding builds the
/// executable (`precision::decode_then_evaluate` then measures the decoded program).
struct Decodable<'a> {
    realized: &'a RealizedProgram,
    chars: &'a Array3<f64>,
}

impl DecodableArtifact for Decodable<'_> {
    type Decoded = Executable;

    fn decode(&self) -> std::result::Result<Executable, String> {
        Executable::new(self.realized, self.chars).map_err(|error| error.to_string())
    }
}

type Status = EvidenceStatus<String, String>;

/// A decoded program measured on the contract.
pub struct Measured {
    pub realized: RealizedProgram,
    pub executable: Executable,
    pub evaluation: Evaluation,
    fidelity: DecodedFidelity<String, String>,
}

fn measure(realized: RealizedProgram, chars: &Array3<f64>, contract: &mut Contract, drop: Option<&ReadDrop<'_>>, previous: Option<&HashMap<usize, Vec<f64>>>) -> Result<Measured> {
    let tolerance = contract.tolerance;
    let mut kept: Option<(Evaluation, Executable)> = None;
    let fidelity = decode_then_evaluate(
        &Decodable { realized: &realized, chars },
        |executable: &Executable| -> std::result::Result<Option<Status>, String> {
            let evaluation = contract.evaluate(executable, false, &[], drop, previous).map_err(|error| error.to_string())?;
            let status = evaluation.status(tolerance).map_err(|error| error.to_string())?;
            kept = Some((evaluation, executable.clone()));
            Ok(Some(status))
        },
        &None,
        |outputs: &Option<Status>, reference: &Option<Status>| {
            outputs.clone().or_else(|| reference.clone()).ok_or_else(|| "the contract produced no status".to_string())
        },
        tolerance,
    )
    .map_err(CyclicProgramError::Declaration)?;
    let (evaluation, executable) =
        kept.ok_or_else(|| CyclicProgramError::Declaration("the decoded program was not evaluated".to_string()))?;
    Ok(Measured { realized, executable, evaluation, fidelity })
}

/// One round of the search.
#[derive(Clone, Debug, PartialEq)]
pub struct Round {
    pub level: usize,
    pub proposals: usize,
    pub batch: usize,
    pub accepted: usize,
    pub bits: u64,
    pub upper: f64,
}

/// A batch `decide_proposal` accepted.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptedBatch {
    pub proposals: Vec<Proposal>,
    pub saving_bits: i128,
    pub upper: f64,
}

/// The search's result.
pub struct SearchOutcome {
    pub sensitivities: [f64; 10],
    pub start_fraction_bits: [i32; 10],
    pub start_bits: u64,
    pub start_upper: f64,
    pub rounds: Vec<Round>,
    pub accepted: Vec<AcceptedBatch>,
    /// `decide_proposal`'s refusals of evaluated batches, by reason.
    pub rejections: std::collections::BTreeMap<&'static str, usize>,
    pub program: Measured,
    /// The selected program on the full contract with per-family statistics.
    pub certificate: Evaluation,
}

struct Searcher<'r> {
    rewrite: &'r ExactRewrite,
    contract: Contract,
    cache: LatticeCache,
    current: Measured,
    refused: std::collections::HashSet<(Proposal, i32)>,
    accepted: Vec<AcceptedBatch>,
    /// `decide_proposal`'s refusals of evaluated batches, by reason.
    rejections: std::collections::BTreeMap<&'static str, usize>,
}

impl Searcher<'_> {
    fn realize(&mut self, structure: &ProgramStructure) -> Result<RealizedProgram> {
        realize(self.rewrite, structure, self.rewrite.abar.view(), &mut self.cache)
    }

    fn bits(realized: &RealizedProgram) -> Result<u64> {
        realized.bits.ok_or_else(|| CyclicProgramError::Declaration("a searched program must be coded".to_string()))
    }

    /// (bits saved, effect) of each proposal: a read drop's certified logit-range bound
    /// `range(r_n) Σ_h ‖w_nhk‖₂` (`‖ζ_hk‖₂ ≤ 1` on every row of every family), otherwise the exact
    /// largest clean-input range over `c` of its logit change. Effects rank proposals and size
    /// batches; acceptance is decided on each batch's exhaustive evaluation.
    fn effects(&mut self, props: Vec<Proposal>) -> Result<Vec<(i128, f64, Proposal)>> {
        let bits = Self::bits(&self.current.realized)?;
        let structure = self.current.realized.structure.clone();
        let mut candidates = Vec::with_capacity(props.len());
        for prop in props {
            let Some(next) = prop.apply(&structure) else { continue };
            let realized = self.realize(&next)?;
            let saving = i128::from(bits) - i128::from(Self::bits(&realized)?);
            if saving <= 0 {
                self.refused.insert(prop.key(&structure));
                continue;
            }
            candidates.push((saving, prop, realized));
        }
        let current = &self.current;
        let p = self.rewrite.p;
        let reference_clean = current.executable.logits(Family::Clean, 0..p * p, &[]);
        let chars = &self.rewrite.chars;
        candidates
            .into_par_iter()
            .map(|(saving, prop, realized)| -> Result<(i128, f64, Proposal)> {
                if let Proposal::DropRead { neuron, plane } = prop
                    && realized.vary[neuron]
                    && let Some(v) = current.executable.neurons.iter().position(|&x| x == neuron)
                {
                    let w = &current.realized.decoded[W];
                    let norm: f64 = (0..realized.structure.law.len())
                        .map(|h| (w[[neuron, h, plane, 0]].powi(2) + w[[neuron, h, plane, 1]].powi(2)).sqrt())
                        .sum();
                    return Ok((saving, current.executable.range[v] * norm, prop));
                }
                let executable = Executable::new(&realized, chars)?;
                let logits = executable.logits(Family::Clean, 0..p * p, &[]);
                let effect = (&logits - &reference_clean)
                    .outer_iter()
                    .map(|row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - row.fold(f64::INFINITY, |a, &b| a.min(b)))
                    .fold(0.0, f64::max);
                Ok((saving, effect, prop))
            })
            .collect()
    }

    /// The read-drop columns when `candidate` differs from the current program only in varying
    /// neurons' read weights.
    fn read_drop(&self, candidate: &RealizedProgram) -> Option<Vec<usize>> {
        let current = &self.current.realized;
        if current.vary != candidate.vary
            || current.structure.embed != candidate.structure.embed
            || current.structure.readout != candidate.structure.readout
            || (0..GROUPS.len()).any(|g| g != W && current.decoded[g] != candidate.decoded[g])
        {
            return None;
        }
        let (old, new) = (&current.decoded[W], &candidate.decoded[W]);
        let changed: Vec<usize> = (0..current.vary.len())
            .filter(|&n| old.index_axis(Axis(0), n) != new.index_axis(Axis(0), n))
            .collect();
        let neurons = &self.current.executable.neurons;
        changed.iter().map(|n| neurons.iter().position(|x| x == n)).collect()
    }

    /// Decides `batch` as one edit, or bisects it; the number of proposals accepted.
    fn decide(&mut self, batch: &[Proposal]) -> Result<usize> {
        let mut structure = self.current.realized.structure.clone();
        for prop in batch {
            if let Some(next) = prop.apply(&structure) {
                structure = next;
            }
        }
        let realized = self.realize(&structure)?;
        let (reference_bits, candidate_bits) = (Self::bits(&self.current.realized)?, Self::bits(&realized)?);
        if candidate_bits < reference_bits {
            let columns = self.read_drop(&realized);
            let drop = columns.map(|columns| ReadDrop { reference: &self.current.executable, columns });
            let previous = self.current.evaluation.row_bounds.clone();
            let chars = &self.rewrite.chars;
            let measured = measure(realized, chars, &mut self.contract, drop.as_ref(), drop.as_ref().map(|_| &previous))?;
            let status = measured.evaluation.status(self.contract.tolerance)?;
            match decide_proposal(
                ProposalKind::Reduce,
                (reference_bits, &self.current.fidelity),
                (candidate_bits, &measured.fidelity),
                status,
            ) {
                Ok(acceptance) => {
                    write_message(self.rewrite, &measured.realized)?;
                    self.accepted.push(AcceptedBatch {
                        proposals: batch.to_vec(),
                        saving_bits: acceptance.saving_bits,
                        upper: measured.evaluation.upper,
                    });
                    self.current = measured;
                    return Ok(batch.len());
                }
                Err(ProposalRejection::ReferenceMissesTolerance(reason)) => {
                    return Err(CyclicProgramError::ReferenceMissesTolerance(reason));
                }
                Err(rejection) => {
                    let name = match rejection {
                        ProposalRejection::CandidateMissesTolerance(_) => "candidate_misses_tolerance",
                        ProposalRejection::NoShorterCode { .. } => "no_shorter_code",
                        ProposalRejection::FidelityRefuted(_) => "fidelity_refuted",
                        ProposalRejection::EstimateIsNotAFidelityBound(_) => "estimate_is_not_a_fidelity_bound",
                        ProposalRejection::NotASupremum(_) => "not_a_supremum",
                        ProposalRejection::ReferenceMissesTolerance(_) => "reference_misses_tolerance",
                    };
                    *self.rejections.entry(name).or_insert(0) += 1;
                }
            }
        }
        if batch.len() == 1 {
            self.refused.insert(batch[0].key(&self.current.realized.structure));
            return Ok(0);
        }
        let half = batch.len() / 2;
        Ok(self.decide(&batch[..half])? + self.decide(&batch[half..])?)
    }
}

/// Coarse-to-fine batched greedy MDL (module documentation, *Search*). Levels run in order: whole
/// planes, whole-group precision and whole-head routing; per-head score and direct-path planes;
/// per-neuron reads inside the surviving planes. Each round ranks the level's untried proposals by
/// bits saved per unit effect and fills a batch in that order while the summed effects fit the
/// logit-range budget [`kl_budget`] of the program's certified maximum; a proposal whose own effect
/// exceeds the budget is not proposed again. The batch is one edit for `fit::decide_proposal` on one
/// exhaustive evaluation; a refused batch is bisected, and a refused single proposal is not
/// proposed again. A level ends when nothing of it is left to try. The selected program is then
/// certified on the full contract with per-family statistics (`exact_planes` exactly).
pub fn search(rewrite: &ExactRewrite, donor: Vec<usize>, tolerance: f64, exact_planes: &[usize], governor: &MemoryGovernor) -> Result<SearchOutcome> {
    let mut contract = Contract::new(rewrite, donor, tolerance, governor)?;
    let sensitivity = sensitivities(rewrite, contract.reference());
    let mut bits = start_precision(&sensitivity, tolerance);
    let mut cache = LatticeCache::default();
    let start = loop {
        let realized = realize(rewrite, &ProgramStructure::full(rewrite, Some(bits)), rewrite.abar.view(), &mut cache)?;
        let measured = measure(realized, &rewrite.chars, &mut contract, None, None)?;
        if measured.evaluation.complete && measured.evaluation.upper <= tolerance {
            break measured;
        }
        if bits.iter().any(|&b| b >= 60) {
            return Err(CyclicProgramError::StartMissesTolerance { max_kl: measured.evaluation.upper, tolerance });
        }
        for b in bits.iter_mut() {
            *b += 1;
        }
    };
    let start_bits = write_message(rewrite, &start.realized)?;
    let start_upper = start.evaluation.upper;
    let mut searcher = Searcher {
        rewrite,
        contract,
        cache,
        current: start,
        refused: std::collections::HashSet::new(),
        accepted: Vec::new(),
        rejections: std::collections::BTreeMap::new(),
    };
    let mut rounds = Vec::new();
    for level in 0..3 {
        loop {
            let structure = searcher.current.realized.structure.clone();
            let props: Vec<Proposal> = proposals(&structure, &searcher.current.realized, level)
                .into_iter()
                .filter(|prop| !searcher.refused.contains(&prop.key(&structure)))
                .collect();
            let mut scored = searcher.effects(props)?;
            if scored.is_empty() {
                break;
            }
            scored.sort_by(|x, y| {
                let ratio = |s: &(i128, f64, Proposal)| if s.1 > 0.0 { s.0 as f64 / s.1 } else { f64::INFINITY };
                ratio(y).total_cmp(&ratio(x)).then(x.2.kind_rank().cmp(&y.2.kind_rank())).then(x.2.cmp(&y.2))
            });
            let budget = kl_budget(searcher.current.evaluation.upper, tolerance);
            let mut batch = Vec::new();
            let mut used = 0.0_f64;
            for (_, effect, prop) in &scored {
                if *effect > budget {
                    searcher.refused.insert(prop.key(&structure));
                } else if used + effect <= budget {
                    batch.push(*prop);
                    used += effect;
                }
            }
            if batch.is_empty() {
                break;
            }
            let accepted = searcher.decide(&batch)?;
            rounds.push(Round {
                level,
                proposals: scored.len(),
                batch: batch.len(),
                accepted,
                bits: Searcher::bits(&searcher.current.realized)?,
                upper: searcher.current.evaluation.upper,
            });
        }
    }
    let certificate = searcher.contract.evaluate(&searcher.current.executable, true, exact_planes, None, None)?;
    Ok(SearchOutcome {
        certificate,
        sensitivities: sensitivity,
        start_fraction_bits: bits,
        start_bits,
        start_upper,
        rounds,
        accepted: searcher.accepted,
        rejections: searcher.rejections,
        program: searcher.current,
    })
}

/// A declared structure measured on the full contract with per-family statistics
/// (`exact_planes` are evaluated exactly), with `fold_at` the routing at which folded `κ` enter.
pub fn certify(rewrite: &ExactRewrite, donor: Vec<usize>, tolerance: f64, structure: &ProgramStructure, fold_at: ArrayView2<'_, f64>, exact_planes: &[usize], governor: &MemoryGovernor) -> Result<(Evaluation, Option<u64>)> {
    let mut contract = Contract::new(rewrite, donor, tolerance, governor)?;
    let realized = realize(rewrite, structure, fold_at, &mut LatticeCache::default())?;
    let bits = match realized.bits {
        Some(_) => Some(write_message(rewrite, &realized)?),
        None => None,
    };
    let executable = Executable::new(&realized, &rewrite.chars)?;
    Ok((contract.evaluate(&executable, true, exact_planes, None, None)?, bits))
}

/// Runs `work` on a pool of `threads` threads.
pub fn with_threads<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> Result<T> {
    if threads == 0 {
        return Err(CyclicProgramError::Declaration("the thread count must be positive".to_string()));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| CyclicProgramError::Threads(error.to_string()))?;
    Ok(pool.install(work))
}


// ------------------------------------------------------------------------------------------ driver

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use memmap2::Mmap;
use serde_json::{Value, json};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            println!("[mpd_modadd_cyclic_baseline_2951] error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|pair| pair[0] == name).map(|pair| pair[1].clone())
}

fn required(args: &[String], name: &str) -> std::result::Result<String, String> {
    flag(args, name).ok_or_else(|| format!("missing {name}"))
}

fn float64(path: &Path) -> std::result::Result<(Vec<usize>, Vec<f64>), String> {
    let file = File::open(path).map_err(|err| format!("open {}: {err}", path.display()))?;
    // SAFETY: the exported array is opened read-only and never written through this
    // mapping for the lifetime of the read.
    let mmap = unsafe { Mmap::map(&file).map_err(|err| format!("mmap {}: {err}", path.display()))? };
    let bad = |what: &str| format!("{}: {what}", path.display());
    if mmap.len() < 10 || &mmap[..6] != b"\x93NUMPY" {
        return Err(bad("not an .npy file"));
    }
    let (len, start) = match mmap[6] {
        1 => (u16::from_le_bytes([mmap[8], mmap[9]]) as usize, 10),
        2 | 3 if mmap.len() >= 12 => (u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize, 12),
        _ => return Err(bad("unsupported .npy version")),
    };
    let header = std::str::from_utf8(mmap.get(start..start + len).ok_or_else(|| bad("truncated header"))?)
        .map_err(|error| bad(&format!("header is not UTF-8: {error}")))?;
    if !header.contains("'descr': '<f8'") || !header.contains("'fortran_order': False") {
        return Err(bad("must hold C-order <f8 values"));
    }
    let open = header.find("'shape': (").ok_or_else(|| bad("no shape"))? + "'shape': (".len();
    let close = open + header[open..].find(')').ok_or_else(|| bad("no shape"))?;
    let shape = header[open..close]
        .split(',')
        .map(str::trim)
        .filter(|extent| !extent.is_empty())
        .map(|extent| extent.parse::<usize>().map_err(|error| bad(&format!("bad shape: {error}"))))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let data = start + len;
    let count: usize = shape.iter().product();
    if data + 8 * count != mmap.len() {
        return Err(bad("size does not match its header"));
    }
    let values = mmap[data..]
        .chunks_exact(8)
        .map(|chunk| {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(chunk);
            f64::from_le_bytes(bytes)
        })
        .collect();
    Ok((shape, values))
}

fn array(dir: &Path, name: &str, shape: &[usize]) -> std::result::Result<ArrayD<f64>, String> {
    let (found, values) = float64(&dir.join(format!("{name}.npy")))?;
    if found.iter().product::<usize>() != shape.iter().product::<usize>() {
        return Err(format!("{name}: exported shape {found:?} does not hold {shape:?}"));
    }
    ArrayD::from_shape_vec(IxDyn(shape), values).map_err(|error| format!("{name}: {error}"))
}

fn program_json(realized: &RealizedProgram) -> Value {
    let structure = &realized.structure;
    let heads = structure.law.len();
    let numbered = |row: ArrayView1<'_, bool>| -> Vec<usize> {
        row.iter().enumerate().filter(|&(k, &on)| on && structure.embed[k]).map(|(k, _)| k + 1).collect()
    };
    json!({
        "bits": realized.bits,
        "fraction_bits": structure.fraction_bits,
        "embed_planes": (0..structure.embed.len()).filter(|&k| structure.embed[k]).map(|k| k + 1).collect::<Vec<_>>(),
        "readout_planes": (0..structure.readout.len()).filter(|&k| structure.readout[k]).map(|k| k + 1).collect::<Vec<_>>(),
        "score_planes": (0..heads).map(|h| if structure.law[h] { None } else { Some(numbered(structure.score.row(h))) }).collect::<Vec<_>>(),
        "direct_planes": (0..heads).map(|h| numbered(structure.direct.row(h))).collect::<Vec<_>>(),
        "law_heads": (0..heads).filter(|&h| structure.law[h]).collect::<Vec<_>>(),
        "folded_heads": (0..heads).filter(|&h| structure.folded[h] || structure.law[h]).collect::<Vec<_>>(),
        "varying_neurons": realized.vary.iter().filter(|&&on| on).count(),
        "read_pairs": structure.reads.indexed_iter().filter(|&((n, k), &on)| on && structure.embed[k] && realized.vary[n]).count(),
        "reals": GROUPS.iter().zip(realized.counts.iter()).map(|(g, c)| (g.to_string(), json!(c))).collect::<serde_json::Map<_, _>>(),
        "reals_total": realized.counts.iter().sum::<usize>(),
        "nonzero_reals": realized.decoded.iter().map(|group| group.iter().filter(|&&v| v != 0.0).count()).sum::<usize>(),
    })
}

fn evaluation_json(evaluation: &Evaluation, tolerance: f64) -> std::result::Result<Value, String> {
    let status = evaluation.status(tolerance).map_err(|error| error.to_string())?;
    Ok(json!({
        "status": format!("{status:?}"),
        "certified_within_tolerance": status.certifies_at_most(tolerance),
        "upper": evaluation.upper,
        "rows": evaluation.rows,
        "bounded": evaluation.bounded,
        "families": evaluation.families.iter().map(|stats| json!({
            "family": stats.family.name(),
            "upper": stats.upper,
            "exact_rows": stats.exact_rows,
            "rows": stats.rows,
            "mean_kl": stats.mean_kl,
            "argmax_agreement": stats.argmax_agreement,
        })).collect::<Vec<_>>(),
    }))
}

/// `--export DIR` (bench/mpd_engine_export_2951.py: float64 `.npy` weights and export.json),
/// `--donor FILE` (float64 `.npy` permutation of the p² inputs), `--epsilon EPS`, `--threads N`,
/// `--exact-planes 1,5,...` (rotate families certified exactly), `--out FILE`.
fn run() -> std::result::Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(required(&args, "--export")?);
    let tolerance: f64 = required(&args, "--epsilon")?.parse().map_err(|error| format!("--epsilon: {error}"))?;
    let threads: usize = required(&args, "--threads")?.parse().map_err(|error| format!("--threads: {error}"))?;
    let out = PathBuf::from(required(&args, "--out")?);
    let exact: Vec<usize> = match flag(&args, "--exact-planes") {
        Some(list) => list
            .split(',')
            .map(|k| k.trim().parse::<usize>().map(|k| k - 1).map_err(|error| format!("--exact-planes: {error}")))
            .collect::<std::result::Result<_, _>>()?,
        None => Vec::new(),
    };
    let manifest: Value = serde_json::from_reader(File::open(dir.join("export.json")).map_err(|error| error.to_string())?)
        .map_err(|error| format!("export.json: {error}"))?;
    let config = &manifest["config"];
    let size = |key: &str| config[key].as_u64().map(|v| v as usize).ok_or_else(|| format!("export.json config.{key} missing"));
    let (p, d, heads, d_head, n) = (size("p")?, size("d_model")?, size("n_heads")?, size("d_head")?, size("d_mlp")?);
    let load = |name: &str, shape: &[usize]| array(&dir, name, shape);
    let embed = load("W_E", &[p + 1, d])?.into_dimensionality::<ndarray::Ix2>().map_err(|e| e.to_string())?;
    let position = load("W_pos", &[3, d])?.into_dimensionality::<ndarray::Ix2>().map_err(|e| e.to_string())?;
    let query = load("W_Q", &[heads, d_head, d])?.into_dimensionality::<ndarray::Ix3>().map_err(|e| e.to_string())?;
    let key = load("W_K", &[heads, d_head, d])?.into_dimensionality::<ndarray::Ix3>().map_err(|e| e.to_string())?;
    let value = load("W_V", &[heads, d_head, d])?.into_dimensionality::<ndarray::Ix3>().map_err(|e| e.to_string())?;
    let output = load("W_O", &[d, heads * d_head])?.into_dimensionality::<ndarray::Ix2>().map_err(|e| e.to_string())?;
    let read_in = load("W_in", &[n, d])?.into_dimensionality::<ndarray::Ix2>().map_err(|e| e.to_string())?;
    let read_in_bias = load("b_in", &[n])?.into_dimensionality::<ndarray::Ix1>().map_err(|e| e.to_string())?;
    let write_out = load("W_out", &[d, n])?.into_dimensionality::<ndarray::Ix2>().map_err(|e| e.to_string())?;
    let write_out_bias = load("b_out", &[d])?.into_dimensionality::<ndarray::Ix1>().map_err(|e| e.to_string())?;
    let unembed = load("W_U", &[p, d])?.into_dimensionality::<ndarray::Ix2>().map_err(|e| e.to_string())?;
    let (donor_shape, donor_values) = float64(&PathBuf::from(required(&args, "--donor")?))?;
    if donor_shape != [p * p] {
        return Err(format!("the donor must be a permutation of the {} inputs; its shape is {donor_shape:?}", p * p));
    }
    let donor: Vec<usize> = donor_values.iter().map(|&v| v as usize).collect();
    let model = CyclicTransformer {
        embed: embed.view(),
        position: position.view(),
        query: query.view(),
        key: key.view(),
        value: value.view(),
        output: output.view(),
        read_in: read_in.view(),
        read_in_bias: read_in_bias.view(),
        write_out: write_out.view(),
        write_out_bias: write_out_bias.view(),
        unembed: unembed.view(),
    };
    let governor = MemoryGovernor::global();
    let start = std::time::Instant::now();
    let (rewrite, outcome) = with_threads(threads, || -> Result<_> {
        let rewrite = ExactRewrite::new(&model)?;
        let outcome = search(&rewrite, donor, tolerance, &exact, governor)?;
        Ok((rewrite, outcome))
    })
    .map_err(|error| error.to_string())?
    .map_err(|error| error.to_string())?;
    let clean = outcome.certificate.families.iter().find(|stats| stats.family == Family::Clean);
    let report = json!({
        "export": dir.display().to_string(),
        "run_sha256": manifest["run_sha256"],
        "tolerance": tolerance,
        "contract": "exhaustive max KL(R || P) over clean, rotate_k (every plane, every shift) and swap_k (every plane, declared donor)",
        "forward_gap": rewrite.forward_gap,
        "total_bits": outcome.program.realized.bits,
        "reals": outcome.program.realized.counts.iter().sum::<usize>(),
        "clean_max_kl": clean.map(|stats| stats.upper),
        "clean_argmax_agreement": clean.map(|stats| stats.argmax_agreement),
        "start_bits": outcome.start_bits,
        "start_upper": outcome.start_upper,
        "start_fraction_bits": outcome.start_fraction_bits,
        "sensitivities": outcome.sensitivities,
        "program": program_json(&outcome.program.realized),
        "certificate": evaluation_json(&outcome.certificate, tolerance)?,
        "rounds": outcome.rounds.iter().map(|round| json!({
            "level": round.level, "proposals": round.proposals, "batch": round.batch,
            "accepted": round.accepted, "bits": round.bits, "upper": round.upper,
        })).collect::<Vec<_>>(),
        "accepted": outcome.accepted.iter().map(|batch| json!({
            "proposals": batch.proposals.iter().map(|prop| format!("{prop:?}")).collect::<Vec<_>>(),
            "saving_bits": batch.saving_bits.to_string(),
            "upper": batch.upper,
        })).collect::<Vec<_>>(),
        "rejections": outcome.rejections,
        "seconds": start.elapsed().as_secs_f64(),
    });
    let text = serde_json::to_string(&report).map_err(|error| error.to_string())?;
    std::fs::write(&out, &text).map_err(|error| format!("write {}: {error}", out.display()))?;
    println!("{}", serde_json::to_string(&json!({
        "total_bits": report["total_bits"], "reals": report["reals"], "clean_max_kl": report["clean_max_kl"],
        "clean_argmax_agreement": report["clean_argmax_agreement"], "certified": report["certificate"]["certified_within_tolerance"],
        "embed_planes": report["program"]["embed_planes"], "seconds": report["seconds"],
    })).map_err(|error| error.to_string())?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_sae::parameter_decomposition::cyclic_action::frequency_edit;
    use ndarray::{Array3, ArrayD};

    /// Deterministic weights of a small cyclic transformer (`p = 7`, `d = 12`, two heads).
    struct Weights {
        embed: Array2<f64>,
        position: Array2<f64>,
        query: Array3<f64>,
        key: Array3<f64>,
        value: Array3<f64>,
        output: Array2<f64>,
        read_in: Array2<f64>,
        read_in_bias: Array1<f64>,
        write_out: Array2<f64>,
        write_out_bias: Array1<f64>,
        unembed: Array2<f64>,
    }

    fn entry(seed: usize, index: usize, scale: f64) -> f64 {
        scale * ((seed * 7919 + index * 104_729) as f64 * 0.618_033_988_7).sin()
    }

    fn weights() -> Weights {
        let (p, d, heads, d_head, n) = (7, 12, 2, 4, 16);
        let m2 = |seed: usize, r: usize, c: usize, scale: f64| Array2::from_shape_fn((r, c), |(i, j)| entry(seed, i * c + j, scale));
        let m3 = |seed: usize, scale: f64| Array3::from_shape_fn((heads, d_head, d), |(h, i, j)| entry(seed, (h * d_head + i) * d + j, scale));
        Weights {
            embed: m2(1, p + 1, d, 1.0),
            position: m2(2, 3, d, 0.5),
            query: m3(3, 0.8),
            key: m3(4, 0.8),
            value: m3(5, 0.6),
            output: m2(6, d, heads * d_head, 0.6),
            read_in: m2(7, n, d, 0.7),
            read_in_bias: Array1::from_shape_fn(n, |i| entry(8, i, 0.3)),
            write_out: m2(9, d, n, 0.7),
            write_out_bias: Array1::from_shape_fn(d, |i| entry(10, i, 0.2)),
            unembed: m2(11, p, d, 1.5),
        }
    }

    fn model(w: &Weights) -> CyclicTransformer<'_> {
        CyclicTransformer {
            embed: w.embed.view(),
            position: w.position.view(),
            query: w.query.view(),
            key: w.key.view(),
            value: w.value.view(),
            output: w.output.view(),
            read_in: w.read_in.view(),
            read_in_bias: w.read_in_bias.view(),
            write_out: w.write_out.view(),
            write_out_bias: w.write_out_bias.view(),
            unembed: w.unembed.view(),
        }
    }

    fn centred(row: ArrayView1<'_, f64>) -> Array1<f64> {
        let mean = row.sum() / row.len() as f64;
        row.mapv(|value| value - mean)
    }

    fn largest_gap(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        (&centred(x.view()) - &centred(y.view())).iter().fold(0.0_f64, |worst, v| worst.max(v.abs()))
    }

    fn donor(p: usize) -> Vec<usize> {
        (0..p * p).map(|r| (r * 5 + 3) % (p * p)).collect()
    }

    /// `R`'s logits with plane `k`'s moved content taken from the donor input `(da, db)`,
    /// straight from the coefficients.
    fn swapped(rw: &ExactRewrite, a: usize, b: usize, k: usize, da: usize, db: usize) -> Array1<f64> {
        let alpha = rw.routing(a, b);
        let donor_alpha = rw.routing(da, db);
        let zeta = |h: usize, plane: usize, t: usize| {
            if plane == k {
                donor_alpha[[h, 0]] * rw.chars[[da, plane, t]] + donor_alpha[[h, 1]] * rw.chars[[db, plane, t]]
            } else {
                alpha[[h, 0]] * rw.chars[[a, plane, t]] + alpha[[h, 1]] * rw.chars[[b, plane, t]]
            }
        };
        let mut act = rw.beta.clone();
        for n in 0..rw.neurons {
            for h in 0..rw.heads {
                for j in 0..3 {
                    act[n] += alpha[[h, j]] * rw.kap[[n, h, j]];
                }
                for plane in 0..rw.planes {
                    for t in 0..2 {
                        act[n] += rw.w[[n, h, plane, t]] * zeta(h, plane, t);
                    }
                }
            }
        }
        act.mapv_inplace(|v| v.max(0.0));
        let mut y = rw.y0.clone();
        for q in 0..rw.planes {
            for t in 0..2 {
                y[[q, t]] += (0..rw.neurons).map(|n| rw.rho[[n, q, t]] * act[n]).sum::<f64>();
                for h in 0..rw.heads {
                    for j in 0..3 {
                        y[[q, t]] += rw.dk[[q, t, h, j]] * alpha[[h, j]];
                    }
                    for plane in 0..rw.planes {
                        for s in 0..2 {
                            y[[q, t]] += rw.dw[[q, t, h, plane, s]] * zeta(h, plane, s);
                        }
                    }
                }
            }
        }
        Array1::from_shape_fn(rw.p, |c| (0..rw.planes).map(|q| y[[q, 0]] * rw.chars[[c, q, 0]] + y[[q, 1]] * rw.chars[[c, q, 1]]).sum())
    }

    #[test]
    fn exact_rewrite_is_the_model_and_the_executable_is_the_rewrite_under_every_family() {
        let w = weights();
        let m = model(&w);
        let rw = ExactRewrite::new(&m).expect("rewrite");
        assert!(rw.forward_gap < 1e-11, "R against the model: {}", rw.forward_gap);
        let p = rw.p;
        let full = realize(&rw, &ProgramStructure::full(&rw, None), rw.abar.view(), &mut LatticeCache::default()).expect("realize");
        let exe = Executable::new(&full, &rw.chars).expect("executable");
        assert!(exe.radius > 0.0 && exe.radius < 1e-9, "radius {}", exe.radius);
        let donor = donor(p);
        let clean = exe.logits(Family::Clean, 0..p * p, &donor);
        let cycle = token_cycle(p, p + 1).expect("cycle");
        let planes = cyclic_planes(w.embed.view(), &cycle).expect("planes");
        for r in 0..p * p {
            let direct = m.logits(r / p, r % p, None);
            assert!(largest_gap(&clean.row(r).to_owned(), &direct) < 1e-11);
        }
        for k in 0..rw.planes {
            let rotated = exe.logits(Family::Rotate(k), 0..Family::Rotate(k).rows(p), &donor);
            let basis = planes.basis(&[k + 1]).expect("basis");
            for r in 0..Family::Rotate(k).rows(p) {
                let (a, b, _) = Family::Rotate(k).row(p, r, &donor);
                let edit = frequency_edit(&cycle, basis.view(), &[k + 1], r / (p * p) + 1).expect("edit");
                let delta = edit.right.dot(&edit.left.row(a));
                let direct = m.logits(a, b, Some(delta.view()));
                assert!(largest_gap(&rotated.row(r).to_owned(), &direct) < 1e-11, "rotate plane {k} row {r}");
            }
            let swaps = exe.logits(Family::Swap(k), 0..p * p, &donor);
            for r in 0..p * p {
                let slow = swapped(&rw, r / p, r % p, k, donor[r] / p, donor[r] % p);
                assert!(largest_gap(&swaps.row(r).to_owned(), &slow) < 1e-11, "swap plane {k} row {r}");
            }
        }
    }

    #[test]
    fn accounted_code_length_is_the_written_message_length() {
        let w = weights();
        let rw = ExactRewrite::new(&model(&w)).expect("rewrite");
        let mut cache = LatticeCache::default();
        let mut structure = ProgramStructure::full(&rw, Some([6, 5, 7, 8, 6, 5, 9, 4, 3, 2]));
        for edit in [
            Proposal::DropEmbedPlane { plane: 1 },
            Proposal::RouteLaw { head: 0 },
            Proposal::FoldKap { head: 1 },
            Proposal::DropRead { neuron: 3, plane: 0 },
            Proposal::DropRead { neuron: 3, plane: 2 },
            Proposal::DropReadoutPlane { plane: 2 },
            Proposal::DropDirectPlane { head: 1, plane: 0 },
        ] {
            structure = edit.apply(&structure).expect("applies");
            let realized = realize(&rw, &structure, rw.abar.view(), &mut cache).expect("realize");
            let written = write_message(&rw, &realized).expect("the written message is the accounted length");
            assert_eq!(Some(written), realized.bits);
        }
        // Neuron 3 reads nothing and every κ is folded, so it is constant and sends no reals.
        let realized = realize(&rw, &structure, rw.abar.view(), &mut cache).expect("realize");
        assert!(!realized.vary[3]);
    }

    #[test]
    fn kl_moved_bounds_a_moved_divergence() {
        let p = 9;
        for trial in 0..200 {
            let z = Array1::from_shape_fn(p, |i| entry(trial, i, 4.0));
            let q = Array1::from_shape_fn(p, |i| z[i] + entry(trial + 999, i, 0.7));
            let delta = Array1::from_shape_fn(p, |i| entry(trial + 5555, i, 0.3));
            let kl = |x: &Array1<f64>, y: &Array1<f64>| {
                let (lx, _) = log_softmax(x.view());
                let (ly, _) = log_softmax(y.view());
                lx.iter().zip(ly.iter()).map(|(a, b)| a.exp() * (a - b)).sum::<f64>()
            };
            let moved = &q + &delta;
            let oscillation = delta.fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - delta.fold(f64::INFINITY, |a, &b| a.min(b));
            assert!(kl(&z, &moved) <= kl_moved(kl(&z, &q), oscillation), "trial {trial}");
            let budget = kl_budget(kl(&z, &q), kl(&z, &q) + 0.1);
            assert!(kl_moved(kl(&z, &q), budget) <= kl(&z, &q) + 0.1 + 1e-12);
        }
    }

    #[test]
    fn search_accepts_shorter_programs_within_the_tolerance_and_certifies_the_result() {
        let w = weights();
        let rw = ExactRewrite::new(&model(&w)).expect("rewrite");
        let tolerance = 0.05;
        let outcome = search(&rw, donor(rw.p), tolerance, &[0], MemoryGovernor::global()).expect("search");
        assert!(!outcome.accepted.is_empty(), "nothing was accepted");
        let bits = outcome.program.realized.bits.expect("coded");
        assert!(bits < outcome.start_bits, "{bits} against the start's {}", outcome.start_bits);
        assert!(outcome.certificate.complete);
        let status = outcome.certificate.status(tolerance).expect("status");
        assert!(status.certifies_at_most(tolerance), "{status:?}");
        assert_eq!(outcome.certificate.families.len(), 1 + 2 * rw.planes);
        let mut previous = outcome.start_bits;
        for round in &outcome.rounds {
            assert!(round.bits <= previous && round.upper <= tolerance);
            previous = round.bits;
        }
        let decoded: ArrayD<f64> = outcome.program.realized.decoded[W].clone();
        assert_eq!(decoded.shape(), &[rw.neurons, rw.heads, rw.planes, 2]);
    }
}
