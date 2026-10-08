//! The explanation as a library of learned functions, fitted end to end by variational minimum
//! description length on interchange experiments of causal abstraction (#2951).
//!
//! # The explanation
//!
//! Every attention head and every MLP of the native model `M` is replaced by a block of learned
//! functions that runs on the explanation's own vectors. `M`'s embedding, norms, attention output
//! projection, final norm and unembedding stay.
//!
//! * A head's block reads its layer's normed stream `x` and writes the head's read (the attention
//!   output projection's input). Its function attends with its own query, key and value maps
//!   `q = Q x`, `k = K x`, `v = V x` at the native rotary angles, scale and causal mask; where `M`
//!   norms each head's query and key (Qwen3), `q = γ_q ⊙ rms(Q x)` and `k = γ_k ⊙ rms(K x)` with
//!   the head's own copy of `M`'s gains `γ`, which are part of the law and not trained. Query heads
//!   that share a key and value in `M` (grouped-query attention) read one shared `K` and `V` here
//!   too, so the explanation stays in the family `M` realizes.
//! * An MLP's block reads its layer's second normed stream and writes the MLP's output. Its
//!   functions are `f_i(x) = φ(g_i·x + c_i) u_i`, with the native pointwise law `φ`,
//!   a gate direction `g_i`, a gate bias `c_i` and an output `u_i`; for a gated MLP (SwiGLU on
//!   Qwen3) `f_i(x) = φ(g_i·x) (b_i·x) u_i` with an up direction `b_i`. Biases appear only where
//!   `M` has them, so a bias-free MLP maps 0 to 0.
//!
//! The library starts at `M`: native neuron `i` is function `i` with its original
//! activation and coefficients, and a head is its own function.
//!
//! # The code length
//!
//! The posterior is Gaussian, every parameter independent of the others.
//!
//! The library's parameters `θ` are partitioned into prior groups `G`: a key-value group's rotary
//! plane (the plane's rows of the shared key and of every query head's query), its value
//! coordinate (one row of the shared value), an MLP function's gate `(g_i, c_i)`, its up direction `b_i` when gated, and its output `u_i`. With the posterior `q(θ) = Π_j N(μ_j, σ_j²)` and the
//! prior `p(θ_G) = N(0, v_G I)`, the code length in nats is
//!
//! `F = E_q[D(θ)] + Σ_G KL(q_G ‖ p_G) + Σ_{active G} (½ ln |G| + L_scale(v_G)) + L_subset`,
//!
//! the bits-back code of the training data given the explanation plus the explanation. `D(θ)`
//! sums `KL(M_e ‖ P_e)` of the next-token distributions over every token of every training
//! experiment `e` (below), so the data term's weight is the number of scored tokens `N` and no
//! tradeoff weight exists. `N` is the amount of `M`'s behaviour the explanation is asked to
//! explain; it is the one choice left, and results are reported along it. An active group sends
//! its variance at the precision of a parameter estimated from `|G|` values, `½ log2 |G|` bits (the
//! two-part code's asymptotic cost of a parameter's precision, a regular model's approximation
//! rather than an exact code for an arbitrary real), after its scale: the integer exponent `z` of
//! the bin `v_G / v⁰_G ∈ [2^(z − ½), 2^(z + ½)]` holding it, relative to the group's reference
//! variance `v⁰_G` (`Explanation::reference`: the mean square of `M`'s values over its cells, which
//! every decoder has from `M` and the explanation's structure; [`mean_squares`]), in the Elias δ
//! code of its signed index
//! (`L_scale`; the precision prices the fraction, the scale the exponent, since `(μ, σ) → (a μ, a
//! σ)` leaves `KL` unchanged). Each prior variance `v_G` is the empirical-Bayes value minimizing the
//! group's charge `KL(q_G ‖ p_G) + ln 2 · L_scale(v_G)` (the precision's `½ ln |G|` does not depend
//! on it; `gam_gpu::tensor::group_prior`). With `S_G = Σ_{j∈G} (μ_j² + σ_j²)` and `Σ_G` the group's
//! posterior covariance, `KL(q_G ‖ N(0, v I)) = ½ (S_G / v + |G| ln v − |G| − ln det Σ_G)` is least
//! at `S_G / |G|` and rises with the distance from it in `ln v`, while `L_scale` is constant within
//! a bin, so within a bin the charge is least at `S_G / |G|` clamped to the bin. The bin holding
//! `S_G / |G|` is compared with the bins toward exponent 0, whose one bit is the shortest code, each
//! at its edge nearest `S_G / |G|`, until the divergence there plus the shortest code is no less
//! than the best charge so far, which no bin further on can undercut. At `v_G` the divergence's
//! derivatives are `μ_j / v_G` in `μ_j` and `σ_j² / v_G − 1` in `ln σ_j`: `v_G` is either
//! `S_G / |G|`, where the divergence is stationary in `v`, or a bin's fixed edge. With a mixture
//! prior (`library_mixture`) the variance minimizing the mixture's divergence weights each sample
//! by the Gaussian component's responsibility; the fit keeps the group's own value as an adaptively
//! updated hyperparameter, so the reported `F` is the code length at that variance, not its minimum
//! over the variance. Which groups are in the explanation is sent once in the enumerative
//! subset code, `L_subset = L_int(k + 1) + ⌈log2 C(n, k)⌉` bits for `k` of `n` groups. A group the data does not inform
//! sits at its prior with zero divergence and posterior mean zero, so the null is recovered;
//! removing it from the explanation is a discrete step of the same `F`.
//!
//! Weight noise costs data only on the tokens where a function's sampled value is nonzero. A
//! ReLU-gated function is zero on a token for every weight sample only where its preactivation is
//! nonpositive over the whole posterior; with a preactivation `Z ~ N(m, s²)` under the weight noise,
//! `E[ReLU(Z)²] = (m² + s²) Φ(m/s) + m s φ(m/s)`, so a negative mean alone does not make the
//! function inactive. Where the mean lies many standard deviations below zero, the function's
//! weights stay imprecise and cheap at little data cost: the code length rewards functions inactive
//! on most inputs without any sparsity penalty, when the native activation is ReLU. The argument
//! does not apply to GELU or SiLU, which are nonzero at every finite preactivation.
//!
//! # The experiments
//!
//! The data are interchange experiments (`interchange`), one fixed collection drawn once from the
//! seed: per training base sequence, one clean experiment and one patched experiment under the
//! same hybrid, with the stated weights of `interchange::sample` (`P` alone or a random
//! block-subset hybrid with probability ½ each; the patched block uniform over the `2L` blocks,
//! then with probability ½ one read variable uniform within the block, else a joint read patch of
//! a random subset of the block's variables, its size uniform from one to all of them and then the
//! subset uniform; then a position). A patch replaces the patched functions' read values by their
//! values on the source, in `M` at `M`'s functions and in `P` at the call sites that replaced them
//! (`interchange::values`). The variables are `M`'s functions (`interchange::reads`), those `P`
//! removes included, one at a time and jointly: a function `P` no longer computes is patched in
//! `M` alone, which asks whether it matters. A base's source is another training sequence, drawn
//! uniformly. Each base's source and experiments come from a stream of draws of its own, seeded by
//! the seed and the base's index (`base_draws`), so the collection, and with it `N`, does not
//! depend on how the bases are packed into batches. The questions do not move as `P` learns or
//! loses functions, so `F`, the convergence
//! test and every removal comparison score the same experiments, and `M`'s targets for them are
//! made on the device whenever a batch is scored (`interchange::targets`).
//!
//! An explanation of some blocks only ([`scoped`]) is `M` everywhere else: its library holds the
//! trainable operators and prior groups of those blocks alone, so only they are charged, and every
//! experiment runs `P`'s blocks exactly there and `M`'s elsewhere (no other hybrid would differ from
//! `M`), with the patch drawn as above over all `2L` blocks.
//!
//! # The fit
//!
//! Each step draws one weight sample `θ = μ + σ ⊙ ε` (`ε` standard normal), runs half of a batch's bases' experiments at it and the other
//! half's at its antithetic twin `μ − σ ⊙ ε` (`antithetic_step`), and takes one step of the
//! improved variational online Newton method (IVON; Shen et al., ICML 2024) on
//! `F / N = E_q[ℓ] + KL(q ‖ p) / N`, `ℓ` the data term per scored token and `N`
//! the scored tokens of every training experiment. The data term's curvature is taken in the
//! Gauss–Newton approximation: per token the diagonal of `Σ_t J_tᵀ F_t J_t`, `F_t` the Fisher
//! matrix of `P_e`'s softmax at token `t`, which is positive semidefinite. The Hessian adds the
//! second derivatives of `P_e`'s logits weighted by `p_P − p_M`, so the two agree where `P_e`'s
//! predictions equal `M_e`'s. The estimate is the squared Gauss–Newton factor `ĥ = u ⊙ u / n`
//! (`interchange::Factor`): `u = Σ_t J_tᵀ b_t` over the batch's `n` scored tokens, `J_t` the
//! Jacobian of `P_e`'s logits at `t` and `b_t = √p_t ⊙ ξ_t − p_t (√p_t · ξ_t)` the Fisher probe
//! there (`p_t` `P_e`'s prediction, `ξ_t` independent random signs over the vocabulary, so
//! `E[b_t b_tᵀ] = F_t`), from a second reverse pass through the step's forward pass. `E[ĥ]` is that
//! diagonal, and each entry of `u ⊙ u` estimates its expectation with relative variance at most 2
//! whatever `P_e` predicts; with a label `y_t` drawn from `p_t` (the seed `p_t − e_{y_t}`) the
//! relative variance is `(1 − 2p)² / (p (1 − p))` on a class of probability `p`. The
//! reparameterization estimate
//! `g ε / σ` of the Hessian's diagonal carries every other weight's noise through the off-diagonal
//! terms (on vpd4l, eight draws showed no signal). `h` is the estimates' average over the last
//! epoch's batches (`β₂ = 1 − 1/B` for `B` training batches), starting from the Laplace start
//! (`laplace_start`): one pass over the collection at a weight sample of the unit-information
//! posterior (`σ² = v_G / N`) draws every batch's factor, `h = Σ_b u_b ⊙ u_b / N`, and
//! `σ² = 1 / (N h + 1 / v_G)`. From the unit-information curvature `1 / v_G` instead, `h` fell by
//! one e-fold per epoch toward the measured curvature, and `F` by a fixed 1.87e7 bits per epoch
//! for 10 epochs (vpd4l, `N = 2^16`; 17 epochs at `2^24`). Under the approximation the data's
//! `h ≥ 0`; a prior term's curvature (below) may take `h` below zero, and the step reads
//! `h⁺ = max(h, 0)`. The approximated objective is stationary in `σ` at `σ = 1 / √(N (h⁺ + δ))`,
//! `δ = 1 / (N v_G)` the group prior's precision per token, so `σ² ≤ v_G` (where the expected
//! curvature is negative the Gaussian family has no stationary `σ`, and `σ² = v_G` is the
//! prior's). At a fixed `v_G` an entry with `−δ < h < 0` would be stationary at
//! `σ² = 1 / (N h + 1 / v_G) > v_G`, but `v_G` is fitted to the posterior (the mean of
//! `μ̄² + σ²` over its group, with its scale code): jointly in `σ` and `v_G` such an entry has no
//! stationary point (along `σ² = v_G → ∞` its term `N h σ² / 2` falls without bound), and
//! admitting `σ² > v_G` raised `v_G`, lowered `δ` and moved the entry toward the pole at `h = −δ`
//! (d20c9ae861, reverted). That stationary point is implicit (`h` is an expectation under `q`, and `v_G` depends
//! on `σ`); setting `σ` from the running `h` and the current `v_G` at every step is an online
//! approximation to it, so `σ` has no step size. The mean moves along IVON's direction
//! `d = G / (h₀⁺ + δ)`, `h₀` the curvature before the step's own Gauss–Newton draw enters it and
//! `G = m / W + δ μ` the full gradient's estimate (the bias-corrected momentum `m / W` of the data
//! term's gradients plus the prior's exact `δ μ`; `Device::posterior_ivon`), by the minimum along
//! `d` of `F`'s local Gauss–Newton quadratic model, not of `F` along the line
//! (`DevicePosterior::step`): the slope `G · d` times the epoch's ratio of an unbiased slope along
//! the previous direction to that direction's own gradient's slope (`G · d` counts the momentum's
//! noise as descent), over the curvature along `d` with the epoch-averaged ratio of the
//! Gauss–Newton factor's curvature along `d` to the diagonal's, an average over directions that
//! change from step to step which stabilizes the length rather than measuring the current
//! direction's ratio. A prior term (`PriorTerm`) is evaluated at both of the step's antithetic
//! samples: its mean gradient joins the data term's in the momentum, and the antithetic Stein
//! estimate of its diagonal curvature (`DevicePosterior::stein_curvature`) joins `ĥ`. The
//! posterior stays on the
//! device through an epoch (`device_posterior`): the sample is written into the explanation's
//! program, the gradient stays where the reverse pass left it, and the IVON step and the groups'
//! divergences run there; the host holds it between epochs, for the
//! held-out evaluation, the checkpoint and the removal step. An epoch visits every training batch
//! once, in a fixed order. A step's data term is measured at weight samples around the iterate,
//! the optimizer's state, not the reported posterior, and a different one at every step, so the
//! fit records it as the descent's log (`Epoch::data_bits`) and never adds a description to it. At
//! each epoch's end the fit scores the end-of-epoch posterior, its data term and description at
//! the same `q`, on the whole training collection, its snapshot
//! (`snapshot_estimates`): forward passes only, each batch at its weight sample on the removal
//! comparisons' noise stream, the same draws at every snapshot, give per batch an estimate of `F`
//! at that posterior. The continuous fit stops descending at the first epoch whose snapshot's mean
//! improvement over the previous epoch's, paired by batch (the same experiments and draws: common
//! random numbers), is not positive, and goes back to the epoch whose snapshot estimate is the
//! lowest since the objective last changed: a decision on one realization of the sampled `F`, not a
//! proof of stationarity. The rule is on `F` itself rather than on the natural gradient in `μ`:
//! under IVON the means settle long before the curvature `h`, and so `σ`, has finished its
//! epoch-scale decay, which only `F` sees.
//!
//! Once the stopping criterion holds, the fit removes groups (`library_removal`): first every group without effect on
//! any experiment (a rotary plane of a head with no value coordinate left, the gate of a function
//! whose output is removed), found exactly from the program's structure; then units of groups (a
//! group with those its removal silences) ranked by their second-order removal effect (below),
//! tested in ranges of the ranked list, a rejected range split until each of its units is removed
//! with an accepted range or rejected alone.
//! Where a proposal deletes functions of an MLP, the MLP's surviving functions' outputs move by the
//! least-squares solution that takes over the deleted functions' output on `P`'s own states
//! (`library_compensation`), and the comparison scores the removal with those outputs. The
//! objective is estimated over the whole training set with one common weight sample per batch for
//! both sides of every comparison, on the fixed collection: acceptance never increases that
//! realization of the sampled `F`, whose expectation over the posterior it estimates without
//! bounding. Every proposal and its outcome go to a JSON-lines log next to the checkpoint. The fit
//! alternates descending and removing; it stops when a round accepts nothing, which says the search
//! found no removal, not that none exists.
//!
//! The removal search orders its proposals by each group's second-order removal effect: the change
//! of the data term at weight samples of the posterior when the group's entries of each sample
//! become exactly zero, less the description it saves. With `θ_b` batch `b`'s sample, `g_b` the
//! batch's data gradient there and `H_b` its Gauss–Newton matrix (the Hessian of `Σ KL(M_e ‖ P_e)`
//! where `P_e`'s predictions equal `M_e`'s), the batch's change is
//! `δ_bG = −g_b,G · θ_b,G + ½ θ_b,Gᵀ H_b θ_b,G`, and the estimate is its sum over the batches,
//! `Σ_b δ_bG` ([`Curvature`]). Both are measured at `θ_b`, one forward pass per training batch
//! reversed twice (`interchange::evaluate_probed`): once for `g_b`, and once for a draw
//! `u_b = Σ_t J_tᵀ b_t` of the Gauss–Newton factor (the Fisher probe at every token, as in the
//! fit), whose `E[(u_b,G · θ_b,G)²] = θ_b,Gᵀ H_b θ_b,G`. Over the noise of `θ_b`, `E[δ_bG]` is to
//! second order
//! `−g_G · μ_G + ½ μ_Gᵀ H μ_G − ½ Σ_{j∈G} H_jj σ_j²` at the mean, the change of the expected data
//! term. The samples are drawn on a noise stream of their own: acceptance scores other samples, so
//! a removal ranked first because its estimate fell low on these draws is not also measured on them.
//! Kept per batch, `u_b,G · θ_b,G` gives a set of groups its cross terms,
//! `½ (Σ_G u_b,G · θ_b,G)²` ([`Curvature::joint`]). The form `θ_Gᵀ H θ_G` keeps the couplings
//! between a group's parameters (an output vector's direction against the downstream metric) that a
//! diagonal curvature drops. A deletion is a finite step and the fit is not shown to be stationary,
//! so the estimate orders proposals; the exact evaluation decides them.
//!
//! # Evaluation
//!
//! At the start and after every epoch a fixed subset of the held-out sequences (the first batch of
//! them) is scored ([`HeldOut`]), its experiments and `M`'s targets for them made once per fit, and
//! all of them only while the evaluations so far, with the last
//! full evaluation's time (estimated from the subset's until one is made), stay within
//! [`EVALUATION_SHARE`] of the training time so far; the fit ends with a full evaluation. The
//! held-out experiments are a fixed sample from the training distribution (`interchange::sample`,
//! each base's from its own draws under a seed of their own): per base one clean and one patched
//! experiment, sources among the held-out sequences. `F` per token is the
//! held-out data term at one weight sample per batch plus the description spread over the training
//! experiments' scored tokens; the divergences per experiment kind are taken at the posterior mean.
//! Per layer it counts the surviving heads and MLP functions and, per token, the functions whose
//! activation is not zero and those whose activation exceeds its posterior noise scale.
//!
//! The explanation is reported at the posterior mean `μ` ([`posterior_mean`]); removed groups are
//! absent only when they fill complete interface blocks.

use crate::{
    artifact::{Argument, Artifact, Callee, Owner},
    device_posterior::{DevicePosterior, Ivon, State},
    device_program::{DeviceTrace, gelu_tanh_constant, law_of},
    interchange::{self, Batch, Experiment, Interchange, Patch, ReadVariable, Targets},
    library_compensation::Compensation,
    library_removal::{self, Evaluation},
    operator_program::{
        FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram, Provenance, Rule, SequenceLayout, SlotValues,
        exact_precision,
    },
    run_check::{LayerNodes, head_projection},
};
use gam_gpu::tensor::{Device, Op, Storage, Tensor};
use gam_runtime::warm_start::Fingerprinter;
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    f64::consts::LN_2,
    io::{BufReader, Read, Write},
    ops::Range,
    path::Path,
    sync::Arc,
    thread::JoinHandle,
    time::Instant,
};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}

/// Entries of one trainable operator: `rows × cols`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cells {
    pub operator: usize,
    pub rows: Vec<usize>,
    pub cols: Range<usize>,
}

/// A number's cost in a budget in bits (`Settings::budget_bits`): 16 bits, the fixed precision of
/// toys' structural code, under which their copy tasks were solved.
const NUMBER_BITS: f64 = 16.0;

/// A prior group: parameters sharing one prior variance (module note).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub name: String,
    pub cells: Vec<Cells>,
}

/// A stage of gated components that share gates (`library_vpd`): the assignment operator
/// `operator` (by name; gates × components, each component's column its assignment over the
/// gates, relaxed in training and 0/1 at evaluation) and per component its candidate gates, its
/// own first: the gates its assignment may move to, so a part of any rank forms where components
/// come to share a gate, and leaves where one moves away.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Share {
    pub operator: String,
    pub candidates: Vec<Vec<usize>>,
}

/// An exact frame for one group of a map's slices (`library_vpd`'s exact explanations): the
/// group's `n` start slices `(u_j, v_j)`, writes `U₀` and reads `V₀` summing to `U₀ V₀ᵀ`, become
/// `C ≥ n` slices with reads `V₀ Aᵀ` and writes `U₀ A⁺ + N`, `A` (`C × n`, full column rank) free,
/// `A⁺ = (AᵀA)⁻¹ Aᵀ` and `N A = 0`, so their sum `(U₀ A⁺ + N) A V₀ᵀ = U₀ V₀ᵀ` is the same at every
/// `A` and `N`: the parts change within the group and the explanation stays exact by construction.
/// It is the design's overcomplete frame local to the group (one `n × n` solve per group), its
/// reads in the span of the group's start reads; the group's last `C − n` slices are its extra
/// slices, zero at the start, which let the group hold more parts than its start's rank. Orthogonal
/// mixing (`U Q`, `V Q`) is the case `C = n`, `A = Qᵀ`. Each slice's read is a row of an operator
/// and its write a column of one or more (one block of its rows per attention head; [`Place`]).
/// The fit pins the posterior means of every mixed operator ([`Scorer::pin_mixings`]): the reads to
/// `V₀ Aᵀ`, and the writes to the posterior's own writes `W` projected onto the frame's exact set,
/// `W + (U₀ − W A) A⁺` (that is `U₀ A⁺ + W (I − A A⁺)`, `N` the posterior's within `N A = 0`). The
/// slices' deviations, and so their description, follow the data as any group's; the reads' means
/// move only with `A`, the writes' with `A` and within `N A = 0`. It chains the operators'
/// gradients to `A` and keeps the writes' part within `N A = 0` for the posterior's step
/// ([`Scorer::take_mixing_gradients`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mix {
    /// Per slice of the group its places, every slice's in one order ([`Place::key`]): the `base`
    /// start slices first, then the extra slices.
    pub slices: Vec<Vec<Place>>,
    /// The number of start slices `n`, whose sum the frame keeps.
    pub base: usize,
}

/// Where a mixed slice's read or a block of its write is held.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Place {
    /// Row `row` of `operator` is the slice's read.
    Read { operator: usize, row: usize },
    /// Column `column` of `operator` is the block of the slice's write from entry `offset`.
    Write { operator: usize, column: usize, offset: usize },
}

impl Place {
    /// The place's operator, side and write offset: the same for every slice of a group at the
    /// same place.
    #[must_use]
    pub fn key(&self) -> (usize, bool, usize) {
        match *self {
            Place::Read { operator, .. } => (operator, false, 0),
            Place::Write { operator, offset, .. } => (operator, true, offset),
        }
    }
}

/// A group's frame in a fit ([`Mix`]): per place of its slices (in [`Mix::slices`]' order) the start
/// slices' values there (entries × `n`), its frame `A` (`C × n`), `A` before the pending move, the
/// gradient in `A` of the step's data term gathered over its passes (bits), and the curvature per
/// token of `F` in each entry of `A`, a running average over one pass of the step's factor draws
/// chained to `A` ([`Scorer::mixing_curvature`]).
#[derive(Clone, Debug)]
struct Mixing {
    mix: Mix,
    bases: Vec<Array2<f64>>,
    frame: Array2<f64>,
    previous: Option<Array2<f64>>,
    gradient: Array2<f64>,
    curvature: Array2<f64>,
}

/// The key a relaxed pass's sampled gates are drawn under, against its weight sample's key: they
/// draw on the gate nodes' streams, the weight noise on the operators'.
const GATE_DRAWS: u64 = 0x6761_7465_7364_7261;

/// The prior variance of a frame's entry about its start `[I; 0]` (`A`, [`Mix`]): unit scale, a start
/// slice's read moved by its own size, or an extra slice's read grown to a start slice's, at one
/// standard deviation; the same for every group.
const FRAME_PRIOR: f64 = 1.0;

/// An extra slice's start read, as a share of a start slice's ([`Mixing::of`]). An extra slice with
/// a zero read and a zero write has a zero gradient in both (each is the other's factor), so its row
/// of `A` starts drawn uniformly, each entry with standard deviation `1/(8√n)` (the row's length
/// about 1/8, the move of the frame's first step, [`Scorer::step_mixings`]).
const FRAME_START: f64 = 0.125;

impl Mixing {
    /// The group `mix`'s frame at its start: `A = [I; 0]`, but for the extra slices' rows, drawn
    /// under `seed` ([`FRAME_START`]).
    fn of(program: &OperatorProgram, mix: &Mix, seed: u64) -> Result<Self, String> {
        let (count, n) = (mix.slices.len(), mix.base);
        if n == 0 || n > count {
            return Err(format!("a mixing group of {count} slices with {n} start slices"));
        }
        let bases = (0..mix.slices[0].len())
            .map(|p| {
                let entries = match &mix.slices[0][p] {
                    Place::Read { operator, .. } => program.operators[*operator].cols.width(),
                    Place::Write { operator, .. } => program.operators[*operator].rows.width(),
                };
                let mut base = Array2::zeros((entries, n));
                for (j, slice) in mix.slices[..n].iter().enumerate() {
                    let values = program.operators[slice[p].key().0].matrix();
                    match slice[p] {
                        Place::Read { row, .. } => base.column_mut(j).assign(&values.row(row)),
                        Place::Write { column, .. } => base.column_mut(j).assign(&values.column(column)),
                    }
                }
                base
            })
            .collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let range = FRAME_START * (3.0 / n as f64).sqrt();
        let frame = Array2::from_shape_fn((count, n), |(i, j)| if i < n { f64::from(u8::from(i == j)) } else { rng.random_range(-range..range) });
        Ok(Self { mix: mix.clone(), bases, frame, previous: None, gradient: Array2::zeros((count, n)), curvature: Array2::zeros((count, n)) })
    }

    /// The frame's start `[I; 0]` at entry `(i, j)`, the centre of its prior.
    fn start(i: usize, j: usize) -> f64 {
        f64::from(u8::from(i == j))
    }

    /// `A⁺ = (AᵀA)⁻¹ Aᵀ` (`n × C`), refused where `A` has lost its column rank.
    fn inverse(&self) -> Result<Array2<f64>, String> {
        let a = &self.frame;
        gam_linalg::decompose::solve(a.t().dot(a).view(), a.t()).map_err(|e| format!("a mixing frame without full column rank: {e}"))
    }

    /// The group's slices at place `p` of an operator's `values` (entries × `C`).
    fn at(&self, p: usize, values: &Array2<f64>) -> Array2<f64> {
        let entries = match self.mix.slices[0][p] {
            Place::Read { .. } => values.ncols(),
            Place::Write { .. } => values.nrows(),
        };
        let mut out = Array2::zeros((entries, self.mix.slices.len()));
        for (j, slice) in self.mix.slices.iter().enumerate() {
            match slice[p] {
                Place::Read { row, .. } => out.column_mut(j).assign(&values.row(row)),
                Place::Write { column, .. } => out.column_mut(j).assign(&values.column(column)),
            }
        }
        out
    }

    /// Sets the group's slices at place `p` of an operator's values `into` to `group` ([`Mixing::at`]).
    fn put(&self, p: usize, group: &Array2<f64>, into: &mut Array2<f64>) {
        for (j, slice) in self.mix.slices.iter().enumerate() {
            match slice[p] {
                Place::Read { row, .. } => into.row_mut(row).assign(&group.column(j)),
                Place::Write { column, .. } => into.column_mut(column).assign(&group.column(j)),
            }
        }
    }

    /// The group's slices at its frame into `values` (each mixed operator's whole values): the
    /// reads `V₀ Aᵀ`, and the writes `W + (U₀ − W A) A⁺` from their values `W` in `writes` (the
    /// posterior's).
    fn write(&self, values: &mut BTreeMap<usize, Array2<f64>>, writes: &BTreeMap<usize, Array2<f64>>) -> Result<(), String> {
        let inverse = self.inverse()?;
        for (p, base) in self.bases.iter().enumerate() {
            let op = self.mix.slices[0][p].key().0;
            let group = match self.mix.slices[0][p] {
                Place::Read { .. } => base.dot(&self.frame.t()),
                Place::Write { .. } => {
                    let w = self.at(p, writes.get(&op).ok_or("a mixed write without the posterior's values")?);
                    &w + &(base - &w.dot(&self.frame)).dot(&inverse)
                }
            };
            self.put(p, &group, values.get_mut(&op).ok_or("a mixed operator without values")?);
        }
        Ok(())
    }

    /// Adds the gradient in `A` of a pass whose gradients in the mixed operators are `gradients`, at
    /// the writes `writes` ([`Mixing::chained`]).
    fn gather(&mut self, gradients: &BTreeMap<usize, Array2<f64>>, writes: &BTreeMap<usize, Array2<f64>>) -> Result<(), String> {
        let g = self.chained(gradients, writes)?;
        self.gradient += &g;
        Ok(())
    }

    /// Each entry `(i, j)` of `A` with its description in nats: `KL(N(a, σ²) ‖ N(a₀, v))`, `a₀` its
    /// start ([`Mixing::start`]), `v` [`FRAME_PRIOR`] and `σ² = 1 / (N h + 1/v)` its Laplace variance
    /// at the curvature `h` per token over `tokens` (`N`) scored tokens.
    fn entry_nats(&self, tokens: f64) -> Vec<((usize, usize), f64)> {
        self.frame
            .indexed_iter()
            .map(|((i, j), &a)| {
                let variance = 1.0 / (tokens * self.curvature[[i, j]].max(0.0) + 1.0 / FRAME_PRIOR);
                let d = a - Self::start(i, j);
                ((i, j), 0.5 * ((FRAME_PRIOR / variance).ln() + (d * d + variance) / FRAME_PRIOR - 1.0))
            })
            .collect()
    }

    /// The gradient in `A` of a pass whose gradients in the mixed operators are `gradients`, at the
    /// writes `writes` (on the frame's exact set, `W A = U₀`): per read place `Gᵀ X` (`X` the start
    /// slices' values, entries × `n`, `G` the place's gradient, entries × `C`), and per write place
    /// `−Wᵀ G A⁺ᵀ`, the writes following `A` as `U₀ A⁺ + W (I − A A⁺)` at their present `W`
    /// (`d(U₀ A⁺ + W − W A A⁺) = (U₀ − W A) dA⁺ − W dA A⁺ = −W dA A⁺`).
    fn chained(&self, gradients: &BTreeMap<usize, Array2<f64>>, writes: &BTreeMap<usize, Array2<f64>>) -> Result<Array2<f64>, String> {
        let inverse = self.inverse()?;
        let mut out = Array2::zeros(self.frame.dim());
        for (p, base) in self.bases.iter().enumerate() {
            let op = self.mix.slices[0][p].key().0;
            let Some(values) = gradients.get(&op) else { continue };
            let g = self.at(p, values);
            match self.mix.slices[0][p] {
                Place::Read { .. } => out += &g.t().dot(base),
                Place::Write { .. } => {
                    let w = self.at(p, writes.get(&op).ok_or("a mixed write without the posterior's values")?);
                    out -= &w.t().dot(&g).dot(&inverse.t());
                }
            }
        }
        Ok(out)
    }

    /// Writes into `free` (operator `op`'s) the part within `N A = 0` of the gradient `gradient` in
    /// the group's writes on `op`, `G (I − A A⁺)`: the directions the posterior moves them in.
    fn free_writes(&self, op: usize, gradient: &Array2<f64>, free: &mut Array2<f64>) -> Result<(), String> {
        let places: Vec<usize> = (0..self.bases.len()).filter(|&p| matches!(self.mix.slices[0][p], Place::Write { operator, .. } if operator == op)).collect();
        if places.is_empty() {
            return Ok(());
        }
        let inverse = self.inverse()?;
        for p in places {
            let g = self.at(p, gradient);
            self.put(p, &(&g - &g.dot(&self.frame).dot(&inverse)), free);
        }
        Ok(())
    }
}

/// A saved frame (rows) as a `C × n` matrix, refused unless it has the group's shape.
fn frame_of(rows: &[Vec<f64>], (count, n): (usize, usize)) -> Result<Array2<f64>, String> {
    if rows.len() != count || rows.iter().any(|r| r.len() != n) {
        return Err(format!("a saved frame of another group's shape (want {count} × {n})"));
    }
    Ok(Array2::from_shape_fn((count, n), |(i, j)| rows[i][j]))
}

/// A component's starting logit for its own gate, the others' 0: as descent's share arm starts
/// (d6ad2537cb), its own gate holds `e⁶ / (e⁶ + K − 1)` of the relaxed assignment (98% at K = 8).
const OWN_LOGIT: f64 = 6.0;

/// A training batch's all-on experiment ([`Scorer::all_on_pass`]): its experiments, their bits,
/// the gradient of their sum and a draw of their Gauss–Newton factor.
struct AllOn {
    experiments: Vec<Experiment>,
    bits: Vec<Vec<f64>>,
    gradient: BTreeMap<usize, Tensor>,
    factor: Option<interchange::Factor>,
}

/// Where a training pass's gate thresholds sit ([`Scorer::train_gates`]): at the iterate, at the
/// posterior's average or at the iterate before the pending move.
#[derive(Clone, Copy)]
enum Center {
    Iterate,
    Average,
    Previous,
}

/// How a fit writes a shared stage's assignment ([`Assignment`]): relaxed (each component's
/// softmax over its candidates) in a training pass, as before the pending move in that move's
/// test, or hardened (each component on its largest candidate alone) in every evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Relaxation {
    Soft,
    Previous,
    Hard,
}

/// The parameter values a pass of `P` runs at ([`Scorer::pass`]).
#[derive(Clone, Copy, Debug)]
enum Values {
    /// The posterior's mean `μ̄`.
    Mean,
    /// The mean rounded to its posterior's precision on the device (`DevicePosterior::rounded_into`).
    Rounded,
    /// The weight sample of a key around the mean.
    Sample(u64),
    /// The step's weight sample of a key around the iterate.
    Iterate(u64),
    /// That sample around the iterate before the pending move.
    Previous(u64),
}

/// The gates a pass of `P` executes ([`Scorer::pass`]): the training law, each gated component's
/// relaxed gate and each shared stage's softened assignment, an aid to the optimization alone; or
/// the explanation as it is scored and exported, every gate hard and every assignment hardened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gates {
    Relaxed,
    Hard,
}

/// A batch's data terms of `F` ([`Scorer::terms`]): its experiments' bits, the gradient of their
/// sum and a draw of its factor, and the all-on experiment's ([`Scorer::all_on_pass`]).
struct Terms {
    bits: Vec<Vec<f64>>,
    gradient: BTreeMap<usize, Tensor>,
    factor: Option<interchange::Factor>,
    all_on: Option<AllOn>,
}

impl Terms {
    /// The batch's data term in bits: its experiments' and its all-on experiment's.
    fn bits(&self) -> f64 {
        self.bits.iter().flatten().sum::<f64>() + self.all_on.as_ref().map_or(0.0, |on| on.bits.iter().flatten().sum::<f64>())
    }

    /// `other`, the terms of another part of the same batch, joined to these: bits after bits,
    /// gradients summed, and each factor the one part's that drew it.
    fn join(&mut self, device: &Device, other: Terms) -> Result<(), String> {
        self.bits.extend(other.bits);
        add_into(device, &mut self.gradient, other.gradient)?;
        self.factor = self.factor.take().or(other.factor);
        self.all_on = match (self.all_on.take(), other.all_on) {
            (Some(mut on), Some(other)) => {
                on.experiments.extend(other.experiments);
                on.bits.extend(other.bits);
                add_into(device, &mut on.gradient, other.gradient)?;
                on.factor = on.factor.take().or(other.factor);
                Some(on)
            }
            (on, other) => on.or(other),
        };
        Ok(())
    }

    /// The all-on experiment's gradient and factor joined to the experiments' (a training step's,
    /// the removal round's curvature): the gradients summed, and the all-on factor draw, scaled to
    /// the experiments' factor weight (its square estimates the curvature per token of the tokens
    /// it sums), added to theirs. Their probes are independent, so the sum's square is unbiased for
    /// the sum of the two curvatures; without it the step's model did not see the term it descends.
    fn combine(&mut self, device: &Device) -> Result<(), String> {
        let Some(on) = self.all_on.as_mut() else { return Ok(()) };
        add_into(device, &mut self.gradient, std::mem::take(&mut on.gradient))?;
        if let (Some(factor), Some(on_factor)) = (self.factor.as_mut(), on.factor.take()) {
            let scored: usize = self.bits.iter().map(Vec::len).sum();
            let on_scored: usize = on.bits.iter().map(Vec::len).sum();
            let ratio = ((on_scored as f64 / on_factor.tokens.max(1) as f64) / (scored as f64 / factor.tokens.max(1) as f64)).sqrt();
            for (op, u) in &on_factor.gradient {
                match factor.gradient.get_mut(op) {
                    Some(total) => device.axpy(total, ratio, u).map_err(error)?,
                    None => {
                        let mut scaled = device.zeros(u.rows(), u.cols()).map_err(error)?;
                        device.axpy(&mut scaled, ratio, u).map_err(error)?;
                        factor.gradient.insert(*op, scaled);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Adds each of `from`'s tensors to `into`'s of the same operator.
fn add_into(device: &Device, into: &mut BTreeMap<usize, Tensor>, from: BTreeMap<usize, Tensor>) -> Result<(), String> {
    for (op, g) in from {
        match into.get_mut(&op) {
            Some(sum) => device.axpy(sum, 1.0, &g).map_err(error)?,
            None => {
                into.insert(op, g);
            }
        }
    }
    Ok(())
}

/// A shared stage's assignment in a fit ([`Share`]): its operator (gates × components), each
/// component's candidate gates (its own first) and the logits of its assignment over them, the
/// logits before the pending move, and the gradient of the step's data term in the logits
/// gathered over the step's passes (bits). The logits are not part of the posterior: what a
/// receiver needs is each component's gate, `ln K` nats for a component of `K` candidates
/// (`Explanation::fixed_nats`, `library_vpd`), so they take mirror-descent steps on the data
/// term's gradient ([`Scorer::step_assignments`]), tested with the posterior's move.
#[derive(Clone, Debug)]
struct Assignment {
    operator: usize,
    gates: usize,
    candidates: Vec<Vec<usize>>,
    logits: Vec<Vec<f64>>,
    previous: Option<Vec<Vec<f64>>>,
    gradient: Vec<Vec<f64>>,
}

impl Assignment {
    fn of(program: &OperatorProgram, share: &Share) -> Result<Self, String> {
        let operator = index_of(program, &share.operator)?;
        let (gates, components) = (program.operators[operator].rows.width(), program.operators[operator].cols.width());
        if share.candidates.len() != components || share.candidates.iter().flatten().any(|g| *g >= gates) {
            return Err(format!("{}: candidates of another stage", share.operator));
        }
        let logits = share.candidates.iter().map(|c| (0..c.len()).map(|k| if k == 0 { OWN_LOGIT } else { 0.0 }).collect()).collect();
        let gradient = share.candidates.iter().map(|c| vec![0.0; c.len()]).collect();
        Ok(Self { operator, gates, candidates: share.candidates.clone(), logits, previous: None, gradient })
    }

    /// Each component's relaxed assignment over its candidates at `logits`.
    fn softmax(logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        logits
            .iter()
            .map(|l| {
                let top = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = l.iter().map(|v| (v - top).exp()).collect();
                let sum: f64 = e.iter().sum();
                e.into_iter().map(|v| v / sum).collect()
            })
            .collect()
    }

    /// The assignment operator's values (gates × components) under `relaxation`.
    fn values(&self, relaxation: Relaxation) -> Array2<f64> {
        let mut out = Array2::zeros((self.gates, self.candidates.len()));
        let logits = match relaxation {
            Relaxation::Previous => self.previous.as_ref().unwrap_or(&self.logits),
            Relaxation::Soft | Relaxation::Hard => &self.logits,
        };
        for (b, (candidates, weights)) in self.candidates.iter().zip(Self::softmax(logits)).enumerate() {
            if relaxation == Relaxation::Hard {
                let best = weights.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map_or(0, |(k, _)| k);
                out[[candidates[best], b]] = 1.0;
            } else {
                for (g, w) in candidates.iter().zip(weights) {
                    out[[*g, b]] += w;
                }
            }
        }
        out
    }

    /// The gradient `g` of a training pass in the operator (gates × components, at the relaxed
    /// assignment) chained to the logits through each component's softmax,
    /// `∂/∂ℓ_bk = a_bk (g_{c_bk, b} − Σ_j a_bj g_{c_bj, b})`, added to the step's.
    fn gather(&mut self, g: &Array2<f64>) {
        let weights = Self::softmax(&self.logits);
        for (b, ((candidates, a), sum)) in self.candidates.iter().zip(&weights).zip(&mut self.gradient).enumerate() {
            let mean: f64 = candidates.iter().zip(a).map(|(c, w)| w * g[[*c, b]]).sum();
            for ((c, w), s) in candidates.iter().zip(a).zip(sum.iter_mut()) {
                *s += w * (g[[*c, b]] - mean);
            }
        }
    }
}

/// The library explanation of a language model: the artifact at its starting point, its trainable
/// operators, its prior groups and its layers.
#[derive(Clone, Debug)]
pub struct Explanation {
    pub artifact: Artifact,
    /// The library's operators of `artifact.program`, ascending.
    pub trainable: Vec<usize>,
    pub groups: Vec<Group>,
    pub layers: Vec<Layer>,
    /// Groups that start outside the explanation, exactly zero (the output a read–write tie
    /// replaces, `library_sharing`).
    pub removed: Vec<usize>,
    /// The nats of the explanation's discrete choices: each exact read–write tie's choice of the
    /// write it reads among its candidates (`library_mixture`).
    pub fixed_nats: f64,
    /// Per group, the reference variance `v⁰_G` its variance's scale is sent against (module
    /// note): the mean square of `M`'s values over its cells in the explanation made at `M`, and
    /// for a group a rewrite or share makes, a value fixed by `M` and the structure. It stays with
    /// the group when the explanation starts from moved values (a warm start), so the scale code
    /// is decodable from `M` and the explanation's structure.
    pub reference: Vec<f64>,
    /// `M`'s read variables (`interchange::reads`): fixed by `M` and the library made at it, and
    /// kept by every explanation derived from that one, so a fit reads them here rather than
    /// building the library again.
    pub reads: Vec<interchange::ReadVariable>,
    /// The stages whose components share gates (`library_vpd`'s gate sharing; [`Share`]).
    pub shares: Vec<Share>,
    /// How a fit scores the gated components ([`GateScoring`]).
    pub scoring: GateScoring,
    /// The orthogonal mixings of groups of slices ([`Mix`]; `library_vpd`'s exact explanations).
    pub mixes: Vec<Mix>,
}

/// How a fit scores an explanation's gated components (`library_vpd::Gate`) through the hard gates
/// (`Gates::Hard`): as compiled (a hard stage's width holds `library_vpd::HARD`), or by the hard
/// gate `H(z)` in place of a learned width's `Φ(z / w)` (`DeviceProgram::set_hard`); a relaxed pass
/// gates as compiled. Either way the scored explanation is the exported one (`posterior_mean`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GateScoring {
    #[default]
    Compiled,
    Hard,
}

/// Per group of `groups`, the mean square of `program`'s values over its cells. A group whose
/// values are all zero takes the mean square of its operators' entries in the groups that are not,
/// and `1` where those are all zero too: either is fixed by `program` and the groups before
/// decoding, so every group has a positive reference.
pub fn mean_squares(program: &OperatorProgram, groups: &[Group]) -> Vec<f64> {
    let mut matrices: BTreeMap<usize, Array2<f64>> = BTreeMap::new();
    for group in groups {
        for cell in &group.cells {
            matrices.entry(cell.operator).or_insert_with(|| program.operators[cell.operator].matrix());
        }
    }
    let squares = |cells: &[Cells]| -> (f64, f64) {
        let (mut count, mut sum) = (0.0, 0.0);
        for cell in cells {
            let matrix = &matrices[&cell.operator];
            for &row in &cell.rows {
                for col in cell.cols.clone() {
                    sum += matrix[[row, col]] * matrix[[row, col]];
                    count += 1.0;
                }
            }
        }
        (count, sum)
    };
    let own: Vec<(f64, f64)> = groups.iter().map(|group| squares(&group.cells)).collect();
    // Per operator, the count and sum of squares of its entries in groups that are not all zero.
    let mut operators: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for (group, (_, sum)) in groups.iter().zip(&own) {
        if *sum > 0.0 {
            for cell in &group.cells {
                let (count, total) = squares(std::slice::from_ref(cell));
                let entry = operators.entry(cell.operator).or_default();
                entry.0 += count;
                entry.1 += total;
            }
        }
    }
    groups
        .iter()
        .zip(&own)
        .map(|(group, &(count, sum))| {
            if sum > 0.0 {
                return sum / count;
            }
            let (n, total) = group.cells.iter().filter_map(|c| operators.get(&c.operator)).fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
            if total > 0.0 { total / n } else { 1.0 }
        })
        .collect()
}

/// A prior term beyond the groups' own Gaussian priors (`library_mixture`): its value at a weight
/// sample enters `F` beside the groups' divergences, its gradient in the sample joins the data
/// term's, and a training step takes its diagonal curvature from its gradients at the step's two
/// antithetic samples (`DevicePosterior::stein_curvature`).
pub trait PriorTerm {
    /// The trainable operators it reads (indices into `Explanation::trainable`).
    fn operators(&self) -> Vec<usize>;
    /// Re-choose its structure from `posterior`, once per epoch.
    fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String>;
    /// Its value in nats at the weight sample `theta` (its operators', by trainable index) of
    /// `posterior`, its gradient in `theta`, and with `learn` one step of its own parameters.
    fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String>;
    /// [`PriorTerm::sample`] at the weight sample of `key` of `device_posterior` as a training step
    /// draws it (around IVON's iterate: the draws of [`DevicePosterior::iterate_into`]), its
    /// gradient by trainable index on `device`. `posterior` holds the groups' activity and the
    /// operators' shapes; a term without a form on the device moves its operators' posterior means
    /// and deviations into it ([`host_term`]).
    fn sample_device(&mut self, device: &Device, device_posterior: &DevicePosterior, posterior: &mut Posterior, key: u64, learn: bool) -> Result<(f64, BTreeMap<usize, Tensor>), String> {
        host_term(self, device, device_posterior, posterior, key, learn)
    }
    /// The nats of the parameters it sends.
    fn cost(&self, posterior: &Posterior) -> Result<f64, String>;
    /// Its state, for a checkpoint, and its state restored from one.
    fn save(&self) -> Result<serde_json::Value, String>;
    fn load(&mut self, value: &serde_json::Value) -> Result<(), String>;
}

/// The weight sample of `key` (`DevicePosterior::sample_into`'s draws) of `posterior`'s trainable
/// operators `operators`, on the host.
pub(crate) fn host_sample(posterior: &Posterior, operators: &[usize], key: u64) -> BTreeMap<usize, Array2<f64>> {
    operators
        .par_iter()
        .map(|&i| {
            let cols = posterior.mean[i].ncols();
            let epsilon = Array2::from_shape_fn(posterior.mean[i].dim(), |(r, c)| f64::from(gam_gpu::tensor::posterior_normal(key, i as u64, (r * cols + c) as u64)));
            (i, &*posterior.mean[i] + &posterior.noise(i, &epsilon))
        })
        .collect()
}

/// The weight sample of `key` a training step draws (`DevicePosterior::iterate_into`: `μ + σ ⊙ ε`
/// around IVON's iterate `μ`) of `device_posterior`'s trainable operators `operators`, on the host;
/// `posterior` takes their posterior means `μ̄` and log standard deviations, at which a prior term
/// reads its groups' variances.
pub(crate) fn step_sample(device_posterior: &DevicePosterior, posterior: &mut Posterior, operators: &[usize], key: u64) -> Result<BTreeMap<usize, Array2<f64>>, String> {
    let mut iterates = Vec::with_capacity(operators.len());
    for &i in operators {
        let (mean, log_sd) = device_posterior.values(i)?;
        posterior.mean[i] = mean.into();
        posterior.log_sd[i] = log_sd.into();
        iterates.push((i, device_posterior.iterate(i)?));
    }
    let posterior = &*posterior;
    Ok(iterates
        .into_par_iter()
        .map(|(i, iterate)| {
            let cols = iterate.ncols();
            let epsilon = Array2::from_shape_fn(iterate.dim(), |(r, c)| f64::from(gam_gpu::tensor::posterior_normal(key, i as u64, (r * cols + c) as u64)));
            (i, iterate + &posterior.noise(i, &epsilon))
        })
        .collect())
}

/// [`PriorTerm::sample`] of `prior` at the weight sample of `key` of `device_posterior` as a
/// training step draws it, on the host: its operators' means and deviations come off the device
/// into `posterior`, the sample is drawn around the iterate (`step_sample`), and the gradient goes
/// back up.
pub fn host_term<P: PriorTerm + ?Sized>(prior: &mut P, device: &Device, device_posterior: &DevicePosterior, posterior: &mut Posterior, key: u64, learn: bool) -> Result<(f64, BTreeMap<usize, Tensor>), String> {
    let operators = prior.operators();
    let theta = step_sample(device_posterior, posterior, &operators, key)?;
    let (value, gradient) = prior.sample(posterior, &theta, learn)?;
    let gradient = gradient.into_iter().map(|(i, g)| Ok((i, device.upload(g.view()).map_err(error)?))).collect::<Result<_, String>>()?;
    Ok((value, gradient))
}

/// `prior`'s value with the parameters it sends, in nats, at the weight sample of `key` of
/// `posterior`, and its gradient in the sample; with `learn`, one step of its own parameters.
fn prior_term(prior: &mut (dyn PriorTerm + 'static), posterior: &Posterior, key: u64, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
    let theta = host_sample(posterior, &prior.operators(), key);
    let (value, gradient) = prior.sample(posterior, &theta, learn)?;
    Ok((value + prior.cost(posterior)?, gradient))
}

/// One layer of the explanation: its native sites (`run_check::layer_nodes`) and the prior groups
/// of its functions.
#[derive(Clone, Debug)]
pub struct Layer {
    pub sites: LayerNodes,
    /// Per head, its planes' groups and its value coordinates' groups (those of its key-value
    /// group, shared with the group's other query heads).
    pub heads: Vec<(Vec<usize>, Vec<usize>)>,
    /// Per MLP function, its groups: its gate's, its up direction's when gated, and its output's.
    pub functions: Vec<Vec<usize>>,
    /// A transcoder layer's group of its MLP block's output at each sequence's first token
    /// (`library_transcoder`, `library.l{l}.mlp.sink`), part of the MLP block.
    pub sink: Option<usize>,
    /// A transcoder layer's groups of its MLP's gate biases, held apart from the gate rows (one per
    /// biased map), part of the MLP block.
    pub thresholds: Vec<usize>,
    /// Per gated component of the attention block its groups (`library_vpd`: slices of the
    /// q, k, v and o maps with one intrinsic gate), and the block's thresholds as one more entry.
    pub components: Vec<Vec<usize>>,
}

/// A library operator: dense, every block present, its reals exactly representable.
fn library_operator(name: &str, rows: Interface, cols: Interface, values: Array2<f64>, source: &str) -> Result<Operator, String> {
    let precision = exact_precision(values.iter().copied()).map_err(error)?;
    Operator::dense(name, rows, cols, values, precision, Provenance::derived(&[&Provenance::native(source)], "library initialization".into()))
        .map_err(error)
}

/// [`library_operator`] holding `source`'s reals as they are: where a checkpoint stores them
/// (`DenseValues::stored`), the library's operator reads them there too, with no float64 copy on
/// the host.
fn library_copy(name: &str, rows: Interface, cols: Interface, source: &Operator) -> Result<Operator, String> {
    match &source.body {
        OperatorBody::Dense { values, present, .. } if present.iter().all(|p| *p) && values.stored().is_some() => {
            let stored = values.stored().cloned().ok_or("stored reals")?;
            let provenance = Provenance::derived(&[&Provenance::native(&source.name)], "library initialization".into());
            Operator::stored(name, rows, cols, stored, provenance).map_err(error)
        }
        _ => library_operator(name, rows, cols, source.matrix(), &source.name),
    }
}

/// The native operator of a bias-free affine node reading `input` alone.
fn single_map(native: &OperatorProgram, node: usize, input: usize) -> Result<Arc<Operator>, String> {
    match &native.nodes[node] {
        Node::Affine { terms, bias: None } if terms.len() == 1 && terms[0].0 == input => Ok(Arc::clone(&native.operators[terms[0].1])),
        other => Err(format!("node {node} is not a bias-free map of node {input}: {other:?}")),
    }
}

/// The head norm on `node` (a head's query or key; `run_check::head_projection`): its RMS norm's
/// `ε` and its gain operator.
fn head_norm(native: &OperatorProgram, node: usize) -> Option<(f64, Arc<Operator>)> {
    if head_projection(native, node) == node {
        return None;
    }
    match &native.nodes[node] {
        Node::Affine { terms, .. } => match native.nodes[terms[0].0] {
            Node::RmsNorm { epsilon, .. } => Some((epsilon, Arc::clone(&native.operators[terms[0].1]))),
            _ => None,
        },
        _ => None,
    }
}

/// The native operator and bias of the affine node `node` reading `input` alone.
fn affine_map(native: &OperatorProgram, node: usize, input: usize) -> Result<(Arc<Operator>, Option<Arc<Operator>>), String> {
    match &native.nodes[node] {
        Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == input => {
            Ok((Arc::clone(&native.operators[terms[0].1]), bias.map(|b| Arc::clone(&native.operators[b]))))
        }
        other => Err(format!("node {node} is not an affine map of node {input}: {other:?}")),
    }
}

/// The library block of a gated MLP `down(φ(A x + c) ⊙ (B x + e))` (SwiGLU on Qwen3): functions
/// `f_i(x) = φ(a_i·x + c_i) (b_i·x + e_i) u_i` with `M`'s law `φ`, its gate `a_i`, up direction
/// `b_i` and output `u_i`, and the biases `c_i`, `e_i` only where `M` has them.
fn gated_mlp(native: &OperatorProgram, artifact: Artifact, layer: &LayerNodes, l: usize, left: usize, right: usize) -> Result<(Artifact, Vec<Owner>), String> {
    let Node::Pointwise { input: gate_pre, laws } = &native.nodes[left] else {
        return Err(format!("layer {l}: the gated product's left factor is not one pointwise law"));
    };
    if laws.iter().any(|law| *law != laws[0]) {
        return Err(format!("layer {l}: the MLP's units have different laws"));
    }
    let x = layer.normed;
    let ((gate, gate_bias), (up, up_bias)) = (affine_map(native, *gate_pre, x)?, affine_map(native, right, x)?);
    let down = single_map(native, layer.mlp, layer.active)?;
    let units = up.rows.clone();
    if gate.rows != units || down.cols != units {
        return Err(format!("layer {l}: the gate, up and down maps disagree on the MLP's units"));
    }
    let name = format!("library.l{l}.mlp");
    let base = artifact.program.operators.len();
    let mut operators = vec![
        library_copy(&format!("{name}.gate"), units.clone(), gate.cols.clone(), &gate)?,
        library_copy(&format!("{name}.up"), units.clone(), up.cols.clone(), &up)?,
        library_copy(&format!("{name}.out"), down.rows.clone(), units.clone(), &down)?,
    ];
    let owners = mlp_owners(&name, units.width(), &gate, gate_bias.as_deref(), Some((&up, up_bias.as_deref())), &down);
    let mut bias = |values: Option<Arc<Operator>>, part: &str, source: &str| -> Result<Option<usize>, String> {
        let Some(values) = values else { return Ok(None) };
        operators.push(library_operator(&format!("{name}.{part}_bias"), units.clone(), Interface::constant(), values.matrix(), source)?);
        Ok(Some(base + operators.len() - 1))
    };
    let (gate_bias, up_bias) = (bias(gate_bias, "gate", &gate.name)?, bias(up_bias, "up", &up.name)?);
    let rule = Rule {
        name: name.clone(),
        inputs: vec![gate.cols.clone()],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, base)], bias: gate_bias },
            Node::Pointwise { input: 1, laws: laws.clone() },
            Node::Affine { terms: vec![(0, base + 1)], bias: up_bias },
            Node::Hadamard { left: 2, right: 3 },
            Node::Affine { terms: vec![(4, base + 2)], bias: None },
        ],
        output: 5,
    };
    Ok((artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(x)], layer.mlp, operators)?, owners))
}

/// The owners of an MLP block `name` of `units` functions (`artifact::Owner`): per function its
/// gate row (and bias), its up row (and bias) when gated, and its output column, each replacing
/// the same row or column of `M`'s map.
fn mlp_owners(name: &str, units: usize, gate: &Operator, gate_bias: Option<&Operator>, up: Option<(&Operator, Option<&Operator>)>, down: &Operator) -> Vec<Owner> {
    let mut out = Vec::new();
    let mut add = |part: &str, native: &Operator, unit: usize, column: bool| {
        let (rows, cols) = if column { (0..native.rows.width(), unit..unit + 1) } else { (unit..unit + 1, 0..native.cols.width()) };
        out.push(Owner {
            operator: format!("{name}.{part}"),
            rows: rows.clone(),
            cols: cols.clone(),
            body: name.to_string(),
            site: name.to_string(),
            native: native.name.clone(),
            native_rows: rows,
            native_cols: cols,
            role: part.to_string(),
            ..Owner::default()
        });
    };
    for i in 0..units {
        add("gate", gate, i, false);
        if let Some(bias) = gate_bias {
            add("gate_bias", bias, i, false);
        }
        if let Some((up, up_bias)) = up {
            add("up", up, i, false);
            if let Some(bias) = up_bias {
                add("up_bias", bias, i, false);
            }
        }
        add("out", down, i, true);
    }
    out
}

/// The operator named `name`, when the program has one (a part a law may lack: a bias, an up map).
fn operator_named(program: &OperatorProgram, name: &str) -> Option<usize> {
    program.operators.iter().position(|op| op.name == name)
}

fn index_of(program: &OperatorProgram, name: &str) -> Result<usize, String> {
    let mut found = program.operators.iter().enumerate().filter(|(_, op)| op.name == name).map(|(i, _)| i);
    match (found.next(), found.next()) {
        (Some(index), None) => Ok(index),
        _ => Err(format!("no unique library operator {name}")),
    }
}

/// The library explanation of the split native language model `native` (`run_check::split_sites`)
/// with its `layers` (`run_check::layer_nodes`), at its starting point (module note).
pub fn explanation(native: &OperatorProgram, layers: &[LayerNodes]) -> Result<Explanation, String> {
    explanation_with(native, layers, &BTreeMap::new())
}

/// [`explanation`] with the MLPs of the layers in `transcoders` replaced by the transcoder features
/// each one's kept file holds (`library_transcoder::mlp`): a function `relu(g·x + c) u` per kept
/// feature, `M`'s own MLP functions in every other layer.
pub fn explanation_with(native: &OperatorProgram, layers: &[LayerNodes], transcoders: &BTreeMap<usize, std::path::PathBuf>) -> Result<Explanation, String> {
    let mut artifact = Artifact::native(native)?;
    let mut planes = Vec::new();
    let mut owners = Vec::new();
    for (l, layer) in layers.iter().enumerate() {
        // Per key-value group (the native key and value projections its query heads read): its
        // name, its query heads, its planes and its value width.
        let mut groups: Vec<(usize, usize, String, Vec<(usize, String)>, Vec<Vec<usize>>, usize)> = Vec::new();
        for (h, &read) in layer.reads.iter().enumerate() {
            let Node::Attend { query, key, value, scale, rotary, causal } = native.nodes[read].clone() else {
                return Err(format!("layer {l} head {h}: the read is not an attention node"));
            };
            let x = layer.normed_stream;
            let (query_norm, key_norm) = (head_norm(native, query), head_norm(native, key));
            let (query, key) = (head_projection(native, query), head_projection(native, key));
            let (q, k, v) = (single_map(native, query, x)?, single_map(native, key, x)?, single_map(native, value, x)?);
            if q.rows.width() != k.rows.width() {
                return Err(format!("layer {l} head {h}: query and key widths differ"));
            }
            // Query and key coordinates in their own groups, so a removed plane is an absent block.
            let coordinates = Interface::uniform(q.rows.width(), 1, LabelKind::Unit, 0).map_err(error)?;
            let name = format!("library.l{l}.h{h}");
            let base = artifact.program.operators.len();
            let mut operators = vec![library_copy(&format!("{name}.q"), coordinates.clone(), q.cols.clone(), &q)?];
            // Query heads that read one key and value in `M` read one key map and one value map:
            // one posterior, every head's use in its gradient, one divergence.
            let group = groups.iter().position(|g| g.0 == key && g.1 == value);
            let (k_op, v_op) = match group {
                Some(g) => (index_of(&artifact.program, &format!("{}.k", groups[g].2))?, index_of(&artifact.program, &format!("{}.v", groups[g].2))?),
                None => {
                    let shared = format!("library.l{l}.kv{}", groups.len());
                    operators.push(library_copy(&format!("{shared}.k"), coordinates.clone(), k.cols.clone(), &k)?);
                    operators.push(library_copy(&format!("{shared}.v"), v.rows.clone(), v.cols.clone(), &v)?);
                    (base + 1, base + 2)
                }
            };
            let mut nodes = vec![
                Node::Param { index: 0 },
                Node::Affine { terms: vec![(0, base)], bias: None },
                Node::Affine { terms: vec![(0, k_op)], bias: None },
                Node::Affine { terms: vec![(0, v_op)], bias: None },
            ];
            // A head norm: the RMS norm of the projection, then the head's own copy of `M`'s gain.
            let mut normed = |projection: usize, norm: Option<(f64, Arc<Operator>)>, part: &str| -> Result<usize, String> {
                let Some((epsilon, gain)) = norm else { return Ok(projection) };
                let values = gain.diagonal().ok_or_else(|| format!("{}: a head norm's gain is not diagonal", gain.name))?;
                let precision = exact_precision(values.iter().copied()).map_err(error)?;
                nodes.push(Node::RmsNorm { input: projection, epsilon });
                nodes.push(Node::Affine { terms: vec![(nodes.len() - 1, base + operators.len())], bias: None });
                operators.push(Operator::diag(format!("{name}.{part}_gain"), coordinates.clone(), values, precision, gain.provenance.clone()).map_err(error)?);
                Ok(nodes.len() - 1)
            };
            let (query_node, key_node) = (normed(1, query_norm, "q")?, normed(2, key_norm, "k")?);
            nodes.push(Node::Attend { query: query_node, key: key_node, value: 3, scale, rotary, causal });
            let rule = Rule { name: name.clone(), inputs: vec![q.cols.clone()], output: nodes.len() - 1, nodes };
            artifact = artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(x)], read, operators)?;
            // The head's query map, and its group's key and value maps as this head reads them.
            let shared = match group {
                Some(g) => groups[g].2.clone(),
                None => format!("library.l{l}.kv{}", groups.len()),
            };
            for (operator, source, role) in [(format!("{name}.q"), &q, "q"), (format!("{shared}.k"), &k, "k"), (format!("{shared}.v"), &v, "v")] {
                let (rows, cols) = (0..source.rows.width(), 0..source.cols.width());
                owners.push(Owner { operator, rows: rows.clone(), cols: cols.clone(), body: name.clone(), site: name.clone(), native: source.name.clone(), native_rows: rows, native_cols: cols, role: role.into(), ..Owner::default() });
            }
            let pairs: Vec<Vec<usize>> = match rotary {
                Some(r) => {
                    let pairs = r.pairs();
                    let rotated: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                    pairs.iter().map(|&(a, b)| vec![a, b]).chain((0..q.rows.width()).filter(|c| !rotated.contains(c)).map(|c| vec![c])).collect()
                }
                None => (0..q.rows.width()).map(|c| vec![c]).collect(),
            };
            match group {
                Some(g) if groups[g].4 == pairs && groups[g].5 == v.rows.width() => groups[g].3.push((h, name)),
                Some(_) => return Err(format!("layer {l} head {h}: the query heads of one key-value group differ in their planes")),
                None => groups.push((key, value, format!("library.l{l}.kv{}", groups.len()), vec![(h, name)], pairs, v.rows.width())),
            }
        }
        planes.extend(groups.into_iter().map(|(_, _, shared, heads, pairs, values)| (l, shared, heads, pairs, values)));
        if let Some(kept) = transcoders.get(&l) {
            let (built, more) = crate::library_transcoder::mlp(native, artifact, layer, l, kept)?;
            artifact = built;
            owners.extend(more);
            continue;
        }
        if let Node::Hadamard { left, right } = native.nodes[layer.active] {
            let (built, more) = gated_mlp(native, artifact, layer, l, left, right)?;
            artifact = built;
            owners.extend(more);
            continue;
        }
        let Node::Pointwise { input: pre, laws } = &native.nodes[layer.active] else {
            return Err(format!("layer {l}: the MLP activation is neither one pointwise law nor a gated product"));
        };
        if laws.iter().any(|law| *law != laws[0]) {
            return Err(format!("layer {l}: the MLP's units have different laws"));
        }
        let (up, up_bias) = match &native.nodes[*pre] {
            Node::Affine { terms, bias } if terms.len() == 1 && terms[0].0 == layer.normed => (Arc::clone(&native.operators[terms[0].1]), *bias),
            other => return Err(format!("layer {l}: the MLP's input map is {other:?}")),
        };
        let down = single_map(native, layer.mlp, layer.active)?;
        let units = up.rows.clone();
        let name = format!("library.l{l}.mlp");
        owners.extend(mlp_owners(&name, units.width(), &up, up_bias.map(|b| native.operators[b].as_ref()), None, &down));
        let base = artifact.program.operators.len();
        let mut operators = vec![
            library_copy(&format!("{name}.gate"), units.clone(), up.cols.clone(), &up)?,
            library_copy(&format!("{name}.out"), down.rows.clone(), units.clone(), &down)?,
        ];
        // A gate bias only where `M` has one, so a bias-free MLP still maps 0 to 0.
        let gate_bias = match up_bias {
            Some(op) => {
                operators.push(library_operator(&format!("{name}.gate_bias"), units.clone(), Interface::constant(), native.operators[op].matrix(), &up.name)?);
                Some(base + 2)
            }
            None => None,
        };
        let rule = Rule {
            name: name.clone(),
            inputs: vec![up.cols.clone()],
            nodes: vec![
                Node::Param { index: 0 },
                Node::Affine { terms: vec![(0, base)], bias: gate_bias },
                Node::Pointwise { input: 1, laws: laws.clone() },
                Node::Affine { terms: vec![(2, base + 1)], bias: None },
            ],
            output: 3,
        };
        artifact = artifact.replace_block(&name, Callee::New(rule), vec![Argument::Native(layer.normed)], layer.mlp, operators)?;
    }
    // Groups by the final operator indices (each replacement renumbers the operators).
    let program = &artifact.program;
    let mut groups = Vec::new();
    let mut trainable = Vec::new();
    let mut out: Vec<Layer> = layers.iter().map(|sites| Layer { sites: sites.clone(), heads: Vec::new(), functions: Vec::new(), sink: None, thresholds: Vec::new(), components: Vec::new() }).collect();
    // Per key-value group, a group per rotary plane holding the plane's rows of the shared key
    // and of every query head's query, and a group per value coordinate of the shared value.
    let mut heads: Vec<Vec<Option<(Vec<usize>, Vec<usize>)>>> = layers.iter().map(|l| vec![None; l.reads.len()]).collect();
    for (l, shared, members, pairs, values) in &planes {
        let (k, v) = (index_of(program, &format!("{shared}.k"))?, index_of(program, &format!("{shared}.v"))?);
        let queries = members.iter().map(|(_, name)| index_of(program, &format!("{name}.q"))).collect::<Result<Vec<_>, _>>()?;
        let d = program.operators[k].cols.width();
        let first = groups.len();
        for (p, rows) in pairs.iter().enumerate() {
            let cells = queries.iter().chain([&k]).map(|&operator| Cells { operator, rows: rows.clone(), cols: 0..d }).collect();
            groups.push(Group { name: format!("{shared}.plane{p}"), cells });
        }
        for j in 0..*values {
            groups.push(Group { name: format!("{shared}.value{j}"), cells: vec![Cells { operator: v, rows: vec![j], cols: 0..d }] });
        }
        for (h, _) in members {
            heads[*l][*h] = Some(((first..first + pairs.len()).collect(), (first + pairs.len()..groups.len()).collect()));
        }
        trainable.extend(queries.into_iter().chain([k, v]));
    }
    for (layer, heads) in out.iter_mut().zip(heads) {
        layer.heads = heads.into_iter().collect::<Option<Vec<_>>>().ok_or("a head in no key-value group")?;
    }
    for (l, layer) in out.iter_mut().enumerate() {
        let name = format!("library.l{l}.mlp");
        let output = index_of(program, &format!("{name}.out"))?;
        let (functions, d) = (program.operators[output].cols.width(), program.operators[output].rows.width());
        layer.functions = vec![Vec::new(); functions];
        // Per function its gate (with its bias), its up direction when gated (likewise), its output.
        // A transcoder block's gate biases are one group of the layer, apart from the gate rows: a
        // feature's threshold `c` is 30–60 times its gate weights in size, so under one isotropic
        // prior with its row the empirical-Bayes variance is set by the weights, `c` costs 300–600
        // bits per feature and the prior pulls the thresholds toward 0, which makes the features
        // fire more (compare, vpd4l: 7.2M of 28M description bits). The layer's thresholds share
        // their own variance instead.
        let apart = transcoders.contains_key(&l);
        for part in ["gate", "up"] {
            let Some(map) = operator_named(program, &format!("{name}.{part}")) else { continue };
            let bias = operator_named(program, &format!("{name}.{part}_bias"));
            for (i, function) in layer.functions.iter_mut().enumerate() {
                let mut cells = vec![Cells { operator: map, rows: vec![i], cols: 0..d }];
                if !apart {
                    cells.extend(bias.map(|b| Cells { operator: b, rows: vec![i], cols: 0..1 }));
                }
                function.push(groups.len());
                groups.push(Group { name: format!("{name}.f{i}.{part}"), cells });
            }
            if let (true, Some(b)) = (apart, bias) {
                layer.thresholds.push(groups.len());
                groups.push(Group { name: format!("{name}.{part}_bias"), cells: vec![Cells { operator: b, rows: (0..functions).collect(), cols: 0..1 }] });
            }
            trainable.push(map);
            trainable.extend(bias);
        }
        for (i, function) in layer.functions.iter_mut().enumerate() {
            function.push(groups.len());
            groups.push(Group { name: format!("{name}.f{i}.out"), cells: vec![Cells { operator: output, rows: (0..d).collect(), cols: i..i + 1 }] });
        }
        trainable.push(output);
        if let Some(sink) = operator_named(program, &format!("{name}.sink")) {
            layer.sink = Some(groups.len());
            groups.push(Group { name: format!("{name}.sink"), cells: vec![Cells { operator: sink, rows: (0..d).collect(), cols: 0..1 }] });
            trainable.push(sink);
        }
    }
    trainable.sort_unstable();
    artifact.owners = owners;
    let reference = mean_squares(&artifact.program, &groups);
    let reads = interchange::reads_of(native, &artifact, out.len())?;
    Ok(Explanation { artifact, trainable, groups, layers: out, removed: Vec::new(), fixed_nats: 0.0, reference, reads, shares: Vec::new(), scoring: GateScoring::Compiled, mixes: Vec::new() })
}

/// The prior groups of each of `explanation`'s `2L` blocks (block `2l` layer `l`'s attention,
/// `2l + 1` its MLP), from its layers' heads and functions.
fn block_groups(explanation: &Explanation) -> Vec<Vec<usize>> {
    explanation
        .layers
        .iter()
        .flat_map(|layer| {
            let attention = layer.heads.iter().flat_map(|(planes, values)| planes.iter().chain(values)).chain(layer.components.iter().flatten()).copied().collect();
            [attention, layer.functions.iter().flatten().chain(&layer.sink).chain(&layer.thresholds).copied().collect()]
        })
        .collect()
}

/// Which of `explanation`'s `2L` blocks its library explains: those holding a trainable operator.
fn scope(explanation: &Explanation) -> Vec<bool> {
    let trainable: std::collections::BTreeSet<usize> = explanation.trainable.iter().copied().collect();
    block_groups(explanation).iter().map(|groups| groups.iter().any(|g| explanation.groups[*g].cells.iter().any(|c| trainable.contains(&c.operator)))).collect()
}

/// `explanation` restricted to `blocks` (block `2l` layer `l`'s attention, `2l + 1` its MLP): `M`
/// everywhere else. Only those blocks' prior groups and their operators stay trainable and charged;
/// the other blocks' library operators keep `M`'s values and no experiment runs them (module note).
/// A group in two blocks (a share across them) is refused.
pub fn scoped(explanation: &Explanation, blocks: &[usize]) -> Result<Explanation, String> {
    let per_block = block_groups(explanation);
    if blocks.is_empty() || blocks.iter().any(|b| *b >= per_block.len()) {
        return Err(format!("blocks {blocks:?} of {}", per_block.len()));
    }
    let mut block_of = vec![None; explanation.groups.len()];
    for (b, groups) in per_block.iter().enumerate() {
        for g in groups {
            if block_of[*g].replace(b).is_some_and(|other| other != b) {
                return Err(format!("{}: a group in two blocks", explanation.groups[*g].name));
            }
        }
    }
    // Old group index to new, for the kept groups in order.
    let mut kept = vec![None; explanation.groups.len()];
    let mut count = 0;
    for (g, block) in block_of.iter().enumerate() {
        if block.is_some_and(|b| blocks.contains(&b)) {
            kept[g] = Some(count);
            count += 1;
        }
    }
    let renumber = |ids: &[usize]| -> Vec<usize> { ids.iter().filter_map(|g| kept[*g]).collect() };
    let groups: Vec<Group> = explanation.groups.iter().zip(&kept).filter(|(_, k)| k.is_some()).map(|(g, _)| g.clone()).collect();
    let mut trainable: Vec<usize> = groups.iter().flat_map(|g| g.cells.iter().map(|c| c.operator)).collect();
    trainable.sort_unstable();
    trainable.dedup();
    let layers = explanation
        .layers
        .iter()
        .enumerate()
        .map(|(l, layer)| Layer {
            sites: layer.sites.clone(),
            heads: if blocks.contains(&(2 * l)) { layer.heads.iter().map(|(planes, values)| (renumber(planes), renumber(values))).collect() } else { Vec::new() },
            functions: if blocks.contains(&(2 * l + 1)) { layer.functions.iter().map(|f| renumber(f)).collect() } else { Vec::new() },
            sink: if blocks.contains(&(2 * l + 1)) { layer.sink.and_then(|g| kept[g]) } else { None },
            thresholds: if blocks.contains(&(2 * l + 1)) { renumber(&layer.thresholds) } else { Vec::new() },
            components: if blocks.contains(&(2 * l)) { layer.components.iter().map(|c| renumber(c)).collect() } else { Vec::new() },
        })
        .collect();
    let reference = explanation.reference.iter().zip(&kept).filter(|(_, k)| k.is_some()).map(|(r, _)| *r).collect();
    Ok(Explanation { trainable, groups, layers, removed: renumber(&explanation.removed), reference, ..explanation.clone() })
}

// ------------------------------------------------------------------------------------- posterior

/// Per prior group, the removal estimates over the batches of the fixed collection: with `θ_b` the
/// weight sample of batch `b` on the ranking's own noise stream, `g_b` the data term's gradient there
/// in nats and `u_b` a draw of the Gauss–Newton factor there (`interchange::fisher_probe`),
/// `s_bG = g_b,G · θ_b,G` and `d_bG = u_b,G · θ_b,G`, and each batch's second-order change of its
/// data term when the group's entries of `θ_b` become zero, `δ_bG = −s_bG + ½ d_bG²` (module note):
/// the sums of `δ_bG`, of its square and of `s_bG` over the batches, and every batch's `d_bG` (in
/// float32), which give a set of groups its Gauss–Newton cross terms ([`Curvature::joint`]).
#[derive(Clone, Debug)]
pub struct Curvature {
    pub(crate) rise: Vec<f64>,
    pub(crate) square: Vec<f64>,
    pub(crate) slope: Vec<f64>,
    pub(crate) dots: Vec<Vec<f32>>,
    /// Per trainable operator asked for (by index into `Explanation::trainable`), `Σ_b g_b` in nats:
    /// the first-order change of the data term when its means move.
    pub(crate) gradient: BTreeMap<usize, Array2<f64>>,
}

impl Curvature {
    /// No batches yet, for `groups` prior groups.
    #[must_use]
    pub fn new(groups: usize) -> Self {
        Self { rise: vec![0.0; groups], square: vec![0.0; groups], slope: vec![0.0; groups], dots: Vec::new(), gradient: BTreeMap::new() }
    }

    /// The batches added.
    #[must_use]
    pub fn batches(&self) -> usize {
        self.dots.len()
    }

    /// Adds one batch: per group `s_bG` in nats (`slope`) and `d_bG` (`dot`).
    pub fn add_batch(&mut self, slope: &[f64], dot: &[f64]) -> Result<(), String> {
        if slope.len() != self.rise.len() || dot.len() != self.rise.len() {
            return Err("a batch of another explanation".into());
        }
        for (g, (s, d)) in slope.iter().zip(dot).enumerate() {
            let change = -s + 0.5 * d * d;
            self.rise[g] += change;
            self.square[g] += change * change;
            self.slope[g] += s;
        }
        self.dots.push(dot.iter().map(|d| *d as f32).collect());
        Ok(())
    }

    /// The second-order change of the data term when `weights` (group, weight) are taken from
    /// every batch's sample together, `Σ_b (−Σ_G w_G s_bG + ½ (Σ_G w_G d_bG)²)`: with unit weights,
    /// the removal of the groups with the Gauss–Newton cross terms between them; a unit removed
    /// through any of its `k` roots enters as each root with weight `1/k`.
    #[must_use]
    pub fn joint(&self, weights: &[(usize, f64)]) -> f64 {
        let linear: f64 = weights.iter().map(|(g, w)| w * self.slope[*g]).sum();
        let quadratic: f64 = self
            .dots
            .iter()
            .map(|dots| {
                let d: f64 = weights.iter().map(|(g, w)| w * f64::from(dots[*g])).sum();
                d * d
            })
            .sum();
        -linear + 0.5 * quadratic
    }

    /// Per group, the standard error of the summed estimate from the spread of its batches'
    /// estimates, `√(B s²)` with `s²` their sample variance over the `B` batches (zero with fewer
    /// than two batches).
    #[must_use]
    pub fn spread(&self) -> Vec<f64> {
        let b = self.batches() as f64;
        self.rise
            .iter()
            .zip(&self.square)
            .map(|(sum, square)| if self.batches() < 2 { 0.0 } else { (b * (square - sum * sum / b).max(0.0) / (b - 1.0)).sqrt() })
            .collect()
    }
}

/// The Gaussian posterior over the library's parameters, every parameter independent of the
/// others, and which groups are active.
#[derive(Clone, Debug)]
pub struct Posterior {
    /// Per trainable operator (in `Explanation::trainable` order), the posterior means `μ`, each
    /// operator's array shared copy-on-write ([`Shared`]): a copy of the posterior (a removal
    /// trial) shares every operator it does not change.
    pub mean: Vec<Shared>,
    /// The logarithms `ln σ` of the posterior standard deviations, shared alike.
    pub log_sd: Vec<Shared>,
    /// Per group, whether it is in the explanation; a removed group's parameters are exactly zero.
    pub active: Vec<bool>,
    /// Per trainable operator, each entry's group (fixed by the explanation, shared by every copy).
    membership: Arc<Vec<Array2<u32>>>,
    /// Per trainable operator, the range of its groups' indices.
    spans: Vec<Range<usize>>,
    /// Per group, the reference variance `v⁰_G` against which its variance's scale is sent
    /// (`Explanation::reference`).
    initial: Vec<f64>,
}

/// One operator's posterior array, shared copy-on-write: a clone shares it, and a write through
/// `DerefMut` copies it first when it is shared, so two posteriors hold the same `Shared` exactly
/// when that operator's values are the same ([`Shared::same`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Shared(Arc<Array2<f64>>);

impl Shared {
    /// Whether `self` and `other` are one array (a copy neither has written since), so their
    /// values are equal without reading them.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// The array, copied only when it is shared.
    #[must_use]
    pub fn into_array(self) -> Array2<f64> {
        Arc::try_unwrap(self.0).unwrap_or_else(|shared| (*shared).clone())
    }
}

impl std::borrow::Borrow<Array2<f64>> for Shared {
    fn borrow(&self) -> &Array2<f64> {
        &self.0
    }
}

impl From<Array2<f64>> for Shared {
    fn from(values: Array2<f64>) -> Self {
        Self(Arc::new(values))
    }
}

impl std::ops::Deref for Shared {
    type Target = Array2<f64>;
    fn deref(&self) -> &Array2<f64> {
        &self.0
    }
}

impl std::ops::DerefMut for Shared {
    /// The array to write, copied first if another posterior shares it.
    fn deref_mut(&mut self) -> &mut Array2<f64> {
        Arc::make_mut(&mut self.0)
    }
}

/// Per group: its size, `Σ (μ² + σ²)` and `Σ ln σ²`.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    count: f64,
    second: f64,
    log_variance: f64,
}

/// A group's prior variance `v_G`, `KL(q_G ‖ p_G)` there in nats, and the bits of `v_G`'s scale
/// (module note).
#[derive(Clone, Copy, Debug)]
struct Prior {
    variance: f64,
    divergence: f64,
    bits: f64,
}

impl Moments {
    /// The group's prior at its reference variance `reference` (`gam_gpu::tensor::group_prior`,
    /// the device's arithmetic): zero for a group with no live entry.
    fn prior(&self, reference: f64) -> Prior {
        let (variance, divergence, bits) = gam_gpu::tensor::group_prior(self.count, self.second, self.log_variance, Some(reference));
        Prior { variance, divergence, bits }
    }
}

impl Posterior {
    /// The posterior at the explanation's starting point: means at the starting values, and each
    /// group's variance `v_G / N` with `v_G` the mean square of its starting values and `N` the
    /// training tokens (the precision of `N` unit-information observations).
    pub fn new(explanation: &Explanation, tokens: usize) -> Result<Self, String> {
        let program = &explanation.artifact.program;
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let mean: Vec<Array2<f64>> = explanation.trainable.iter().map(|op| program.operators[*op].matrix()).collect();
        let mut membership: Vec<Array2<u32>> = mean.iter().map(|m| Array2::from_elem(m.dim(), u32::MAX)).collect();
        let mut spans: Vec<Range<usize>> = vec![usize::MAX..0; mean.len()];
        if u32::try_from(explanation.groups.len()).is_err() {
            return Err("too many prior groups".into());
        }
        for (g, group) in explanation.groups.iter().enumerate() {
            for cell in &group.cells {
                let i = *position.get(&cell.operator).ok_or_else(|| format!("{}: operator {} is not trainable", group.name, cell.operator))?;
                for &row in &cell.rows {
                    for col in cell.cols.clone() {
                        let slot = membership[i].get_mut((row, col)).ok_or_else(|| format!("{}: entry ({row}, {col}) outside its operator", group.name))?;
                        if *slot != u32::MAX {
                            return Err(format!("{}: entry ({row}, {col}) of operator {} is in two groups", group.name, cell.operator));
                        }
                        *slot = g as u32;
                    }
                }
                spans[i] = spans[i].start.min(g)..spans[i].end.max(g + 1);
            }
        }
        if membership.iter().any(|m| m.iter().any(|g| *g == u32::MAX)) {
            return Err("a trainable entry is in no prior group".into());
        }
        if tokens == 0 {
            return Err("no training tokens".into());
        }
        // The starting variance of each group from its means alone.
        let mut squares = vec![(0.0, 0.0); explanation.groups.len()];
        for (mean, membership) in mean.iter().zip(&membership) {
            for (value, group) in mean.iter().zip(membership.iter()) {
                let entry = &mut squares[*group as usize];
                entry.0 += 1.0;
                entry.1 += value * value;
            }
        }
        if explanation.removed.iter().any(|g| *g >= explanation.groups.len()) {
            return Err("a removed group the explanation does not have".into());
        }
        if let Some(g) = (0..squares.len()).find(|g| !squares[*g].1.is_finite()) {
            return Err(format!("{}: a group starting at a nonfinite value", explanation.groups[g].name));
        }
        if explanation.reference.len() != explanation.groups.len() {
            return Err("one reference variance per prior group required".into());
        }
        // A group starting at zero is valid: its deviations start from its reference variance,
        // which `M` fixes for every group (`Explanation::reference`).
        for (g, group) in explanation.groups.iter().enumerate() {
            if explanation.removed.contains(&g) || squares[g].1 > 0.0 {
                continue;
            }
            let reference = explanation.reference[g];
            if !(reference > 0.0 && reference.is_finite()) {
                return Err(format!("{}: a group starting at zero without a reference variance", group.name));
            }
            squares[g].1 = squares[g].0 * reference;
        }
        let log_sd = membership
            .iter()
            .map(|membership| {
                membership.mapv(|group| {
                    let (count, sum) = squares[group as usize];
                    0.5 * (sum / count / tokens as f64).ln()
                })
            })
            .map(Shared::from)
            .collect();
        let initial = explanation.reference.clone();
        let mean = mean.into_iter().map(Shared::from).collect();
        let mut posterior = Self { mean, log_sd, active: vec![true; explanation.groups.len()], membership: Arc::new(membership), spans, initial };
        posterior.remove(&explanation.removed);
        Ok(posterior)
    }

    /// The posterior over parameter arrays `mean` whose entries' prior groups are `membership`
    /// (one id array per mean array, ids below the group count, every group holding an entry),
    /// every group in the explanation, at [`Posterior::new`]'s start: `reference` holds each
    /// group's reference variance `v⁰_G` (one per group, positive, as `Explanation::reference`
    /// gives [`Posterior::new`]), and a group's standard deviations start at `√(v_G / N)` for
    /// `tokens` training tokens `N`, `v_G` the mean square of its starting means, or `v⁰_G` where
    /// those are all zero (a Gaussian of mean zero and positive variance). A parameterization that
    /// is not a library explanation (the toy accounts of `mpd_toy_gate_2951`) is priced by this
    /// posterior's code length.
    pub fn from_parts(mean: Vec<Array2<f64>>, membership: Vec<Array2<u32>>, reference: Vec<f64>, tokens: usize) -> Result<Self, String> {
        if membership.len() != mean.len() || membership.iter().zip(&mean).any(|(ids, m)| ids.dim() != m.dim()) {
            return Err("one group array per mean array, of its shape, required".into());
        }
        if tokens == 0 {
            return Err("no training tokens".into());
        }
        if let Some(g) = reference.iter().position(|v| !(*v > 0.0 && v.is_finite())) {
            return Err(format!("group {g}: a reference variance that is not positive and finite"));
        }
        let groups = reference.len();
        let mut squares = vec![(0.0, 0.0); groups];
        let mut spans: Vec<Range<usize>> = vec![usize::MAX..0; mean.len()];
        for ((values, ids), span) in mean.iter().zip(&membership).zip(&mut spans) {
            for (value, id) in values.iter().zip(ids.iter()) {
                let g = *id as usize;
                let entry = squares.get_mut(g).ok_or_else(|| format!("group {g} of {groups}"))?;
                entry.0 += 1.0;
                entry.1 += value * value;
                *span = span.start.min(g)..span.end.max(g + 1);
            }
        }
        if let Some(g) = squares.iter().position(|(count, sum)| !(*count > 0.0 && sum.is_finite())) {
            return Err(format!("group {g} holds no entry or starts at a nonfinite value"));
        }
        // A group starting at zero starts from its reference variance (`Posterior::new`).
        for ((count, sum), v0) in squares.iter_mut().zip(&reference) {
            if *sum == 0.0 {
                *sum = *count * *v0;
            }
        }
        let log_sd = membership
            .iter()
            .map(|ids| {
                ids.mapv(|id| {
                    let (count, sum) = squares[id as usize];
                    0.5 * (sum / count / tokens as f64).ln()
                })
            })
            .map(Shared::from)
            .collect();
        let mean = mean.into_iter().map(Shared::from).collect();
        Ok(Self { mean, log_sd, active: vec![true; groups], membership: Arc::new(membership), spans, initial: reference })
    }

    fn moments(&self) -> Vec<Moments> {
        let partial: Vec<(Range<usize>, Vec<Moments>)> = (0..self.mean.len())
            .into_par_iter()
            .map(|i| {
                let span = self.spans[i].clone();
                let mut local = vec![Moments::default(); span.len()];
                for ((mu, s), group) in self.mean[i].iter().zip(self.log_sd[i].iter()).zip(self.membership[i].iter()) {
                    let g = *group as usize;
                    if !self.active[g] {
                        continue;
                    }
                    let m = &mut local[g - span.start];
                    m.count += 1.0;
                    m.second += mu * mu + (2.0 * s).exp();
                    m.log_variance += 2.0 * s;
                }
                (span, local)
            })
            .collect();
        let mut out = vec![Moments::default(); self.active.len()];
        for (span, local) in partial {
            for (g, m) in span.zip(local) {
                out[g].count += m.count;
                out[g].second += m.second;
                out[g].log_variance += m.log_variance;
            }
        }
        out
    }

    /// Per group, its prior variance, divergence and scale bits (module note; zero for a removed
    /// group).
    fn priors(&self) -> Vec<Prior> {
        self.moments().iter().zip(&self.initial).map(|(m, reference)| m.prior(*reference)).collect()
    }

    /// Per group, the reference variance `v⁰_G` its variance's scale is sent against
    /// (`Explanation::reference`).
    #[must_use]
    pub fn references(&self) -> &[f64] {
        &self.initial
    }

    /// Per group, its prior variance `v_G` (module note; zero for a removed group): the minimizer
    /// of `KL(q_G ‖ N(0, v I)) + ln 2 · L_scale(v)`, which the device's steps also use.
    pub fn variances(&self) -> Vec<f64> {
        self.priors().iter().map(|p| p.variance).collect()
    }

    /// Per group, its second-order data change on removal summed over the batches,
    /// `Σ_b δ_bG` (module note, [`Curvature`]), and zero for a removed group: a proposal score,
    /// not a bound on the finite change.
    pub fn removal_data(&self, curvature: &Curvature) -> Vec<f64> {
        (0..self.active.len()).map(|g| if self.active[g] { curvature.rise[g] } else { 0.0 }).collect()
    }

    /// Over the active groups' entries, the mean `ln σ` and the mean `|μ|` (absent when no entry
    /// is active: a removal round may remove every group), and over the active groups, the sum of
    /// the prior variances `v_G`: what moves
    /// `Σ_G KL(q_G ‖ p_G) = ½ Σ_G (S_G / v_G + |G| ln v_G − |G| − Σ_{j∈G} ln σ_j²)` between epochs.
    pub fn spread(&self) -> (Option<f64>, Option<f64>, f64) {
        let (mut entries, mut log_sd, mut magnitude) = (0.0, 0.0, 0.0);
        for i in 0..self.mean.len() {
            for ((mu, s), group) in self.mean[i].iter().zip(self.log_sd[i].iter()).zip(self.membership[i].iter()) {
                if self.active[*group as usize] {
                    entries += 1.0;
                    log_sd += s;
                    magnitude += mu.abs();
                }
            }
        }
        let variances = self.priors().iter().zip(&self.active).filter(|(_, a)| **a).map(|(p, _)| p.variance).sum();
        let average = |sum: f64| (entries > 0.0).then(|| sum / entries);
        (average(log_sd), average(magnitude), variances)
    }

    /// Per group, `KL(q_G ‖ p_G)` in nats at its prior variance (zero for a removed group).
    pub fn divergences(&self) -> Vec<f64> {
        self.priors().iter().zip(&self.active).map(|(p, active)| if *active { p.divergence } else { 0.0 }).collect()
    }

    /// Per group, `KL(q_G ‖ p_G)` plus its variance's precision `½ ln |G|` and its scale's
    /// `ln 2 · L_scale(v_G)`, in nats (zero for a removed group).
    pub fn costs(&self) -> Vec<f64> {
        self.moments()
            .iter()
            .zip(&self.active)
            .zip(&self.initial)
            .map(|((m, active), reference)| {
                if !*active {
                    return 0.0;
                }
                let prior = m.prior(*reference);
                prior.divergence + 0.5 * m.count.ln() + prior.bits * LN_2
            })
            .collect()
    }

    /// The nats of which groups are in the explanation (module note).
    pub fn subset_nats(&self) -> f64 {
        let active = self.active.iter().filter(|a| **a).count();
        crate::codec::subset_code_len_bits(self.active.len(), active).map_or(f64::INFINITY, |bits| bits as f64 * LN_2)
    }

    /// The posterior with each mean rounded to its precision: to the nearest multiple of
    /// `2^⌊log2 σ_j⌋`, the coarsest power of two not above its standard deviation (a removed entry
    /// stays zero).
    #[must_use]
    pub fn rounded(&self) -> Self {
        let mut out = self.clone();
        for (i, mean) in out.mean.iter_mut().enumerate() {
            let variances = self.marginal_variances(i);
            ndarray::Zip::from(&mut **mean).and(&variances).for_each(|mu, v| {
                if *v > 0.0 && v.is_finite() {
                    let step = (0.5 * v.ln() / LN_2).floor().exp2();
                    *mu = (*mu / step).round() * step;
                }
            });
        }
        out
    }

    /// `Σ_G KL(q_G ‖ p_G)`, the active groups' variances and which groups are active, in nats.
    pub fn description(&self) -> f64 {
        self.costs().iter().sum::<f64>() + self.subset_nats()
    }


    /// Operator `i`'s noise for standard normal draws `epsilon`: `σ ⊙ ε` (zero in removed
    /// groups, whose `σ` is zero).
    #[must_use]
    pub fn noise(&self, i: usize, epsilon: &Array2<f64>) -> Array2<f64> {
        let mut scaled = epsilon.clone();
        ndarray::Zip::from(&mut scaled).and(&*self.log_sd[i]).for_each(|e, s| *e *= s.exp());
        scaled
    }

    /// Operator `i`'s posterior variances.
    #[must_use]
    pub fn marginal_variances(&self, i: usize) -> Array2<f64> {
        self.log_sd[i].mapv(|s| (2.0 * s).exp())
    }

    /// The posterior means with every removed group zeroed.
    pub fn means(&self) -> Vec<Array2<f64>> {
        self.mean
            .iter()
            .zip(self.membership.iter())
            .map(|(mean, membership)| {
                let mut out = (**mean).clone();
                ndarray::Zip::from(&mut out).and(membership).for_each(|v, g| {
                    if !self.active[*g as usize] {
                        *v = 0.0;
                    }
                });
                out
            })
            .collect()
    }

    /// Remove `groups` from the explanation: their parameters become exactly zero.
    pub fn remove(&mut self, groups: &[usize]) {
        for g in groups {
            self.active[*g] = false;
        }
        let active = &self.active;
        // Only the operators holding a removed group are written: a copy of the posterior (a
        // removal trial) shares every other operator with its source.
        let touched: Vec<bool> = self.spans.iter().map(|span| groups.iter().any(|g| span.contains(g))).collect();
        self.mean.par_iter_mut().zip(self.log_sd.par_iter_mut()).zip(self.membership.par_iter()).zip(touched.par_iter()).filter(|(_, touched)| **touched).for_each(|(((mean, log_sd), membership), _)| {
            ndarray::Zip::from(&mut **mean).and(&mut **log_sd).and(membership).for_each(|m, s, g| {
                if !active[*g as usize] {
                    *m = 0.0;
                    *s = f64::NEG_INFINITY;
                }
            });
        });
    }
}

// ------------------------------------------------------------------------------------- the fit

/// The optimizer's settings and the run's resources (none of them is part of the objective). Read
/// through `SettingsRecord`, which also accepts and drops the keys of retired steps.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "SettingsRecord")]
pub struct Settings {
    /// Base sequences per step.
    pub batch_sequences: usize,
    /// The seed of the weight noise and of the experiments' draws.
    pub seed: u64,
    /// The device's numeric buffers for operators (each of the native and the explanation).
    pub numeric_bytes: usize,
    /// Rows of vocabulary logits formed at once.
    pub head_tile_rows: usize,
    /// When set, the fit ends once its epoch count (counted from `M`, a start's epochs included)
    /// reaches this, with no removal round: a comparison of arms at one budget of steps.
    #[serde(default)]
    pub epochs: Option<usize>,
    /// The families of each base's patched experiment, one drawn uniformly per base
    /// (`interchange::Family`): `read` (a read patch, `interchange::sample`), `swap`, `zero`,
    /// `scale`, `push` and `cut` (operations on sites every explanation shares with `M`,
    /// `interchange::Interchange::draw_ops`). Empty (the default) is `read` alone.
    #[serde(default)]
    pub families: Vec<interchange::Family>,
    /// The declared per-token execution budget `K`: the fit minimizes `F` subject to
    /// `E_q[k(x)] ≤ K`, `k(x)` the parts executed on a token (`library_complexity`), by dual
    /// ascent on its multiplier `λ`. Bits and parts per token have no derivable exchange rate (a
    /// receiver can compute the trace, so `k` is not a code length of anything it needs), so the
    /// budget is a declared constraint, not a term of `F`. None (the default) is no budget.
    #[serde(default)]
    pub budget: Option<f64>,
    /// With `budget_bits`, `k(x)` counts the description bits of the parts executed on a token in
    /// place of their rank (`library_vpd`'s gated components, [`GatedStage::bits`]): a part's bits
    /// are [`NUMBER_BITS`] per number of its groups (its slices' reads and writes and its direction
    /// row), its share of its stage's thresholds and widths, and its index among the stage's parts,
    /// `log₂ n`; `K` is in bits per token. The budget measures the parts' structure at a fixed
    /// precision, so `K` does not move with the posterior's sharpness; `F`'s description stays its
    /// bits-back code.
    #[serde(default)]
    pub budget_bits: bool,
}

/// [`Settings`] as configs and checkpoints hold them, unknown keys refused, and the keys of steps
/// the fit no longer has accepted and dropped: `rate` (IVON's fixed fraction of the Newton step,
/// replaced by the line step), `trust_rate`, `line_search`, `split_filter`,
/// `deterministic` (the 2^16 and 2^24 A/B arms), `decoder`, `half_factor` (now the step's),
/// `one_sample`, `rotated` (the rotated posterior of f80fd69565, which its A/B in 1a8361c2c8
/// retired), `cross_fit` and `full_antithetic` (4948bbc723's cross-fitted mean step and
/// 5759e6a350's scoring of the whole batch at both antithetic samples, which their paired A/Bs
/// retired), and `seed_bf16` and `train_bf16` (the bfloat16 arms of fitperf-seedab: every
/// evaluation with a gradient now runs in the reverse passes' arithmetic, `Scorer::reversed`), and
/// `measured_beta2` (6d3f4b137d's measured curvature gains, which their paired A/B retired), so
/// that configs and checkpoints written before still read.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SettingsRecord {
    batch_sequences: usize,
    #[serde(default)]
    beta1: Option<serde::de::IgnoredAny>,
    seed: u64,
    numeric_bytes: usize,
    head_tile_rows: usize,
    #[serde(default)]
    momentum_rule: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    epoch_ratio: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    preconditioned: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    epochs: Option<usize>,
    #[serde(default)]
    families: Vec<interchange::Family>,
    #[serde(default)]
    budget: Option<f64>,
    #[serde(default)]
    budget_bits: bool,
    #[serde(default)]
    measured_beta2: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    full_antithetic: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    seed_bf16: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    train_bf16: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    cross_fit: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    rate: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    trust_rate: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    line_search: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    split_filter: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    deterministic: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    decoder: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    half_factor: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    one_sample: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    rotated: Option<serde::de::IgnoredAny>,
}

impl From<SettingsRecord> for Settings {
    fn from(r: SettingsRecord) -> Self {
        let retired = [("rate", r.rate.is_some()), ("trust_rate", r.trust_rate.is_some()), ("line_search", r.line_search.is_some()), ("split_filter", r.split_filter.is_some()), ("deterministic", r.deterministic.is_some()), ("decoder", r.decoder.is_some()), ("preconditioned", r.preconditioned.is_some()), ("half_factor", r.half_factor.is_some()), ("one_sample", r.one_sample.is_some()), ("rotated", r.rotated.is_some()), ("beta1", r.beta1.is_some()), ("epoch_ratio", r.epoch_ratio.is_some()), ("momentum_rule", r.momentum_rule.is_some()), ("cross_fit", r.cross_fit.is_some()), ("full_antithetic", r.full_antithetic.is_some()), ("seed_bf16", r.seed_bf16.is_some()), ("train_bf16", r.train_bf16.is_some()), ("measured_beta2", r.measured_beta2.is_some())];
        for (key, present) in retired {
            if present {
                log::info!("library settings: the retired key `{key}` is ignored");
            }
        }
        Settings {
            batch_sequences: r.batch_sequences,
            seed: r.seed,
            numeric_bytes: r.numeric_bytes,
            head_tile_rows: r.head_tile_rows,
            epochs: r.epochs,
            families: r.families,
            budget: r.budget,
            budget_bits: r.budget_bits,
        }
    }
}

impl Settings {
    fn validate(&self) -> Result<(), String> {
        if self.batch_sequences == 0
            || self.numeric_bytes == 0
            || self.head_tile_rows == 0
            || self.budget.is_some_and(|k| !(k >= 0.0))
        {
            return Err("invalid library fit settings".into());
        }
        Ok(())
    }
}

/// One layer's survivors and activity on the held-out sequences at the posterior mean.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerCount {
    /// Heads with an active value coordinate; active rotary planes and value coordinates.
    pub heads: usize,
    pub planes: usize,
    pub values: usize,
    /// MLP functions whose groups are all active.
    pub functions: usize,
    /// Mean per token of the surviving functions whose activation `a = φ(z) y` is not zero, and of
    /// those whose activation exceeds its posterior noise scale `√((φ′(z) y)² s_z² + φ(z)² s_y²)`,
    /// `s_z² = Σ_j σ²_{g_j} x_j² + σ²_c` the posterior variance of the gate value `z = g·x + c` at
    /// the token's input `x` and `s_y²` likewise of a gated law's up value `y = b·x + e` (an
    /// ungated law has `y = 1`, `s_y = 0`).
    pub nonzero_per_token: f64,
    pub resolved_per_token: f64,
}

/// The held-out evaluation (module note). Divergences are `KL(M_e ‖ P_e)` per scored token, in
/// bits; a kind no experiment drew is none.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HeldOut {
    /// `F` per scored token: the held-out data term at one weight sample per batch plus the
    /// description over the training experiments' scored tokens. Every data term here is F's
    /// (`Scorer::terms`): the experiments' bits and the all-on experiment's, through the scored
    /// explanation's hard gates, over the experiments' scored tokens.
    pub objective_bits_per_token: f64,
    pub data_bits_per_token: f64,
    /// The held-out data term per scored token on the same experiments at the posterior mean, and
    /// at the mean rounded to its posterior precision ([`Posterior::rounded`]): the two
    /// candidates for the reported artifact, judged against the samples' `data_bits_per_token`.
    #[serde(default)]
    pub mean_bits_per_token: f64,
    #[serde(default)]
    pub rounded_bits_per_token: f64,
    /// `Σ_G KL(q_G ‖ p_G)`, and the active groups' variances (precision and scale) with the code of
    /// which groups are active, in bits.
    pub divergence_bits: f64,
    pub variance_bits: f64,
    /// The explanation's discrete choices (`Explanation::fixed_nats`), and the prior term's value at
    /// one weight sample with the parameters it sends (`PriorTerm`), in bits.
    #[serde(default)]
    pub choice_bits: f64,
    #[serde(default)]
    pub prior_bits: f64,
    /// At the posterior mean: the sample's clean and patched experiments per hybrid size `k = 1..=2L` (`k = 2L`
    /// is the explanation alone), and the patched ones by kind: one read variable, or a joint
    /// patch of several.
    pub clean: Vec<Option<f64>>,
    pub patched: Vec<Option<f64>>,
    pub read_patch: Option<f64>,
    pub joint_patch: Option<f64>,
    pub layers: Vec<LayerCount>,
    /// The held-out data term per scored token at the posterior mean with every gated component
    /// forced on (`library_vpd`; none for other explanations): zero exactly while the components'
    /// slices still sum to `M`'s maps. A native weight edit acts on `M`'s maps, so a drift from
    /// them is an error in every weight edit's prediction.
    #[serde(default)]
    pub all_on_bits_per_token: Option<f64>,
}

/// The largest share of the fit's training time its per-epoch held-out evaluations may take: the
/// full held-out set is scored after an epoch only while the evaluations stay within it.
pub const EVALUATION_SHARE: f64 = 0.1;

/// One epoch of the continuous fit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Epoch {
    pub epoch: usize,
    /// Mean over the epoch's steps of the data term over the training collection at each step's
    /// weight samples around the iterate (the optimizer's state, not the reported posterior), in
    /// bits: the log's record of the descent, at no one posterior and with no description. `F` of
    /// one posterior, its data term and description at the same `q`, is the snapshot's.
    pub data_bits: f64,
    /// The snapshot's estimate of `F` at the end-of-epoch posterior, the mean of its per-batch
    /// estimates on the whole training collection at fixed draws, and the snapshot's seconds
    /// (module note): what the stop and the best epoch are decided on.
    #[serde(default)]
    pub snapshot_bits: f64,
    #[serde(default)]
    pub snapshot_seconds: f64,
    /// The snapshot's mean improvement over the previous epoch's, paired by batch, and its
    /// standard error, in bits.
    pub improvement_bits: Option<f64>,
    pub standard_error_bits: Option<f64>,
    pub active_groups: usize,
    /// `KL(M_e ‖ P_e)` per scored token at the weight samples over the clean and the patched
    /// training experiments, in bits.
    pub clean_bits_per_token: f64,
    pub patched_bits_per_token: f64,
    /// The epoch's training seconds: its steps and its snapshot.
    pub seconds: f64,
    /// The held-out evaluation after the epoch's steps on the fixed subset, and on every held-out
    /// sequence when the schedule ran it (module note).
    pub held_out: HeldOut,
    pub held_out_full: Option<HeldOut>,
    /// With a budget (`Settings::budget`): `K`, the epoch's mean over its steps of the expected
    /// parts executed per token `Ê[k]` (at each step's weight sample around the iterate), and the
    /// multiplier `λ` (nats of `F` per part per token) at the epoch's end.
    #[serde(default)]
    pub budget: Option<f64>,
    #[serde(default)]
    pub expected_parts: Option<f64>,
    #[serde(default)]
    pub multiplier: Option<f64>,
}

/// One removal step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Removal {
    pub candidates: usize,
    pub removed: usize,
    /// The groups removed as without effect on any experiment, which `removed` counts.
    #[serde(default)]
    pub dead: usize,
    /// `F` before and after, in bits.
    pub before_bits: f64,
    pub after_bits: f64,
    /// Every evaluated proposal `(groups, F − F_before)` in bits, `F_before` the state it was
    /// proposed on, and each unit the search rejected alone `(its first group, its change of F)`:
    /// every other unit was removed (`library_removal`'s module note).
    pub evaluations: Vec<(usize, f64)>,
    #[serde(default)]
    pub singles: Vec<(usize, f64)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub settings: Settings,
    pub training_sequences: usize,
    /// The scored tokens of every training experiment, `N`.
    pub scored_tokens: usize,
    /// The realized count of each experiment family in the training collection
    /// (`interchange::census`).
    pub families: BTreeMap<String, usize>,
    pub groups: usize,
    pub parameters: usize,
    /// The held-out evaluation at the starting point, on the fixed subset, and of the finished fit,
    /// on every held-out sequence.
    pub start: HeldOut,
    pub end: HeldOut,
    /// The reported artifact: the posterior mean, or the mean rounded to its posterior precision,
    /// whichever has the lower held-out data term at the end ([`HeldOut::mean_bits_per_token`]).
    pub representative: Representative,
    /// The literals every evaluation ran with: f32 where the device held f32 (every operator
    /// rounded to f32 as it was uploaded), float64 on the host. The saved artifacts hold the same
    /// ([`Fit::artifact`]), so the reported fidelity is the saved artifact's.
    pub literals: Literals,
    pub epochs: Vec<Epoch>,
    pub removals: Vec<Removal>,
    pub active_groups: usize,
    /// `F` where the fit stopped, in bits.
    pub objective_bits: f64,
    pub seconds: f64,
}

pub struct Fit {
    pub posterior: Posterior,
    pub report: Report,
}

/// The literals a fit's evaluations ran with, and its saved artifacts hold (`Report::literals`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Literals {
    F32,
    F64,
}

impl Literals {
    /// The literals of a fit on `device`.
    #[must_use]
    pub fn of(device: &Device) -> Self {
        if device.float64() { Self::F64 } else { Self::F32 }
    }

    /// `artifact` with these literals: rounded to f32, or as it is.
    pub fn apply(self, artifact: Artifact) -> Result<Artifact, String> {
        match self {
            Self::F32 => artifact.f32_literals(),
            Self::F64 => Ok(artifact),
        }
    }
}

/// Which point of the posterior the reported artifact holds (`Report::representative`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Representative {
    Mean,
    Rounded,
}

impl Fit {
    /// The posterior whose means the reported artifact holds ([`posterior_mean`]).
    #[must_use]
    pub fn representative(&self) -> Posterior {
        match self.report.representative {
            Representative::Mean => self.posterior.clone(),
            Representative::Rounded => self.posterior.rounded(),
        }
    }

    /// The reported artifact of `explanation`: the representative's means with the literals the
    /// fit's evaluations ran with (`Report::literals`), so it is the artifact the held-out
    /// evaluation scored.
    pub fn artifact(&self, explanation: &Explanation) -> Result<Artifact, String> {
        self.report.literals.apply(posterior_mean(explanation, &self.representative())?)
    }
}

/// A family of complete sequences of equal length.
pub fn sequence_family(sequences: &[&[u32]]) -> Result<FamilyInputs, String> {
    let length = sequences.first().ok_or("no sequences")?.len();
    if length == 0 || sequences.iter().any(|s| s.len() != length) {
        return Err("sequences must be nonempty and of equal length".into());
    }
    let mut tokens = Vec::with_capacity(sequences.len() * length);
    let (mut sequence, mut position) = (Vec::new(), Vec::new());
    for (i, s) in sequences.iter().enumerate() {
        tokens.extend_from_slice(s);
        sequence.extend(std::iter::repeat_n(i as u32, length));
        position.extend(0..length as u32);
    }
    Ok(FamilyInputs { rows: tokens.len(), slots: vec![SlotValues::Tokens(tokens)], layout: Some(SequenceLayout { sequence, position }) })
}

/// A batch of experiments: its base sequences, each base's source (indices into the sequences),
/// and the seed every base's draws derive from ([`base_draws`]).
#[derive(Clone, Debug)]
struct Draw {
    bases: Vec<usize>,
    sources: Vec<usize>,
    seed: u64,
}

impl Draw {
    fn batch(&self, sequences: &[Vec<u32>]) -> Result<Batch, String> {
        let pick = |indices: &[usize]| indices.iter().map(|i| sequences[*i].clone()).collect();
        Batch::new(pick(&self.bases), pick(&self.sources))
    }

    /// Per base one clean and one patched experiment over `variables` in `blocks` blocks
    /// (`interchange::sample` of the one base), each base's from its own draws after its source
    /// ([`base_draws`]), so a base's experiments do not depend on the batch it is in.
    fn experiments(&self, sequences: &[Vec<u32>], variables: &[ReadVariable], blocks: usize) -> Result<Vec<Experiment>, String> {
        let length = self.bases.first().map_or(0, |b| sequences[*b].len());
        let mut out = Vec::with_capacity(2 * self.bases.len());
        for (k, &base) in self.bases.iter().enumerate() {
            let (_, mut rng) = base_draws(self.seed, sequences.len(), base);
            for mut e in interchange::sample(&mut rng, 1, variables, blocks, length)? {
                (e.base, e.source) = (k, k);
                out.push(e);
            }
        }
        Ok(out)
    }
}

/// The version of the experiment collection's draws a checkpoint records ([`Progress::collection`]):
/// 1 draws every base's source and experiments from its own stream ([`base_draws`]). Before it
/// (0, a checkpoint naming none), each batch drew its bases' experiments from a seed of its own,
/// which the batch size changed.
const COLLECTION: u32 = 1;

/// Base `base`'s draws among `count` sequences under `seed`: its source, uniform among the other
/// sequences, and the generator positioned after it, from which its experiments are drawn
/// ([`Draw::experiments`]). The generator is seeded by `seed` and the base's index alone (the
/// SplitMix64 hash of `seed` mixed with the index's, `gam_linalg::utils::splitmix64_hash`), so the
/// collection does not depend on how the bases are packed into batches.
fn base_draws(seed: u64, count: usize, base: usize) -> (usize, StdRng) {
    use gam_linalg::utils::splitmix64_hash;
    let mut rng = StdRng::seed_from_u64(splitmix64_hash(seed ^ splitmix64_hash(base as u64)));
    let j = rng.random_range(0..count - 1);
    (if j >= base { j + 1 } else { j }, rng)
}

/// One of `batches` batches drawn uniformly from a collection of `tokens` scored tokens: the factor
/// turning the batch's data term into an unbiased estimate of the collection's (`B`), and the one
/// turning its gradient and squared Gauss–Newton factor into the collection's per token (`B / N`),
/// whatever the batch's share of the tokens. An epoch's mean of the estimates is the collection's
/// data term exactly.
fn batch_weights(batches: usize, tokens: usize) -> (f64, f64) {
    let b = batches as f64;
    (b, b / tokens as f64)
}

/// The `count` sequences in order, in batches of `size` bases, each base's source drawn uniformly
/// among the other sequences from its own draws under `seed` ([`base_draws`]).
fn draws(count: usize, size: usize, seed: u64) -> Result<Vec<Draw>, String> {
    if count < 2 || size == 0 {
        return Err("a source needs another sequence, and batches must be nonempty".into());
    }
    let all: Vec<usize> = (0..count).collect();
    Ok(all
        .chunks(size)
        .map(|bases| Draw { bases: bases.to_vec(), sources: bases.iter().map(|&i| base_draws(seed, count, i).0).collect(), seed })
        .collect())
}

/// An affine node of an MLP in the explanation's flat program: the node, and its operator and
/// bias operator.
struct Map {
    node: usize,
    operator: usize,
    bias: Option<usize>,
}

impl Map {
    /// The node of `flat` applying operator `name` (and its bias `{name}_bias`, when it exists) to
    /// its first input, and that input. A shared block adds further terms (`library_sharing`).
    fn of(flat: &OperatorProgram, name: &str) -> Result<(Self, usize), String> {
        let (operator, bias) = (index_of(flat, name)?, operator_named(flat, &format!("{name}_bias")));
        let found = flat.nodes.iter().enumerate().find_map(|(n, node)| match node {
            Node::Affine { terms, bias: b } if *b == bias && terms.first().is_some_and(|t| t.1 == operator) => Some((n, terms[0].0)),
            _ => None,
        });
        let (node, input) = found.ok_or_else(|| format!("no node applies {name}"))?;
        Ok((Self { node, operator, bias }, input))
    }
}

/// An MLP's nodes in the explanation's flat program: the gate's input `x`, the gate `z = g·x + c`,
/// the activation `φ(z)` and its law, and for a gated law the up value `y = b·x + e`.
struct Mlp {
    input: usize,
    gate: Map,
    activation: usize,
    law: Law,
    up: Option<Map>,
}

impl Mlp {
    /// None for a layer whose MLP is gated components with no one gate map (`library_vpd`).
    fn of(flat: &OperatorProgram, l: usize) -> Result<Option<Self>, String> {
        let name = format!("library.l{l}.mlp");
        if operator_named(flat, &format!("{name}.gate")).is_none() && operator_named(flat, &format!("library.l{l}.attn.read")).is_some() {
            return Ok(None);
        }
        let (gate, input) = Map::of(flat, &format!("{name}.gate"))?;
        let found = flat.nodes.iter().enumerate().find_map(|(n, node)| match node {
            Node::Pointwise { input, laws } if *input == gate.node => laws.first().map(|law| (n, *law)),
            _ => None,
        });
        let (activation, law) = found.ok_or_else(|| format!("layer {l}: no activation node"))?;
        let up = match operator_named(flat, &format!("{name}.up")) {
            Some(_) => Some(Map::of(flat, &format!("{name}.up"))?.0),
            None => None,
        };
        Ok(Some(Self { input, gate, activation, law, up }))
    }
}

/// A stage of gated components (`library_vpd`), whose gate node is `z = input·Aᵀ + c` in the flat
/// program (`A` the identity over the components' read norms for an own gate, the gate rows `g`
/// for a direction gate): its input node, the threshold operator `c` and the direction operator
/// `g` (none for an own gate), and per component the read groups of the slices it runs when on
/// (its rank in rank-one equivalents is the number of those still active).
struct GatedStage {
    input: usize,
    threshold: usize,
    direction: Option<usize>,
    slices: Vec<Vec<usize>>,
    /// Per component its own prior groups (its slices' read and write groups and its direction
    /// row), and the stage's groups its components share (thresholds and widths)
    /// ([`GatedStage::account`]).
    groups: Vec<Vec<usize>>,
    shared: Vec<usize>,
    /// The stage's operators' prefix (`library.l{l}.attn`, `.o`, `.mlp.fc`, `.mlp.dn`).
    prefix: String,
    /// The node holding each component's gate pre-activation: the gate node, or in a shared
    /// stage the gates' pre-activations through the assignment (`library_vpd`).
    component_gate: usize,
    /// A shared stage's assignment operator.
    assign: Option<usize>,
    /// The gates' widths `w` (the Gated nodes' `Φ(z / w)`).
    width: usize,
}

impl GatedStage {
    /// Layer `l`'s stages of the explanation `explanation` compiled into `flat`: the attention's
    /// input (its components' q, k, v slices and the o slices they carry), the own-gated o slices,
    /// the MLP's input (c_fc slices and the down slices they carry) and the own-gated down slices.
    fn of(flat: &OperatorProgram, explanation: &Explanation, l: usize) -> Result<Vec<Self>, String> {
        let program = &explanation.artifact.program;
        let name = format!("library.l{l}");
        // Each single-row read group by its operator and row.
        let mut read_group: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (g, group) in explanation.groups.iter().enumerate() {
            if group.name.starts_with(&format!("{name}.")) && group.name.ends_with(".read") && group.cells.len() == 1 && group.cells[0].rows.len() == 1 {
                read_group.insert((group.cells[0].operator, group.cells[0].rows[0]), g);
            }
        }
        // Per row group of operator `op` its read groups.
        let blocks = |op: usize| -> Result<Vec<Vec<usize>>, String> {
            let mut row = 0;
            let mut out = Vec::new();
            for group in program.operators[op].rows.groups() {
                out.push((row..row + group.width).map(|r| read_group.get(&(op, r)).copied().ok_or_else(|| format!("{name}: no read group of row {r}"))).collect::<Result<Vec<_>, _>>()?);
                row += group.width;
            }
            Ok(out)
        };
        // The carried write-side slices: row `r` of the selection picks the component, carrier `r`.
        let carry = |slices: &mut [Vec<usize>], select: &str, carriers: &[Vec<usize>]| -> Result<usize, String> {
            let Some(op) = operator_named(program, select) else { return Ok(0) };
            let m = program.operators[op].matrix();
            for (r, row) in m.rows().into_iter().enumerate() {
                let c = row.iter().position(|v| *v != 0.0).ok_or("an empty gate selection")?;
                slices[c].extend(&carriers[r]);
            }
            Ok(m.nrows())
        };
        let gate_node = |threshold: usize| -> Result<(usize, usize), String> {
            flat.nodes
                .iter()
                .enumerate()
                .find_map(|(n, node)| match node {
                    Node::Affine { terms, bias: Some(b) } if *b == threshold && terms.len() == 1 => Some((n, terms[0].0)),
                    _ => None,
                })
                .ok_or_else(|| format!("{name}: no gate node of threshold {threshold}"))
        };
        let mut stages = Vec::new();
        let mut stage = |prefix: &str, slices: Vec<Vec<usize>>| -> Result<(), String> {
            let threshold = index_of(flat, &format!("{prefix}.threshold"))?;
            let (gate, input) = gate_node(threshold)?;
            let direction = operator_named(flat, &format!("{prefix}.direction"));
            if program.operators[threshold].rows.width() != slices.len() {
                return Err(format!("{prefix}: {} thresholds for {} components", program.operators[threshold].rows.width(), slices.len()));
            }
            let assign = operator_named(flat, &format!("{prefix}.assign"));
            let component_gate = match assign {
                // The first node through the assignment: the gates' pre-activations (its own
                // stage's, or with the followed stage's components after them) to each component.
                Some(a) => flat.nodes.iter().position(|n| matches!(n, Node::Transposed { operator, .. } if *operator == a)).ok_or_else(|| format!("{prefix}: no gate through the assignment"))?,
                None => gate,
            };
            let width = index_of(flat, &format!("{prefix}.width"))?;
            stages.push(Self { input, threshold, direction, slices, groups: Vec::new(), shared: Vec::new(), prefix: prefix.to_string(), component_gate, assign, width });
            Ok(())
        };
        // A stage's read holds the components carried in from other blocks first
        // (`library_vpd`'s blocks across blocks, [`GatedStage::carry_across`]): its own stage's
        // rows follow them.
        let carried_in = |consumer: &str| -> usize { program.operators.iter().filter(|o| o.name.starts_with(&format!("{consumer}.from."))).map(|o| o.rows.width()).sum() };
        let mut attention = blocks(index_of(program, &format!("{name}.attn.read"))?)?.split_off(carried_in(&format!("{name}.attn")));
        let o = match operator_named(program, &format!("{name}.o.read")) {
            Some(op) => blocks(op)?.split_off(carried_in(&format!("{name}.o"))),
            None => Vec::new(),
        };
        let carried = carry(&mut attention, &format!("{name}.o.select_gate"), &o)?;
        stage(&format!("{name}.attn"), attention)?;
        if o.len() > carried {
            stage(&format!("{name}.o"), o[carried..].to_vec())?;
        }
        // An MLP with no component (`{name}.mlp.zero`) has no stage.
        if let Some(fc_read) = operator_named(program, &format!("{name}.mlp.fc_read")) {
            let mut up = blocks(fc_read)?.split_off(carried_in(&format!("{name}.mlp.fc")));
            let down = blocks(index_of(program, &format!("{name}.mlp.dn_read"))?)?.split_off(carried_in(&format!("{name}.mlp.dn")));
            let carried = carry(&mut up, &format!("{name}.mlp.dn_select_gate"), &down)?;
            stage(&format!("{name}.mlp.fc"), up)?;
            if down.len() > carried {
                stage(&format!("{name}.mlp.dn"), down[carried..].to_vec())?;
            }
        }
        Ok(stages)
    }

    /// The slices carried across blocks (`library_vpd`): each selection `{consumer}.from.{home}`
    /// picks, for the consumer stage's leading read rows (by home in home order), the home stage's
    /// component whose gate runs them, and those rows' slices join that component's.
    fn carry_across(stages: &mut [Vec<Self>], flat: &OperatorProgram, explanation: &Explanation) -> Result<(), String> {
        let program = &explanation.artifact.program;
        let mut read_group: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for (g, group) in explanation.groups.iter().enumerate() {
            if group.name.ends_with(".read") && group.cells.len() == 1 && group.cells[0].rows.len() == 1 {
                read_group.insert((group.cells[0].operator, group.cells[0].rows[0]), g);
            }
        }
        // Per consumer its selections, by home `(layer, stage)`.
        let mut by_consumer: BTreeMap<String, Vec<((usize, usize), String, usize)>> = BTreeMap::new();
        for (op, operator) in program.operators.iter().enumerate() {
            let Some((consumer, home)) = operator.name.split_once(".from.") else { continue };
            let order = home
                .strip_prefix("library.l")
                .and_then(|rest| rest.split_once('.'))
                .and_then(|(layer, stage)| Some((layer.parse::<usize>().ok()?, if stage == "attn" { 0 } else { 2 })))
                .ok_or_else(|| format!("{}: no home stage", operator.name))?;
            by_consumer.entry(consumer.to_string()).or_default().push((order, home.to_string(), op));
        }
        for (consumer, mut selections) in by_consumer {
            selections.sort();
            let read = match consumer.rsplit_once('.') {
                Some((stem, "fc")) => format!("{stem}.fc_read"),
                Some((stem, "dn")) => format!("{stem}.dn_read"),
                _ => format!("{consumer}.read"),
            };
            let op = index_of(program, &read)?;
            let mut rows = Vec::new();
            let mut row = 0;
            for group in program.operators[op].rows.groups() {
                rows.push((row..row + group.width).map(|r| read_group.get(&(op, r)).copied().ok_or_else(|| format!("{read}: no read group of row {r}"))).collect::<Result<Vec<_>, _>>()?);
                row += group.width;
            }
            let mut offset = 0;
            for ((layer, _), home, select) in selections {
                let threshold = index_of(flat, &format!("{home}.threshold"))?;
                let stage = stages.get_mut(layer).and_then(|s| s.iter_mut().find(|s| s.threshold == threshold)).ok_or_else(|| format!("{consumer}: no home stage {home}"))?;
                let m = program.operators[select].matrix();
                for (r, picks) in m.rows().into_iter().enumerate() {
                    let c = picks.iter().position(|v| *v != 0.0).ok_or("an empty gate selection")?;
                    stage.slices[c].extend(&rows[offset + r]);
                }
                offset += m.nrows();
            }
        }
        Ok(())
    }

    /// Each component's own groups and the stage's shared ones (`GatedStage::groups`, `shared`), from
    /// its slices' read groups (with their writes) and the stage's named groups.
    fn account(&mut self, named: &BTreeMap<&str, usize>, explanation: &Explanation) {
        let prefix = self.prefix.clone();
        self.groups = self
            .slices
            .iter()
            .enumerate()
            .map(|(b, reads)| {
                let mut own = Vec::with_capacity(2 * reads.len() + 1);
                for &g in reads {
                    own.push(g);
                    let write = explanation.groups[g].name.strip_suffix(".read").map(|stem| format!("{stem}.write"));
                    own.extend(write.and_then(|w| named.get(w.as_str()).copied()));
                }
                own.extend(named.get(format!("{prefix}.g{b}").as_str()).copied());
                own
            })
            .collect();
        self.shared = ["thresholds", "widths"].iter().filter_map(|s| named.get(format!("{prefix}.{s}").as_str()).copied()).collect();
    }

    /// Per component its active slices, its rank in rank-one equivalents.
    fn ranks(&self, active: &[bool]) -> Vec<f64> {
        self.slices.iter().map(|groups| groups.iter().filter(|g| active[**g]).count() as f64).collect()
    }

    /// Per component its description in bits (`Settings::budget_bits`), from each group's bits
    /// `bits`: its own active groups', an equal share of the stage's shared groups', and its index
    /// among the stage's components, `log₂ n` (descent's rot arm, 489cf5e569).
    fn bits(&self, active: &[bool], bits: &[f64]) -> Vec<f64> {
        let n = self.groups.len() as f64;
        let shared = self.shared.iter().filter(|g| active[**g]).map(|g| bits[*g]).sum::<f64>() / n;
        self.groups.iter().map(|own| own.iter().filter(|g| active[**g]).map(|g| bits[*g]).sum::<f64>() + shared + n.log2()).collect()
    }
}

/// `M` and the explanation compiled for the experiments, with the explanation's MLP nodes.
struct Scorer {
    experiments: Interchange,
    mlps: Vec<Option<Mlp>>,
    /// Per layer its stages of gated components (`library_vpd`), none for a layer of functions.
    stages: Vec<Vec<GatedStage>>,
    /// The hard gates' stages (`library_vpd::Gate::Hard`: a width that is no parameter), each its
    /// width and threshold operators and its gates ([`Scorer::train_gates`]), and how the
    /// explanation's gates are scored ([`GateScoring`]).
    hard_gates: Vec<(usize, usize, usize)>,
    scoring: GateScoring,
    /// Every gated stage's threshold operator and its rows ([`Scorer::all_on_pass`]).
    thresholds: Vec<(usize, usize)>,
    /// Each trainable operator's position in `Explanation::trainable`.
    position: BTreeMap<usize, usize>,
    /// The blocks the explanation explains ([`scope`]), when not all of them ([`scoped`]): every
    /// experiment runs `P` exactly there.
    scope: Option<Vec<bool>>,
    /// The families of each base's patched experiment (`Settings::families`; empty is `read`).
    families: Vec<interchange::Family>,
    /// Each batch's drawn edits (by its seed and bases), drawn once: a draw runs `M` on the batch
    /// to find the parts firing there, and a fit asks for a batch's experiments many times.
    edits: std::cell::RefCell<BTreeMap<(u64, Vec<usize>), Vec<(usize, Patch, usize)>>>,
    /// The shared stages' assignments (`Explanation::shares`), the relaxation and logits version
    /// last written into `P`'s program, the logits' version, and the mirror-descent step size of
    /// their next step ([`Scorer::step_assignments`]; none before the first).
    assignments: Vec<Assignment>,
    written: Option<(Relaxation, u64)>,
    version: u64,
    assignment_step: Option<f64>,
    /// Per shared stage (its assignment operator) the budget count's derivative per token in its
    /// assignment at the step's relaxed assignment (`complexity_terms`).
    assignment_budget: Vec<(usize, Array2<f64>)>,
    /// The explanation's frames ([`Mixing`]), every mixed operator's values with each group's slices
    /// at their start, the mixed writes' operators' means as [`Scorer::pin_mixings`] last pinned
    /// them, and the step size of the frames' next step ([`Scorer::step_mixings`]; none before the
    /// first).
    mixings: Vec<Mixing>,
    mixed: BTreeMap<usize, Array2<f64>>,
    mixed_writes: BTreeMap<usize, Array2<f64>>,
    mix_step: Option<f64>,
    /// The fit's scored tokens `N` (the frames' Laplace code, [`Mixing::entry_nats`]).
    mix_tokens: f64,
    /// With a budget in bits (`Settings::budget_bits`), each group's bits, [`NUMBER_BITS`] per
    /// number, which the counts weigh their parts by.
    group_bits: Option<Vec<f64>>,
    /// Whether relaxed passes draw their gates (`DeviceProgram::set_sampled`), as every fit's do;
    /// off, they take the expected gate `Φ(z / w)` (a test of a derivative through it).
    sample_gates: bool,
}

impl Scorer {
    /// `f` on the experiments with `P` evaluating, its forward and its head's logits, in the reverse
    /// passes' arithmetic (`interchange::factor_arithmetic`: bfloat16 on CUDA in f32), `P`'s own
    /// restored after, error or not. Every evaluation of `P` with a gradient or a Gauss–Newton draw
    /// (a training step, the Laplace start's pass, the removal curvature) runs there; every scoring
    /// without one (the snapshot the stop and the best epoch are decided on, held-out scoring,
    /// removal comparisons) in `P`'s own, so an epoch's running estimates (`Epoch::data_bits`,
    /// `clean_bits_per_token`, `patched_bits_per_token`) are
    /// the steps' own. On vpd4l at N = 2^20 (fitperf-seedab, RTX 4090) bfloat16 steps took 66 ms
    /// against 96 and reached a lower snapshot `F` at every epoch of seed 1.
    fn reversed<R>(&mut self, f: impl FnOnce(&mut Interchange) -> R) -> R {
        let own = self.experiments.program_mut().arithmetic();
        let step = interchange::factor_arithmetic(self.experiments.program_mut().device(), own);
        if step != own {
            self.experiments.program_mut().set_arithmetic(step);
        }
        let out = f(&mut self.experiments);
        if step != own {
            self.experiments.program_mut().set_arithmetic(own);
        }
        out
    }

    /// The experiments of `explanation` against `M`, the split native program `native`: the read
    /// variables patched are `M`'s functions (`interchange::reads`), each model patching its own
    /// value of each, so every explanation of the model is asked the same questions.
    fn new(device: &Device, native: &OperatorProgram, explanation: &Explanation, settings: &Settings) -> Result<Self, String> {
        let sites: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let reads = explanation.reads.clone();
        // A shared stage's assignment operator takes gradients with the posterior's operators and
        // is written by the fit itself ([`Assignment`]).
        let assignments = explanation.shares.iter().map(|share| Assignment::of(&explanation.artifact.program, share)).collect::<Result<Vec<_>, String>>()?;
        // A mixed operator (`Explanation::mixes`) is written by the fit from its groups' frames, and
        // takes gradients for them; each group's extra slices start under the fit's seed.
        let mixings = explanation
            .mixes
            .iter()
            .enumerate()
            .map(|(k, mix)| Mixing::of(&explanation.artifact.program, mix, gam_linalg::utils::splitmix64_hash(settings.seed ^ gam_linalg::utils::splitmix64_hash(k as u64))))
            .collect::<Result<Vec<_>, String>>()?;
        let mixed: BTreeMap<usize, Array2<f64>> = explanation.mixes.iter().flat_map(|m| m.slices.iter().flatten()).map(|p| p.key().0).map(|op| (op, explanation.artifact.program.operators[op].matrix())).collect();
        if mixed.keys().any(|op| !explanation.trainable.contains(op)) {
            return Err("a mixed operator outside the posterior: its slices' means are pinned there".into());
        }
        let sides: std::collections::BTreeSet<(usize, bool)> = explanation.mixes.iter().flat_map(|m| m.slices.iter().flatten()).map(|p| (p.key().0, p.key().1)).collect();
        if sides.iter().any(|&(op, write)| write && sides.contains(&(op, false))) {
            return Err("an operator holding both mixed reads and mixed writes".into());
        }
        let differentiated: Vec<usize> = explanation.trainable.iter().copied().chain(assignments.iter().map(|a| a.operator)).collect();
        let group_bits = settings.budget_bits.then(|| explanation.groups.iter().map(|g| NUMBER_BITS * g.cells.iter().map(|c| (c.rows.len() * c.cols.len()) as f64).sum::<f64>()).collect());
        let mut experiments =
            Interchange::new(device, native, &sites, &explanation.artifact, &differentiated, reads, settings.numeric_bytes, settings.head_tile_rows)?;
        // Every scoring of the fit (its steps, held-out evaluations and removal comparisons) is of
        // one fixed collection of experiments, so `M`'s targets are kept on the host while the
        // process's memory budget admits them.
        experiments.keep_targets(gam_runtime::resource::MemoryGovernor::global());
        let (flat, _, _) = interchange::sites(&explanation.artifact, &sites)?;
        let mlps: Vec<Option<Mlp>> = (0..sites.len()).map(|l| Mlp::of(&flat, l)).collect::<Result<_, _>>()?;
        let mut stages: Vec<Vec<GatedStage>> = mlps.iter().enumerate().map(|(l, mlp)| if mlp.is_some() { Ok(Vec::new()) } else { GatedStage::of(&flat, explanation, l) }).collect::<Result<_, String>>()?;
        GatedStage::carry_across(&mut stages, &flat, explanation)?;
        let named: BTreeMap<&str, usize> = explanation.groups.iter().enumerate().map(|(i, g)| (g.name.as_str(), i)).collect();
        for stage in stages.iter_mut().flatten() {
            stage.account(&named, explanation);
        }
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        let scope = Some(scope(explanation)).filter(|blocks| !blocks.iter().all(|b| *b));
        let program = &explanation.artifact.program;
        let hard_gates = stages.iter().flatten().filter(|s| !position.contains_key(&s.width)).map(|s| (s.width, s.threshold, program.operators[s.width].rows.width())).collect();
        let scoring = explanation.scoring;
        let thresholds = stages.iter().flatten().map(|s| (s.threshold, program.operators[s.threshold].rows.width())).collect();
        let mut scorer = Self { experiments, mlps, stages, hard_gates, scoring, thresholds, position, scope, families: settings.families.clone(), edits: std::cell::RefCell::new(BTreeMap::new()), assignments, written: None, version: 0, assignment_step: None, assignment_budget: Vec::new(), mixings, mixed, mixed_writes: BTreeMap::new(), mix_step: None, mix_tokens: 0.0, group_bits, sample_gates: true };
        scorer.train_gates(None)?;
        Ok(scorer)
    }

    /// The gates' law for a pass ([`library_vpd::Gate`]): with `posterior`, a relaxed pass
    /// ([`Gates::Relaxed`]): every hard stage's width holds its thresholds' posterior deviations
    /// `σ_b` and its thresholds their means about `center` (the iterate, the average or the iterate
    /// before the pending move, whatever the pass samples around), so its gate is `Φ(z_b / σ_b)`,
    /// the hard gate's expectation with the threshold integrated exactly and the reads and a
    /// direction by the pass's weight sample; a learned width's gate is `Φ(z / w)`. With none, the
    /// law the explanation is scored by ([`Gates::Hard`]): every hard stage at `library_vpd::HARD`
    /// (the hard gate `H(z_b)`), and a learned width's gate by its scoring ([`GateScoring`]).
    fn train_gates(&mut self, posterior: Option<(&DevicePosterior, Center)>) -> Result<(), String> {
        if posterior.is_none() {
            self.experiments.explanation_mut().set_sampled(None);
        }
        if self.scoring == GateScoring::Hard {
            self.experiments.explanation_mut().set_hard(posterior.is_none());
        }
        if self.hard_gates.is_empty() {
            return Ok(());
        }
        for (width, threshold, gates) in self.hard_gates.clone() {
            let (deviations, mean) = match posterior {
                Some((posterior, center)) => {
                    let j = self.at(threshold)?;
                    let (average, log_sd) = posterior.values(j)?;
                    let mean = match center {
                        Center::Iterate => posterior.iterate(j)?,
                        Center::Average => average,
                        Center::Previous => posterior.previous_iterate(j)?,
                    };
                    (log_sd.mapv(f64::exp), Some(mean))
                }
                None => (Array2::from_elem((gates, 1), crate::library_vpd::HARD), None),
            };
            let program = self.experiments.explanation_mut();
            let device = program.device().clone();
            program.replace_dense_parameter(width, device.upload(deviations.view()).map_err(error)?)?;
            if let Some(mean) = mean {
                program.replace_dense_parameter(threshold, device.upload(mean.view()).map_err(error)?)?;
            }
        }
        self.experiments.explanation_mut().refresh_fused()
    }


    /// Writes `values` and `gates` into `P`'s program ([`Self::pass`]): the shared stages'
    /// assignments (softened in a relaxed pass, as before the pending move at its values; hardened
    /// in a hard one), the posterior's values, and in a relaxed pass the gates' training law about
    /// the center `values` samples around ([`Self::train_gates`]), until [`Self::rest`].
    fn set(&mut self, posterior: &DevicePosterior, values: Values, gates: Gates) -> Result<(), String> {
        self.write_assignments(match (gates, values) {
            (Gates::Hard, _) => Relaxation::Hard,
            (Gates::Relaxed, Values::Previous(_)) => Relaxation::Previous,
            (Gates::Relaxed, _) => Relaxation::Soft,
        })?;
        let program = self.experiments.explanation_mut();
        match values {
            Values::Mean => posterior.mean_into(program)?,
            Values::Rounded => posterior.rounded_into(program)?,
            Values::Sample(key) => posterior.sample_into(program, key)?,
            Values::Iterate(key) => posterior.iterate_into(program, key)?,
            Values::Previous(key) => posterior.previous_into(program, key)?,
        }
        if gates == Gates::Relaxed {
            let center = match values {
                Values::Iterate(_) => Center::Iterate,
                Values::Previous(_) => Center::Previous,
                Values::Mean | Values::Rounded | Values::Sample(_) => Center::Average,
            };
            self.train_gates(Some((posterior, center)))?;
            // Each gate drawn on with probability Φ(z / w) (`DeviceProgram::set_sampled`), under the
            // pass's key: the move's two sides draw alike.
            let key = match values {
                Values::Iterate(k) | Values::Previous(k) | Values::Sample(k) => k,
                Values::Mean | Values::Rounded => 0,
            };
            if self.sample_gates {
                self.experiments.explanation_mut().set_sampled(Some(gam_linalg::utils::splitmix64_hash(key ^ GATE_DRAWS)));
            }
        }
        Ok(())
    }

    /// Ends a pass through `gates` ([`Self::set`]): a relaxed pass's gates back to the scored law.
    fn rest(&mut self, gates: Gates) -> Result<(), String> {
        match gates {
            Gates::Relaxed => self.train_gates(None),
            Gates::Hard => Ok(()),
        }
    }

    /// One pass of `P` on `experiments` (on `batch`) against `M`'s `targets` for them, at `values`
    /// through `gates`, with every gated component on where `all_on` (each stage's thresholds raised
    /// out of reach, `z = read + 10³⁰`, until the next write of the values): with `gradient` the
    /// gradient of its bits' sum per operator and, with `probe`, a draw of its Gauss–Newton factor
    /// (`interchange::Factor`) under that key, left on the device. The values, the gates and the
    /// experiments are each the caller's: a weight sample does not soften the gates. A relaxed
    /// pass is a training pass and runs in the reverse passes' arithmetic ([`Self::reversed`]), as
    /// the step it is compared with does; a hard one in `P`'s own.
    fn pass(
        &mut self,
        posterior: &DevicePosterior,
        (batch, experiments, targets): (&Batch, &[Experiment], &Targets),
        (values, gates, all_on): (Values, Gates, bool),
        (gradient, probe): (bool, Option<u64>),
    ) -> Result<interchange::Evaluation, String> {
        self.set(posterior, values, gates)?;
        if all_on {
            for (t, rows) in self.thresholds.clone() {
                let program = self.experiments.explanation_mut();
                let on = program.device().upload(Array2::from_elem((rows, 1), 1e30).view()).map_err(error)?;
                program.replace_dense_parameter(t, on)?;
            }
            self.experiments.explanation_mut().refresh_fused()?;
        }
        let evaluation = match gates {
            Gates::Relaxed => self.reversed(|e| e.evaluate_probed(batch, experiments, Some(targets), gradient, probe)),
            Gates::Hard => self.experiments.evaluate_probed(batch, experiments, Some(targets), gradient, probe),
        };
        self.rest(gates)?;
        let evaluation = evaluation?;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err(if all_on { "nonfinite divergence with every gated component on" } else { "nonfinite explanation divergence" }.into());
        }
        Ok(evaluation)
    }

    /// The data terms of `F` on a batch ([`Terms`]): `P`'s bits on `experiments` against `targets`
    /// and the all-on experiment's on their clean sequences ([`Self::all_on_pass`]), both at
    /// `values` through `gates`; with `gradient` each one's gradient and, with `probe`, a draw of
    /// each one's factor (the all-on one's under the SplitMix64 output after `probe`). Every scoring
    /// of `F`'s data term runs here: a training step's halves and its move's test, the snapshot the
    /// best epoch is chosen by, the removal round's curvature and comparisons, and the held-out
    /// evaluation. The shared stages' assignments keep their gradients (a training step takes them,
    /// [`Self::take_assignment_gradients`]).
    fn terms(
        &mut self,
        posterior: &DevicePosterior,
        (batch, experiments, targets): (&Batch, &[Experiment], &Targets),
        (values, gates): (Values, Gates),
        (gradient, probe): (bool, Option<u64>),
    ) -> Result<Terms, String> {
        let mut evaluation = self.pass(posterior, (batch, experiments, targets), (values, gates, false), (gradient, probe))?;
        let factor = evaluation.factor.take();
        let all_on = self.all_on_pass(posterior, (batch, experiments), (values, gates), (gradient, probe.map(gam_linalg::utils::splitmix64_hash)))?;
        Ok(Terms { bits: evaluation.bits, gradient: evaluation.gradient, factor, all_on })
    }

    /// The all-on experiment of a batch (`library_vpd`'s gated components): its clean experiments
    /// of `P` alone run again with every gated component forced on, at `values` through `gates`,
    /// scored against `M`'s targets. With every part on the explanation is the sum of its parts, so
    /// this is `KL(M ‖ Σ parts)` on the batch: a term of F's data term whose value is the drift of
    /// the parts from `M`'s maps. Without it the parts drift freely (vpd4l grouped direction gates,
    /// learned widths: held-out KL with every part on 1.38 → 12.1 bits per token over two epochs at
    /// 2^22, decomp-vpd4l-h at 591bb575c2), and a native weight edit acts on the drifted parts.
    /// Returns the experiments, their bits and, with `gradient`, the gradient of their sum per
    /// trainable operator (thresholds' and assignments' left out: forced, they take none) and, with
    /// `probe`, a draw of its factor; none for an explanation without gated components or a batch
    /// without a clean experiment of `P` alone.
    fn all_on_pass(
        &mut self,
        posterior: &DevicePosterior,
        (batch, experiments): (&Batch, &[Experiment]),
        (values, gates): (Values, Gates),
        (gradient, probe): (bool, Option<u64>),
    ) -> Result<Option<AllOn>, String> {
        let clean: Vec<Experiment> = experiments.iter().filter(|e| e.patch.is_none() && e.explained.iter().all(|x| *x)).cloned().collect();
        if self.thresholds.is_empty() || clean.is_empty() {
            return Ok(None);
        }
        let targets = self.experiments.targets(batch, &clean)?;
        let mut evaluation = self.pass(posterior, (batch, &clean, &targets), (values, gates, true), (gradient, probe))?;
        let forced: Vec<usize> = self.thresholds.iter().map(|(t, _)| *t).chain(self.assignments.iter().map(|a| a.operator)).collect();
        for op in &forced {
            evaluation.gradient.remove(op);
            if let Some(f) = evaluation.factor.as_mut() {
                f.gradient.remove(op);
            }
        }
        Ok(Some(AllOn { experiments: clean, bits: evaluation.bits, gradient: evaluation.gradient, factor: evaluation.factor }))
    }

    /// Writes the shared stages' assignments into `P`'s program under `relaxation`, unless they
    /// are there already.
    fn write_assignments(&mut self, relaxation: Relaxation) -> Result<(), String> {
        if self.assignments.is_empty() || self.written == Some((relaxation, self.version)) {
            return Ok(());
        }
        for a in &self.assignments {
            let values = self.experiments.models().1.program.device().upload(a.values(relaxation).view()).map_err(error)?;
            self.experiments.explanation_mut().replace_dense_parameter(a.operator, values)?;
        }
        self.experiments.explanation_mut().refresh_fused()?;
        self.written = Some((relaxation, self.version));
        Ok(())
    }

    /// Takes the assignments' gradients out of a training pass's `gradients` (and its
    /// Gauss–Newton factor's), chained to their logits ([`Assignment::gather`]); an evaluation's
    /// are dropped.
    fn take_assignment_gradients(&mut self, gradients: &mut BTreeMap<usize, Tensor>, factor: Option<&mut interchange::Factor>, training: bool) -> Result<(), String> {
        let mut factor = factor;
        for a in &mut self.assignments {
            if let Some(f) = factor.as_deref_mut() {
                f.gradient.remove(&a.operator);
            }
            if let Some(g) = gradients.remove(&a.operator)
                && training
            {
                let g = self.experiments.models().1.program.device().download(&g).map_err(error)?;
                a.gather(&g);
            }
        }
        Ok(())
    }

    /// Removes the shared stages' assignment operators from a pass's gradient or factor `map`: the
    /// posterior's maps hold its own operators alone.
    fn strip(&self, map: &mut BTreeMap<usize, Tensor>) {
        for a in &self.assignments {
            map.remove(&a.operator);
        }
    }

    /// The budget's pull on the assignments (`Settings::budget`): `factor` times the count's
    /// derivative per token in each, chained to its logits and added to the step's gathered
    /// gradient, in the data term's units (as the budget's pull on the posterior's operators is).
    fn pull_assignments(&mut self, factor: f64) {
        let pulls = std::mem::take(&mut self.assignment_budget);
        for (op, g) in &pulls {
            if let Some(a) = self.assignments.iter_mut().find(|a| a.operator == *op) {
                a.gather(&g.mapv(|v| v * factor));
            }
        }
        self.assignment_budget = pulls;
    }

    /// Clears the step's gathered assignment gradients (before a step's passes).
    fn clear_assignment_gradients(&mut self) {
        for a in &mut self.assignments {
            a.gradient.iter_mut().flatten().for_each(|g| *g = 0.0);
        }
    }

    /// One mirror-descent step of every assignment's logits along the step's gathered gradient,
    /// `ℓ ← ℓ − α ḡ`, `ḡ` the gradient per token in nats (`scale`: the posterior step's, `B / N`
    /// times ln 2): the logits before it are kept until the posterior's move is tested
    /// ([`Scorer::accept_assignments`], [`Scorer::revert_assignments`]). The step size starts where
    /// the largest logit moves by one nat and doubles at each accepted move, halves at each
    /// rejected one (the test of the joint move on the next batch, `step_accepted`).
    fn step_assignments(&mut self, scale: f64) {
        if self.assignments.is_empty() {
            return;
        }
        let largest = self.assignments.iter().flat_map(|a| a.gradient.iter().flatten()).fold(0.0f64, |m, g| m.max((g * scale).abs()));
        if largest == 0.0 {
            return;
        }
        let step = *self.assignment_step.get_or_insert(1.0 / largest);
        for a in &mut self.assignments {
            a.previous = Some(a.logits.clone());
            for (l, g) in a.logits.iter_mut().flatten().zip(a.gradient.iter().flatten()) {
                *l -= step * scale * g;
            }
        }
        self.version += 1;
    }

    fn accept_assignments(&mut self) {
        let moved = self.assignments.iter_mut().fold(false, |moved, a| a.previous.take().is_some() || moved);
        if moved && let Some(step) = self.assignment_step.as_mut() {
            *step *= 2.0;
        }
    }

    fn revert_assignments(&mut self) {
        let mut reverted = false;
        for a in &mut self.assignments {
            if let Some(previous) = a.previous.take() {
                a.logits = previous;
                reverted = true;
            }
        }
        if reverted {
            self.version += 1;
            if let Some(step) = self.assignment_step.as_mut() {
                *step *= 0.5;
            }
        }
    }

    /// The assignments' logits, as a checkpoint keeps them.
    fn saved_assignments(&self) -> Option<Vec<Vec<Vec<f64>>>> {
        (!self.assignments.is_empty()).then(|| self.assignments.iter().map(|a| a.logits.clone()).collect())
    }

    fn restore_assignments(&mut self, saved: &[Vec<Vec<f64>>]) -> Result<(), String> {
        if saved.len() != self.assignments.len() || saved.iter().zip(&self.assignments).any(|(s, a)| s.len() != a.logits.len() || s.iter().zip(&a.logits).any(|(x, y)| x.len() != y.len())) {
            return Err("a checkpoint's assignments of other shared stages".into());
        }
        for (a, s) in self.assignments.iter_mut().zip(saved) {
            a.logits = s.clone();
            a.previous = None;
        }
        self.version += 1;
        Ok(())
    }

    /// Pins the posterior means of the mixed operators ([`Mix`]): their start values with each group's
    /// slices at its frame, the writes projected from the posterior's onto the frame's exact set
    /// ([`Mixing::write`]); it keeps the pinned writes, at which the passes until the next pin take
    /// their gradients in the frames.
    fn pin_mixings(&mut self, posterior: &mut DevicePosterior) -> Result<(), String> {
        if self.mixings.is_empty() {
            return Ok(());
        }
        let ops: std::collections::BTreeSet<usize> = self.mixings.iter().flat_map(|m| m.mix.slices[0].iter()).filter(|p| p.key().1).map(|p| p.key().0).collect();
        let writes = ops.into_iter().map(|op| Ok((op, posterior.iterate(self.at(op)?)?))).collect::<Result<BTreeMap<_, _>, String>>()?;
        let mut values = self.mixed.clone();
        for mixing in &self.mixings {
            mixing.write(&mut values, &writes)?;
        }
        let pinned = values.iter().map(|(&op, v)| Ok((self.at(op)?, v.clone()))).collect::<Result<Vec<_>, String>>()?;
        posterior.pin(&pinned)?;
        self.mixed_writes = writes.into_keys().map(|op| (op, values[&op].clone())).collect();
        Ok(())
    }

    /// Takes the mixed operators' gradients out of a training pass's `gradients`, chained to the
    /// frames ([`Mixing::gather`]): the reads' means move only with the frames, and the writes keep,
    /// for the posterior's step, their gradients' part within `N A = 0` ([`Mixing::free_writes`]).
    /// Their factor draws stay, so their deviations follow the data's curvature.
    fn take_mixing_gradients(&mut self, gradients: &mut BTreeMap<usize, Tensor>) -> Result<(), String> {
        if self.mixings.is_empty() {
            return Ok(());
        }
        if self.mixed_writes.is_empty() {
            return Err("a pass of mixed operators before their first pin".into());
        }
        let device = self.experiments.models().1.program.device().clone();
        let mut host = BTreeMap::new();
        for op in self.mixed.keys() {
            if let Some(g) = gradients.remove(op) {
                host.insert(*op, device.download(&g).map_err(error)?);
            }
        }
        for mixing in &mut self.mixings {
            mixing.gather(&host, &self.mixed_writes)?;
        }
        for op in self.mixed_writes.keys() {
            if let Some(g) = host.get(op) {
                let mut free = Array2::zeros(g.dim());
                for mixing in &self.mixings {
                    mixing.free_writes(*op, g, &mut free)?;
                }
                gradients.insert(*op, device.upload(free.view()).map_err(error)?);
            }
        }
        Ok(())
    }

    /// The frames' curvature per token: the step's factor draws in the mixed operators (`factor`,
    /// weighed by `weight` as the posterior weighs them) chained to each frame, squared, into a
    /// running average over one pass (`β₂`).
    fn mixing_curvature(&mut self, factor: &BTreeMap<usize, Tensor>, weight: f64, beta2: f64) -> Result<(), String> {
        if self.mixings.is_empty() {
            return Ok(());
        }
        let device = self.experiments.models().1.program.device().clone();
        let mut host = BTreeMap::new();
        for op in self.mixed.keys() {
            if let Some(u) = factor.get(op) {
                host.insert(*op, device.download(u).map_err(error)?);
            }
        }
        for m in &mut self.mixings {
            let u = m.chained(&host, &self.mixed_writes)?;
            m.curvature = &m.curvature * beta2 + &(&u * &u * (weight * (1.0 - beta2)));
        }
        Ok(())
    }

    /// The frames' description in nats: every entry's Laplace code ([`Mixing::entry_nats`]),
    /// a term of `F` as every parameter's description is.
    fn mixing_nats(&self) -> f64 {
        self.mixings.iter().flat_map(|m| m.entry_nats(self.mix_tokens)).map(|(_, nats)| nats).sum()
    }

    /// The change of the frames' description by the pending move of their entries, in nats: at the
    /// same deviations, the means' term `Σ ((a − a₀)² − (a_prev − a₀)²) / 2v`.
    fn mixing_change(&self) -> f64 {
        self.mixings
            .iter()
            .filter_map(|m| m.previous.as_ref().map(|p| (m, p)))
            .map(|(m, p)| {
                m.frame
                    .indexed_iter()
                    .zip(p.iter())
                    .map(|(((i, j), a), b)| {
                        let start = Mixing::start(i, j);
                        ((a - start).powi(2) - (b - start).powi(2)) / (2.0 * FRAME_PRIOR)
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    fn clear_mixing_gradients(&mut self) {
        for m in &mut self.mixings {
            m.gradient.fill(0.0);
        }
    }

    /// One gradient step of every frame along the step's gathered gradient, `A ← A − α ḡ` (`ḡ` per
    /// token in nats, `scale` the posterior step's); the frames before it are kept until the
    /// posterior's move is tested ([`Scorer::accept_mixings`], [`Scorer::revert_mixings`]). The step
    /// size starts where the largest entry of `A` moves by 1/8 and, as the assignments' step,
    /// doubles at each accepted move and halves at each rejected one.
    fn step_mixings(&mut self, scale: f64) {
        // The description's pull `(a − a₀) / v` (nats over the collection), in the gathered
        // gradient's units: `F`'s per token is `scale` times them.
        let tokens = self.mix_tokens;
        for m in &mut self.mixings {
            let pull = Array2::from_shape_fn(m.frame.dim(), |(i, j)| (m.frame[[i, j]] - Mixing::start(i, j)) / (FRAME_PRIOR * tokens * scale));
            m.gradient += &pull;
        }
        let largest = self.mixings.iter().flat_map(|m| m.gradient.iter()).fold(0.0f64, |m, g| m.max((g * scale).abs()));
        if largest == 0.0 {
            return;
        }
        let step = *self.mix_step.get_or_insert(0.125 / largest);
        for m in &mut self.mixings {
            m.previous = Some(m.frame.clone());
            m.frame.scaled_add(-step * scale, &m.gradient);
        }
    }

    fn accept_mixings(&mut self) {
        let moved = self.mixings.iter_mut().fold(false, |moved, m| m.previous.take().is_some() || moved);
        if moved && let Some(step) = self.mix_step.as_mut() {
            *step *= 2.0;
        }
    }

    fn revert_mixings(&mut self) {
        let mut reverted = false;
        for m in &mut self.mixings {
            if let Some(previous) = m.previous.take() {
                m.frame = previous;
                reverted = true;
            }
        }
        if reverted {
            if let Some(step) = self.mix_step.as_mut() {
                *step *= 0.5;
            }
        }
    }

    /// The frames, as a checkpoint and a report keep them.
    fn saved_mixings(&self) -> Option<Vec<Vec<Vec<f64>>>> {
        (!self.mixings.is_empty()).then(|| self.mixings.iter().map(|m| m.frame.rows().into_iter().map(|r| r.to_vec()).collect()).collect())
    }

    fn restore_mixings(&mut self, saved: &[Vec<Vec<f64>>]) -> Result<(), String> {
        if saved.len() != self.mixings.len() {
            return Err("a checkpoint's frames of other groups".into());
        }
        for (m, frame) in self.mixings.iter_mut().zip(saved) {
            m.frame = frame_of(frame, m.frame.dim())?;
            m.previous = None;
        }
        Ok(())
    }

    /// With push among the families, the scorer with the pushed directions set from the seed and
    /// each shared site's typical norm measured on `M`'s runs of the first batch of the training
    /// `sequences` (the unit of a push's size, the same for every explanation and fit). With weight
    /// edits among them, the table they draw from: [`WEIGHT_CANDIDATES`] native edits of `M`
    /// (`weight_edit::candidates`, the edits driver's strong families), each measured on
    /// `M` alone on the first [`WEIGHT_SCREEN`] training sequences (`KL(M_e ‖ M)` per token), and
    /// [`WEIGHT_EDITS`] of them kept stratified by that effect (`weight_edit::stratified`, bins at
    /// `weight_edit::EFFECT_EDGES`), so edits that move `M` much are as common as those that
    /// barely do. The table depends on `M`, the seed and the sequences alone: every explanation
    /// faces the same edits.
    fn prepared(mut self, native: &OperatorProgram, sequences: &[Vec<u32>], settings: &Settings) -> Result<Self, String> {
        if self.families.contains(&interchange::Family::Push) {
            let n = settings.batch_sequences.min(sequences.len());
            let batch = Batch::new(sequences[..n].to_vec(), sequences[..n].to_vec())?;
            self.experiments.set_directions(interchange::DIRECTIONS, settings.seed);
            self.experiments.measure_typical(&batch)?;
        }
        if self.families.contains(&interchange::Family::Weight) {
            let started = Instant::now();
            let k = WEIGHT_SCREEN.min(sequences.len());
            let screen = Batch::new(sequences[..k].to_vec(), sequences[..k].to_vec())?;
            let kept = self.experiments.draw_weight_edits(native, &screen, settings.seed, (WEIGHT_CANDIDATES, WEIGHT_EDITS))?;
            let mut census: BTreeMap<String, usize> = BTreeMap::new();
            for d in &kept {
                let bin = crate::weight_edit::EFFECT_EDGES.iter().filter(|e| d.effect.unwrap_or(0.0) >= **e).count();
                *census.entry(format!("{:?} bin {bin}", d.kind)).or_default() += 1;
            }
            log::info!("library weight edits: {} of {WEIGHT_CANDIDATES} candidates kept, {census:?}; {:.1} s", kept.len(), started.elapsed().as_secs_f64());
        }
        Ok(self)
    }

    fn layers(&self) -> usize {
        self.mlps.len()
    }

    fn at(&self, operator: usize) -> Result<usize, String> {
        self.position.get(&operator).copied().ok_or_else(|| format!("operator {operator} is not trainable"))
    }

    /// Before a batch: the device buffers kept from the last batch's tensors go back to the driver
    /// (`Device::release_recycled`), so the buffers kept are those of this batch's shapes.
    fn next_batch(&self) {
        self.experiments.models().1.program.device().release_recycled();
    }

    /// The batch's experiments from `draw`: the fixed collection's for that batch.
    /// With edit families (`Settings::families`), each base's patched experiment takes a family
    /// drawn uniformly from them, from draws of the base's own keyed by its index as its source's
    /// and experiments' are ([`base_draws`]), so packing changes neither: an edit replaces its
    /// read patch (or joins a base without one) under the same hybrid.
    fn experiments(&self, draw: &Draw, sequences: &[Vec<u32>]) -> Result<Vec<Experiment>, String> {
        use gam_linalg::utils::splitmix64_hash;
        let mut experiments = draw.experiments(sequences, self.experiments.variables(), 2 * self.layers())?;
        if self.families.iter().any(|f| *f != interchange::Family::Read) {
            let key = (draw.seed, draw.bases.clone());
            let cached = self.edits.borrow().get(&key).cloned();
            let drawn = match cached {
                Some(drawn) => drawn,
                None => {
                    let length = draw.bases.first().map_or(0, |b| sequences[*b].len());
                    let mut drawn: Vec<(usize, Patch, usize)> = Vec::new();
                    for (n, &base) in draw.bases.iter().enumerate() {
                        let mut rng = StdRng::seed_from_u64(splitmix64_hash(splitmix64_hash(draw.seed ^ 0xED17) ^ splitmix64_hash(base as u64)));
                        let family = self.families[rng.random_range(0..self.families.len())];
                        if family != interchange::Family::Read {
                            let (patch, position) = self.experiments.draw_ops(&mut rng, family, length)?;
                            drawn.push((n, patch, position));
                        }
                    }
                    self.edits.borrow_mut().insert(key, drawn.clone());
                    drawn
                }
            };
            for (n, patch, position) in drawn {
                let clean = experiments.iter().position(|e| e.base == n && e.patch.is_none()).ok_or("a base without its clean experiment")?;
                let edit = Experiment { base: n, source: n, explained: experiments[clean].explained.clone(), patch: Some(patch), position };
                match experiments.iter().position(|e| e.base == n && e.patch.is_some()) {
                    Some(at) => experiments[at] = edit,
                    None => experiments.insert(clean + 1, edit),
                }
            }
        }
        if let Some(scope) = &self.scope {
            for e in &mut experiments {
                e.explained.clone_from(scope);
            }
        }
        Ok(experiments)
    }

}

/// The key of the Gauss–Newton factor's Fisher probe (`interchange::evaluate_probed`) at the
/// weight sample of `seed`: the SplitMix64 output after it (`gam_linalg::utils::splitmix64_hash`),
/// so the probe's signs and the sample's noise, both Philox draws, are under different keys.
fn probe_key(seed: u64) -> u64 {
    gam_linalg::utils::splitmix64_hash(seed)
}

/// The noise seed of batch `batch` in stream `epoch`: streams `1, 2, …` are the training steps' of
/// epochs `0, 1, …` (`training_key`), stream 0 the removal comparisons', the epochs' snapshots'
/// (`snapshot_estimates`) and the held-out evaluation's.
fn noise_seed(seed: u64, epoch: usize, batch: usize) -> u64 {
    seed.wrapping_add((epoch as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)).wrapping_add((batch as u64).wrapping_mul(0x8CB9_2BA7_2F3D_8DD7))
}

/// The weight noise of training batch `batch` in epoch `epoch`, drawn afresh every epoch (stream
/// `1 + epoch`): the steps estimate the gradient of `F`, the expectation over the posterior, and
/// not of one fixed Monte Carlo sample of it, whose minimizer fits that sample's noise. Common
/// random numbers are kept where a paired comparison needs them: the epochs' snapshots, the
/// removal comparisons and the held-out evaluation share stream 0 (`noise_seed` with epoch 0),
/// and two fits compared arm against arm draw the same keys at the same epoch and batch. A resumed
/// fit draws the keys its epoch and batch give, as the uninterrupted fit did.
fn training_key(seed: u64, epoch: usize, batch: usize) -> u64 {
    noise_seed(seed, 1 + epoch, batch)
}

/// A training step's scoring with antithetic weight noise inside the step: the batch's bases split
/// in two, the first half's experiments scored at the weight sample of `key` and the second half's
/// at its negation (`gam_gpu::tensor::ANTITHETIC`: `ε` and `−ε`), with the gradients summed over
/// both. Each half's sample is one of the posterior, so the step's estimate of `F` and of its
/// gradient keep their expectations. The gradient's term linear in the noise is `(H_A − H_B) σ ε`
/// over the halves `A` and `B` (each half's Hessian on its own experiments) rather than
/// `(H_A + H_B) σ ε`: it cancels only as far as the two halves' curvatures agree, and the momentum
/// averages what remains. The products are one scoring's of the whole batch. A batch of one base is
/// scored at the sample of `key` alone.
///
/// Scoring the whole batch at both samples and averaging cancels `H σ ε` exactly, and it lowers
/// `F` more per epoch: on vpd4l's one-block fit (block 3, 1024 sequences, L40, MATS audit-new-*)
/// held-out F after 3 epochs was 2.555 and 2.783 bits per token (seeds 1 and 2) against the split's
/// 2.980 and 3.055. It costs 1.5× the time per epoch, and at equal time it lost (audit-z-*): 2.578
/// and 2.859 after 3 epochs in 169 s against the split's 2.399 and 2.596 after 5 in 175 s.
///
/// The Gauss–Newton factor is one half's alone, its curvature estimate per token of the tokens it
/// sums; which half supplies it is drawn from the batch's key, each with probability ½
/// ([`factor_half`]). The batches are fixed runs of the training sequences, so a fixed half would
/// leave the same sequences out of the curvature at every step; with the half drawn, every
/// training sequence contributes with probability ½ and the curvature average stays unbiased for
/// the whole collection's. The key, and so the half, is drawn afresh every epoch with the weight
/// noise (`training_key`). Each entry of a draw `u` is a sum of independent zero-mean terms,
/// one per token, close to normal, so `u_i²` has a relative standard deviation near √2 however
/// many tokens it sums: half the tokens estimate the curvature about as precisely, for half the
/// factor pass. Its A/B (fitperf-halffactor-ab2: vpd4l, N = 2^20, RTX 4090, 3 epochs, seeds 1-2)
/// measured F after epoch 2 at 25.45e6 and 24.09e6 bits against 25.73e6 and 24.81e6 with both
/// halves' factors, and 7% less wall time per step.
/// The parts a training step scores a batch's experiments in ([`antithetic_step`]): the experiments of the first and of the second half of the
/// batch's bases (the antithetic pair), or all of them together when a half has none. `M`'s
/// targets are kept per part (`Interchange::targets`), so every scoring of a training batch asks
/// for them by these parts ([`part_targets`]).
fn step_parts(batch: &Batch, experiments: Vec<Experiment>) -> Vec<Vec<Experiment>> {
    let half = batch.base.len() / 2;
    let (first, second): (Vec<Experiment>, Vec<Experiment>) = experiments.into_iter().partition(|e| e.base < half);
    if first.is_empty() || second.is_empty() {
        vec![first.into_iter().chain(second).collect()]
    } else {
        vec![first, second]
    }
}

/// A training batch's experiments in the order of its [`step_parts`], and `M`'s targets for them,
/// asked for part by part.
fn part_targets(scorer: &Scorer, batch: &Batch, experiments: Vec<Experiment>) -> Result<(Vec<Experiment>, Targets), String> {
    let parts = step_parts(batch, experiments);
    let targets = Targets::joined(parts.iter().map(|part| scorer.experiments.targets(batch, part)).collect::<Result<Vec<_>, String>>()?);
    Ok((parts.into_iter().flatten().collect(), targets))
}

/// Per gate row of the budget's terms, F's push against the count per part and the row's weight:
/// `(max(0, −⟨∂F, ∂Ê⟩) / |∂Ê|², |∂Ê|²)` along the row's parameters (a threshold's entry, a
/// direction's row), `∂F` the step's gradient `gradients` in bits times `scale ln 2`; ascending.
fn balance_rates(terms: &[(usize, Tensor, Tensor)], gradients: &BTreeMap<usize, Tensor>, (device, explanation): (&Device, &Explanation), scale: f64) -> Result<Vec<(f64, f64)>, String> {
    let mut ratios = Vec::new();
    for (i, mean, _) in terms {
        let Some(g) = gradients.get(&explanation.trainable[*i]) else { continue };
        let (data, count) = (device.download(g).map_err(error)?, device.download(mean).map_err(error)?);
        if data.dim() != count.dim() {
            continue;
        }
        for (f, k) in data.rows().into_iter().zip(count.rows()) {
            let (along, square) = (f.dot(&k) * scale * LN_2, k.dot(&k));
            if square > 0.0 && along.is_finite() {
                ratios.push(((-along).max(0.0) / square, square));
            }
        }
    }
    ratios.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(ratios)
}

/// Each component's unit of a gated stage's threshold shift ([`project`]): its learned width, or
/// for a hard gate its threshold's posterior deviation.
fn gate_units(scorer: &Scorer, device_posterior: &DevicePosterior, stage: &GatedStage) -> Result<Vec<f64>, String> {
    Ok(match scorer.at(stage.width) {
        Ok(w) => device_posterior.iterate(w)?.column(0).iter().map(|v| v.abs()).collect(),
        Err(_) => device_posterior.values(scorer.at(stage.threshold)?)?.1.column(0).iter().map(|s| s.exp()).collect(),
    })
}

/// The budget as a projection (`Settings::budget`): where the count at the step's sample on the
/// batch's forward exceeds `limit`, every gated stage's thresholds (`library_vpd`) rise by one
/// shift `Δ` times each gate's unit ([`gate_units`]) and every ReLU function's gate bias falls by
/// `Δ` times its posterior deviation, `Δ` bisected
/// until the count is at `limit`, the posterior's means pinned there ([`DevicePosterior::pin`]):
/// projected descent onto the constraint set `E[k] ≤ K`, run after every step, so the count never
/// exceeds `K` and λ only steers which gates trade. A count below `K` is left as it is: the budget
/// is an inequality, and a fit whose free count is below it is not pushed up to it. Returns `Δ`
/// and the hard count before and after it.
fn project(scorer: &mut Scorer, device_posterior: &mut DevicePosterior, explanation: &Explanation, active: &[bool], batch: &Batch, key: u64, limit: f64) -> Result<(f64, f64, f64), String> {
    let (family, trace) = count_trace(scorer, device_posterior, batch, (key, false))?;
    let mut count_at = |shift: f64| -> Result<f64, String> { Ok(count_terms(scorer, device_posterior, explanation, active, (&family, &trace), (false, shift, false))?.0) };
    // A bracket by doubling toward the budget, then bisection.
    let at_zero = count_at(0.0)?;
    if at_zero <= limit {
        return Ok((0.0, at_zero, at_zero));
    }
    let direction = 1.0;
    let (mut near, mut far) = (0.0, direction);
    while (count_at(far)? > limit) == (direction > 0.0) {
        (near, far) = (far, 2.0 * far);
        if far.abs() > 1e6 {
            return Err(format!("library budget: no threshold shift brings the count to K {limit} (at no shift {at_zero})"));
        }
    }
    for _ in 0..40 {
        let middle = 0.5 * (near + far);
        if (count_at(middle)? > limit) == (direction > 0.0) {
            near = middle;
        } else {
            far = middle;
        }
    }
    let shift = far;
    let count = count_at(shift)?;
    let mut pinned = Vec::new();
    for stage in scorer.stages.iter().flatten() {
        let i = scorer.at(stage.threshold)?;
        let start = device_posterior.iterate(i)?;
        let unit = gate_units(scorer, device_posterior, stage)?;
        pinned.push((i, Array2::from_shape_fn(start.dim(), |(b, c)| start[[b, c]] - shift * unit[b])));
    }
    for b in scorer.mlps.iter().flatten().filter(|m| m.law == Law::Relu && m.up.is_none()).filter_map(|m| m.gate.bias) {
        let j = scorer.at(b)?;
        let (start, log_sd) = (device_posterior.iterate(j)?, device_posterior.values(j)?.1);
        pinned.push((j, &start - &log_sd.mapv(|s| shift * s.exp())));
    }
    device_posterior.pin(&pinned)?;
    Ok((shift, at_zero, count))
}

/// The batch's expected parts executed per token under the posterior (`library_complexity`) and,
/// per gated MLP (a ReLU law without an up map), its gate's and its threshold's derivatives of
/// that mean `(trainable index, ∂Ê/∂μ, ∂Ê/∂σ²)`. The layers' inputs come from one forward of `P`
/// on the batch's bases at the step's weight sample `key` around the iterate; the gate's own
/// entries are integrated exactly at the iterate's means and deviations. Every surviving function
/// of another law and every attention head with a surviving value executes on every token and
/// counts one; a gated layer's fixed positions (where its block's output is not its functions',
/// `library_transcoder`'s first token) count nothing of it.
fn complexity_terms(scorer: &mut Scorer, device_posterior: &DevicePosterior, explanation: &Explanation, active: &[bool], batch: &Batch, (key, previous): (u64, bool)) -> Result<(f64, Vec<(usize, Tensor, Tensor)>), String> {
    let (family, trace) = count_trace(scorer, device_posterior, batch, (key, previous))?;
    count_terms(scorer, device_posterior, explanation, active, (&family, &trace), (previous, 0.0, true))
}

/// The forward the budget's count reads its gates' inputs from: a training pass at the iterate's
/// sample of `key`, or with `previous` around the iterate before the pending move (its test,
/// `step_accepted`), on the batch's bases.
fn count_trace(scorer: &mut Scorer, device_posterior: &DevicePosterior, batch: &Batch, (key, previous): (u64, bool)) -> Result<(FamilyInputs, DeviceTrace), String> {
    let values = if previous { Values::Previous(key) } else { Values::Iterate(key) };
    scorer.set(device_posterior, values, Gates::Relaxed)?;
    let family = sequence_family(&batch.base.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
    // The count's derivatives are a step's, so its forward and products run in the reverse passes'
    // arithmetic, as every evaluation of P with a gradient does (`Scorer::reversed`, b290535ec5).
    let trace = scorer.reversed(|experiments| experiments.models().1.program.forward(&family));
    scorer.rest(Gates::Relaxed)?;
    Ok((family, trace?))
}

/// [`complexity_terms`] on the forward `trace` of `family` ([`count_trace`]), every gated stage's
/// thresholds lowered by `shift` times each gate's unit ([`project`]); every gate's own means
/// (thresholds, widths, directions) on the side of the move `previous` names.
fn count_terms(
    scorer: &mut Scorer,
    device_posterior: &DevicePosterior,
    explanation: &Explanation,
    active: &[bool],
    (family, trace): (&FamilyInputs, &DeviceTrace),
    (previous, shift, relaxed): (bool, f64, bool),
) -> Result<(f64, Vec<(usize, Tensor, Tensor)>), String> {
    let center = |i: usize| if previous { device_posterior.previous_iterate(i) } else { device_posterior.iterate(i) };
    let (_, p) = scorer.experiments.models();
    let (program, d) = (p.program, p.program.device());
    let arithmetic = interchange::factor_arithmetic(d, program.arithmetic());
    let positions = &family.layout.as_ref().ok_or("a sequence layout")?.position;
    let rows = family.rows;
    let alive = |g: &usize| active[*g];
    let mut count = 0.0;
    let mut terms = Vec::new();
    // Per shared stage the count's derivative per token in its assignment.
    let mut assignment_terms = Vec::new();
    for (l, (layer, mlp)) in explanation.layers.iter().zip(&scorer.mlps).enumerate() {
        count += (layer.heads.iter().filter(|(_, values)| values.iter().any(alive)).count() * rows) as f64;
        let Some(mlp) = mlp else {
            // Gated components (`library_vpd`): each counts its rank where its gate is on, on the
            // device (`gated_expected`).
            // The previous stage's per-component pre-activations and variances, which a stage whose
            // components follow its components reads.
            let mut followed: Option<(Tensor, Tensor, Vec<f64>)> = None;
            for stage in &scorer.stages[l] {
                let rank = match &scorer.group_bits {
                    Some(bits) => stage.bits(active, bits),
                    None => stage.ranks(active),
                };
                let j = scorer.at(stage.threshold)?;
                let unit = gate_units(scorer, device_posterior, stage)?;
                let bias = (center(j)?.column(0).iter().zip(&unit).map(|(m, u)| m - shift * u).collect::<Vec<f64>>(), device_posterior.values(j)?.1.column(0).mapv(|s| (2.0 * s).exp()).to_vec());
                // A direction gate's means and variances stay on the device (the posterior's iterate
                // and `e^{2s}`); its count's derivatives come back as device tensors.
                let direction = match stage.direction {
                    Some(g) => {
                        let i = scorer.at(g)?;
                        let (mean, log_sd) = if previous { device_posterior.previous_and_log_sd(i)? } else { device_posterior.iterate_and_log_sd(i)? };
                        Some((i, mean, d.gate_function(gam_gpu::tensor::GateFunction::Variance, log_sd, None).map_err(error)?))
                    }
                    None => None,
                };
                let per = 1.0 / rows as f64;
                let gate = direction.as_ref().map(|(_, mean, variance)| (*mean, variance));
                // A shared stage counts each component through its assignment, as the pass wrote it.
                let relaxation = if previous { Relaxation::Previous } else { Relaxation::Soft };
                let assign = match stage.assign {
                    Some(op) => Some(scorer.assignments.iter().find(|a| a.operator == op).ok_or("a shared stage without its assignment")?.values(relaxation)),
                    None => None,
                };
                // The gates' widths enter the count (what the pass executes) but take no pull from
                // it: nothing keeps a width positive.
                // A hard gate has no width (`library_vpd::Gate::Hard`): its count is `Φ(m / σ)`.
                // The count the budget holds (`relaxed` false) is the hard gate's under the posterior,
                // E_q[H(z)] = Φ(μ_z / sd_q(z)): what the explanation executes when it is evaluated
                // (hard), with the posterior's own noise and no learned width. The count the step
                // descends (`relaxed`) reads the learned widths too, `Φ(μ_z / √(w² + σ²))`: the
                // hard count's derivative vanishes on every gate a few deviations from its
                // threshold, so a gate the data leaves free to close but firmly on took no pull
                // (toys' gated copy under a budget in bits: no gate closed while λ reached 4e5).
                let width = match scorer.at(stage.width) {
                    Ok(w) if relaxed => center(w)?.column(0).iter().map(|v| v.abs()).collect(),
                    _ => vec![0.0; explanation.artifact.program.operators[stage.width].rows.width()],
                };
                let extra = followed.as_ref().map(|(m, s2, w)| (m, s2, w.as_slice()));
                let (expected, gate_terms, assigned, components) = gated_expected(d, arithmetic, trace.value(stage.input)?, gate, (&bias.0, &bias.1, &width), &rank, (assign.as_ref(), extra))?;
                followed = Some(components);
                if let (Some(op), Some(g)) = (stage.assign, assigned) {
                    assignment_terms.push((op, g * per));
                }
                if let (Some((i, _, _)), Some((mean, variance))) = (&direction, gate_terms) {
                    terms.push((*i, d.scaled(per, &mean).map_err(error)?, d.scaled(per, &variance).map_err(error)?));
                }
                count += expected.count;
                terms.push((j, d.upload((expected.bias_mean.insert_axis(ndarray::Axis(1)) * per).view()).map_err(error)?, d.upload((expected.bias_variance.insert_axis(ndarray::Axis(1)) * per).view()).map_err(error)?));
            }
            continue;
        };
        if layer.functions.is_empty() {
            continue;
        }
        if scorer.group_bits.is_some() {
            return Err("a budget in bits counts gated components (library_vpd), not functions".into());
        }
        let surviving: Vec<bool> = layer.functions.iter().map(|groups| groups.iter().all(alive)).collect();
        if mlp.law != Law::Relu || mlp.up.is_some() {
            count += (surviving.iter().filter(|s| **s).count() * rows) as f64;
            continue;
        }
        let rule = explanation.artifact.program.rules.iter().find(|r| r.name == format!("library.l{l}.mlp"));
        let fixed: Vec<u32> = rule
            .and_then(|r| r.nodes.iter().find_map(|n| match n {
                Node::Select { positions, .. } => Some(positions.clone()),
                _ => None,
            }))
            .unwrap_or_default();
        let kept: Vec<usize> = (0..rows).filter(|r| !fixed.contains(&positions[*r])).collect();
        let x = d.download(trace.value(mlp.input)?).map_err(|e| e.to_string())?.select(ndarray::Axis(0), &kept);
        let i = scorer.at(mlp.gate.operator)?;
        let mean = center(i)?;
        let variance = device_posterior.values(i)?.1.mapv(|s| (2.0 * s).exp());
        let bias = match mlp.gate.bias {
            Some(b) => {
                let j = scorer.at(b)?;
                let log_sd = device_posterior.values(j)?.1.column(0).to_owned();
                // A function's bias moves by `shift` of its posterior deviation ([`project`]).
                let mean = &center(j)?.column(0) - &log_sd.mapv(|s| shift * s.exp());
                Some((j, mean, log_sd.mapv(|s| (2.0 * s).exp())))
            }
            None => None,
        };
        let expected = crate::library_complexity::expected(&crate::library_complexity::Gate {
            x: x.view(),
            mean: mean.view(),
            variance: variance.view(),
            bias: bias.as_ref().map(|(_, m, v)| (m.view(), v.view())),
            alive: &surviving,
        })?;
        count += expected.count;
        let per = 1.0 / rows as f64;
        terms.push((i, d.upload((expected.mean * per).view()).map_err(error)?, d.upload((expected.variance * per).view()).map_err(error)?));
        if let Some((j, _, _)) = bias {
            terms.push((j, d.upload((expected.bias_mean.insert_axis(ndarray::Axis(1)) * per).view()).map_err(error)?, d.upload((expected.bias_variance.insert_axis(ndarray::Axis(1)) * per).view()).map_err(error)?));
        }
    }
    // The step's own (not the move's test's) for the assignment's pull ([`Scorer::pull_assignments`]).
    if !previous {
        scorer.assignment_budget = assignment_terms;
    }
    Ok((count / rows as f64, terms))
}

/// A gated stage's expected rank executed on the rows of its gate's input `input` (rows × parts
/// of read norms for an own gate, rows × d of the stage's input for a direction gate `gate`, its
/// rows' means and variances), with the thresholds' means and variances `bias` and per component
/// its rank: `library_complexity::own` and `library_complexity::weighted` on the device
/// (`Device::gate_function`), their counts and the thresholds' derivatives downloaded, and for a
/// direction gate its rows' derivatives. On the host they took about 5 s of a 5.5 s step of the
/// vpd4l grouped direction arm at 4,096 rows (decomp-vpd4l-b), against 0.35–0.57 s for a step
/// without the budget. A shared stage (`assign`, gates × components, as the pass wrote it) has
/// its gates' pre-activations `m` (an own gate's `Σ_b A_mb n_b + μ_c`, `n_b` the squared read
/// norms its input holds) and variances `s²`, and each component takes `m A` and `s² (A ⊙ A)`,
/// as its Gated node does; the gates' derivatives gather the components' through `A`, and the
/// count's derivative in `A` itself is returned (summed over the rows; none for an unshared stage).
/// A stage whose components may follow the previous stage's (`library_vpd`'s down components
/// following c_fc components) has an assignment whose rows past its own gates are those
/// components, read from `extra` (their per-component pre-activations, variances and widths, the
/// fourth return of the previous stage's call); the followed stage's thresholds take no pull
/// through its followers here. The gates are soft, `Φ(z / w)` with the gates' widths `widths`
/// (each component's `w A`), and `E[Φ(z / w)] = Φ(m / √(w² + s²))` for `z ~ N(m, s²)`: the count is
/// of what the pass executes (with `w = 0`, a hard gate, of `s²` alone).
fn gated_expected(
    d: &Device,
    arithmetic: gam_gpu::tensor::Arithmetic,
    input: &Tensor,
    gate: Option<(&Tensor, &Tensor)>,
    (bias_mean, bias_variance, widths): (&[f64], &[f64], &[f64]),
    rank: &[f64],
    (assign, extra): (Option<&Array2<f64>>, Option<(&Tensor, &Tensor, &[f64])>),
) -> Result<(crate::library_complexity::Expected, Option<(Tensor, Tensor)>, Option<Array2<f64>>, (Tensor, Tensor, Vec<f64>)), String> {
    use gam_gpu::tensor::GateFunction;
    let rows = input.rows();
    let parts = rank.len();
    let gates = bias_mean.len();
    let row = |values: &[f64]| d.upload_vec(1, values.len(), values.to_vec()).map_err(error);
    // The assignment and its entries' squares, on the device: its rows over the stage's own gates
    // (`shared`), and in a stage whose components may follow the previous stage's components
    // (cross-stage sharing: the assignment's rows past the stage's gates) those rows with the
    // previous stage's per-component pre-activations and variances `extra` (`cross`).
    let upload = |a: ndarray::ArrayView2<f64>| -> Result<(Tensor, Tensor), String> { Ok((d.upload(a).map_err(error)?, d.upload(a.mapv(|v| v * v).view()).map_err(error)?)) };
    let (shared, cross) = match assign {
        Some(a) if a.dim() == (gates, parts) => (Some(upload(a.view())?), None),
        Some(a) if a.ncols() == parts && a.nrows() > gates => {
            let (m_extra, s2_extra, _) = extra.ok_or("library budget: a cross-stage assignment without the previous stage's components")?;
            if m_extra.cols() != a.nrows() - gates {
                return Err(format!("library budget: {} rows past the gates for {} components of the previous stage", a.nrows() - gates, m_extra.cols()));
            }
            let (bottom, bottom2) = upload(a.slice(ndarray::s![gates.., ..]))?;
            (Some(upload(a.slice(ndarray::s![..gates, ..]))?), Some((bottom, bottom2, m_extra, s2_extra)))
        }
        Some(a) => return Err(format!("library budget: an assignment of {:?} for {gates} gates and {parts} components", a.dim())),
        None if gates != parts => return Err(format!("library budget: {gates} thresholds for {parts} components")),
        None => (None, None),
    };
    // An own gate pools its members' read norms through the assignment only where the stage's
    // components share the stage's own gates.
    let pooled = shared.is_some() && cross.is_none();
    // m = x μ_gᵀ + μ_c and s² = x² σ²_gᵀ + σ²_c (a direction gate), or m = ‖V_bᵀx‖ + μ_c and
    // s² = σ²_c (an own gate; through the assignment, m = n Aᵀ + μ_c), per gate.
    let squares = match gate {
        Some(_) => {
            let mut squares = d.empty(rows, input.cols()).map_err(error)?;
            d.hadamard(&mut squares, input, input, false).map_err(error)?;
            Some(squares)
        }
        None => None,
    };
    let (mut m, mut s2) = match (gate, &squares) {
        (Some((mean, variance)), Some(squares)) => {
            // In the input's storage (the posterior may hold another).
            let held = |t: &Tensor| -> Result<Option<Tensor>, String> { if t.storage() == input.storage() { Ok(None) } else { d.convert(t).map(Some).map_err(error) } };
            let (mean_held, variance_held) = (held(mean)?, held(variance)?);
            let (mut m, mut s2) = (d.empty(rows, gates).map_err(error)?, d.empty(rows, gates).map_err(error)?);
            d.gemm(&mut m, 1.0, input, Op::N, mean_held.as_ref().unwrap_or(mean), Op::T, 0.0, arithmetic).map_err(error)?;
            d.gemm(&mut s2, 1.0, squares, Op::N, variance_held.as_ref().unwrap_or(variance), Op::T, 0.0, arithmetic).map_err(error)?;
            (m, s2)
        }
        _ => {
            if input.cols() != parts {
                return Err(format!("library budget: {} read norms for {parts} components", input.cols()));
            }
            let m = match &shared {
                Some((a, _)) if pooled => {
                    let mut m = d.empty(rows, gates).map_err(error)?;
                    d.gemm(&mut m, 1.0, input, Op::N, a, Op::T, 0.0, arithmetic).map_err(error)?;
                    m
                }
                _ => d.copy(input).map_err(error)?,
            };
            (m, d.zeros(rows, gates).map_err(error)?)
        }
    };
    d.add_row(&mut m, 1.0, &row(bias_mean)?).map_err(error)?;
    d.add_row(&mut s2, 1.0, &row(bias_variance)?).map_err(error)?;
    // Per component: through the assignment where the stage is shared.
    let (m_gate, s2_gate) = (m, s2);
    let (m, s2) = match &shared {
        Some((a, a2)) => {
            let (mut mb, mut s2b) = (d.empty(rows, parts).map_err(error)?, d.empty(rows, parts).map_err(error)?);
            d.gemm(&mut mb, 1.0, &m_gate, Op::N, a, Op::N, 0.0, arithmetic).map_err(error)?;
            d.gemm(&mut s2b, 1.0, &s2_gate, Op::N, a2, Op::N, 0.0, arithmetic).map_err(error)?;
            if let Some((bottom, bottom2, m_extra, s2_extra)) = &cross {
                d.gemm(&mut mb, 1.0, m_extra, Op::N, bottom, Op::N, 1.0, arithmetic).map_err(error)?;
                d.gemm(&mut s2b, 1.0, s2_extra, Op::N, bottom2, Op::N, 1.0, arithmetic).map_err(error)?;
            }
            (mb, s2b)
        }
        None => (d.copy(&m_gate).map_err(error)?, d.copy(&s2_gate).map_err(error)?),
    };
    // Each component's width `w_b = Σ_g A_gb w_g` over the gates it may take (its gate's own without
    // sharing; a followed component's width past the stage's gates), as its Gated node's: the pass
    // executes `Φ(z / w)`, and `E[Φ(z / w)] = Φ(m / √(w² + s²))` for `z ~ N(m, s²)`.
    if widths.len() != gates {
        return Err(format!("library budget: {} widths for {gates} gates", widths.len()));
    }
    let followed_widths = extra.map(|(_, _, w)| w).unwrap_or(&[]);
    let component_width: Vec<f64> = match assign {
        Some(a) => (0..parts).map(|b| (0..a.nrows()).map(|g| a[[g, b]] * if g < gates { widths[g] } else { followed_widths.get(g - gates).copied().unwrap_or(0.0) }).sum()).collect(),
        None => widths.to_vec(),
    };
    let variance = d.copy(&s2).map_err(error)?;
    let mut s2 = s2;
    d.add_row(&mut s2, 1.0, &row(&component_width.iter().map(|w| w * w).collect::<Vec<_>>())?).map_err(error)?;
    let s = d.gate_function(GateFunction::Sqrt, &s2, None).map_err(error)?;
    // Per row and component, weighted by its rank: P = Φ(m/s), ∂P/∂m = φ(m/s)/s and
    // 2 ∂P/∂s² = −φ(m/s) m / s³.
    let weight = row(rank)?;
    let weighted = |f: &Tensor| -> Result<Tensor, String> {
        let mut out = d.empty(rows, parts).map_err(error)?;
        d.scale_columns(&mut out, f, &weight, false).map_err(error)?;
        Ok(out)
    };
    let probability = weighted(&d.gate_function(GateFunction::Cdf, &m, Some(&s)).map_err(error)?)?;
    let slope = weighted(&d.gate_function(GateFunction::CdfSlope, &m, Some(&s)).map_err(error)?)?;
    let spread = weighted(&d.gate_function(GateFunction::Ratio, &d.gate_function(GateFunction::CdfScaleSlope, &m, Some(&s)).map_err(error)?, Some(&s)).map_err(error)?)?;
    // The assignment's path through the components' widths, `∂(w_b²)/∂A_gb = 2 w_b w_g`:
    // `w_g w_b Σ_rows 2 ∂P/∂s²_b` (summed over the rows, half of `spread`'s doubling).
    let width_spread: ndarray::Array1<f64> = match assign {
        Some(_) => {
            let ones = d.upload_vec(1, rows, vec![1.0; rows]).map_err(error)?;
            let mut out = d.empty(1, parts).map_err(error)?;
            d.gemm(&mut out, 1.0, &ones, Op::N, &spread, Op::N, 0.0, arithmetic).map_err(error)?;
            d.download(&out).map_err(error)?.row(0).iter().zip(&component_width).map(|(p, w)| p * w).collect()
        }
        None => ndarray::Array1::zeros(0),
    };
    // Back to the gates through the assignment: ∂/∂m_g = Σ_b A_gb ∂/∂m_b, ∂/∂s²_g = Σ_b A_gb² ∂/∂s²_b.
    let (slope_gate, spread_gate) = match &shared {
        Some((a, a2)) => {
            let (mut sg, mut pg) = (d.empty(rows, gates).map_err(error)?, d.empty(rows, gates).map_err(error)?);
            d.gemm(&mut sg, 1.0, &slope, Op::N, a, Op::T, 0.0, arithmetic).map_err(error)?;
            d.gemm(&mut pg, 1.0, &spread, Op::N, a2, Op::T, 0.0, arithmetic).map_err(error)?;
            (sg, pg)
        }
        None => (d.copy(&slope).map_err(error)?, d.copy(&spread).map_err(error)?),
    };
    // The count's derivative in the assignment (gates × components, summed over the rows): through
    // each component's pre-activation `m A` (`mᵀ ∂/∂m_b`), its variance `s² (A ⊙ A)`
    // (`2 A_gb (s²ᵀ ∂/∂s²_b)`, `spread` being `2 ∂/∂s²`), and an own gate's pooling `n Aᵀ` of the
    // read norms (`(∂/∂m_g)ᵀ n`).
    let assigned = match (&shared, assign) {
        (Some(_), Some(host)) => {
            let (mut direct, mut varied) = (d.empty(gates, parts).map_err(error)?, d.empty(gates, parts).map_err(error)?);
            d.gemm(&mut direct, 1.0, &m_gate, Op::T, &slope, Op::N, 0.0, arithmetic).map_err(error)?;
            d.gemm(&mut varied, 1.0, &s2_gate, Op::T, &spread, Op::N, 0.0, arithmetic).map_err(error)?;
            let mut total = d.download(&direct).map_err(error)? + d.download(&varied).map_err(error)? * &host.slice(ndarray::s![..gates, ..]);
            if let Some((_, _, m_extra, s2_extra)) = &cross {
                // The rows past the gates: the previous stage's components followed.
                let followed = host.nrows() - gates;
                let (mut direct, mut varied) = (d.empty(followed, parts).map_err(error)?, d.empty(followed, parts).map_err(error)?);
                d.gemm(&mut direct, 1.0, m_extra, Op::T, &slope, Op::N, 0.0, arithmetic).map_err(error)?;
                d.gemm(&mut varied, 1.0, s2_extra, Op::T, &spread, Op::N, 0.0, arithmetic).map_err(error)?;
                let bottom = d.download(&direct).map_err(error)? + d.download(&varied).map_err(error)? * &host.slice(ndarray::s![gates.., ..]);
                total = ndarray::concatenate(ndarray::Axis(0), &[total.view(), bottom.view()]).map_err(error)?;
            }
            let all_widths: Vec<f64> = widths.iter().chain(followed_widths).copied().collect();
            total += &Array2::from_shape_fn(total.dim(), |(g, b)| all_widths.get(g).copied().unwrap_or(0.0) * width_spread[b]);
            if pooled && gate.is_none() {
                let mut pooled = d.empty(gates, parts).map_err(error)?;
                d.gemm(&mut pooled, 1.0, &slope_gate, Op::T, input, Op::N, 0.0, arithmetic).map_err(error)?;
                total += &d.download(&pooled).map_err(error)?;
            }
            Some(total)
        }
        _ => None,
    };
    let (slope, spread) = (slope_gate, spread_gate);
    let ones = d.upload_vec(1, rows, vec![1.0; rows]).map_err(error)?;
    let column_sums = |t: &Tensor| -> Result<ndarray::Array1<f64>, String> {
        let mut out = d.empty(1, t.cols()).map_err(error)?;
        d.gemm(&mut out, 1.0, &ones, Op::N, t, Op::N, 0.0, arithmetic).map_err(error)?;
        Ok(d.download(&out).map_err(error)?.row(0).to_owned())
    };
    let count = column_sums(&probability)?.sum();
    let bias_mean = column_sums(&slope)?;
    let bias_variance = column_sums(&spread)? * 0.5;
    let gate_terms = match &squares {
        Some(squares) => {
            let (mut mean, mut variance) = (d.empty(gates, input.cols()).map_err(error)?, d.empty(gates, input.cols()).map_err(error)?);
            d.gemm(&mut mean, 1.0, &slope, Op::T, input, Op::N, 0.0, arithmetic).map_err(error)?;
            d.gemm(&mut variance, 0.5, &spread, Op::T, squares, Op::N, 0.0, arithmetic).map_err(error)?;
            Some((mean, variance))
        }
        None => None,
    };
    let expected = crate::library_complexity::Expected { count, rows, mean: Array2::zeros((gates, 0)), variance: Array2::zeros((gates, 0)), bias_mean, bias_variance };
    Ok((expected, gate_terms, assigned, (m, variance, component_width)))
}

/// The test of the posterior's pending move (`DevicePosterior::pending_divergence`) on this step's
/// batch: the change of the Lagrangian per token the move made, measured on the batch at the
/// step's own weight samples on both sides (`new`: the batch's data bits and expected parts per
/// token at the moved iterate, from the step itself; the same draws around the iterate before
/// the move, `DevicePosterior::previous_into`), `ΔL = (B ln 2 Δbits + ΔKL + λ ΔÊ) / N` with `B`
/// the batches, `N` the training tokens, `ΔKL` the move's change of the prior's divergence and
/// `λ` the budget's multiplier (nats of the whole `F` per part per token, as the step's pull
/// takes it). A batch the move was not made on, so the test is not the move's
/// own fit. The data term's change is measured per sequence of the batch (paired: both sides on
/// the same experiments and draws), so the batch's `ΔL` has a standard error from the spread of
/// its sequences' changes, `B ln 2 √(n var_s) / N` over its `n` sequences. A move is rejected only
/// when `ΔL` exceeds that standard error: a single batch cannot tell a small true gain from its
/// own noise, and at `ΔL > 0` alone 30% of grouped_direction's moves were rejected by its sixth
/// epoch at 2^22 (decomp-vpd4l-f, 42e03873f6). Returns the decision, `ΔL` and its standard
/// error. A line step's quadratic model
/// can be wrong (toys: TMS-id grouped own gates, η −9.5e-3 with ρ̄ NaN, the mean's KL from 4e-4 to
/// 2e182 over epochs 2–6; resid_mlp_2l per-slice gates, η changing sign, 1.9 → 3.8e19); a move
/// that raises the measured objective is not taken.
fn step_accepted(
    scorer: &mut Scorer,
    device_posterior: &DevicePosterior,
    explanation: &Explanation,
    active: &[bool],
    (batch, experiments, key): (&Batch, &[Experiment], u64),
    (bits, expected): (&[Vec<f64>], Option<f64>),
    all_on: Option<&AllOn>,
    (scale, tokens, lambda): (f64, usize, f64),
) -> Result<(bool, f64, f64), String> {
    let divergence = device_posterior.pending_divergence().ok_or("no pending move")?;
    // The same parts and draws as the step's (`antithetic_step`).
    let parts = step_parts(batch, experiments.to_vec());
    let keys = [key, key ^ gam_gpu::tensor::ANTITHETIC];
    let mut previous: Vec<f64> = Vec::with_capacity(experiments.len());
    let mut previous_on: Vec<(usize, f64)> = Vec::new();
    for (part, k) in parts.iter().zip(keys) {
        let targets = scorer.experiments.targets(batch, part)?;
        // The step's own side is a training pass; this side is one too, before the move.
        let terms = scorer.terms(device_posterior, (batch, part, &targets), (Values::Previous(k), Gates::Relaxed), (false, None))?;
        previous.extend(terms.bits.iter().map(|b| b.iter().sum::<f64>()));
        if let Some(on) = terms.all_on {
            previous_on.extend(on.experiments.iter().zip(&on.bits).map(|(e, b)| (e.base, b.iter().sum::<f64>())));
        }
    }
    if previous.len() != bits.len() {
        return Err(format!("the move's test scored {} experiments against the step's {}", previous.len(), bits.len()));
    }
    // Per sequence (base) of the batch, its experiments' change of the data bits.
    let mut by_base: BTreeMap<usize, f64> = BTreeMap::new();
    for ((e, new), old) in experiments.iter().zip(bits).zip(&previous) {
        *by_base.entry(e.base).or_default() += new.iter().sum::<f64>() - old;
    }
    // The all-on experiment, both sides (`Scorer::all_on_pass`).
    if let Some(on) = all_on {
        for (e, new) in on.experiments.iter().zip(&on.bits) {
            *by_base.entry(e.base).or_default() += new.iter().sum::<f64>();
        }
    }
    for (base, old) in previous_on {
        *by_base.entry(base).or_default() -= old;
    }
    let n = by_base.len() as f64;
    let total: f64 = by_base.values().sum();
    let spread = if n > 1.0 { by_base.values().map(|d| (d - total / n).powi(2)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    let standard_error = scale * LN_2 * (n * spread).sqrt() / tokens as f64;
    let budget = match expected {
        Some(moved) if lambda > 0.0 => lambda * (moved - complexity_terms(scorer, device_posterior, explanation, active, batch, (key, true))?.0),
        _ => 0.0,
    };
    let change = (scale * LN_2 * total + divergence + budget + scorer.mixing_change()) / tokens as f64;
    if !change.is_finite() {
        return Err(format!("a nonfinite change of the objective at the move's test ({change})"));
    }
    Ok((change <= standard_error, change, standard_error))
}

/// Returns the experiments in the order of their bits.
fn antithetic_step(scorer: &mut Scorer, (device, device_posterior): (&Device, &DevicePosterior), batch: &Batch, experiments: Vec<Experiment>, key: u64) -> Result<(Vec<Experiment>, Terms), String> {
    // One part at its key: a training pass with its gradient, the shared stages' assignments'
    // taken out, and with `probed` a draw of its factor.
    let half = |scorer: &mut Scorer, part: &[Experiment], key: u64, probed: bool| -> Result<Terms, String> {
        let targets = scorer.experiments.targets(batch, part)?;
        let mut terms = scorer.terms(device_posterior, (batch, part, &targets), (Values::Iterate(key), Gates::Relaxed), (true, probed.then(|| probe_key(key))))?;
        scorer.take_assignment_gradients(&mut terms.gradient, terms.factor.as_mut(), true)?;
        scorer.take_mixing_gradients(&mut terms.gradient)?;
        if let Some(on) = terms.all_on.as_mut() {
            scorer.take_mixing_gradients(&mut on.gradient)?;
        }
        Ok(terms)
    };
    let mut parts = step_parts(batch, experiments);
    if parts.len() == 1 {
        let all = parts.pop().ok_or("a batch's experiments")?;
        let terms = half(scorer, &all, key, true)?;
        return Ok((all, terms));
    }
    let (second, first) = (parts.pop().ok_or("a batch's second half")?, parts.pop().ok_or("a batch's first half")?);
    let probed_first = factor_half(key);
    let mut terms = half(scorer, &first, key, probed_first)?;
    let other = half(scorer, &second, key ^ gam_gpu::tensor::ANTITHETIC, !probed_first)?;
    terms.join(device, other)?;
    Ok((first.into_iter().chain(second).collect(), terms))
}

/// Whether the first half of a training batch's bases supplies the step's Gauss–Newton factor
/// (`antithetic_step`), drawn from the batch's `key`: the top bit of the SplitMix64 output after
/// the probe's key ([`probe_key`]), so apart from the weight noise's and the probe's draws, each
/// half with probability ½.
fn factor_half(key: u64) -> bool {
    probe_key(probe_key(key)) >> 63 == 0
}

/// The momentum's decay: an average over about 199 gradients (`n_eff = (1 + β₁) / (1 − β₁)`). On
/// vpd4l's one-block case at equal tokens (batch 8, 3 epochs) it ended at held-out F 2.595 bits per
/// token against 2.730 at 0.9 (frontier-gn-b3-noise-*), and at 2^24 one epoch reached 2.158 against
/// 2.230 (frontier-gn-n32768-m99 against -line4). A decay set each epoch from the gradient-noise
/// scale IVON's state measures, `β₁ = (B − 1) / (B + 1)` (4d64ef9bd5), lost to it on the same draws
/// (MATS 16780, 16781 at dc12e86db5): 4.127, 3.497, 2.954 against 4.037, 3.298, 2.866 after epochs
/// 0–2 (it measured B ≈ 145 batches, β₁ = 0.986).
const MOMENTUM_DECAY: f64 = 0.99;

/// The native weight edits a fit with weight edits among its families draws from
/// (`Scorer::prepared`, `interchange::Interchange::draw_weight_edits`): the candidates drawn, the
/// edits kept, and the sequences each candidate's effect is measured on.
pub const WEIGHT_CANDIDATES: usize = 1024;
pub const WEIGHT_EDITS: usize = 256;
pub const WEIGHT_SCREEN: usize = 2;

/// A running mean of bits over scored tokens.
#[derive(Clone, Copy, Debug, Default)]
struct Mean {
    bits: f64,
    tokens: usize,
}

impl Mean {
    fn add(&mut self, bits: &[f64]) {
        self.bits += bits.iter().sum::<f64>();
        self.tokens += bits.len();
    }

    fn mean(&self) -> Option<f64> {
        (self.tokens > 0).then(|| self.bits / self.tokens as f64)
    }
}

/// The held-out batches of `sequences` and their experiments: a fixed sample from the training
/// distribution, one clean and one patched experiment per base, each base's from its own draws
/// ([`base_draws`]). They are drawn from their own seed, the SplitMix64 output after the fit's
/// (`gam_linalg::utils::splitmix64_hash`): drawn from the fit's seed itself, held-out base `b`
/// would repeat training base `b`'s interventions exactly, so held-out scores would test unseen
/// bases under seen interventions only.
fn held_out_experiments(scorer: &Scorer, sequences: &[Vec<u32>], settings: &Settings) -> Result<Vec<(Draw, Vec<Experiment>)>, String> {
    draws(sequences.len(), settings.batch_sequences, gam_linalg::utils::splitmix64_hash(settings.seed))?
        .into_iter()
        .map(|draw| {
            let experiments = scorer.experiments(&draw, sequences)?;
            Ok((draw, experiments))
        })
        .collect()
}

/// A held-out batch ([`held_batches`]): its bases and sources, its experiments and `M`'s targets
/// for them.
type HeldBatch = (Batch, Vec<Experiment>, Targets);

/// The held-out batches of `sequences` ([`held_out_experiments`]) with `M`'s targets for each.
fn held_batches(scorer: &Scorer, sequences: &[Vec<u32>], settings: &Settings) -> Result<Vec<HeldBatch>, String> {
    held_out_experiments(scorer, sequences, settings)?
        .into_iter()
        .map(|(draw, experiments)| {
            let batch = draw.batch(sequences)?;
            let targets = scorer.experiments.targets(&batch, &experiments)?;
            Ok((batch, experiments, targets))
        })
        .collect()
}

/// The held-out evaluation of `posterior` (held on the host, and as `device_posterior` on the
/// device) on `sequences` (module note), its batches and `M`'s targets made here; `tokens` is `N`.
fn held_out(
    scorer: &mut Scorer,
    explanation: &Explanation,
    posteriors: (&Posterior, &DevicePosterior),
    sequences: &[Vec<u32>],
    settings: &Settings,
    tokens: usize,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<HeldOut, String> {
    let batches = held_batches(scorer, sequences, settings)?;
    held_out_on(scorer, explanation, posteriors, (sequences, &batches), settings, tokens, prior)
}

/// [`held_out`] on `batches`, the held-out batches of `sequences` made beforehand
/// ([`held_batches`]): the fit's fixed subset, whose experiments and targets depend on `M`, the
/// batch and its experiments alone, has them made once per fit and scored after every epoch.
fn held_out_on(
    scorer: &mut Scorer,
    explanation: &Explanation,
    (posterior, device_posterior): (&Posterior, &DevicePosterior),
    (sequences, batches): (&[Vec<u32>], &[HeldBatch]),
    settings: &Settings,
    tokens: usize,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<HeldOut, String> {
    let blocks = 2 * scorer.layers();
    let (mut clean, mut patched) = (vec![Mean::default(); blocks], vec![Mean::default(); blocks]);
    let (mut read, mut joint, mut sampled) = (Mean::default(), Mean::default(), Mean::default());
    let (mut at_mean, mut at_rounded) = (Mean::default(), Mean::default());
    let size = |e: &Experiment| e.explained.iter().filter(|x| **x).count();
    // The mean, the sample and the rounded posterior (rounded on the device), each through the
    // scored explanation's hard gates (`Gates::Hard`), are scored against each batch's targets
    // from `M`; each data term with its all-on experiment's bits, as F's (`Scorer::terms`).
    let (mut on_mean, mut on_sampled, mut on_rounded) = (Mean::default(), Mean::default(), Mean::default());
    for (b, (batch, experiments, targets)) in batches.iter().enumerate() {
        let terms = scorer.terms(device_posterior, (batch, experiments, targets), (Values::Mean, Gates::Hard), (false, None))?;
        terms.bits.iter().for_each(|b| at_mean.add(b));
        if let Some(on) = &terms.all_on {
            on.bits.iter().for_each(|b| on_mean.add(b));
        }
        for (e, bits) in experiments.iter().zip(&terms.bits) {
            match &e.patch {
                None => clean[size(e) - 1].add(bits),
                Some(patch) => {
                    patched[size(e) - 1].add(bits);
                    match patch {
                        Patch::Read { .. } => read.add(bits),
                        Patch::Reads { .. } => joint.add(bits),
                        Patch::Ops { .. } | Patch::Weights { .. } => {}
                    }
                }
            }
        }
        let terms = scorer.terms(device_posterior, (batch, experiments, targets), (Values::Sample(noise_seed(settings.seed, 0, b)), Gates::Hard), (false, None))?;
        terms.bits.iter().for_each(|b| sampled.add(b));
        if let Some(on) = &terms.all_on {
            on.bits.iter().for_each(|b| on_sampled.add(b));
        }
        let terms = scorer.terms(device_posterior, (batch, experiments, targets), (Values::Rounded, Gates::Hard), (false, None))?;
        terms.bits.iter().for_each(|b| at_rounded.add(b));
        if let Some(on) = &terms.all_on {
            on.bits.iter().for_each(|b| on_rounded.add(b));
        }
    }
    // A data term per scored token: the experiments' bits and the all-on experiment's over the
    // experiments' scored tokens (`N` counts those alone, as the training collection's does).
    let per_token = |data: &Mean, on: &Mean| (data.tokens > 0).then(|| (data.bits + on.bits) / data.tokens as f64);
    let all_on = on_mean.mean();
    let data = per_token(&sampled, &on_sampled).ok_or("no held-out tokens")?;
    let divergence: f64 = posterior.divergences().iter().sum();
    let gaussian = posterior.description();
    let prior_nats = match prior {
        Some(prior) => prior_term(prior, posterior, noise_seed(settings.seed, 0, 0), false)?.0,
        None => 0.0,
    };
    let description = gaussian + explanation.fixed_nats + prior_nats + scorer.mixing_nats();
    Ok(HeldOut {
        objective_bits_per_token: data + description / LN_2 / tokens as f64,
        data_bits_per_token: data,
        mean_bits_per_token: per_token(&at_mean, &on_mean).ok_or("no held-out tokens")?,
        rounded_bits_per_token: per_token(&at_rounded, &on_rounded).ok_or("no held-out tokens")?,
        divergence_bits: divergence / LN_2,
        variance_bits: (gaussian - divergence) / LN_2,
        choice_bits: explanation.fixed_nats / LN_2,
        prior_bits: prior_nats / LN_2,
        clean: clean.iter().map(Mean::mean).collect(),
        patched: patched.iter().map(Mean::mean).collect(),
        read_patch: read.mean(),
        joint_patch: joint.mean(),
        layers: activity(scorer, explanation, (posterior, device_posterior), sequences, settings)?,
        all_on_bits_per_token: all_on,
    })
}

/// Per layer, the survivors of `posterior` and its functions' activity on `sequences` at the
/// posterior mean (`LayerCount`), counted on the device (`Device::resolved_counts`).
fn activity(scorer: &mut Scorer, explanation: &Explanation, (posterior, device_posterior): (&Posterior, &DevicePosterior), sequences: &[Vec<u32>], settings: &Settings) -> Result<Vec<LayerCount>, String> {
    scorer.set(device_posterior, Values::Mean, Gates::Hard)?;
    let variance = |op: usize| -> Result<Array2<f64>, String> { Ok(posterior.log_sd[scorer.at(op)?].mapv(|s| (2.0 * s).exp())) };
    let (_, p) = scorer.experiments.models();
    let (program, d) = (p.program, p.program.device());
    // The counts are summed in float64 where the device holds it.
    let wide = match d.with_storage(Storage::F64) {
        Ok(wide) => wide,
        Err(_) => d.clone(),
    };
    let active = |g: &usize| posterior.active[*g];
    let mut out = Vec::with_capacity(scorer.layers());
    // Per map its weights' posterior variances and its bias's (zero without one), on the device.
    let variances = |map: &Map| -> Result<(Tensor, Tensor), String> {
        let weights = d.upload(variance(map.operator)?.view()).map_err(error)?;
        let bias = match map.bias {
            Some(b) => variance(b)?.column(0).to_vec(),
            None => vec![0.0; program.widths()[map.node]],
        };
        Ok((weights, d.upload_vec(1, bias.len(), bias).map_err(error)?))
    };
    // Per layer its surviving functions' flags, its law per function, and its maps' variances.
    let mut gates = Vec::with_capacity(scorer.layers());
    for (layer, mlp) in explanation.layers.iter().zip(&scorer.mlps) {
        let Some(mlp) = mlp else {
            // Gated components (`library_vpd`), counted below from their gates.
            out.push(LayerCount { heads: 0, planes: 0, values: 0, functions: 0, nonzero_per_token: 0.0, resolved_per_token: 0.0 });
            gates.push(None);
            continue;
        };
        let surviving: Vec<u32> = layer.functions.iter().map(|groups| u32::from(groups.iter().all(active))).collect();
        out.push(LayerCount {
            heads: layer.heads.iter().filter(|(_, values)| values.iter().any(active)).count(),
            planes: layer.heads.iter().map(|(planes, _)| planes.iter().filter(|g| active(g)).count()).sum(),
            values: layer.heads.iter().map(|(_, values)| values.iter().filter(|g| active(g)).count()).sum(),
            functions: surviving.iter().filter(|s| **s == 1).count(),
            nonzero_per_token: 0.0,
            resolved_per_token: 0.0,
        });
        // A layer whose MLP the explanation leaves to `M` ([`scoped`]) has no functions to count.
        if layer.functions.is_empty() {
            gates.push(None);
            continue;
        }
        let codes = d.upload_indices(&vec![law_of(mlp.law).code(); surviving.len()]).map_err(error)?;
        let alive = d.upload_indices(&surviving).map_err(error)?;
        gates.push(Some((alive, codes, variances(&mlp.gate)?, mlp.up.as_ref().map(variances).transpose()?)));
    }
    // Per layer the positions where its block runs `M`'s MLP instead of its functions (a transcoder
    // layer's first token, `library_transcoder`): its functions are computed there but not used,
    // so those tokens are not counted.
    let fixed: Vec<Vec<u32>> = (0..explanation.layers.len())
        .map(|l| {
            let rule = explanation.artifact.program.rules.iter().find(|r| r.name == format!("library.l{l}.mlp"));
            rule.and_then(|r| r.nodes.iter().find_map(|n| match n {
                Node::Select { positions, .. } => Some(positions.clone()),
                _ => None,
            }))
            .unwrap_or_default()
        })
        .collect();
    let mut rows = vec![0usize; out.len()];
    for chunk in sequences.chunks(settings.batch_sequences) {
        let family = sequence_family(&chunk.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        let trace = program.forward(&family)?;
        let positions = &family.layout.as_ref().ok_or("a sequence layout")?.position;
        for (l, ((count, mlp), gates)) in out.iter_mut().zip(&scorer.mlps).zip(&gates).enumerate() {
            let skipped: Vec<usize> = positions.iter().enumerate().filter(|(_, p)| fixed[l].binary_search(p).is_ok()).map(|(r, _)| r).collect();
            rows[l] += family.rows - skipped.len();
            if mlp.is_none() {
                // Gated components at the posterior mean: the rank of each component whose gate is
                // on, per token (the slices executed, in rank-one equivalents).
                let mut executed = 0.0;
                let mut surviving = 0;
                for stage in &scorer.stages[l] {
                    let rank = stage.ranks(&posterior.active);
                    surviving += rank.iter().filter(|r| **r > 0.0).count();
                    let z = d.download(trace.value(stage.component_gate)?).map_err(error)?;
                    executed += z.rows().into_iter().map(|row| row.iter().zip(&rank).filter(|(z, _)| **z > 0.0).map(|(_, r)| r).sum::<f64>()).sum::<f64>();
                }
                count.functions = surviving;
                count.nonzero_per_token += executed;
                count.resolved_per_token += executed;
                continue;
            }
            let Some(mlp) = mlp else { continue };
            let Some((alive, codes, gate, up)) = gates else { continue };
            let x = trace.value(mlp.input)?;
            let mut squares = d.empty(x.rows(), x.cols()).map_err(error)?;
            d.hadamard(&mut squares, x, x, false).map_err(error)?;
            // `s²` of each function's value at every token: `Σ_k σ²_k x_k²` plus the bias's `σ²`.
            let noise = |(weights, bias): &(Tensor, Tensor)| -> Result<Tensor, String> {
                let mut s2 = d.empty(x.rows(), weights.rows()).map_err(error)?;
                d.gemm(&mut s2, 1.0, &squares, Op::N, weights, Op::T, 0.0, program.arithmetic()).map_err(error)?;
                d.add_row(&mut s2, 1.0, bias).map_err(error)?;
                Ok(s2)
            };
            let (z, a) = (trace.value(mlp.gate.node)?, trace.value(mlp.activation)?);
            // The activations with the skipped tokens' rows zero: nothing counts there.
            let masked = if skipped.is_empty() {
                None
            } else {
                let mut masked = d.copy(a).map_err(error)?;
                let zero = d.zeros(1, a.cols()).map_err(error)?;
                for &r in &skipped {
                    d.set_rows(&mut masked, r, &zero).map_err(error)?;
                }
                Some(masked)
            };
            let a = masked.as_ref().unwrap_or(a);
            let sz2 = noise(gate)?;
            let mut sums = wide.zeros(1, 3).map_err(error)?;
            match (&mlp.up, up) {
                (Some(map), Some(variances)) => {
                    // A gated law: the value φ(z) y, its slope φ'(z) y against the gate's noise and
                    // φ(z) against the up direction's.
                    let y = trace.value(map.node)?;
                    let mut value = d.empty(a.rows(), a.cols()).map_err(error)?;
                    d.hadamard(&mut value, a, y, false).map_err(error)?;
                    let slope = d.law_slopes(y, z, codes, gelu_tanh_constant()).map_err(error)?;
                    d.resolved_counts((&value, &slope, a), (&sz2, Some(&noise(variances)?)), alive, &mut sums).map_err(error)?;
                }
                _ => {
                    let ones = d.broadcast_rows(&d.upload_vec(1, z.cols(), vec![1.0; z.cols()]).map_err(error)?, z.rows()).map_err(error)?;
                    let slope = d.law_slopes(&ones, z, codes, gelu_tanh_constant()).map_err(error)?;
                    d.resolved_counts((a, &slope, a), (&sz2, None), alive, &mut sums).map_err(error)?;
                }
            }
            let sums = wide.download(&sums).map_err(error)?;
            count.nonzero_per_token += sums[[0, 1]];
            count.resolved_per_token += sums[[0, 2]];
        }
    }
    for (count, rows) in out.iter_mut().zip(rows) {
        count.nonzero_per_token /= rows as f64;
        count.resolved_per_token /= rows as f64;
    }
    Ok(out)
}

/// The line arm's ratio before any step ([`Progress::ratio`]).
fn unit_ratio() -> (f64, u64) {
    (1.0, 0)
}

/// Where a fit stands at the end of an epoch: with the posterior and the optimizer's moments, all a
/// fit needs to continue exactly as if it had not stopped.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Progress {
    identity: Identity,
    settings: Settings,
    tokens: usize,
    shapes: Vec<(usize, usize)>,
    /// The held-out evaluation at the starting point, on the fixed subset.
    start: Option<HeldOut>,
    /// The next epoch, and the optimizer steps taken.
    epoch: usize,
    step: i32,
    /// The steps the posterior's mean averages IVON's iterate over ([`DevicePosterior::averaged`]).
    #[serde(default)]
    averaged: u64,
    /// The line arm's ratio of the joint curvature to the diagonal one and the steps it averages
    /// ([`DevicePosterior::line_ratio`]).
    #[serde(default = "unit_ratio")]
    ratio: (f64, u64),
    /// The line step's averages of the fresh gradient's slope along the previous direction and of
    /// that direction's own gradient's slope, and the draws they average
    /// ([`DevicePosterior::line_slope`]); no draws in a checkpoint written before they were kept.
    #[serde(default)]
    slope: (f64, f64, u64),
    /// Per trainable operator, its momentum's bias correction `W`
    /// ([`DevicePosterior::momentum_weights`]), which the step's direction reads.
    #[serde(default)]
    weights: Option<Vec<f64>>,
    /// A checkpoint written before the corrections were kept per operator holds one pair
    /// `(W, W2)` (`W2` the sum of the weights' squares, which the step no longer reads): `W` for
    /// every operator. A checkpoint with neither resumes with `β₁ = 0.9`'s `W` over its steps, the
    /// decay those fits used ([`saved_weights`]). Read, never written.
    #[serde(default, skip_serializing)]
    momentum_weights: Option<(f64, f64)>,
    /// The best epoch since the objective last changed: its snapshot's estimate of `F` in nats and
    /// the epoch, whose posterior is the checkpoint beside the fit's with extension `best.bin` (the
    /// removal round starts from it).
    #[serde(default)]
    best: Option<(f64, usize)>,
    epochs: Vec<Epoch>,
    removals: Vec<Removal>,
    /// The budget's multiplier `λ` (`Settings::budget`), carried across a resume.
    #[serde(default)]
    multiplier: f64,
    /// Whether the budget has bound (`Settings::budget`): from then on `multiplier` integrates its
    /// violation.
    #[serde(default)]
    engaged: bool,
    /// The last epoch's snapshot, its per-batch estimates of `F` in nats, when convergence is being
    /// judged.
    previous: Option<Vec<f64>>,
    /// The version of the experiment collection's draws ([`COLLECTION`]; 0 in a checkpoint written
    /// before it was recorded), which a resumed fit must share.
    #[serde(default)]
    collection: u32,
    active: Vec<bool>,
    done: bool,
    seconds: f64,
    /// The training and the per-epoch evaluation seconds so far, and the last full held-out
    /// evaluation's seconds (the schedule's inputs).
    #[serde(default)]
    training_seconds: f64,
    #[serde(default)]
    evaluation_seconds: f64,
    #[serde(default)]
    full_seconds: f64,
    /// The prior term's state, when the fit has one.
    #[serde(default)]
    prior: Option<serde_json::Value>,
    /// The shared stages' assignment logits ([`Assignment`]), when the explanation has any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    assignments: Option<Vec<Vec<Vec<f64>>>>,
    /// The frames ([`Mixing`]), when the explanation has any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frames: Option<Vec<Vec<Vec<f64>>>>,
    /// The precision of each of the payload's arrays per operator ([`CHECKPOINT_ARRAYS`] of them,
    /// or [`LEGACY_ARRAYS`]): the storage the device posterior holds each in; none for a checkpoint
    /// written in float64 throughout, in the legacy layout.
    #[serde(default)]
    precision: Option<Vec<Precision>>,
}

/// What a checkpoint belongs to: a fit resumes from it only when every field agrees, so a
/// checkpoint of another model, dataset or library can neither resume nor skip training. Each
/// field is a SHA-256 in hexadecimal. The arithmetic a fit runs in (the device and its storage) is
/// not part of it: a checkpoint resumes on another device, continuing the same objective from the
/// values it holds (each array restored exactly in its saved precision), in that device's rounding
/// from there on, so a resume on another arithmetic is the uninterrupted fit to that rounding, not
/// bit for bit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    /// The export the native model was imported from (`export.json`'s digest).
    pub export: String,
    /// The training sequences' token ids, then the held-out sequences', in order.
    pub tokens: String,
    /// The structure of the native program and of the explanation's program: their nodes,
    /// rules, operators' names and interfaces, and outputs (the values are the export's).
    pub program: String,
    /// The prior groups: every group's name and member entries.
    pub groups: String,
    /// The shared-parameter map: per trainable operator, every node of every rule and of the
    /// program that applies it.
    pub sharing: String,
    /// What else defines the explanation: the values of its fixed operators that are not `M`'s
    /// tensors (a tie's selection and scatter, a transport, a merged body's unit scales), the
    /// trainable operators' starting values, the ownership map with its factors, the layers' groups
    /// and sites, the trainable operators, and each group's scale reference. A checkpoint that names
    /// none is of another identity.
    #[serde(default)]
    pub definition: String,
}

/// The explanation-only fields of a fit's identity: its prior groups, its shared-parameter map and
/// what else defines it ([`Identity`]).
fn explanation_identity(explanation: &Explanation) -> (String, String, String) {
    let mut groups = Fingerprinter::new();
    groups.absorb_str(b"removed", &format!("{:?}", explanation.removed));
    groups.absorb_u64(b"fixed", explanation.fixed_nats.to_bits());
    for group in &explanation.groups {
        groups.absorb_str(b"group", &group.name);
        for cell in &group.cells {
            groups.absorb_str(b"cells", &format!("{}|{:?}|{:?}", cell.operator, cell.rows, cell.cols));
        }
    }
    let mut sharing = Fingerprinter::new();
    let explained = &explanation.artifact.program;
    let bodies = std::iter::once(("program".to_string(), &explained.nodes)).chain(explained.rules.iter().map(|r| (r.name.clone(), &r.nodes)));
    let mut uses: BTreeMap<usize, Vec<String>> = explanation.trainable.iter().map(|op| (*op, Vec::new())).collect();
    for (body, nodes) in bodies {
        for (n, node) in nodes.iter().enumerate() {
            for op in node.operators() {
                if let Some(list) = uses.get_mut(&op) {
                    list.push(format!("{body}:{n}"));
                }
            }
        }
    }
    for (op, list) in &uses {
        sharing.absorb_str(b"uses", &format!("{op}|{}", list.join(",")));
    }
    let mut definition = Fingerprinter::new();
    let trainable: std::collections::BTreeSet<usize> = explanation.trainable.iter().copied().collect();
    for (i, op) in explained.operators.iter().enumerate() {
        // A fixed operator derived rather than taken from `M`, and every trainable operator's
        // starting values (two explanations of one shape warmed differently are different fits).
        if trainable.contains(&i) || !op.provenance.derivation.is_empty() {
            definition.absorb_str(b"fixed", &op.name);
            let values: Vec<u8> = op.matrix().iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
            definition.absorb_bytes(b"values", &values);
        }
    }
    definition.absorb_str(b"owners", &format!("{:?}", explanation.artifact.owners));
    definition.absorb_str(b"layers", &layers_definition(&explanation.layers));
    definition.absorb_str(b"trainable", &format!("{:?}", explanation.trainable));
    // Only where some stage shares gates, so explanations without one keep their identity.
    if !explanation.shares.is_empty() {
        definition.absorb_str(b"shares", &format!("{:?}", explanation.shares));
    }
    let reference: Vec<u8> = explanation.reference.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    definition.absorb_bytes(b"reference", &reference);
    (groups.finalize().to_hex(), sharing.finalize().to_hex(), definition.finalize().to_hex())
}

fn program_structure(hasher: &mut Fingerprinter, tag: &[u8], program: &OperatorProgram) {
    hasher.absorb_tag(tag);
    for node in &program.nodes {
        hasher.absorb_str(b"node", &format!("{node:?}"));
    }
    for rule in &program.rules {
        hasher.absorb_str(b"rule", &format!("{}|{:?}|{:?}|{}", rule.name, rule.inputs, rule.nodes, rule.output));
    }
    for op in &program.operators {
        hasher.absorb_str(b"operator", &format!("{}|{:?}|{:?}", op.name, op.rows, op.cols));
    }
    hasher.absorb_u64(b"output", program.output as u64);
}

/// The identity of a fit of `explanation` (made from `native`, imported from the export whose
/// digest is `export`) on the training `sequences` with the `held` sequences held out.
pub fn identity(export: &str, native: &OperatorProgram, explanation: &Explanation, sequences: &[Vec<u32>], held: &[Vec<u32>]) -> Identity {
    let mut tokens = Fingerprinter::new();
    for (tag, set) in [(&b"training"[..], sequences), (&b"held out"[..], held)] {
        tokens.absorb_u64(tag, set.len() as u64);
        for sequence in set {
            let bytes: Vec<u8> = sequence.iter().flat_map(|t| t.to_le_bytes()).collect();
            tokens.absorb_bytes(b"sequence", &bytes);
        }
    }
    let mut program = Fingerprinter::new();
    program_structure(&mut program, b"native", native);
    program_structure(&mut program, b"explanation", &explanation.artifact.program);
    let (groups, sharing, definition) = explanation_identity(explanation);
    Identity { export: export.to_string(), tokens: tokens.finalize().to_hex(), program: program.finalize().to_hex(), groups, sharing, definition }
}

/// The layers as an explanation's definition hashes them: [`Layer`]'s debug form without the sink
/// field, which is appended only to a layer that holds a sink group. Hashing the derived debug form
/// made every field added to `Layer` a change of every explanation's identity: the sink field
/// (`None` for every library on main) made each checkpoint written before it "a checkpoint of
/// another fit", though no explanation changed. A layer without a sink hashes as before the field
/// existed, so those checkpoints resume and are scored again.
fn layers_definition(layers: &[Layer]) -> String {
    let each: Vec<String> = layers
        .iter()
        .map(|layer| {
            let sink = layer.sink.map_or(String::new(), |group| format!(", sink: Some({group})"));
            let thresholds = if layer.thresholds.is_empty() { String::new() } else { format!(", thresholds: {:?}", layer.thresholds) };
            let components = if layer.components.is_empty() { String::new() } else { format!(", components: {:?}", layer.components) };
            format!("Layer {{ sites: {:?}, heads: {:?}, functions: {:?}{sink}{thresholds}{components} }}", layer.sites, layer.heads, layer.functions)
        })
        .collect();
    format!("[{}]", each.join(", "))
}

/// Read only the length-delimited JSON header. Neither identity checks nor restoration need a
/// second, checkpoint-sized byte buffer beside the posterior and its optimizer state.
fn checkpoint_header<T: serde::de::DeserializeOwned>(path: &Path) -> Result<(T, BufReader<std::fs::File>, u64), String> {
    let file = std::fs::File::open(path).map_err(error)?;
    let file_bytes = file.metadata().map_err(error)?.len();
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut prefix = [0_u8; 8];
    reader
        .read_exact(&mut prefix)
        .map_err(|e| if e.kind() == std::io::ErrorKind::UnexpectedEof { "a truncated checkpoint".to_string() } else { error(e) })?;
    let header_bytes = u64::from_le_bytes(prefix);
    let payload_start = header_bytes.checked_add(8).filter(|end| *end <= file_bytes).ok_or("a truncated checkpoint")?;
    // `take` prevents a malformed JSON header from consuming binary coefficient bytes. JSON
    // decoding allocates its actual fields, not the advertised header or payload byte count.
    let header = serde_json::from_reader((&mut reader).take(header_bytes)).map_err(error)?;
    Ok((header, reader, file_bytes - payload_start))
}

/// The checkpoint at `path`'s identity.
fn checkpoint_identity(path: &Path) -> Result<Identity, String> {
    #[derive(Deserialize)]
    struct Header {
        identity: Identity,
    }
    let (header, _, _): (Header, _, _) =
        checkpoint_header(path).map_err(|e| format!("{}: a checkpoint without this fit's identity: {e}", path.display()))?;
    Ok(header.identity)
}

/// Refuse a checkpoint at `path` of another fit than `identity`'s, naming every field that
/// differs; a path with no checkpoint passes. A driver calls this before it writes anything.
pub fn check_checkpoint(path: &Path, identity: &Identity) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    let found = checkpoint_identity(path)?;
    check_checkpoint_identity(path, &found, identity)
}

fn check_checkpoint_identity(path: &Path, found: &Identity, identity: &Identity) -> Result<(), String> {
    let differing: Vec<&str> = [
        ("export", found.export == identity.export),
        ("tokens", found.tokens == identity.tokens),
        ("program", found.program == identity.program),
        ("groups", found.groups == identity.groups),
        ("sharing", found.sharing == identity.sharing),
        ("definition", found.definition == identity.definition),
    ]
    .into_iter()
    .filter(|(_, same)| !same)
    .map(|(field, _)| field)
    .collect();
    if differing.is_empty() {
        Ok(())
    } else {
        Err(format!("{}: a checkpoint of another fit (its {} differ)", path.display(), differing.join(", ")))
    }
}

/// The arrays a checkpoint holds per trainable operator: the posterior's mean `μ̄`, `ln σ`, IVON's
/// state (the gradient's momentum and the curvature estimate) and IVON's iterate `μ` whose Polyak
/// average `μ̄` is, each little-endian in its [`Precision`].
const CHECKPOINT_ARRAYS: usize = 5;

/// The arrays per operator of a checkpoint written before the gradient's second moment was dropped
/// from IVON's state: the second moment after the curvature, read and discarded.
const LEGACY_ARRAYS: usize = 6;

/// The precision a checkpoint array is written in: the storage the device holds it in, so that
/// writing and reading it is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum Precision {
    F64,
    F32,
    /// bfloat16: the 16 high bits of the value's f32.
    Bf16,
}

impl Precision {
    fn of(storage: Storage) -> Self {
        match storage {
            Storage::F64 => Self::F64,
            Storage::F32 => Self::F32,
            Storage::Bf16 => Self::Bf16,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }

    /// Each of a payload's arrays per operator in its precision: those `precision` names
    /// ([`CHECKPOINT_ARRAYS`], or [`LEGACY_ARRAYS`]), or float64 in the legacy layout for a
    /// checkpoint that names none.
    fn of_payload(precision: Option<&[Self]>) -> Result<Vec<Self>, String> {
        match precision {
            None => Ok(vec![Self::F64; LEGACY_ARRAYS]),
            Some(arrays) if arrays.len() == CHECKPOINT_ARRAYS || arrays.len() == LEGACY_ARRAYS => Ok(arrays.to_vec()),
            Some(arrays) => Err(format!("a checkpoint of {} arrays per operator", arrays.len())),
        }
    }
}

/// The payload bytes of a checkpoint of operators of `shapes` with arrays in `precision`.
fn checkpoint_payload_bytes(shapes: &[(usize, usize)], precision: &[Precision]) -> Option<u64> {
    let per_cell = precision.iter().map(|p| p.bytes() as u64).sum::<u64>();
    shapes.iter().try_fold(0_u64, |total, &(rows, cols)| {
        let cells = u64::try_from(rows).ok()?.checked_mul(u64::try_from(cols).ok()?)?;
        total.checked_add(cells.checked_mul(per_cell)?)
    })
}

/// One operator's arrays of a checkpoint payload in `precision` ([`Precision::of_payload`]) from
/// `reader`: `mean` and `log_sd` in place, then IVON's momentum and curvature, and its iterate (a
/// legacy payload's gradient second moment read and dropped).
fn read_checkpoint_operator(reader: &mut impl Read, (mean, log_sd): (&mut Array2<f64>, &mut Array2<f64>), precision: &[Precision]) -> Result<([Array2<f64>; 2], Array2<f64>), String> {
    let dim = mean.dim();
    read_checkpoint_array(reader, mean, precision[0])?;
    read_checkpoint_array(reader, log_sd, precision[1])?;
    let mut moments: [Array2<f64>; 2] = std::array::from_fn(|_| Array2::zeros(dim));
    for (array, precision) in moments.iter_mut().zip(&precision[2..4]) {
        read_checkpoint_array(reader, array, *precision)?;
    }
    if precision.len() == LEGACY_ARRAYS {
        read_checkpoint_array(reader, &mut Array2::zeros(dim), precision[4])?;
    }
    let mut iterate = Array2::zeros(dim);
    read_checkpoint_array(reader, &mut iterate, precision[precision.len() - 1])?;
    Ok((moments, iterate))
}

/// Per trainable operator of `operators`, its momentum's bias correction as a checkpoint holds it:
/// its `weights`, else its legacy pair's `W` for every operator, else `β₁ = 0.9`'s after its
/// `steps` steps, the decay the fits that saved neither used.
fn saved_weights(weights: Option<Vec<f64>>, legacy: Option<(f64, f64)>, steps: u64, operators: usize) -> Vec<f64> {
    match (weights, legacy) {
        (Some(weights), _) => weights,
        (None, Some((w, _))) => vec![w; operators],
        (None, None) => vec![gam_gpu::tensor::PosteriorStep::constant_weight(0.9, steps); operators],
    }
}

/// Decode in fixed-size byte tiles directly into the destination, including nonstandard array
/// layouts. The wire order remains ndarray's logical iteration order used by the writer; every
/// precision widens exactly.
fn read_checkpoint_array(reader: &mut impl Read, array: &mut Array2<f64>, precision: Precision) -> Result<(), String> {
    let width = precision.bytes();
    let mut bytes = [0_u8; 64 * 1024];
    let mut remaining = array.len();
    let mut values = array.iter_mut();
    while remaining != 0 {
        let count = remaining.min(bytes.len() / width);
        reader.read_exact(&mut bytes[..count * width]).map_err(error)?;
        for (value, encoded) in values.by_ref().take(count).zip(bytes[..count * width].chunks_exact(width)) {
            *value = match precision {
                Precision::F64 => f64::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6], encoded[7]]),
                Precision::F32 => f64::from(f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]])),
                Precision::Bf16 => f64::from(f32::from_bits(u32::from(u16::from_le_bytes([encoded[0], encoded[1]])) << 16)),
            };
        }
        remaining -= count;
    }
    Ok(())
}

/// `array`'s values appended to `out` in `precision`, in ndarray's logical order; a value the
/// precision does not hold exactly (one the device could not have held) is refused.
fn write_checkpoint_array(out: &mut Vec<u8>, array: &Array2<f64>, precision: Precision) -> Result<(), String> {
    out.reserve(array.len() * precision.bytes());
    for &value in array {
        let narrow = value as f32;
        let exact = match precision {
            Precision::F64 => true,
            Precision::F32 => f64::from(narrow) == value || value.is_nan(),
            Precision::Bf16 => (f64::from(narrow) == value && narrow.to_bits() & 0xffff == 0) || value.is_nan(),
        };
        if !exact {
            return Err(format!("a checkpoint value {value:e} not held in {precision:?}"));
        }
        match precision {
            Precision::F64 => out.extend_from_slice(&value.to_le_bytes()),
            Precision::F32 => out.extend_from_slice(&narrow.to_le_bytes()),
            Precision::Bf16 => out.extend_from_slice(&((narrow.to_bits() >> 16) as u16).to_le_bytes()),
        }
    }
    Ok(())
}

/// A checkpoint streamed from the device to disk: the fit's thread takes one operator's arrays off
/// the device at a time and hands them to the writer thread ([`Writer`]), which writes them while
/// the next is taken and finishes the file while the fit goes on. At most two operators' arrays
/// are on the host, never the whole payload.
struct Snapshot;

impl Snapshot {
    /// Save `progress` and `posterior` as they are now to `path`; `progress` records the payload's
    /// precision. The write starts once the one before it has ended. With `link`, the written files
    /// are also linked there (`Snapshot::write`).
    fn save(progress: &mut Progress, posterior: &DevicePosterior, (path, link): (&Path, Option<&Path>), writer: &mut Writer) -> Result<(), String> {
        let precision = posterior.storages().map(Precision::of);
        progress.precision = Some(precision.to_vec());
        progress.averaged = posterior.averaged();
        progress.ratio = posterior.line_ratio();
        progress.slope = posterior.line_slope();
        progress.weights = Some(posterior.momentum_weights().to_vec());
        progress.momentum_weights = None;
        let bytes = checkpoint_payload_bytes(&progress.shapes, &precision).ok_or("a checkpoint too large to address")?;
        let (header, json) = (serde_json::to_vec(progress).map_err(error)?, serde_json::to_vec_pretty(progress).map_err(error)?);
        // One operator's arrays wait while the writer writes the one before.
        let (send, receive) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
        let (path, link) = (path.to_path_buf(), link.map(Path::to_path_buf));
        writer.start(move || Self::write(&path, &header, &json, receive, bytes, link.as_deref()))?;
        const STOPPED: &str = "the checkpoint writer stopped";
        let stream = move |chunk: Vec<u8>| -> Result<(), String> { send.send(chunk).map_err(|_| STOPPED.to_string()) };
        let taken = (|| -> Result<(), String> {
            for i in 0..progress.shapes.len() {
                let (mean, log_sd, [momentum, curvature]) = posterior.operator(i)?;
                let mut chunk = Vec::new();
                for (array, precision) in [&mean, &log_sd, &momentum, &curvature, &posterior.iterate(i)?].into_iter().zip(precision) {
                    write_checkpoint_array(&mut chunk, array, precision)?;
                }
                stream(chunk)?;
            }
            Ok(())
        })();
        // The payload ends here: the writer finishes the file, or refuses a payload that ends short
        // and leaves the last whole checkpoint in place. A writer that stopped first reports its
        // own failure.
        drop(stream);
        match taken {
            Ok(()) => Ok(()),
            Err(e) => Err(writer.wait().err().filter(|_| e == STOPPED).unwrap_or(e)),
        }
    }

    /// Write the checkpoint atomically: the progress `header` as JSON after its length, then the
    /// payload's `bytes` as they arrive. The progress alone also goes to the path with extension
    /// `json`, readable while the fit runs. With `link`, both files are then linked there (copied
    /// where the file system refuses a link), each replacing the one before atomically; a later
    /// checkpoint replaces `path`'s directory entry, never the linked file.
    fn write(path: &Path, header: &[u8], json: &[u8], payload: std::sync::mpsc::Receiver<Vec<u8>>, bytes: u64, link: Option<&Path>) -> Result<(), String> {
        let partial = path.with_extension("partial");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&partial).map_err(error)?);
        file.write_all(&(header.len() as u64).to_le_bytes()).map_err(error)?;
        file.write_all(header).map_err(error)?;
        let mut written = 0u64;
        for chunk in payload {
            file.write_all(&chunk).map_err(error)?;
            written += chunk.len() as u64;
        }
        if written != bytes {
            drop(file);
            std::fs::remove_file(&partial).map_err(error)?;
            return Err(format!("a checkpoint payload of {written} bytes where {bytes} were due"));
        }
        file.into_inner().map_err(error)?.sync_all().map_err(error)?;
        std::fs::rename(&partial, path).map_err(error)?;
        let partial = path.with_extension("json.partial");
        std::fs::write(&partial, json).map_err(error)?;
        std::fs::rename(&partial, path.with_extension("json")).map_err(error)?;
        if let Some(link) = link {
            for (from, to) in [(path.to_path_buf(), link.to_path_buf()), (path.with_extension("json"), link.with_extension("json"))] {
                let linking = to.with_extension("linking");
                if linking.exists() {
                    std::fs::remove_file(&linking).map_err(error)?;
                }
                if std::fs::hard_link(&from, &linking).is_err() {
                    std::fs::copy(&from, &linking).map_err(error)?;
                }
                std::fs::rename(&linking, &to).map_err(error)?;
            }
        }
        Ok(())
    }
}

/// The thread writing a fit's last save: at most one write is in flight, and a save first waits
/// for the one before it, so the files on disk are always one whole save.
#[derive(Default)]
struct Writer {
    pending: Option<JoinHandle<Result<(), String>>>,
}

impl Writer {
    /// Wait for the write in flight, with its outcome.
    fn wait(&mut self) -> Result<(), String> {
        match self.pending.take() {
            Some(job) => job.join().map_err(|_| "the checkpoint writer panicked".to_string())?,
            None => Ok(()),
        }
    }

    /// Start `job` once the write in flight is done.
    fn start(&mut self, job: impl FnOnce() -> Result<(), String> + Send + 'static) -> Result<(), String> {
        self.wait()?;
        self.pending = Some(std::thread::spawn(job));
        Ok(())
    }
}

impl Drop for Writer {
    /// A fit that stops early still finishes the write it started.
    fn drop(&mut self) {
        if let Err(e) = self.wait() {
            log::error!("library checkpoint: {e}");
        }
    }
}

/// Restore a checkpoint of this fit into `posterior`, with IVON's state per operator (the momentum
/// and the curvature) and its iterate, or refuse one of another fit.
fn load_checkpoint(path: &Path, expected: &Progress, posterior: &mut Posterior) -> Result<(Progress, Vec<[Array2<f64>; 2]>, Vec<Array2<f64>>, Vec<Array2<f64>>), String> {
    let (progress, mut reader, payload_bytes): (Progress, _, _) = checkpoint_header(path)?;
    check_checkpoint_identity(path, &progress.identity, &expected.identity)?;
    if progress.collection != expected.collection {
        return Err(format!(
            "{}: a checkpoint of experiment collection {}, where this fit draws collection {} (every base's source and experiments from its own draws, which the batch size does not change): it was trained on other experiments",
            path.display(),
            progress.collection,
            expected.collection
        ));
    }
    let same_settings =
        serde_json::to_value(&progress.settings).map_err(error)? == serde_json::to_value(&expected.settings).map_err(error)?;
    if !same_settings
        || progress.tokens != expected.tokens
        || progress.shapes != expected.shapes
        || progress.active.len() != expected.active.len()
    {
        return Err(format!("{}: a checkpoint of another fit", path.display()));
    }
    let precision = Precision::of_payload(progress.precision.as_deref()).map_err(|e| format!("{}: {e}", path.display()))?;
    if checkpoint_payload_bytes(&progress.shapes, &precision) != Some(payload_bytes)
        || posterior.mean.len() != progress.shapes.len()
        || posterior.log_sd.len() != progress.shapes.len()
        || posterior.mean.iter().zip(&progress.shapes).any(|(array, shape)| array.dim() != *shape)
        || posterior.log_sd.iter().zip(&progress.shapes).any(|(array, shape)| array.dim() != *shape)
    {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    let (mut moments, mut iterates) = (Vec::with_capacity(posterior.mean.len()), Vec::with_capacity(posterior.mean.len()));
    for i in 0..posterior.mean.len() {
        let (state, iterate) = read_checkpoint_operator(&mut reader, (&mut *posterior.mean[i], &mut *posterior.log_sd[i]), &precision)?;
        moments.push(state);
        iterates.push(iterate);
    }
    if reader.read(&mut [0_u8; 1]).map_err(error)? != 0 {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    let held = posterior.mean.clone();
    posterior.active = progress.active.clone();
    Ok((progress, moments, held.into_iter().map(Shared::into_array).collect(), iterates))
}

/// A checkpoint of `explanation` read whole: the posterior (means, log standard deviations and
/// active groups) and the start it holds
/// ([`checkpoint_start`]); a checkpoint whose identity names another explanation is refused.
fn read_checkpoint(explanation: &Explanation, path: &Path) -> Result<(Posterior, Start), String> {
    #[derive(Deserialize)]
    struct Header {
        identity: Identity,
        tokens: usize,
        shapes: Vec<(usize, usize)>,
        active: Vec<bool>,
        #[serde(default)]
        precision: Option<Vec<Precision>>,
        step: i32,
        epoch: usize,
        #[serde(default)]
        averaged: u64,
        #[serde(default = "unit_ratio")]
        ratio: (f64, u64),
        #[serde(default)]
        slope: (f64, f64, u64),
        #[serde(default)]
        weights: Option<Vec<f64>>,
        #[serde(default)]
        momentum_weights: Option<(f64, f64)>,
        #[serde(default)]
        prior: Option<serde_json::Value>,
    }
    let (header, mut reader, payload_bytes): (Header, _, _) = checkpoint_header(path)?;
    // The checkpoint must be a fit of this explanation: the same groups, shared parameters,
    // starting values and correspondence to `M` (the identity's explanation-only fields).
    let (groups, sharing, definition) = explanation_identity(explanation);
    let differing: Vec<&str> = [("groups", header.identity.groups == groups), ("sharing", header.identity.sharing == sharing), ("definition", header.identity.definition == definition)]
        .into_iter()
        .filter(|(_, same)| !same)
        .map(|(field, _)| field)
        .collect();
    if !differing.is_empty() {
        return Err(format!("{}: a checkpoint of another explanation (its {} differ)", path.display(), differing.join(", ")));
    }
    let precision = Precision::of_payload(header.precision.as_deref()).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut posterior = Posterior::new(explanation, header.tokens)?;
    if header.shapes != posterior.mean.iter().map(|m| m.dim()).collect::<Vec<_>>() || header.active.len() != posterior.active.len() {
        return Err(format!("{}: a checkpoint of another explanation", path.display()));
    }
    if checkpoint_payload_bytes(&header.shapes, &precision) != Some(payload_bytes) {
        return Err(format!("{}: a checkpoint of the wrong size", path.display()));
    }
    let (mut moments, mut iterate) = (Vec::with_capacity(header.shapes.len()), Vec::with_capacity(header.shapes.len()));
    for i in 0..header.shapes.len() {
        let (state, values) = read_checkpoint_operator(&mut reader, (&mut *posterior.mean[i], &mut *posterior.log_sd[i]), &precision)?;
        moments.push(state);
        iterate.push(values);
    }
    let held = posterior.mean.clone();
    posterior.active = header.active;
    let steps = u64::try_from(header.step).map_err(error)?;
    let weights = saved_weights(header.weights, header.momentum_weights, steps, header.shapes.len());
    let start = Start {
        mean: held.into_iter().map(Shared::into_array).collect(),
        log_sd: posterior.log_sd.iter().map(|s| (**s).clone()).collect(),
        active: posterior.active.clone(),
        state: Some(Optimizer { moments, iterate, averaged: header.averaged, steps, weights, ratio: header.ratio, slope: header.slope, prior: header.prior }),
        epoch: header.epoch,
    };
    Ok((posterior, start))
}

/// The posterior of a fit checkpoint of `explanation` (`OUT/checkpoint.bin`, [`fit`]): its means,
/// log standard deviations and active groups, for reading a fit that is still running.
pub fn checkpoint_posterior(explanation: &Explanation, path: &Path) -> Result<Posterior, String> {
    read_checkpoint(explanation, path).map(|(posterior, _)| posterior)
}

/// The posterior-mean artifact ([`posterior_mean`]) of a fit checkpoint of `explanation` at
/// `path`, with `literals` (those of the fit's device, [`Literals::of`]): the explanation a
/// running fit holds, read from its last save.
pub fn checkpoint_artifact(explanation: &Explanation, path: &Path, literals: Literals) -> Result<Artifact, String> {
    let mut artifact = literals.apply(posterior_mean(explanation, &checkpoint_posterior(explanation, path)?)?)?;
    // The shared stages' assignments as an evaluation holds them: each component on its gate.
    if let Some(saved) = saved_assignments(path)? {
        if saved.len() != explanation.shares.len() {
            return Err(format!("{}: assignments of other shared stages", path.display()));
        }
        for (share, logits) in explanation.shares.iter().zip(saved) {
            let mut a = Assignment::of(&artifact.program, share)?;
            if logits.len() != a.logits.len() {
                return Err(format!("{}: an assignment of another stage", path.display()));
            }
            a.logits = logits;
            let old = std::sync::Arc::clone(&artifact.program.operators[a.operator]);
            let values = a.values(Relaxation::Hard);
            let precision = exact_precision(values.iter().copied()).map_err(error)?;
            artifact.program.operators[a.operator] = std::sync::Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).map_err(error)?);
        }
    }
    Ok(artifact)
}

/// The frames a checkpoint at `path` keeps, when it keeps any.
fn saved_mixings(path: &Path) -> Result<Option<Vec<Vec<Vec<f64>>>>, String> {
    #[derive(Deserialize)]
    struct Saved {
        #[serde(default)]
        frames: Option<Vec<Vec<Vec<f64>>>>,
    }
    let (saved, _, _): (Saved, _, _) = checkpoint_header(path)?;
    Ok(saved.frames)
}

/// The shared stages' assignment logits a checkpoint at `path` keeps, when it keeps any.
fn saved_assignments(path: &Path) -> Result<Option<Vec<Vec<Vec<f64>>>>, String> {
    #[derive(Deserialize)]
    struct Saved {
        #[serde(default)]
        assignments: Option<Vec<Vec<Vec<f64>>>>,
    }
    let (saved, _, _): (Saved, _, _) = checkpoint_header(path)?;
    Ok(saved.assignments)
}

/// IVON's state a [`Start`] continues (a checkpoint's): per trainable operator (in
/// `Explanation::trainable` order) the gradient's momentum and the curvature estimate; IVON's
/// iterate `μ` (as the device holds it) whose Polyak average the start's means are, and the steps
/// `averaged` that average spans; the steps taken; per operator the momentum's bias correction `W`
/// (`gam_gpu::tensor::PosteriorStep::weight`); the line step's curvature ratio `ratio`
/// ([`DevicePosterior::line_ratio`]) and slope averages `slope` ([`DevicePosterior::line_slope`]),
/// each with its draws; and the prior term's state ([`PriorTerm::save`]), when the fit had one.
#[derive(Debug, PartialEq)]
pub struct Optimizer {
    pub moments: Vec<[Array2<f64>; 2]>,
    pub iterate: Vec<Array2<f64>>,
    pub averaged: u64,
    pub steps: u64,
    pub weights: Vec<f64>,
    pub ratio: (f64, u64),
    pub slope: (f64, f64, u64),
    pub prior: Option<serde_json::Value>,
}

/// Where [`fit_from`] starts in place of `M`: per trainable operator (in `Explanation::trainable`
/// order) the posterior means and log standard deviations, which prior groups are in the
/// explanation (a removed group's entries are zeroed), and, to continue an optimizer, IVON's state
/// with the steps taken ([`Optimizer`]). Without a state the fit's step count starts at zero, and
/// it makes the Laplace start (`laplace_start`) from the start's means: its standard deviations
/// and IVON's curvature come from one pass at a sample of the given posterior. `epoch` is the next
/// epoch, whose batches' weight noise the fit draws.
pub struct Start {
    pub mean: Vec<Array2<f64>>,
    pub log_sd: Vec<Array2<f64>>,
    pub active: Vec<bool>,
    pub state: Option<Optimizer>,
    pub epoch: usize,
}

/// The start a fit checkpoint of `explanation` holds: its posterior, IVON's state with the steps
/// taken, the line step's averages, the prior term's state and the next epoch. [`fit_from`]
/// continues from it exactly as resuming the checkpoint would, with the convergence test begun
/// afresh.
pub fn checkpoint_start(explanation: &Explanation, path: &Path) -> Result<Start, String> {
    read_checkpoint(explanation, path).map(|(_, start)| start)
}

/// Fit the library explanation of `native` to its interchange experiments on the training
/// `sequences`, evaluating on the `held_out` sequences after every epoch (module note). `native` is
/// the split native program the explanation was built from. With `checkpoint`, the fit is saved
/// there after every epoch and removal step, and resumed from it when it exists.
pub fn fit(
    device: &Device,
    native: &OperatorProgram,
    explanation: &Explanation,
    sequences: &[Vec<u32>],
    held: &[Vec<u32>],
    settings: &Settings,
    export: &str,
    checkpoint: Option<&Path>,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<Fit, String> {
    fit_from(device, native, explanation, sequences, held, settings, export, checkpoint, prior, None)
}

/// [`fit`] from `start` in place of `M` when one is given (a checkpoint's [`checkpoint_start`], a
/// posterior whose groups no checkpoint has, or a start set from the curvature). A checkpoint of
/// this fit, when one exists, is resumed and `start` is not used; one of this fit's problem from an
/// older experiment collection is the start instead.
pub fn fit_from(
    device: &Device,
    native: &OperatorProgram,
    explanation: &Explanation,
    sequences: &[Vec<u32>],
    held: &[Vec<u32>],
    settings: &Settings,
    export: &str,
    checkpoint: Option<&Path>,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
    start: Option<Start>,
) -> Result<Fit, String> {
    settings.validate()?;
    let mut prior = prior;
    let length = sequences.first().map_or(0, Vec::len);
    if length == 0 || held.len() < 2 || sequences.iter().chain(held).any(|s| s.len() != length) {
        return Err("training and held-out sequences (at least two) must be nonempty and of one length".into());
    }
    let started = Instant::now();
    let draws = draws(sequences.len(), settings.batch_sequences, settings.seed)?;
    if draws.len() < 2 {
        return Err("the convergence test needs at least two training batches".into());
    }
    let mut scorer = Scorer::new(device, native, explanation, settings)?.prepared(native, sequences, settings)?;
    // The fixed collection: its scored tokens N and its realized families.
    let (mut tokens, mut families) = (0, BTreeMap::new());
    for draw in &draws {
        let experiments = scorer.experiments(draw, sequences)?;
        tokens += experiments.iter().map(|e| length - e.position).sum::<usize>();
        for (family, count) in interchange::census(&experiments, scorer.experiments.variables()) {
            *families.entry(family.to_string()).or_insert(0) += count;
        }
    }
    log::info!("library training collection: {tokens} scored tokens, families {families:?}");
    scorer.mix_tokens = tokens as f64;
    let mut posterior = Posterior::new(explanation, tokens)?;
    let parameters = posterior.mean.iter().map(|m| m.len()).sum();
    let mut progress = Progress {
        identity: identity(export, native, explanation, sequences, held),
        settings: settings.clone(),
        tokens,
        shapes: posterior.mean.iter().map(|m| m.dim()).collect(),
        start: None,
        epoch: 0,
        step: 0,
        averaged: 0,
        ratio: unit_ratio(),
        slope: (0.0, 0.0, 0),
        weights: None,
        momentum_weights: None,
        best: None,
        epochs: Vec::new(),
        removals: Vec::new(),
        multiplier: 0.0,
        engaged: false,
        previous: None,
        collection: COLLECTION,
        active: posterior.active.clone(),
        done: false,
        seconds: 0.0,
        training_seconds: 0.0,
        evaluation_seconds: 0.0,
        full_seconds: 0.0,
        prior: None,
        precision: None,
        assignments: None,
        frames: None,
    };
    // The fixed held-out subset: the first batch of held-out bases (at least the two a source
    // needs), with `M`'s targets for its experiments made once for every evaluation of it.
    let subset = &held[..settings.batch_sequences.clamp(2, held.len())];
    let subset_batches = held_batches(&scorer, subset, settings)?;
    // A checkpoint at `checkpoint` of this fit's problem (its identity) drawn from an older
    // experiment collection ([`COLLECTION`]) was trained on other experiments, so it is not
    // resumed: its posterior and IVON's state start this fit as its [`checkpoint_start`] would, in
    // place of `start`. The file stays beside it as `<name>.collection<c>.bin`, a hard link the
    // fit's first save leaves (a save replaces `checkpoint`'s directory entry), and a fit stopped
    // before that save starts from it again.
    let mut start = start;
    let mut warm = false;
    if let Some(path) = checkpoint.filter(|p| p.exists()) {
        #[derive(Deserialize)]
        struct Header {
            identity: Identity,
            #[serde(default)]
            collection: u32,
        }
        let (header, _, _): (Header, _, _) = checkpoint_header(path)?;
        if header.collection != COLLECTION {
            check_checkpoint_identity(path, &header.identity, &progress.identity)?;
            let aside = path.with_extension(format!("collection{}.bin", header.collection));
            if !aside.exists() {
                std::fs::hard_link(path, &aside).or_else(|_| std::fs::copy(path, &aside).map(|_| ())).map_err(error)?;
            }
            start = Some(read_checkpoint(explanation, path)?.1);
            warm = true;
            log::info!("library fit started from {} (experiment collection {}, kept as {})", path.display(), header.collection, aside.display());
        }
    }
    // IVON's state the fit continues, a start's or a checkpoint's, and the posterior's means the
    // device holds with it (the Polyak average of the state's iterate).
    let (mut resumed, mut held_means): (Option<Optimizer>, Option<Vec<Array2<f64>>>) = (None, None);
    if let Some(start) = start {
        let shapes: Vec<(usize, usize)> = posterior.mean.iter().map(|m| m.dim()).collect();
        let fits = |arrays: &[Array2<f64>]| arrays.iter().map(Array2::dim).eq(shapes.iter().copied());
        let state_fits = start.state.as_ref().is_none_or(|state| {
            state.moments.len() == shapes.len() && state.moments.iter().zip(&shapes).all(|(m, d)| m.iter().all(|a| a.dim() == *d)) && fits(&state.iterate) && state.weights.len() == shapes.len()
        });
        if !fits(&start.mean) || !fits(&start.log_sd) || start.active.len() != posterior.active.len() || !state_fits {
            return Err("a start of another explanation".into());
        }
        // One clock: the steps a state was taken over, or none without one.
        if let Some(state) = &start.state {
            held_means = Some(start.mean.clone());
            progress.step = i32::try_from(state.steps).map_err(error)?;
            progress.prior = state.prior.clone();
        }
        posterior.mean = start.mean.into_iter().map(Shared::from).collect();
        posterior.log_sd = start.log_sd.into_iter().map(Shared::from).collect();
        posterior.remove(&(0..start.active.len()).filter(|g| !start.active[*g]).collect::<Vec<_>>());
        progress.active = posterior.active.clone();
        progress.epoch = start.epoch;
        resumed = start.state;
    }
    if let Some(path) = checkpoint.filter(|p| p.exists() && !warm) {
        let (loaded, moments, held, iterate) = load_checkpoint(path, &progress, &mut posterior)?;
        held_means = Some(held);
        progress = loaded;
        if let Some(saved) = &progress.assignments {
            scorer.restore_assignments(saved)?;
        }
        if let Some(saved) = &progress.frames {
            scorer.restore_mixings(saved)?;
        }
        match (prior.as_deref_mut(), &progress.prior) {
            (Some(prior), Some(state)) => prior.load(state)?,
            (None, None) => {}
            _ => return Err(format!("{}: a checkpoint of a fit with another prior term", path.display())),
        }
        let steps = u64::try_from(progress.step).map_err(error)?;
        let weights = saved_weights(progress.weights.clone(), progress.momentum_weights, steps, progress.shapes.len());
        resumed = Some(Optimizer { moments, iterate, averaged: progress.averaged, steps, weights, ratio: progress.ratio, slope: progress.slope, prior: progress.prior.clone() });
        log::info!("library fit resumed at epoch {} from {}", progress.epoch, path.display());
    } else if let Some(state) = resumed.as_ref().and_then(|resumed| resumed.prior.as_ref()) {
        // A start's prior state: the fit's prior term continues from it, as from its checkpoint.
        match prior.as_deref_mut() {
            Some(prior) => prior.load(state)?,
            None => return Err("a start of a fit with a prior term, continued without one".into()),
        }
    }
    let resumed_seconds = progress.seconds;
    let fresh = resumed.is_none();
    let state = resumed.as_ref().map(|resumed| State::Saved { moments: &resumed.moments, weights: &resumed.weights });
    let mut device_posterior = DevicePosterior::new(device, explanation, &posterior, tokens as f64, state, u64::try_from(progress.step).map_err(error)?)?;
    // A resumed fit holds exactly the device's means of the checkpoint, the iterate they average
    // with the steps they span, and the line step's averages: it goes on as the fit that was not
    // stopped.
    if let (Some(resumed), Some(held)) = (resumed.take(), held_means.take()) {
        device_posterior.restore(&held, &resumed.iterate, resumed.averaged)?;
        device_posterior.set_line_ratio(resumed.ratio);
        device_posterior.set_line_slope(resumed.slope);
    }
    if fresh {
        let timed = Instant::now();
        let sums = laplace_sums(&mut scorer, &device_posterior, &draws, sequences, settings)?;
        // The unit-information start's state goes before the Laplace start's is made, which
        // takes its deviations and curvature one operator at a time.
        drop(device_posterior);
        device_posterior = DevicePosterior::new(device, explanation, &posterior, tokens as f64, Some(State::Zero), 0)?;
        laplace_start(&scorer, explanation, &mut posterior, sums, tokens, |i, log_sd, h| device_posterior.set_start(i, log_sd, h))?;
        device_posterior.settle()?;
        log::info!("library Laplace start: {:.1} s", timed.elapsed().as_secs_f64());
    }
    // The frames' slices at their start (the extra slices' reads, `FRAME_START`), or at a resumed
    // fit's frames, where its means already are.
    scorer.pin_mixings(&mut device_posterior)?;
    // The curvature estimate `h` estimates the Gauss–Newton diagonal per token of the whole
    // training collection, the mean of its `B` batches' diagonals. `β₂ = 1 − 1/B` makes its running
    // average span about one pass, so each batch weighs about once whatever the batch size. One
    // factor draw per batch estimates its entry of the batch's diagonal with relative variance at
    // most 2 under a random-sign probe, so the pass's average of `B` independent draws has relative
    // variance about `2 / B` per entry: the batch size sets how many draws the average holds, not
    // what it estimates. A gain `1 − β₂` set per operator each epoch from the Kalman gain of a
    // local-level model of the draws (its noise `R` and drift `Q` measured from the innovations'
    // lag-0 and lag-1 moments, 6d3f4b137d) lost its paired A/B: vpd4l's one-block fit, held-out F
    // after epochs 0–2, 4.189, 3.518, 3.013 and 4.560, 3.731, 3.288 bits per token (seeds 1 and 2)
    // against 4.189, 3.408, 2.854 and 4.560, 3.662, 3.038 at equal time (MATS audit-b2-*). It
    // measured `R/h² ≈ 1.4–2.5` but `Q/h²` at or near 0, so it averaged over far more draws than an
    // epoch and the curvature lagged the moving posterior.
    let ivon = Ivon { beta1: MOMENTUM_DECAY, beta2: 1.0 - 1.0 / draws.len() as f64 };
    // The fit's thread streams the checkpoint off the device to the writer thread, which finishes
    // writing it while the device trains on ([`Snapshot`]). The current explanation is read from
    // the checkpoint where it is wanted ([`checkpoint_artifact`]): a posterior-mean artifact made
    // at every save held the means, their group map and the encoded artifact on the host beside
    // the checkpoint, about 28 bytes per trainable parameter that the fit never reads back.
    let mut writer = Writer::default();
    let save = |progress: &mut Progress, posterior: &Posterior, (device_posterior, link): (&DevicePosterior, Option<&Path>), writer: &mut Writer| -> Result<(), String> {
        progress.active = posterior.active.clone();
        progress.seconds = resumed_seconds + started.elapsed().as_secs_f64();
        let Some(path) = checkpoint else { return Ok(()) };
        Snapshot::save(progress, device_posterior, (path, link), writer)
    };
    if let Some(prior) = prior.as_deref_mut()
        && progress.prior.is_none()
    {
        prior.epoch(explanation, &posterior)?;
    }
    if progress.start.is_none() {
        // The start obeys the budget too: the subset now; the full set's time, until a full
        // evaluation is made, estimated from the subset's in proportion to the sequences.
        let timed = Instant::now();
        let start = held_out_on(&mut scorer, explanation, (&posterior, &device_posterior), (subset, &subset_batches), settings, tokens, prior.as_deref_mut())?;
        let seconds = timed.elapsed().as_secs_f64();
        progress.evaluation_seconds += seconds;
        progress.full_seconds = seconds * held.len() as f64 / subset.len() as f64;
        log::info!("library start: {start:?}");
        progress.start = Some(start);
        device_posterior.values_into(&mut posterior)?;
        progress.prior = prior.as_deref().map(PriorTerm::save).transpose()?;
        // The start's own checkpoint beside the fit's (extension `start.bin`), kept whatever the
        // descent does next: a valid posterior to begin other fits or a removal round from. It is
        // the fit's first checkpoint, linked there.
        let start = checkpoint.map(|path| path.with_extension("start.bin"));
        save(&mut progress, &posterior, (&device_posterior, start.as_deref()), &mut writer)?;
    }
    // The posterior at the end of the epoch with the lowest snapshot estimate of `F` since the
    // objective last changed (a start or a removal), with that estimate.
    // The best epoch's posterior, written whole where the removal round can read it back: beside
    // the fit's checkpoint, or for a fit without one a file of its own, removed at the end.
    static FITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    // The best epoch's snapshot evaluation, the removal round's start (not checkpointed: a resumed
    // fit scores it again).
    let mut best_evaluation: Option<Evaluation> = None;
    // A budget is held by projection after every step ([`project`]) on gated components' thresholds
    // and ReLU functions' gate biases; an explanation with neither (functions whose gates have no
    // bias, the tiny decoder's) keeps the multiplier's integral rule.
    let projected = scorer.stages.iter().any(|s| !s.is_empty()) || scorer.mlps.iter().flatten().any(|m| m.law == Law::Relu && m.up.is_none() && m.gate.bias.is_some());
    // Whether the last projection bound (the count at its target), where λ steers, and the first
    // projected step's count and step, from which the target eases to `K`.
    let mut bound = false;
    let mut eased: Option<(f64, i32)> = None;
    let best_path = checkpoint.map_or_else(
        || std::env::temp_dir().join(format!("library_best_{}_{}.bin", std::process::id(), FITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed))),
        |path| path.with_extension("best.bin"),
    );
    while !progress.done {
        let epoch = progress.epoch;
        let epoch_started = Instant::now();
        if let Some(prior) = prior.as_deref_mut()
            && epoch > 0
        {
            prior.epoch(explanation, &posterior)?;
        }
        // The steps' data terms at their samples around the iterate: the descent's record, no
        // estimate of `F` (`Epoch::data_bits`); the snapshot below scores `F`.
        let mut data_sum = 0.0;
        let (mut clean, mut patched) = (Mean::default(), Mean::default());
        // The steps' expected parts executed per token and their count (`Settings::budget`).
        let mut parts = (0.0, 0usize);
        // With a projected budget, the hard counts before each step's projection.
        let mut hard = (0.0, 0usize);
        for (b, draw) in draws.iter().enumerate() {
            let step_started = Instant::now();
            scorer.next_batch();
            let batch = draw.batch(sequences)?;
            let experiments = scorer.experiments(draw, sequences)?;
            let key = training_key(settings.seed, epoch, b);
            scorer.clear_assignment_gradients();
            scorer.clear_mixing_gradients();
            // The step's data terms, with its all-on experiment (`Scorer::terms`): the all-on bits join
            // the data term, and its gradient and factor the step's (`Terms::combine`).
            let (experiments, mut terms) = antithetic_step(&mut scorer, (device, &device_posterior), &batch, experiments, key)?;
            terms.combine(device)?;
            let Terms { bits, gradient: mut gradients, factor, all_on } = terms;
            let factor = factor.ok_or("no Gauss–Newton factor")?;
            for (e, bits) in experiments.iter().zip(&bits) {
                if e.patch.is_some() { patched.add(bits) } else { clean.add(bits) }
            }
            let scored = bits.iter().map(Vec::len).sum::<usize>();
            let (scale, weight) = batch_weights(draws.len(), tokens);
            data_sum += scale * LN_2 * bits.iter().flatten().sum::<f64>();
            let on_note = all_on.as_ref().map_or(String::new(), |on| {
                let tokens: usize = on.bits.iter().map(Vec::len).sum();
                format!(", all on {:.4} bits per token", on.bits.iter().flatten().sum::<f64>() / tokens.max(1) as f64)
            });
            if let Some(on) = &all_on {
                data_sum += scale * LN_2 * on.bits.iter().flatten().sum::<f64>();
            }
            // The prior term at the step's two antithetic samples, around the iterate as the data
            // step's (`PriorTerm::sample_device`), and its seconds: its value and gradient averaged
            // over the pair, the gradient joining the data term's (which the step weighs by `scale`
            // in nats), and the antithetic Stein estimate of its diagonal curvature per token
            // (`DevicePosterior::stein_curvature`) joining the step's curvature estimate. The twin
            // is evaluated first and without learning, so both see the prior's parameters before
            // the step's one step of them.
            let mut prior_seconds = 0.0;
            let mut prior_curvature = BTreeMap::new();
            if let Some(prior) = prior.as_deref_mut() {
                let timed = Instant::now();
                let (_, twin) = prior.sample_device(device, &device_posterior, &mut posterior, key ^ gam_gpu::tensor::ANTITHETIC, false)?;
                let (_, own) = prior.sample_device(device, &device_posterior, &mut posterior, key, true)?;
                let mut reached: Vec<usize> = own.keys().chain(twin.keys()).copied().collect();
                reached.sort_unstable();
                reached.dedup();
                for i in reached {
                    let op = explanation.trainable[i];
                    let (plus, minus) = (own.get(&i), twin.get(&i));
                    prior_curvature.insert(op, device_posterior.stein_curvature(i, (plus, minus), key)?);
                    for g in [plus, minus].into_iter().flatten() {
                        match gradients.get_mut(&op) {
                            Some(total) => device.axpy(total, 0.5 / (scale * LN_2), g).map_err(error)?,
                            None => {
                                let mut scaled = device.zeros(g.rows(), g.cols()).map_err(error)?;
                                device.axpy(&mut scaled, 0.5 / (scale * LN_2), g).map_err(error)?;
                                gradients.insert(op, scaled);
                            }
                        }
                    }
                }
                prior_seconds = timed.elapsed().as_secs_f64();
            }
            // The execution budget (`Settings::budget`, `library_complexity`): the step descends
            // the Lagrangian `F + λ (E_q[k] − K)`. `λ ∂Ê/∂μ` joins the data gradient (in its units,
            // as the prior term's does), and by Price's theorem `∂E_q[f]/∂σ² = ½ E_q[∂²f]` the
            // term's expected curvature per token `2 λ ∂Ê/∂σ² / N` joins the step's curvature.
            // `λ̂ = max(0, −⟨g_F, g_k⟩) / |g_k|²` (`g_k = ∂Ê/∂μ`, both measured on the step) is the
            // multiplier at which the term's gradient cancels F's push on the count: descending
            // `g_F` raises `Ê` iff `⟨g_F, g_k⟩ < 0`, and where F lowers the count on its own no
            // multiplier is needed. `λ̄` is its running mean over one pass of the `B` batches (the
            // plain mean of the steps since the budget bound, until there are `B` of them: one
            // step's `λ̂` is often 0 and swings 40-fold from step to step). From
            // the first step that finds `Ê > K` on, `λ = λ̄ Ê / K`, applied on the step itself: above
            // the budget the term outweighs F's push and the count falls, below it F's push wins
            // and the count rises, and the only point where the descent holds the count still
            // (`λ = λ̂`) is `Ê = K`. A budget that never binds leaves `λ = 0`, the step bit for bit
            // the budget-free one. The multiplier follows the count at once and holds no memory of
            // past violations. Integral rules lagged: `λ ← λ + λ̂₀ (Ê − K)/(K B)` (3732bc2c8d) rose
            // to 4.5e7 nats per part per token on toys' resid_mlp_2l (per-slice own gates,
            // K = 110), and with the count at 84 it had fallen only to 4.2e7 while the posterior
            // mean's KL went from 1e-24 to 1.4e5 bits per token; re-measuring `λ̂` in its rate
            // (3a279507d3) moved it from 69 to 2,813 within three steps of a tiny vpd4l fit.
            let mut parts_note = String::new();
            let budget = match settings.budget.filter(|k| k.is_finite()) {
                Some(limit) => Some((limit, complexity_terms(&mut scorer, &device_posterior, explanation, &posterior.active, &batch, (key, false))?)),
                None => None,
            };
            // The previous step's move stands only if this batch's measured objective is lower at
            // it (`step_accepted`); a rejected move is undone and this batch takes no step.
            if device_posterior.pending_divergence().is_some() {
                let new = (bits.as_slice(), budget.as_ref().map(|(_, (expected, _))| *expected));
                let lambda = progress.multiplier;
                let predicted = device_posterior.pending_predicted().unwrap_or(0.0);
                let (accepted, change, standard_error) = step_accepted(&mut scorer, &device_posterior, explanation, &posterior.active, (&batch, &experiments, key), new, all_on.as_ref(), (scale, tokens, lambda))?;
                // The trust region's ratio test, where the batch can test the model: the predicted
                // decrease beyond the measurement's standard error. Below it a single batch's ratio is
                // its noise over the prediction: tested on every resolved change, the factor fell to
                // 7e-9 within 276 steps (decomp-vpd4l-i, 4b76a52dcc), quartered on each chance rise.
                if predicted > standard_error {
                    device_posterior.trust_update(-change / predicted);
                }
                if accepted {
                    device_posterior.accept();
                    scorer.accept_assignments();
                    scorer.accept_mixings();
                } else {
                    device_posterior.revert()?;
                    scorer.revert_assignments();
                    scorer.revert_mixings();
                    scorer.pin_mixings(&mut device_posterior)?;
                    progress.step += 1;
                    log::info!("library step {epoch}.{b}: the last move rejected (its change of the objective on this batch {change:.4e} ± {standard_error:.2e} nats per token), undone; no step on this batch");
                    continue;
                }
            }
            if let Some((limit, (expected, terms))) = budget {
                parts.0 += expected;
                parts.1 += 1;
                // The multiplier: from the first step whose count exceeds the budget, λ starts at
                // the robust balance of F's push on the gates against the count's, the median over
                // the gates of F's push against the count, max(0, −∂F/∂z_b ∂Ê/∂z_b) / (∂Ê/∂z_b)², each
                // gate weighted by (∂Ê/∂z_b)² (a gate whose count does not move says nothing of the
                // balance, and one F already pushes off needs no pull: the median of the unsigned,
                // unweighted |∂F/∂z_b| / |∂Ê/∂z_b| started λ near 400 on the tiny decoder at B = 128,
                // the slope-weighted one near 300, where the count holds `K` near 4; read off each
                // gate's row of its parameters: a
                // threshold's or bias's entry, whose derivative is its pre-activation's, or the norms
                // of a gate's weight row, whose derivative is its pre-activation's times its input),
                // and then integrates the violation in log space, log λ ← log λ + (Ê − K) / (K H) per
                // step: symmetric, and still only where Ê = K. `H` is the slower of the step's two
                // averages, one pass of `B` batches (the curvature's, β₂ = 1 − 1/B) and the
                // momentum's `1 / (1 − β₁)` steps: the pull reaches the means through the momentum,
                // so the count answers a change of λ only over that many steps, and a multiplier
                // moving faster sees no answer and swings (with `H = B` on the tiny decoder, B = 2
                // against the momentum's 100 steps, λ rose from 24 to 68 before the count fell, the
                // count then sat at 8.4–9.4 against K 11.8 while λ fell to 0.02, and rose to 14.6
                // after; the budget test at 48eaf0af20). A step's relative violation counts at most
                // one: λ moves by at most a factor e per horizon, however far the count is from `K`
                // (toys' TMS at its true K, the count 63 against 7.96, took λ to 3e25 at one
                // relative violation of 7 per pass). A cap at the λ whose step would move the count
                // to `K` under the measured response (a Newton step on the constraint) does not hold:
                // on the tiny decoder the predicted response of the budget's pull alone,
                // `η λ Σ σ² (∂Ê/∂μ)²` (`σ²` the posterior's variances), capped λ near 2 with the count
                // at 17.4 against K 11.8, where F's push needs about 20, and with the data's push
                // `B ln 2 ⟨g_F, ∂Ê/∂μ⟩_σ²` of one batch included it turned negative. The
                // budget's terms reach the thresholds and gate rows alone, never a learned width:
                // the count is the hard gate's under the posterior and reads no width. The global ratio
                // −⟨g_F, g_k⟩ / |g_k|² (560f12d2d3's λ̄ Ê / K) explodes where most gates are
                // saturated and |g_k| is tiny: a toy fit (resid_mlp_1l, K = 55) took λ to 8,000 and
                // diverged with the count flat.
                if projected {
                    // The count is held at `K` by the projection after every step ([`project`]); λ
                    // steers which gates trade: the slope-weighted median of the positive rates
                    // max(0, −⟨∂F, ∂Ê⟩) / |∂Ê|², F's push against the count where the projection holds
                    // it (269b6645c2's rule), measured at every step. An integral rule on λ with a
                    // projection only at the start (8ed727551e) let every MLP gate close within
                    // epochs and none reopen (the learned tiny library, GHA seeds 3–7), and on toys'
                    // gated copy under a budget in bits λ reached 4e5 while no gate closed.
                    // Where the constraint is slack (the last projection did not bind), λ is zero: the
                    // pull kept on below `K` closed every MLP gate of the learned tiny library and then
                    // the always-on parts (GHA, seeds 3–6).
                    let mut ratios = balance_rates(&terms, &gradients, (device, explanation), scale)?;
                    ratios.retain(|r| r.0 > 0.0);
                    let half = 0.5 * ratios.iter().map(|r| r.1).sum::<f64>();
                    let mut below = 0.0;
                    progress.engaged = true;
                    progress.multiplier = if bound {
                        ratios.iter().find(|r| {
                            below += r.1;
                            below >= half
                        }).map_or(0.0, |r| r.0)
                    } else {
                        0.0
                    };
                } else if !progress.engaged && limit > 0.0 && expected > limit {
                    let ratios = balance_rates(&terms, &gradients, (device, explanation), scale)?;
                    let excess = ratios.iter().map(|r| r.1).sum::<f64>() * (expected - limit) / expected;
                    let mut below = 0.0;
                    let start = ratios.iter().find(|r| {
                        below += r.1;
                        below >= excess && r.0 > 0.0
                    });
                    if let Some(&(balance, _)) = start {
                        progress.engaged = true;
                        progress.multiplier = balance;
                        log::info!("library budget bound at {expected:.4} parts per token (K {limit}): λ starts at {balance:.4e}, the rate at the excess share of {} gates' weights", ratios.len());
                    }
                } else if progress.engaged && limit > 0.0 {
                    let horizon = (draws.len() as f64).max(1.0 / (1.0 - ivon.beta1));
                    progress.multiplier *= (((expected - limit) / limit).clamp(-1.0, 1.0) / horizon).exp();
                }
                let lambda = if progress.engaged { progress.multiplier } else { 0.0 };
                for (i, mean, variance) in &terms {
                    let op = explanation.trainable[*i];
                    // At λ = 0 the term adds nothing, and the step is the budget-free one bit for bit.
                    if lambda == 0.0 {
                        continue;
                    }
                    // On the device, in the fit's storage: `λ/(scale ln 2)` times the count's derivative
                    // into the gradient and `2λ/N` times its variance term into the curvature.
                    let (pull, bend) = (device.convert(mean).map_err(error)?, device.convert(variance).map_err(error)?);
                    match gradients.get_mut(&op) {
                        Some(total) => device.axpy(total, lambda / (scale * LN_2), &pull).map_err(error)?,
                        None => {
                            gradients.insert(op, device.scaled(lambda / (scale * LN_2), &pull).map_err(error)?);
                        }
                    }
                    match prior_curvature.get_mut(&op) {
                        Some(total) => device.axpy(total, 2.0 * lambda / tokens as f64, &bend).map_err(error)?,
                        None => {
                            prior_curvature.insert(op, device.scaled(2.0 * lambda / tokens as f64, &bend).map_err(error)?);
                        }
                    }
                }
                if lambda != 0.0 {
                    scorer.pull_assignments(lambda / (scale * LN_2));
                }
                parts_note = format!(", parts per token {expected:.4} (K {limit}), λ {:.4e}", progress.multiplier);
            }
            progress.step += 1;
            // The factor's scale: its square estimates the curvature per token of the tokens it
            // sums (one antithetic half's, `antithetic_step`).
            let factor_weight = weight * scored as f64 / factor.tokens as f64;
            let posterior_started = Instant::now();
            scorer.mixing_curvature(&factor.gradient, factor_weight, ivon.beta2)?;
            device_posterior.step(&gradients, weight * LN_2, (&factor.gradient, factor_weight), &prior_curvature, &ivon)?;
            scorer.step_assignments(weight * LN_2);
            scorer.step_mixings(weight * LN_2);
            scorer.pin_mixings(&mut device_posterior)?;
            if let Some(limit) = settings.budget.filter(|k| k.is_finite() && projected) {
                // The target eases from the first step's count to `K` with the horizon `H` of the
                // step's averages, `K + (C₀ − K) e^{−t/H}`: a projection of the whole excess at once
                // shut gates regardless of their worth (toys' gated copy at its true K, 32,100 bits
                // per token at the start against 7,850: its held-out KL went from 0.13 bits per token
                // without the budget to 1,100).
                let horizon = (draws.len() as f64).max(1.0 / (1.0 - ivon.beta1));
                let target = match eased {
                    // The first step measures the count it eases from.
                    None => f64::INFINITY,
                    Some((start, at)) => limit + (start - limit).max(0.0) * (-f64::from(progress.step - at) / horizon).exp(),
                };
                let (shift, before, count) = project(&mut scorer, &mut device_posterior, explanation, &posterior.active, &batch, key, target)?;
                eased.get_or_insert((before, progress.step));
                bound = shift > 0.0;
                hard.0 += before;
                hard.1 += 1;
                parts_note.push_str(&format!(", hard {before:.4} projected by {shift:.3e} widths to {count:.4}"));
            }
            let posterior_seconds = posterior_started.elapsed().as_secs_f64();
            let (eta, rho, draws_averaged, ratio) = device_posterior.step_state();
            log::info!("library line step {epoch}.{b}: η {eta:.4e}, ρ̄ {rho:.4e} over {draws_averaged} draws, r̄ {ratio:.4e}, trust {:.3e}; posterior step {posterior_seconds:.3} s", device_posterior.trust());
            // A nonfinite step state is a failed step: it fails here, at the step that made it,
            // never later as a null in a checkpoint record (toys' TMS-id fit ran on with ρ̄ NaN for
            // epochs while its mean's KL went from 4e-4 to 2e182).
            if !(eta.is_finite() && rho.is_finite() && ratio.is_finite() && progress.multiplier.is_finite()) {
                return Err(format!("step {epoch}.{b}: nonfinite step state (η {eta}, ρ̄ {rho}, r̄ {ratio}, λ {})", progress.multiplier));
            }
            let prior_note = if prior.is_some() { format!(" (prior: {prior_seconds:.3} s)") } else { String::new() };
            log::info!("library step {epoch}.{b}: data {:.6} bits per scored token at the iterate's samples, {:.2} s{prior_note}{parts_note}{on_note}", bits.iter().flatten().sum::<f64>() / scored as f64, step_started.elapsed().as_secs_f64());
        }
        let count = draws.len() as f64;
        // The end-of-epoch posterior scored on the whole collection at the draws every snapshot
        // shares: the estimates the stop and the best epoch are decided on (module note).
        let snapshot_started = Instant::now();
        let (snapshot, evaluated) = snapshot_estimates(&mut scorer, &device_posterior, &mut posterior, explanation, &Evidence { draws: &draws, sequences, settings }, prior.as_deref_mut())?;
        let snapshot_seconds = snapshot_started.elapsed().as_secs_f64();
        let snapshot_mean = snapshot.iter().sum::<f64>() / count;
        let (improvement, standard_error) = match &progress.previous {
            Some(before) => {
                let differences: Vec<f64> = before.iter().zip(&snapshot).map(|(a, b)| a - b).collect();
                let mean = differences.iter().sum::<f64>() / count;
                let variance = differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (count - 1.0);
                (Some(mean), Some((variance / count).sqrt()))
            }
            None => (None, None),
        };
        let to_bits = |nats: f64| nats / LN_2;
        let record = Epoch {
            epoch,
            data_bits: to_bits(data_sum / count),
            snapshot_bits: to_bits(snapshot_mean),
            snapshot_seconds,
            improvement_bits: improvement.map(to_bits),
            standard_error_bits: standard_error.map(to_bits),
            active_groups: posterior.active.iter().filter(|a| **a).count(),
            clean_bits_per_token: clean.mean().unwrap_or(f64::NAN),
            patched_bits_per_token: patched.mean().unwrap_or(f64::NAN),
            seconds: epoch_started.elapsed().as_secs_f64(),
            held_out: {
                progress.training_seconds += epoch_started.elapsed().as_secs_f64();
                let timed = Instant::now();
                let evaluation = held_out_on(&mut scorer, explanation, (&posterior, &device_posterior), (subset, &subset_batches), settings, tokens, prior.as_deref_mut())?;
                progress.evaluation_seconds += timed.elapsed().as_secs_f64();
                evaluation
            },
            held_out_full: if progress.evaluation_seconds + progress.full_seconds <= EVALUATION_SHARE * progress.training_seconds {
                let timed = Instant::now();
                let evaluation = held_out(&mut scorer, explanation, (&posterior, &device_posterior), held, settings, tokens, prior.as_deref_mut())?;
                progress.full_seconds = timed.elapsed().as_secs_f64();
                progress.evaluation_seconds += progress.full_seconds;
                Some(evaluation)
            } else {
                None
            },
            budget: settings.budget,
            expected_parts: settings.budget.map(|_| if hard.1 > 0 { hard.0 / hard.1 as f64 } else { parts.0 / parts.1.max(1) as f64 }),
            multiplier: settings.budget.map(|_| progress.multiplier),
        };
        log::info!("library fit epoch {epoch}: {record:?}");
        let (log_sd, magnitude, variances) = posterior.spread();
        let absent = || "none (no active entry)".to_string();
        log::info!(
            "library posterior after epoch {epoch}: mean ln σ {}, mean |μ| {}, Σ v_G {variances:.6e}",
            log_sd.map_or_else(absent, |v| format!("{v:.5}")),
            magnitude.map_or_else(absent, |v| format!("{v:.6e}"))
        );
        // Nothing nonfinite is checkpointed: the epoch fails at its record instead. The check
        // applies to the values that exist: with every group removed (a removal round may remove
        // them all, and the descent goes on) the spread's averages are absent, not nonfinite.
        let held = &record.held_out;
        let values = [record.data_bits, record.snapshot_bits, held.objective_bits_per_token, held.data_bits_per_token, held.mean_bits_per_token, held.rounded_bits_per_token, held.divergence_bits, variances];
        if values.iter().chain(log_sd.iter()).chain(magnitude.iter()).any(|v| !v.is_finite()) {
            return Err(format!("epoch {epoch}: nonfinite record (data, snapshot, held-out F, data, mean, rounded, divergence, Σ v_G: {values:?}; mean ln σ {log_sd:?}, mean |μ| {magnitude:?})"));
        }
        progress.epochs.push(record);
        progress.previous = Some(snapshot);
        progress.epoch += 1;
        let budget = settings.epochs.is_some_and(|last| progress.epoch >= last);
        // The descent stops at the first epoch whose snapshot's mean improvement over the last
        // one's, paired batch by batch, is not positive, and the removal round starts from the
        // best epoch's posterior (the lowest snapshot estimate since the objective last changed).
        // A fit with a budget of epochs (`Settings::epochs`) runs exactly that many and removes
        // nothing: arms compared at one budget take the same steps.
        let is_best = progress.best.is_none_or(|(b, _)| snapshot_mean < b);
        if is_best {
            progress.best = Some((snapshot_mean, epoch));
            best_evaluation = Some(evaluated);
        }
        let mut keep_best = false;
        if budget {
            progress.done = true;
        } else if settings.epochs.is_none() && improvement.is_some_and(|i| i <= 0.0) {
            if let Some((bits, at)) = progress.best.take() {
                log::info!("library fit stops after epoch {epoch}: back to the best epoch's posterior ({:.6e} bits)", bits / LN_2);
                if at != epoch {
                    // Back to every state the best snapshot's `F` was scored at: the posterior's
                    // means and deviations, its active groups and the prior term's state (its
                    // parameters and structure), so the restored state scores the recorded best.
                    // The optimizer continues from there rather than from the best epoch's own
                    // state: IVON's iterate is the restored mean, the momentum and its bias
                    // correction of every operator whose values changed are zeroed (its gradients
                    // were taken along the abandoned path, `DevicePosterior::set_values`), and the
                    // curvature `h`, the step-length averages `ρ̄` and `r̄` and the step count carry
                    // on: averages of measurements of `F` near the posterior over the last epoch,
                    // which the removal round below changes and the next epoch re-estimates.
                    writer.wait()?;
                    let (restored, start) = read_checkpoint(explanation, &best_path)?;
                    posterior = restored;
                    if let Some(saved) = saved_assignments(&best_path)? {
                        scorer.restore_assignments(&saved)?;
                    }
                    if let Some(saved) = saved_mixings(&best_path)? {
                        scorer.restore_mixings(&saved)?;
                    }
                    match (prior.as_deref_mut(), start.state.and_then(|state| state.prior)) {
                        (Some(prior), Some(state)) => prior.load(&state)?,
                        (None, None) => {}
                        _ => return Err("the best epoch's snapshot holds another prior term's state".into()),
                    }
                    device_posterior.set_values(&posterior)?;
                }
            }
            let log = checkpoint.map(|path| path.with_extension("removals.jsonl"));
            let evidence = Evidence { draws: &draws, sequences, settings };
            let removal = remove(&mut scorer, &mut device_posterior, &mut posterior, (&evidence, best_evaluation.take()), explanation, prior.as_deref_mut(), log.as_deref())?;
            log::info!("library removal after epoch {epoch}: {} of {} candidates, {} without effect", removal.removed, removal.candidates, removal.dead);
            // The removed groups' entries are exactly zero with `ln σ = −∞`, which the device step
            // leaves alone; the objective left its last trial on the device.
            device_posterior.set_values(&posterior)?;
            scorer.pin_mixings(&mut device_posterior)?;
            progress.done = removal.removed == 0;
            progress.removals.push(removal);
            // The objective changed discretely: convergence is judged afresh.
            progress.previous = None;
        } else if is_best {
            // The best posterior so far, kept on disk rather than as a copy on the host, with the
            // prior term's state its snapshot was scored at: the epoch's checkpoint below, linked
            // at `best_path`, or for a fit without one a file of its own.
            keep_best = true;
            if checkpoint.is_none() {
                let mut kept = progress.clone();
                kept.active = posterior.active.clone();
                kept.prior = prior.as_deref().map(PriorTerm::save).transpose()?;
                kept.assignments = scorer.saved_assignments();
                kept.frames = scorer.saved_mixings();
                Snapshot::save(&mut kept, &device_posterior, (&best_path, None), &mut writer)?;
            }
        }
        progress.prior = prior.as_deref().map(PriorTerm::save).transpose()?;
        progress.assignments = scorer.saved_assignments();
        progress.frames = scorer.saved_mixings();
        save(&mut progress, &posterior, (&device_posterior, keep_best.then_some(best_path.as_path())), &mut writer)?;
    }
    writer.wait()?;
    for written in [best_path.clone(), best_path.with_extension("json")] {
        if written.exists() {
            std::fs::remove_file(&written).map_err(error)?;
        }
    }
    // A fit ended by its budget of epochs has no removal round: its last epoch's snapshot, the
    // estimate of `F` at the posterior it ends with.
    let objective_bits = progress.removals.last().map(|r| r.after_bits).or_else(|| settings.epochs.and(progress.epochs.last()).map(|e| e.snapshot_bits)).unwrap_or(f64::NAN);
    let end = held_out(&mut scorer, explanation, (&posterior, &device_posterior), held, settings, tokens, prior.as_deref_mut())?;
    log::info!("library end: {end:?}");
    writer.wait()?;
    let representative = if end.rounded_bits_per_token <= end.mean_bits_per_token { Representative::Rounded } else { Representative::Mean };
    Ok(Fit {
        report: Report {
            settings: settings.clone(),
            training_sequences: sequences.len(),
            scored_tokens: tokens,
            families,
            groups: explanation.groups.len(),
            parameters,
            start: progress.start.ok_or("no starting evaluation")?,
            end,
            representative,
            literals: Literals::of(device),
            active_groups: posterior.active.iter().filter(|a| **a).count(),
            objective_bits,
            seconds: resumed_seconds + started.elapsed().as_secs_f64(),
            epochs: progress.epochs,
            removals: progress.removals,
        },
        posterior,
    })
}

/// The Laplace start of a fresh fit. One pass over the training collection draws every batch's
/// Gauss–Newton factor `u_b` (`interchange::fisher_probe`) at a weight sample of the
/// unit-information start, and
/// `h = Σ_b u_b ⊙ u_b / N` (the batches' estimates `B / N · u_b ⊙ u_b`, averaged) estimates the
/// Gauss–Newton diagonal per token. Each entry's deviation becomes `σ² = 1 / (N h + 1 / v_G)`, the
/// minimum in `σ` of the data term's Gauss–Newton model `½ N h σ²` plus `KL(q ‖ p)`, instead of the
/// epochs IVON's curvature average needs to fall from the start's `1 / v_G` to `h`. Returns the
/// batches' sums `Σ_b u_b ⊙ u_b` per operator (by id), on the device they were summed on;
/// [`laplace_start`] reads them one operator at a time.
fn laplace_sums(scorer: &mut Scorer, device_posterior: &DevicePosterior, draws: &[Draw], sequences: &[Vec<u32>], settings: &Settings) -> Result<(Device, BTreeMap<usize, Tensor>), String> {
    let device = scorer.experiments.models().1.program.device().clone();
    // `Σ_b u_b ⊙ u_b` summed on the device, in float64 where it holds float64, and read once.
    let wide = match device.with_storage(Storage::F64) {
        Ok(wide) => wide,
        Err(_) => device.clone(),
    };
    let mut sums: BTreeMap<usize, Tensor> = BTreeMap::new();
    for (b, draw) in draws.iter().enumerate() {
        scorer.next_batch();
        let batch = draw.batch(sequences)?;
        let experiments = scorer.experiments(draw, sequences)?;
        // The factor alone, at the batch's weight sample with its probe keyed from the same seed:
        // the probe at `P`'s own predictions needs neither `M`'s targets nor the divergence's
        // gradient.
        let key = noise_seed(settings.seed, 0, b);
        scorer.set(device_posterior, Values::Sample(key), Gates::Relaxed)?;
        let factor = scorer.reversed(|e| e.fisher_probe_resident(&batch, &experiments, probe_key(key)));
        scorer.rest(Gates::Relaxed)?;
        let mut factor = factor?;
        scorer.strip(&mut factor);
        for (op, u) in &factor {
            let u = wide.convert(u).map_err(error)?;
            match sums.get_mut(op) {
                Some(sum) => wide.hadamard(sum, &u, &u, true).map_err(error)?,
                None => {
                    let mut sum = wide.zeros(u.rows(), u.cols()).map_err(error)?;
                    wide.hadamard(&mut sum, &u, &u, false).map_err(error)?;
                    sums.insert(*op, sum);
                }
            }
        }
    }
    Ok((wide, sums))
}

/// The Laplace start from the batches' sums ([`laplace_sums`]), one operator at a time: its
/// curvature `h` read off the device, its deviations set from it in `posterior`, and both handed
/// to `set` (the device posterior's [`DevicePosterior::set_start`]) before the next operator's are
/// read, so no more than one operator's curvature is on the host.
fn laplace_start(
    scorer: &Scorer,
    explanation: &Explanation,
    posterior: &mut Posterior,
    (wide, sums): (Device, BTreeMap<usize, Tensor>),
    tokens: usize,
    mut set: impl FnMut(usize, &Array2<f64>, &Array2<f64>) -> Result<(), String>,
) -> Result<(), String> {
    let mut at = BTreeMap::new();
    for (op, sum) in sums {
        at.insert(scorer.at(op)?, sum);
    }
    let variance = posterior.variances();
    let n = tokens as f64;
    let gate: Vec<bool> = explanation.groups.iter().map(|g| g.name.ends_with(".widths")).collect();
    // A gated component's slice (`library_vpd`'s `.read` and `.write` groups) starts its deviation
    // at a hundredth of the slice's own scale (the root mean square of its group at the start,
    // `Explanation::reference`), and the step's curvature at the value that holds it there,
    // `h = (1/σ² − 1/v_G) / N`; the running average then moves `h` toward the measured curvature
    // over about one pass, so a slice's deviation grows only where the data allow it. From the
    // Laplace curvature a slice the gated runs seldom reach starts near its prior, and with every
    // part on (the all-on experiment) the sample's noise summed over every slice: the all-on KL at
    // the first step's sample was 11.3 bits per token where the mean's is 1.32 (vpd4l tiny fit,
    // grouped own gates, 14985361f0). descent's prototype starts its slices this way.
    // A gate's threshold and its direction's row (`.thresholds`, `.g{b}`) start the same way: the
    // budget counts the hard gate under the posterior, `Φ(μ_z / sd_q(z))`, and from the thresholds'
    // prior deviations it counted 3,822 parts per token at the start where the hard gates at the
    // mean execute 364 (vpd4l grouped direction gates with learned widths, decomp-vpd4l-i at
    // c55f1fadd8); the count then fell as the curvature shrank the deviations over the first pass,
    // whatever λ (the learned tiny library: from K to 33–37 parts against K near 65 at λ under 1).
    let direction = |name: &str| name.rsplit_once(".g").is_some_and(|(_, b)| !b.is_empty() && b.bytes().all(|c| c.is_ascii_digit()));
    let slice: Vec<Option<f64>> = explanation
        .groups
        .iter()
        .zip(&explanation.reference)
        .map(|(g, r)| (g.name.ends_with(".read") || g.name.ends_with(".write") || g.name.ends_with(".thresholds") || direction(&g.name)).then(|| 0.01 * r.sqrt()))
        .collect();
    for i in 0..posterior.mean.len() {
        let mut h = match at.remove(&i) {
            Some(sum) => wide.download(&sum).map_err(error)?,
            None => Array2::zeros(posterior.mean[i].dim()),
        };
        h.mapv_inplace(|square| square / n);
        // A gate's threshold or width (`library_vpd`'s `.thresholds` and `.widths` groups) starts
        // its deviation at its group's prior, `σ² = v_G`: the curvature of a gate at its start is
        // not a valid Laplace curvature (a nearly hard gate's is a step's), and from it the
        // thresholds' deviations started near 10⁻³ and only shrank (vpd4l, decomp-vpd4l-b, -c at
        // 6387505b50). The step's curvature estimate keeps the measured `h`: with it zeroed there
        // (9b53f32bda) IVON's direction `G / (h + δ)` put most of its length on the gates, and the
        // line step's joint-to-diagonal curvature ratio ρ̄ started near 1e9 and held η near 1e-8
        // for the first epoch (decomp-vpd4l-f, -g at 5c54cd5f20).
        ndarray::Zip::from(&mut *posterior.log_sd[i]).and(&mut h).and(&posterior.membership[i]).for_each(|s, h, group| {
            let v = variance[*group as usize];
            if *s == f64::NEG_INFINITY || !(v > 0.0) {
                return;
            }
            match slice[*group as usize] {
                Some(sd) if sd > 0.0 && sd * sd < v => {
                    *s = sd.ln();
                    *h = (1.0 / (sd * sd) - 1.0 / v) / n;
                }
                _ => {
                    let h = if gate[*group as usize] { 0.0 } else { *h };
                    *s = -0.5 * (n * h + 1.0 / v).ln();
                }
            }
        });
        set(i, &posterior.log_sd[i], &h)?;
    }
    Ok(())
}

/// The noise stream of the removal estimates' weight samples: one no epoch draws (training takes
/// `1, 2, …`, the removal's evaluations `0`), so a removal is accepted on samples its ranking did
/// not see.
const RANKING_STREAM: usize = usize::MAX;

/// The removal estimates' [`Curvature`]: per training batch of the fixed collection, at the batch's
/// weight sample on the noise stream `stream` (the fit's: [`RANKING_STREAM`]), the data term's
/// gradient and one draw of the Gauss–Newton factor (module note), each batch's group sums made
/// on the device (`DevicePosterior::add_removal`); for the operators `moved` (by id), the data
/// gradient summed over the batches on the device.
fn removal_curvature(scorer: &mut Scorer, posterior: &DevicePosterior, draws: &[Draw], sequences: &[Vec<u32>], settings: &Settings, stream: usize, moved: &[usize]) -> Result<Curvature, String> {
    let mut curvature = Curvature::new(posterior.group_count());
    // Seconds in each part of the pass: the sample, the batch's experiments, M's targets, the
    // scoring with its two reverse passes, and the per-group sums.
    let (mut seconds, started) = ([0.0_f64; 5], Instant::now());
    let device = scorer.experiments.models().1.program.device().clone();
    let mut sums: BTreeMap<usize, Tensor> = BTreeMap::new();
    let mut pending = Vec::new();
    for (b, draw) in draws.iter().enumerate() {
        scorer.next_batch();
        let key = noise_seed(settings.seed, stream, b);
        let mut timed = Instant::now();
        let mut lap = |part: usize, timed: &mut Instant| {
            seconds[part] += timed.elapsed().as_secs_f64();
            *timed = Instant::now();
        };
        lap(0, &mut timed);
        let batch = draw.batch(sequences)?;
        let experiments = scorer.experiments(draw, sequences)?;
        lap(1, &mut timed);
        let (experiments, targets) = part_targets(scorer, &batch, experiments)?;
        lap(2, &mut timed);
        // F's data terms at the batch's sample (`Scorer::terms`, the all-on experiment's joined,
        // `Terms::combine`), reversed twice: the divergence's gradient (in bits) and a draw of the
        // Gauss–Newton factor.
        let mut evaluation = scorer.terms(posterior, (&batch, &experiments, &targets), (Values::Sample(key), Gates::Relaxed), (true, Some(probe_key(key))))?;
        evaluation.combine(&device)?;
        let mut factor = evaluation.factor.take().ok_or("no Gauss–Newton factor")?;
        scorer.strip(&mut evaluation.gradient);
        scorer.strip(&mut factor.gradient);
        lap(3, &mut timed);
        posterior.add_removal((&evaluation.gradient, LN_2), &factor.gradient, key, &mut curvature, &mut pending)?;
        lap(4, &mut timed);
        for op in moved {
            let Some(g) = evaluation.gradient.get(op) else { continue };
            match sums.get_mut(op) {
                Some(sum) => device.axpy(sum, 1.0, g).map_err(error)?,
                None => {
                    sums.insert(*op, device.copy(g).map_err(error)?);
                }
            }
        }
    }
    posterior.read_removal(&mut pending, &mut curvature)?;
    for (op, sum) in &sums {
        curvature.gradient.insert(scorer.at(*op)?, device.download(sum).map_err(error)? * LN_2);
    }
    log::info!(
        "library removal curvature: {} batches {:.1} s: sample {:.1} s, experiments {:.1} s, targets {:.1} s, scoring and reverses {:.1} s, group sums {:.1} s",
        draws.len(),
        started.elapsed().as_secs_f64(),
        seconds[0],
        seconds[1],
        seconds[2],
        seconds[3],
        seconds[4]
    );
    Ok(curvature)
}

/// `E_q[D]` over every training batch in nats, one weight sample per batch from the removal seeds,
/// with the groups `removed` (and the already removed ones) zeroed, on the fixed collection; with
/// `prior`, plus its value at each batch's sample, averaged, and the parameters it sends. Returned
/// per batch (each with its share of the prior's value), the prior's cost as the rest. With
/// `against`, an accepted posterior's per-batch values and the decrease of the rest of `F` other
/// than the prior's cost from it, the batches are scored in decreasing accepted value
/// (`library_removal::order`) and stop once a rise of `F` is certain (`library_removal::settled`),
/// the evaluation then incomplete (unscored batches NaN); without a prior term only, whose
/// per-batch value has no floor.
fn expected_divergence(
    scorer: &mut Scorer,
    device_posterior: &mut DevicePosterior,
    posterior: &Posterior,
    draws: &[Draw],
    sequences: &[Vec<u32>],
    removed: &[usize],
    settings: &Settings,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
    against: Option<(&[f64], f64)>,
) -> Result<Evaluation, String> {
    let timed = Instant::now();
    // The removal objective's trial arrives with its groups removed: no copy of it.
    let trial = if removed.is_empty() {
        std::borrow::Cow::Borrowed(posterior)
    } else {
        let mut trial = posterior.clone();
        trial.remove(removed);
        std::borrow::Cow::Owned(trial)
    };
    let cloned = timed.elapsed().as_secs_f64();
    // The trial goes to the device once; each batch's weight sample is drawn there.
    device_posterior.set_values(&trial)?;
    let uploaded = timed.elapsed().as_secs_f64();
    let (evaluation, (preparing, targeting, scoring)) = collection_divergence(scorer, device_posterior, &trial, (draws, sequences, settings), prior, against)?;
    log::info!(
        "library removal evaluation: {:.2} s: trial clone {cloned:.2} s, set_values {:.2} s, {} of {} batches {:.2} s (experiments {preparing:.2} s, targets {targeting:.2} s, scoring {:.2} s)",
        timed.elapsed().as_secs_f64(),
        uploaded - cloned,
        evaluation.scored(),
        draws.len(),
        preparing + targeting + scoring,
        scoring
    );
    Ok(evaluation)
}

/// [`expected_divergence`] of the posterior `device_posterior` holds, `trial` on the host, which
/// the prior term's samples are drawn from: the device posterior's state is only read, each
/// batch's weight sample drawn around its mean `μ̄` on the removal comparisons' noise stream
/// (`noise_seed(seed, 0, b)`). Returns the evaluation and the seconds spent making the batches'
/// experiments, their targets, and scoring them.
fn collection_divergence(
    scorer: &mut Scorer,
    device_posterior: &DevicePosterior,
    trial: &Posterior,
    (draws, sequences, settings): (&[Draw], &[Vec<u32>], &Settings),
    prior: Option<&mut (dyn PriorTerm + 'static)>,
    against: Option<(&[f64], f64)>,
) -> Result<(Evaluation, (f64, f64, f64)), String> {
    let mut prior = prior;
    let cost = prior.as_deref().map_or(Ok(0.0), |p| p.cost(trial))?;
    let n = draws.len();
    let against = against.filter(|_| prior.is_none());
    let sequence: Vec<usize> = against.map_or_else(|| (0..n).collect(), |(accepted, _)| library_removal::order(accepted));
    let mut batches = vec![f64::NAN; n];
    let (mut rise, mut slack) = (0.0, against.map_or(0.0, |(accepted, _)| accepted.iter().sum::<f64>()));
    let (mut preparing, mut targeting, mut scoring) = (0.0, 0.0, 0.0);
    for (k, b) in sequence.into_iter().enumerate() {
        scorer.next_batch();
        let draw = &draws[b];
        // Removal zeroes entries, so the remaining entries see the same noise as the full posterior.
        let key = noise_seed(settings.seed, 0, b);
        let started = Instant::now();
        let experiments = scorer.experiments(draw, sequences)?;
        let batch = draw.batch(sequences)?;
        preparing += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let (experiments, targets) = part_targets(scorer, &batch, experiments)?;
        targeting += started.elapsed().as_secs_f64();
        let started = Instant::now();
        // F's data terms through the scored explanation's hard gates (`Scorer::terms`): the
        // snapshot the best epoch is chosen by and the removal round's comparisons score them here.
        let mut nats = scorer.terms(device_posterior, (&batch, &experiments, &targets), (Values::Sample(key), Gates::Hard), (false, None))?.bits() * LN_2;
        if let Some(prior) = prior.as_deref_mut() {
            nats += prior.sample(trial, &host_sample(trial, &prior.operators(), key), false)?.0 / draws.len() as f64;
        }
        scoring += started.elapsed().as_secs_f64();
        batches[b] = nats;
        if let Some((accepted, budget)) = against {
            rise += nats - accepted[b];
            slack -= accepted[b];
            if k + 1 < n && library_removal::settled(rise, slack, budget - cost) {
                break;
            }
        }
    }
    let complete = batches.iter().all(|b| !b.is_nan());
    Ok((Evaluation { batches, rest: cost, complete }, (preparing, targeting, scoring)))
}

/// The snapshot of the posterior `device_posterior` holds at an epoch's end (module note): per
/// training batch `b` of `evidence`, the estimate `B D_b + R` of `F` in nats, `B` the batches,
/// `D_b` the batch's data term at its weight sample on the removal comparisons' noise stream
/// (`noise_seed(seed, 0, b)`, the same draws at every snapshot) with its share of the prior term's
/// value there ([`collection_divergence`]), and `R` the rest of `F` at the posterior: the groups'
/// description, the explanation's discrete choices and the prior term's parameters. The mean over
/// the batches is the removal comparisons' `F` of the posterior. Forward passes only: the device
/// posterior's state (its iterate, its average and IVON's state) is left as it is; `posterior` is
/// set to the device's values. Also the evaluation itself, its rest the whole `R`: what a removal
/// round's objective returns for this posterior ([`remove`]).
fn snapshot_estimates(
    scorer: &mut Scorer,
    device_posterior: &DevicePosterior,
    posterior: &mut Posterior,
    explanation: &Explanation,
    evidence: &Evidence,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
) -> Result<(Vec<f64>, Evaluation), String> {
    let &Evidence { draws, sequences, settings } = evidence;
    device_posterior.values_into(posterior)?;
    let (mut evaluation, (preparing, targeting, scoring)) = collection_divergence(scorer, device_posterior, posterior, (draws, sequences, settings), prior, None)?;
    log::info!("library snapshot: {} batches, experiments {preparing:.2} s, targets {targeting:.2} s, scoring {scoring:.2} s", draws.len());
    // Summed as a removal round's objective sums it (`remove`), so the evaluation it takes over is
    // the one the round would have made, to the last bit.
    let rest = evaluation.rest + (posterior.description() + explanation.fixed_nats + scorer.mixing_nats());
    if !rest.is_finite() {
        return Err("a nonfinite posterior divergence".into());
    }
    let count = draws.len() as f64;
    let estimates = evaluation.batches.iter().map(|data| count * data + rest).collect();
    evaluation.rest = rest;
    Ok((estimates, evaluation))
}

/// A removal round's fixed evidence: the training batches of the sequences under the fit's
/// settings.
struct Evidence<'a> {
    draws: &'a [Draw],
    sequences: &'a [Vec<u32>],
    settings: &'a Settings,
}

/// The removal step (`library_removal`) on `posterior`, scored by `F` on the round's
/// fixed `evidence`, compensated in the MLPs it deletes functions of (`library_compensation`),
/// logged to `log`. `start`, when given, is `posterior`'s own evaluation (the best epoch's
/// snapshot, [`snapshot_estimates`]: the same posterior on the same draws and noise stream), which
/// the round's first scoring returns in place of scoring it again.
fn remove(
    scorer: &mut Scorer,
    device_posterior: &mut DevicePosterior,
    posterior: &mut Posterior,
    (evidence, start): (&Evidence, Option<Evaluation>),
    explanation: &Explanation,
    prior: Option<&mut (dyn PriorTerm + 'static)>,
    log: Option<&Path>,
) -> Result<Removal, String> {
    let mut prior = prior;
    let (fixed, Evidence { draws, sequences, settings }) = (explanation.fixed_nats + scorer.mixing_nats(), evidence);
    let timed = Instant::now();
    let compensation = Compensation::new(&mut scorer.experiments, explanation, posterior, sequences, settings.batch_sequences)?;
    let compensated = timed.elapsed().as_secs_f64();
    let moved: Vec<usize> = compensation.outputs().iter().map(|i| explanation.trainable[*i]).collect();
    let curvature = removal_curvature(scorer, device_posterior, draws, sequences, settings, RANKING_STREAM, &moved)?;
    log::info!("library removal setup: compensation Gram {compensated:.1} s, curvature {:.1} s", timed.elapsed().as_secs_f64() - compensated);
    let mut start = start;
    let mut objective = |trial: &Posterior, accepted: Option<&Evaluation>| -> Result<Evaluation, String> {
        if accepted.is_none()
            && let Some(evaluation) = start.take()
        {
            return Ok(evaluation);
        }
        let rest = trial.description() + fixed;
        let against = accepted.map(|a| (a.batches.as_slice(), a.rest - rest));
        let mut evaluation = expected_divergence(scorer, device_posterior, trial, draws, sequences, &[], settings, prior.as_deref_mut(), against)?;
        evaluation.rest += rest;
        Ok(evaluation)
    };
    library_removal::round(explanation, posterior, Some(&compensation), &curvature, &mut objective, log)
}

/// A removal step run on a posterior outside a fit ([`removal_step`]): the fit's training and
/// held-out sequences and settings, and the search's log.
pub struct Step<'a> {
    pub sequences: &'a [Vec<u32>],
    pub held: &'a [Vec<u32>],
    pub settings: &'a Settings,
    pub log: Option<&'a Path>,
}

/// One removal step of [`fit`] on `posterior` outside a fit (a checkpoint's): the same training
/// collection, weight noise, compensation and acceptance, with the held-out evaluation before and
/// after.
pub fn removal_step(device: &Device, native: &OperatorProgram, explanation: &Explanation, posterior: &mut Posterior, step: Step) -> Result<(Removal, HeldOut, HeldOut), String> {
    let Step { sequences, held, settings, log } = step;
    settings.validate()?;
    let length = sequences.first().map_or(0, Vec::len);
    let draws = draws(sequences.len(), settings.batch_sequences, settings.seed)?;
    let mut scorer = Scorer::new(device, native, explanation, settings)?.prepared(native, sequences, settings)?;
    let mut tokens = 0;
    for draw in &draws {
        tokens += scorer.experiments(draw, sequences)?.iter().map(|e| length - e.position).sum::<usize>();
    }
    let evaluate = |scorer: &mut Scorer, posterior: &Posterior| -> Result<HeldOut, String> {
        let device_posterior = DevicePosterior::new(device, explanation, posterior, tokens as f64, None, 0)?;
        held_out(scorer, explanation, (posterior, &device_posterior), held, settings, tokens, None)
    };
    let before = evaluate(&mut scorer, posterior)?;
    let evidence = Evidence { draws: &draws, sequences, settings };
    let mut device_posterior = DevicePosterior::new(device, explanation, posterior, tokens as f64, None, 0)?;
    let removal = remove(&mut scorer, &mut device_posterior, posterior, (&evidence, None), explanation, None, log)?;
    let after = evaluate(&mut scorer, posterior)?;
    Ok((removal, before, after))
}

/// The posterior a fresh fit starts from on `sequences` under `settings` ([`fit`]): the
/// unit-information posterior at `M`'s values with the Laplace start's deviations from one
/// pass of the Gauss–Newton factor (`laplace_start`).
pub fn start_posterior(device: &Device, native: &OperatorProgram, explanation: &Explanation, sequences: &[Vec<u32>], settings: &Settings) -> Result<Posterior, String> {
    settings.validate()?;
    let length = sequences.first().map_or(0, Vec::len);
    let draws = draws(sequences.len(), settings.batch_sequences, settings.seed)?;
    let mut scorer = Scorer::new(device, native, explanation, settings)?.prepared(native, sequences, settings)?;
    let mut tokens = 0;
    for draw in &draws {
        tokens += scorer.experiments(draw, sequences)?.iter().map(|e| length - e.position).sum::<usize>();
    }
    // Seconds of the start's parts: the posterior made on the host, sent to the device, the
    // curvature pass over the training batches, and the deviations made on the host.
    let mut seconds = [0.0_f64; 4];
    let mut timed = Instant::now();
    let mut lap = |part: usize, timed: &mut Instant| {
        seconds[part] = timed.elapsed().as_secs_f64();
        *timed = Instant::now();
    };
    let mut posterior = Posterior::new(explanation, tokens)?;
    lap(0, &mut timed);
    let device_posterior = DevicePosterior::new(device, explanation, &posterior, tokens as f64, None, 0)?;
    device.synchronize().map_err(error)?;
    lap(1, &mut timed);
    let sums = laplace_sums(&mut scorer, &device_posterior, &draws, sequences, settings)?;
    device.synchronize().map_err(error)?;
    lap(2, &mut timed);
    laplace_start(&scorer, explanation, &mut posterior, sums, tokens, |_, _, _| Ok(()))?;
    lap(3, &mut timed);
    log::info!(
        "library start parts: posterior on the host {:.1} s, to the device {:.1} s, curvature pass {:.1} s, deviations on the host {:.1} s ({} parameters)",
        seconds[0],
        seconds[1],
        seconds[2],
        seconds[3],
        posterior.mean.iter().map(|m| m.len()).sum::<usize>()
    );
    Ok(posterior)
}

/// `posterior` with the removals a round accepted applied in order (each `(without effect,
/// groups)` of `accepted`, from its journal): groups without effect removed plainly, every other
/// proposal compensated as the round made it (`library_compensation`, its Gram from `posterior`
/// before the first), with the held-out evaluation before and after.
pub fn removal_replay(device: &Device, native: &OperatorProgram, explanation: &Explanation, posterior: &mut Posterior, step: Step, accepted: &[(bool, Vec<usize>)]) -> Result<(HeldOut, HeldOut), String> {
    let Step { sequences, held, settings, .. } = step;
    settings.validate()?;
    let length = sequences.first().map_or(0, Vec::len);
    let draws = draws(sequences.len(), settings.batch_sequences, settings.seed)?;
    let mut scorer = Scorer::new(device, native, explanation, settings)?.prepared(native, sequences, settings)?;
    let mut tokens = 0;
    for draw in &draws {
        tokens += scorer.experiments(draw, sequences)?.iter().map(|e| length - e.position).sum::<usize>();
    }
    let evaluate = |scorer: &mut Scorer, posterior: &Posterior| -> Result<HeldOut, String> {
        let device_posterior = DevicePosterior::new(device, explanation, posterior, tokens as f64, None, 0)?;
        held_out(scorer, explanation, (posterior, &device_posterior), held, settings, tokens, None)
    };
    let before = evaluate(&mut scorer, posterior)?;
    let compensation = Compensation::new(&mut scorer.experiments, explanation, posterior, sequences, settings.batch_sequences)?;
    for (dead, groups) in accepted {
        if *dead {
            posterior.remove(groups);
        } else {
            *posterior = compensation.proposal(posterior, groups)?;
        }
    }
    let after = evaluate(&mut scorer, posterior)?;
    Ok((before, after))
}

/// Per group set of `sets`, removed from `posterior` alone, the changes of the data term and of
/// the description in nats on the removal step's training collection at its weight samples,
/// without compensation and with it (`library_compensation`): what a unit's prediction estimates.
pub fn removal_changes(device: &Device, native: &OperatorProgram, explanation: &Explanation, posterior: &Posterior, step: Step, sets: &[Vec<usize>]) -> Result<Vec<[(f64, f64); 2]>, String> {
    let Step { sequences, settings, .. } = step;
    settings.validate()?;
    let length = sequences.first().map_or(0, Vec::len);
    let draws = draws(sequences.len(), settings.batch_sequences, settings.seed)?;
    let mut scorer = Scorer::new(device, native, explanation, settings)?.prepared(native, sequences, settings)?;
    let mut tokens = 0;
    for draw in &draws {
        tokens += scorer.experiments(draw, sequences)?.iter().map(|e| length - e.position).sum::<usize>();
    }
    let mut device_posterior = DevicePosterior::new(device, explanation, posterior, tokens as f64, None, 0)?;
    let compensation = Compensation::new(&mut scorer.experiments, explanation, posterior, sequences, settings.batch_sequences)?;
    let base = expected_divergence(&mut scorer, &mut device_posterior, posterior, &draws, sequences, &[], settings, None, None)?.total();
    let mut changes = Vec::with_capacity(sets.len());
    for set in sets {
        let mut plain = posterior.clone();
        plain.remove(set);
        let compensated = compensation.proposal(posterior, set)?;
        let mut pair = [(0.0, 0.0); 2];
        for (change, trial) in pair.iter_mut().zip([&plain, &compensated]) {
            let data = expected_divergence(&mut scorer, &mut device_posterior, trial, &draws, sequences, &[], settings, None, None)?.total();
            *change = (data - base, trial.description() - posterior.description());
        }
        changes.push(pair);
    }
    Ok(changes)
}

// ----------------------------------------------------------------------------- the reported artifact

/// The explanation at the posterior mean: each library operator holds `μ`, and the blocks only
/// removed groups touch are absent (no literals). Partial interface blocks remain
/// present, so their zeroed entries still cost ordinary serialized literals.
pub fn posterior_mean(explanation: &Explanation, posterior: &Posterior) -> Result<Artifact, String> {
    let mut artifact = mean_artifact(explanation.artifact.clone(), &explanation.trainable, posterior.means(), &posterior.membership, &posterior.active)?;
    // The artifact is the explanation as it is scored: a learned width's gate is scored hard
    // (`GateScoring::Hard`), so the widths hold `library_vpd::HARD` and the artifact's own law is
    // the hard gate `H(z)` for every reader; `Φ(z / w)` is the training law alone.
    if explanation.scoring == GateScoring::Hard {
        let widths: std::collections::BTreeSet<usize> = explanation.groups.iter().filter(|g| g.name.ends_with(".widths")).flat_map(|g| g.cells.iter().map(|c| c.operator)).collect();
        for op in widths {
            let old = Arc::clone(&artifact.program.operators[op]);
            let values = Array2::from_elem((old.rows.width(), old.cols.width()), crate::library_vpd::HARD);
            let precision = exact_precision(values.iter().copied()).map_err(error)?;
            artifact.program.operators[op] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).map_err(error)?);
        }
    }
    Ok(artifact)
}

/// [`posterior_mean`] of an explanation's `artifact` whose `trainable` operators hold `means`, each
/// entry in the group `membership` names, the groups active where `active` says.
fn mean_artifact(mut artifact: Artifact, trainable: &[usize], means: Vec<Array2<f64>>, membership: &[Array2<u32>], active: &[bool]) -> Result<Artifact, String> {
    for ((op, values), membership) in trainable.iter().zip(means).zip(membership) {
        let source = &artifact.program.operators[*op];
        let (rows, cols) = (source.rows.clone(), source.cols.clone());
        let mut present = Array2::from_elem((rows.group_count(), cols.group_count()), false);
        for r in 0..rows.group_count() {
            for c in 0..cols.group_count() {
                present[[r, c]] = rows.range(r).any(|i| cols.range(c).any(|j| active[membership[[i, j]] as usize]));
            }
        }
        let precision = exact_precision(values.iter().copied()).map_err(error)?;
        let operator = Operator::blocks(source.name.clone(), rows, cols, values, present, precision, source.provenance.clone()).map_err(error)?;
        if !matches!(operator.body, OperatorBody::Dense { .. }) {
            return Err("a library operator lost its dense body".into());
        }
        artifact.program.operators[*op] = Arc::new(operator);
    }
    artifact.program.interfaces().map_err(error)?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    /// A shared stage's assignment ([`super::Assignment`]): the relaxed operator's columns are each
    /// component's softmax over its candidates, the hardened one puts each component on its largest
    /// candidate alone, and the gradient chained to the logits ([`super::Assignment::gather`])
    /// equals central differences of `⟨G, A(ℓ)⟩` (1e-8).
    #[test]
    fn an_assignment_chains_its_gradient_to_its_logits() {
        use super::{Assignment, Relaxation};
        let candidates = vec![vec![0, 1, 2], vec![1, 0], vec![2, 0, 1]];
        let logits = vec![vec![0.3, -1.2, 0.8], vec![1.5, -0.4], vec![-0.2, 0.9, 0.1]];
        let mut a = Assignment { operator: 0, gates: 3, candidates: candidates.clone(), logits: logits.clone(), previous: None, gradient: candidates.iter().map(|c| vec![0.0; c.len()]).collect() };
        let soft = a.values(Relaxation::Soft);
        for b in 0..3 {
            assert!((soft.column(b).sum() - 1.0).abs() < 1e-15, "component {b}'s assignment sums to one");
            let e: Vec<f64> = logits[b].iter().map(|v| v.exp()).collect();
            let z: f64 = e.iter().sum();
            for (k, g) in candidates[b].iter().enumerate() {
                assert!((soft[[*g, b]] - e[k] / z).abs() < 1e-15);
            }
        }
        let hard = a.values(Relaxation::Hard);
        assert_eq!(hard, ndarray::array![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], "each component on its largest candidate");
        let g = ndarray::array![[0.7, -0.3, 1.1], [-0.5, 0.2, 0.4], [0.9, -1.4, -0.6]];
        a.gather(&g);
        let h = 1e-6;
        for b in 0..3 {
            for k in 0..candidates[b].len() {
                let at = |shift: f64| {
                    let mut l = logits.clone();
                    l[b][k] += shift;
                    let moved = Assignment { logits: l, ..a.clone() };
                    (&g * &moved.values(Relaxation::Soft)).sum()
                };
                let numeric = (at(h) - at(-h)) / (2.0 * h);
                assert!((numeric - a.gradient[b][k]).abs() < 1e-8, "logit ({b}, {k}): chained {} against {numeric}", a.gradient[b][k]);
            }
        }
    }

    /// The definition hashes a layer without a sink exactly as `Layer`'s debug form did before the
    /// sink field existed (so earlier checkpoints keep their identity), and a sink when there is one.
    #[test]
    fn a_layer_without_a_sink_is_defined_as_before_the_sink_field() {
        let export = crate::test_support::tiny_qwen3_export("layers_definition", 2);
        let imported = crate::import::import_language_model(&export, 2, 6).unwrap();
        std::fs::remove_dir_all(&export).unwrap();
        let native = crate::run_check::split_sites(&imported.program).unwrap();
        let sites = crate::run_check::layer_nodes(&native, 2).unwrap();
        let layers: Vec<super::Layer> = sites
            .iter()
            .enumerate()
            .map(|(l, sites)| super::Layer { sites: sites.clone(), heads: vec![(vec![l, 1], vec![2])], functions: vec![vec![3, 4], vec![l]], sink: None, thresholds: Vec::new(), components: Vec::new() })
            .collect();
        // `Layer`'s derived debug form before the sink field is today's without that field.
        assert_eq!(super::layers_definition(&layers), format!("{layers:?}").replace(", sink: None, thresholds: [], components: []", ""));
        assert!(!super::layers_definition(&layers).contains("sink"));
        let mut sunk = layers.clone();
        sunk[1].sink = Some(7);
        assert_eq!(super::layers_definition(&sunk), format!("{sunk:?}").replace(", sink: None", "").replace(", thresholds: []", "").replace(", components: []", ""));
        assert_ne!(super::layers_definition(&sunk), super::layers_definition(&layers));
    }

    use super::*;
    use crate::{
        import::import_language_model,
        run_check::{layer_nodes, split_sites},
    };

    /// A library operator copied from a stored native operator reads the checkpoint's reals where
    /// they are, the same reals as a copy of the native operator's float64 values; from an operator
    /// held on the host, it holds them on the host.
    #[test]
    fn a_library_copy_of_a_stored_operator_keeps_it_stored() {
        let (rows, cols) = (3usize, 4usize);
        let values: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.375 - 1.5).collect();
        let header = format!(r#"{{"w":{{"dtype":"F32","shape":[{rows},{cols}],"data_offsets":[0,{}]}}}}"#, 4 * rows * cols);
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        values.iter().for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
        let path = std::env::temp_dir().join(format!("library_copy_{}.safetensors", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let file = crate::safetensors::SafetensorsFile::open(&path).unwrap();
        let (r, c) = (Interface::native(rows).unwrap(), Interface::native(cols).unwrap());
        let native = Operator::stored("w", r.clone(), c.clone(), file.stored("w", rows, cols).unwrap(), Provenance::native("w")).unwrap();
        std::fs::remove_file(&path).unwrap();
        let copy = library_copy("library.w", r.clone(), c.clone(), &native).unwrap();
        let held = library_operator("library.w", r.clone(), c.clone(), native.matrix(), "w").unwrap();
        let OperatorBody::Dense { values: copied, .. } = &copy.body else { panic!("a dense copy") };
        assert!(copied.stored().is_some(), "the copy reads the checkpoint");
        assert_eq!(copy, held, "the same operator as a float64 copy");
        let from_host = library_copy("library.w", r, c, &held).unwrap();
        let OperatorBody::Dense { values: hosted, .. } = &from_host.body else { panic!("a dense copy") };
        assert!(hosted.stored().is_none() && from_host.matrix() == held.matrix());
    }

    /// `KL(M_e ‖ P_e)` per scored token in bits for `experiments` on `batch` at the explanation's
    /// host values `theta`, and with `gradient` the gradient of its sum in every trainable
    /// operator, on the host.
    fn score(scorer: &mut Scorer, batch: &Batch, experiments: &[Experiment], theta: &[Array2<f64>], gradient: bool) -> Result<(Vec<Vec<f64>>, Vec<Array2<f64>>), String> {
        let targets = scorer.experiments.targets(batch, experiments)?;
        scorer.experiments.load(theta)?;
        let evaluation = scorer.experiments.evaluate_resident(batch, experiments, &targets, gradient)?;
        if evaluation.bits.iter().flatten().any(|b| !b.is_finite()) {
            return Err("nonfinite explanation divergence".into());
        }
        if !gradient {
            return Ok((evaluation.bits, Vec::new()));
        }
        let d = scorer.experiments.models().1.program.device();
        let mut gradients = Vec::with_capacity(theta.len());
        for (op, values) in scorer.position.iter().map(|(op, i)| (*op, &theta[*i])) {
            gradients.push(match evaluation.gradient.get(&op) {
                Some(g) => d.download(g).map_err(error)?,
                None => Array2::zeros(values.dim()),
            });
        }
        Ok((evaluation.bits, gradients))
    }

    /// The weight sample of `key` of every trainable operator of `posterior` (the device's draws,
    /// `DevicePosterior::sample_into`), on the host.
    fn sample(posterior: &Posterior, key: u64) -> Vec<Array2<f64>> {
        host_sample(posterior, &(0..posterior.mean.len()).collect::<Vec<_>>(), key).into_values().collect()
    }

    /// The derivatives in `μ` and in `ln σ` of `data(θ) + Σ_G KL_G`, per trainable operator, from
    /// `data`'s gradient at the sample `θ = μ + σ ⊙ noise` (zero in removed groups); the host
    /// reference of the device step.
    fn derivatives(posterior: &Posterior, data: &[Array2<f64>], noise: &[Array2<f64>]) -> Vec<(Array2<f64>, Array2<f64>)> {
        let variance = posterior.variances();
        (0..data.len())
            .into_par_iter()
            .map(|i| {
                let (mut mean, mut log_sd) = (data[i].clone(), Array2::zeros(data[i].dim()));
                ndarray::Zip::from(&mut mean).and(&mut log_sd).and(&noise[i]).and(&*posterior.mean[i]).and(&*posterior.log_sd[i]).and(&posterior.membership[i]).for_each(
                    |gm, gs, e, mu, s, group| {
                        let v = variance[*group as usize];
                        if v > 0.0 {
                            let sd = s.exp();
                            *gs = *gm * e * sd + sd * sd / v - 1.0;
                            *gm += mu / v;
                        } else {
                            *gm = 0.0;
                        }
                    },
                );
                (mean, log_sd)
            })
            .collect()
    }

    /// The tiny two-layer decoder export with MLP law `law`: its split program, layers, and its six
    /// sequences of twelve tokens.
    fn tiny(tag: &str, law: &str) -> (OperatorProgram, Vec<LayerNodes>, FamilyInputs, Vec<Vec<u32>>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let path = dir.join("export.json");
        let mut record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        record["config"]["mlp_act"] = law.into();
        std::fs::write(&path, record.to_string()).unwrap();
        let imported = import_language_model(&dir, 6, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let family = imported.family;
        let SlotValues::Tokens(tokens) = &family.slots[0] else { panic!("token slot") };
        let sequences = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        (native, layers, family, sequences)
    }

    /// The tiny decoder made like Qwen3: SiLU-gated MLPs, an RMS norm with a gain on every head's
    /// query and key, and one key-value head shared by both query heads.
    fn tiny_qwen3(tag: &str) -> (OperatorProgram, Vec<LayerNodes>, FamilyInputs, Vec<Vec<u32>>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let path = dir.join("export.json");
        let mut record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let (d, mlp, head) = (8, 16, 4);
        let mut rng = StdRng::seed_from_u64(11);
        for l in 0..2 {
            for (name, shape, centre) in [
                ("attn.k_proj", [head, d], 0.0),
                ("attn.v_proj", [head, d], 0.0),
                ("mlp.gate_proj", [mlp, d], 0.0),
                ("attn.q_norm.gain", [1, head], 1.0),
                ("attn.k_norm.gain", [1, head], 1.0),
            ] {
                let name = format!("blocks.{l}.{name}");
                let bytes: Vec<u8> = (0..shape[0] * shape[1]).flat_map(|_| (centre + rng.random::<f64>() - 0.5).to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{name}.f64")), bytes).unwrap();
                record["files"][name] = serde_json::json!({"shape": shape});
            }
        }
        let config = &mut record["config"];
        config["n_kv_heads"] = 1.into();
        config["mlp_act"] = "silu".into();
        config["mlp_gated"] = true.into();
        config["qk_norm"] = true.into();
        std::fs::write(&path, record.to_string()).unwrap();
        let imported = import_language_model(&dir, 6, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let family = imported.family;
        let SlotValues::Tokens(tokens) = &family.slots[0] else { panic!("token slot") };
        let sequences = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        (native, layers, family, sequences)
    }

    #[test]
    fn compact_targets_of_a_tied_qwen3_head_are_the_full_logit_divergence() {
        use crate::resident_causal_fit::fixed_head_target::{Head, ResidentHead, Teacher};
        use gam_gpu::tensor::Arithmetic;
        // M's head is its token embedding read transposed (tied) after the final norm. `P` is M
        // with the first MLP's output map scaled, so the two next-token distributions differ.
        let (native, _, family, _) = tiny_qwen3("library_tied_head");
        let mut changed = native.clone();
        let down = changed.operators.iter().position(|op| op.name == "blocks.0.down_proj").unwrap();
        let op = Arc::clone(&changed.operators[down]);
        changed.operators[down] = Arc::new(library_operator(&op.name, op.rows.clone(), op.cols.clone(), op.matrix() * 1.7, &op.name).unwrap());
        let logits = |program: &OperatorProgram| program.execute(&family, false).unwrap().values[program.output].clone();
        let (p, q) = (logits(&native), logits(&changed));
        let log_softmax = |row: ndarray::ArrayView1<'_, f64>| {
            let peak = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let partition = peak + row.iter().map(|x| (x - peak).exp()).sum::<f64>().ln();
            row.mapv(|x| x - partition)
        };
        let full: Vec<f64> = p
            .rows()
            .into_iter()
            .zip(q.rows())
            .map(|(p, q)| {
                let (lp, lq) = (log_softmax(p), log_softmax(q));
                lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
            })
            .collect();
        let device = Device::host();
        let head = Head::of(&native).unwrap();
        let target = Teacher::new(&device, &native, 16, 1 << 26).unwrap().target(&family, None).unwrap();
        let prefix = crate::device_program::DeviceProgram::compile_values(&device, &head.prefix(&changed)).unwrap();
        let trace = prefix.forward(&family).unwrap();
        let (compact, _, _) =
            ResidentHead::new(&device, &head, 16).unwrap().score(&device, trace.value(prefix.hidden()).unwrap(), &target, false, None, Arithmetic::F64).unwrap();
        assert_eq!(compact.len(), full.len());
        assert!(full.iter().sum::<f64>() > 1e-3, "the scaled MLP moves the distributions");
        for (row, (a, b)) in compact.iter().zip(&full).enumerate() {
            assert!((a - b).abs() <= 1e-10 * (1.0 + b.abs()), "row {row}: compact KL {a} against the full-logit KL {b}");
        }
    }

    #[test]
    fn the_starting_library_of_a_qwen3_decoder_is_the_model_in_every_experiment() {
        let (native, layers, family, sequences) = tiny_qwen3("library_start_qwen3");
        let explanation = explanation(&native, &layers).unwrap();
        explanation.artifact.validate_coverage(&native).unwrap();
        assert_eq!(explanation.artifact.blocks.len(), 2 * (2 + 1), "two heads and one MLP per layer");
        // Per key-value group (one, read by both query heads) two rotary planes and four value rows;
        // per MLP function its gate, up direction and output.
        assert_eq!(explanation.groups.len(), 2 * ((2 + 4) + 16 * 3));
        assert!(explanation.layers.iter().all(|l| l.functions.iter().all(|f| f.len() == 3)));
        assert!(explanation.layers.iter().all(|l| l.heads[0] == l.heads[1]), "both query heads hold their group's planes and values");
        for l in 0..2 {
            assert_eq!(key_value(&explanation.artifact.program, l, 0), key_value(&explanation.artifact.program, l, 1), "one key and value map per key-value group");
        }
        let expected = native.execute(&family, false).unwrap().values[native.output].clone();
        let actual = explanation.artifact.execute(&family).unwrap().values[explanation.artifact.program.output].clone();
        let scale = expected.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let difference = expected.iter().zip(&actual).fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(difference <= 1e-12 * scale, "the starting library differs from the native model by {difference}");
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let cells: usize = explanation.groups.iter().flat_map(|g| &g.cells).map(|c| c.rows.len() * c.cols.len()).sum();
        assert_eq!(cells, posterior.mean.iter().map(|m| m.len()).sum::<usize>(), "the groups partition the parameters");
        let settings = settings();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 72.0, None, 0).unwrap();
        // Every cut and a read patch must be drawn: each held-out base draws one clean and one
        // patched experiment, and six bases leave some of the families out, so the six token
        // rows are cycled into 96 bases (a family drawn with chance p per experiment is missed
        // with chance (1 - p)^96).
        let sequences: Vec<Vec<u32>> = sequences.iter().cycle().take(96).cloned().collect();
        let evaluation = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings, 72, None).unwrap();
        // Every family the held-out sample drew (which hybrid sizes it holds depends on its seed).
        assert!(evaluation.read_patch.is_some() && evaluation.clean.iter().chain(&evaluation.patched).filter(|b| b.is_some()).count() >= 2);
        for bits in evaluation.clean.iter().chain(&evaluation.patched).chain([&evaluation.read_patch]).flatten() {
            assert!(bits.abs() < 1e-10, "the starting library diverges from the model by {bits} bits per token");
        }
        for (count, layer) in evaluation.layers.iter().zip(&explanation.layers) {
            assert_eq!(count.functions, layer.functions.len());
            assert!(count.resolved_per_token <= count.nonzero_per_token && count.nonzero_per_token <= layer.functions.len() as f64);
        }
    }

    #[test]
    fn a_qwen3_library_fit_never_increases_its_objective() {
        let (native, layers, _, sequences) = tiny_qwen3("library_fit_qwen3");
        let explanation = explanation(&native, &layers).unwrap();
        let (train, held) = sequences.split_at(4);
        let fit = fit(&Device::host(), &native, &explanation, train, held, &settings(), "tiny", None, None).unwrap();
        let report = &fit.report;
        assert_eq!(report.removals.last().unwrap().removed, 0, "the fit ends when no removal is accepted");
        assert!(report.removals.iter().all(|r| r.after_bits <= r.before_bits), "a removal never increases the objective");
        assert!(report.epochs.iter().all(|e| e.snapshot_bits.is_finite() && e.data_bits.is_finite() && e.held_out.objective_bits_per_token.is_finite()));
        let fitted = posterior_mean(&explanation, &fit.posterior).unwrap();
        fitted.validate_coverage(&native).unwrap();
        // After training, both query heads still read one key and one value map, moved from M's.
        for l in 0..2 {
            let ((k, v), other) = (key_value(&fitted.program, l, 0), key_value(&fitted.program, l, 1));
            assert_eq!((k, v), other);
            assert_ne!(fitted.program.operators[k].matrix(), explanation.artifact.program.operators[k].matrix(), "training moves the shared key");
        }
    }

    /// The key and value operators head `h` of layer `l` reads.
    fn key_value(program: &OperatorProgram, l: usize, h: usize) -> (usize, usize) {
        let rule = program.rules.iter().find(|r| r.name == format!("library.l{l}.h{h}")).unwrap();
        let map = |node: usize| match &rule.nodes[node] {
            Node::Affine { terms, .. } => terms[0].1,
            other => panic!("{other:?}"),
        };
        (map(2), map(3))
    }

    #[test]
    fn every_parameter_block_names_the_native_parameter_it_replaces_and_survives_the_bytes() {
        for (native, layers, _, _) in [tiny("library_owners", "gelu_tanh"), tiny_qwen3("library_owners_qwen3")] {
            let explanation = explanation(&native, &layers).unwrap();
            let program = &explanation.artifact.program;
            let owners = &explanation.artifact.owners;
            for owner in owners {
                let op = &program.operators[index_of(program, &owner.operator).unwrap()];
                let source = native.operators.iter().find(|o| o.name == owner.native).expect("a native owner of M");
                assert!(owner.rows.end <= op.rows.width() && owner.cols.end <= op.cols.width());
                assert!(owner.native_rows.end <= source.rows.width() && owner.native_cols.end <= source.cols.width());
                assert_eq!((owner.rows.len(), owner.cols.len()), (owner.native_rows.len(), owner.native_cols.len()));
                // At the start, P's block is M's block.
                let (ours, theirs) = (op.matrix(), source.matrix());
                for (r, nr) in owner.rows.clone().zip(owner.native_rows.clone()) {
                    for (c, nc) in owner.cols.clone().zip(owner.native_cols.clone()) {
                        assert_eq!(ours[[r, c]], theirs[[nr, nc]], "{} against {}", owner.operator, owner.native);
                    }
                }
            }
            // Every trainable entry has an owner.
            for &op in &explanation.trainable {
                let name = &program.operators[op].name;
                let (rows, cols) = (program.operators[op].rows.width(), program.operators[op].cols.width());
                for r in 0..rows {
                    for c in 0..cols {
                        assert!(owners.iter().any(|o| &o.operator == name && o.rows.contains(&r) && o.cols.contains(&c)), "{name} ({r}, {c}) has no owner");
                    }
                }
            }
            // A key-value group's shared key is owned once per query head reading it.
            let key_sites: Vec<&str> = owners.iter().filter(|o| o.operator == "library.l0.kv0.k").map(|o| o.site.as_str()).collect();
            let heads_reading = program.rules.iter().filter(|r| r.name.starts_with("library.l0.h") && r.nodes.iter().any(|n| matches!(n, Node::Affine { terms, .. } if terms.iter().any(|t| program.operators[t.1].name == "library.l0.kv0.k")))).count();
            assert_eq!(key_sites.len(), heads_reading);
            let bytes = explanation.artifact.f32_literals().unwrap().to_bytes().unwrap();
            let decoded = Artifact::from_bytes(&bytes, &native.declarations).unwrap();
            assert_eq!(&decoded.owners, owners);
        }
    }

    #[test]
    fn the_reported_objective_pays_for_the_explanation_s_choices() {
        let (native, layers, _, sequences) = tiny("library_choice_bits", "gelu_tanh");
        let mut explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 72.0, None, 0).unwrap();
        let without = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings, 72, None).unwrap();
        explanation.fixed_nats = 1000_f64.ln();
        let with = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings, 72, None).unwrap();
        assert!((with.choice_bits - 1000_f64.log2()).abs() < 1e-12);
        let added = (with.objective_bits_per_token - without.objective_bits_per_token) * 72.0;
        assert!((added - 1000_f64.log2()).abs() < 1e-9, "F per token gains the choice's bits over the scored tokens: {added}");
    }

    #[test]
    fn a_tied_gate_is_differentiated_through_both_of_its_uses() {
        let (native, layers, _, sequences) = tiny("library_tie_gradient", "gelu_tanh");
        let start = explanation(&native, &layers).unwrap();
        let tie = crate::library_sharing::Tie { source: (0, 3), target: (1, 5), scale: 0.7 };
        let explanation = crate::library_sharing::tie(&start, &[tie]).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let theta = sample(&posterior, noise_seed(settings.seed, 0, 0));
        let (_, gradients) = score(&mut scorer, &batch, &experiments, &theta, true).unwrap();
        let at = |name: &str| explanation.trainable.iter().position(|op| explanation.artifact.program.operators[*op].name == name).unwrap();
        let (gate, scale) = (at("library.l1.mlp.gate"), at("library.l0.mlp.tie1.f3.scale"));
        let mut bits = |theta: &[Array2<f64>]| -> f64 {
            score(&mut scorer, &batch, &experiments, theta, false).unwrap().0.iter().flatten().sum()
        };
        for (i, entry) in [(gate, (5, 0)), (gate, (5, 6)), (scale, (0, 0))] {
            let h = 1e-5;
            let (mut up, mut down) = (theta.clone(), theta.clone());
            up[i][entry] += h;
            down[i][entry] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!((gradients[i][entry] - central).abs() <= 1e-5 * (1.0 + central.abs()), "tied gradient {} against {central}", gradients[i][entry]);
        }
    }

    #[test]
    fn a_block_shared_across_layers_is_differentiated_through_both_of_its_uses() {
        let (native, layers, _, sequences) = tiny("library_block_gradient", "gelu_tanh");
        let start = explanation(&native, &layers).unwrap();
        let row = crate::library_sharing::tie_row(&start, "gate", (1, 7), crate::library_sharing::RowSource::Row { layer: 0, part: "gate", function: 4 }, 0.8).unwrap();
        let explanation = crate::library_sharing::tie_column(&row, (1, 7), (0, 4), -0.6).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let theta = sample(&posterior, noise_seed(settings.seed, 0, 0));
        let (_, gradients) = score(&mut scorer, &batch, &experiments, &theta, true).unwrap();
        let at = |name: &str| explanation.trainable.iter().position(|op| explanation.artifact.program.operators[*op].name == name).unwrap();
        let (gate, out) = (at("library.l0.mlp.gate"), at("library.l0.mlp.out"));
        let (row_scale, column_scale) = (at("library.l1.mlp.f7.gate.from_l0_gate4.scale"), at("library.l1.mlp.f7.out.from_l0_4.scale"));
        let mut bits = |theta: &[Array2<f64>]| -> f64 { score(&mut scorer, &batch, &experiments, theta, false).unwrap().0.iter().flatten().sum() };
        for (i, entry) in [(gate, (4, 0)), (gate, (4, 5)), (out, (2, 4)), (out, (6, 4)), (row_scale, (0, 0)), (column_scale, (0, 0))] {
            let h = 1e-5;
            let (mut up, mut down) = (theta.clone(), theta.clone());
            up[i][entry] += h;
            down[i][entry] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!((gradients[i][entry] - central).abs() <= 1e-5 * (1.0 + central.abs()), "shared block gradient {} against {central}", gradients[i][entry]);
        }
    }

    #[test]
    fn a_key_value_group_shared_across_layers_is_differentiated_through_both_of_its_uses() {
        use crate::library_sharing::{Member, share_query_key, share_value};
        let (native, layers, _, sequences) = tiny_qwen3("library_group_gradient");
        let start = explanation(&native, &layers).unwrap();
        // Layer 1's query heads read layer 0's query-key maps, swapped (unscaled: the heads norm
        // their queries), and its value map is layer 0's through the output projections' transport.
        let shared = share_query_key(&start, &[Member { layer: 0, group: 0, queries: vec![0, 1] }, Member { layer: 1, group: 0, queries: vec![1, 0] }]).unwrap();
        let explanation = share_value(&shared, (1, 0), (0, 0), 0.7).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let theta = sample(&posterior, noise_seed(settings.seed, 0, 0));
        let (_, gradients) = score(&mut scorer, &batch, &experiments, &theta, true).unwrap();
        let at = |name: &str| explanation.trainable.iter().position(|op| explanation.artifact.program.operators[*op].name == name).unwrap();
        let entries = [
            (at("library.l0.h0.q"), (1, 3)),
            (at("library.l0.h1.q"), (2, 5)),
            (at("library.l0.kv0.k"), (0, 6)),
            (at("library.l0.kv0.v"), (3, 2)),
            (at("library.l1.kv0.v_from_l0_kv0.scale"), (0, 0)),
        ];
        let mut bits = |theta: &[Array2<f64>]| -> f64 { score(&mut scorer, &batch, &experiments, theta, false).unwrap().0.iter().flatten().sum() };
        for (i, entry) in entries {
            let h = 1e-5;
            let (mut up, mut down) = (theta.clone(), theta.clone());
            up[i][entry] += h;
            down[i][entry] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!((gradients[i][entry] - central).abs() <= 1e-5 * (1.0 + central.abs()), "shared group gradient {} against {central}", gradients[i][entry]);
        }
    }

    #[test]
    fn a_shared_key_is_differentiated_through_every_query_head_reading_it() {
        let (native, layers, _, sequences) = tiny_qwen3("library_shared_key_gradient");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let theta = sample(&posterior, noise_seed(settings.seed, 0, 0));
        let (_, gradients) = score(&mut scorer, &batch, &experiments, &theta, true).unwrap();
        let (k, _) = key_value(&explanation.artifact.program, 1, 0);
        let i = explanation.trainable.iter().position(|op| *op == k).unwrap();
        let mut bits = |theta: &[Array2<f64>]| -> f64 {
            score(&mut scorer, &batch, &experiments, theta, false).unwrap().0.iter().flatten().sum()
        };
        for at in [(0, 0), (1, 3), (3, 7)] {
            let h = 1e-5;
            let (mut up, mut down) = (theta.clone(), theta.clone());
            up[i][at] += h;
            down[i][at] -= h;
            let central = (bits(&up) - bits(&down)) / (2.0 * h);
            assert!((gradients[i][at] - central).abs() <= 1e-5 * (1.0 + central.abs()), "shared key gradient {} against {central}", gradients[i][at]);
        }
    }

    /// The budget's step terms on the tiny decoder with ReLU functions: the derivative of the
    /// batch's expected parts per token in a gate mean of the last layer (whose count no later
    /// layer's input carries) matches the change of the count itself under that mean's tangent,
    /// central differences of `complexity_terms` with the posterior moved along it (relative 1e-5).
    #[test]
    fn the_budget_terms_derivative_matches_the_counts_tangent() {
        let (native, layers, _, sequences) = tiny("library_budget_tangent", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let key = training_key(settings.seed, 0, 0);
        let mut terms_at = |posterior: &Posterior| {
            let device_posterior = DevicePosterior::new(&device, &explanation, posterior, 72.0, None, 0).unwrap();
            complexity_terms(&mut scorer, &device_posterior, &explanation, &posterior.active, &batch, (key, false)).unwrap()
        };
        let (count, terms) = terms_at(&posterior);
        let terms: Vec<(usize, Array2<f64>, Array2<f64>)> = terms.iter().map(|(i, m, v)| (*i, device.download(m).unwrap(), device.download(v).unwrap())).collect();
        assert!(count > 0.0);
        // The last layer's gate: its trainable index is the largest among the gates' terms whose
        // derivative has the shape of a gate (more than one column).
        let (i, gradient, _) = terms.iter().filter(|(_, m, _)| m.ncols() > 1).max_by_key(|(i, _, _)| *i).unwrap();
        let (r, c) = gradient.indexed_iter().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap().0;
        assert!(gradient[[r, c]] != 0.0);
        let h = 1e-6;
        let moved = |delta: f64| {
            let mut p = posterior.clone();
            p.mean[*i][[r, c]] += delta;
            p
        };
        let numeric = (terms_at(&moved(h)).0 - terms_at(&moved(-h)).0) / (2.0 * h);
        assert!((numeric - gradient[[r, c]]).abs() <= 1e-5 * (1.0 + numeric.abs()), "∂Ê/∂μ {} against the count's tangent {numeric}", gradient[[r, c]]);
    }

    /// The tiny decoder as VPD-style gated slices with learned widths (`library_vpd`,
    /// `Gate::Learned`): each MLP input column of each layer its own component with an own gate at a
    /// threshold its reads cross (`τ` 0.5, width 0.5), the rest one always-on component per layer;
    /// with its native program and its sequences of 12 tokens.
    fn learned_tiny(tag: &str) -> (OperatorProgram, Explanation, Vec<Vec<u32>>) {
        learned_tiny_mixed(tag, false)
    }

    /// [`learned_tiny`], with `mixing` exact (`library_vpd`'s mixing, [`Mix`]): in each layer the q
    /// slices 0–2 with one extra slice (in the always-on component), the c_fc slices in threes each
    /// with one extra slice (in the component of the group's first) and the down slices in fours,
    /// every slice's mean pinned.
    fn learned_tiny_mixed(tag: &str, mixing: bool) -> (OperatorProgram, Explanation, Vec<Vec<u32>>) {
        learned_tiny_with(tag, mixing, None)
    }

    /// [`learned_tiny`] with the first layer's MLP components on direction gates (`g = e_i`,
    /// threshold 0.5, so component `i` runs where input coordinate `i` exceeds 0.5) and its
    /// component 0 a block across blocks: with `cross` true it also runs the second layer's q slice
    /// 1, o slice 1 and down slice 0 (taken from their components); with `cross` false component 0
    /// and all those slices are absent.
    fn cross_tiny(tag: &str, cross: bool) -> (OperatorProgram, Explanation, Vec<Vec<u32>>) {
        learned_tiny_with(tag, false, Some(cross))
    }

    fn learned_tiny_with(tag: &str, mixing: bool, cross: Option<bool>) -> (OperatorProgram, Explanation, Vec<Vec<u32>>) {
        let dir = crate::test_support::tiny_export(tag, 2);
        let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("export.json")).unwrap()).unwrap();
        let factors = std::env::temp_dir().join(format!("gam_mpd_{tag}_factors_{}", std::process::id()));
        std::fs::create_dir_all(&factors).unwrap();
        let (mut files, mut sites, mut components) = (serde_json::Map::new(), Vec::new(), Vec::new());
        let (mut groups, mut extra): (Vec<Vec<[usize; 2]>>, Vec<[usize; 2]>) = (Vec::new(), Vec::new());
        for l in 0..2 {
            let (mut always, mut mlp): (Vec<[usize; 2]>, Vec<Vec<[usize; 2]>>) = (Vec::new(), Vec::new());
            for (k, name) in ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "mlp.c_fc", "mlp.down_proj"].into_iter().enumerate() {
                let shape = &record["files"][format!("blocks.{l}.{name}")]["shape"];
                let (r, c) = (shape[0].as_u64().unwrap() as usize, shape[1].as_u64().unwrap() as usize);
                let w = crate::import::read_f64_shaped(&dir.join(format!("blocks.{l}.{name}.f64")), r, c).unwrap();
                let site = format!("h.{l}.{name}");
                for (suffix, values) in [("U", w.t().to_owned()), ("V", Array2::eye(c))] {
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(factors.join(format!("{site}.{suffix}.f64")), bytes).unwrap();
                    files.insert(format!("{site}.{suffix}"), serde_json::json!({"shape": [values.nrows(), values.ncols()]}));
                }
                sites.push(site);
                match k {
                    // Component i: the MLP's input column i and its hidden units j ≡ i.
                    4 => mlp = (0..c).map(|i| vec![[6 * l + k, i]]).collect(),
                    5 => {
                        let n = mlp.len();
                        (0..c).for_each(|j| mlp[j % n].push([6 * l + k, j]));
                    }
                    _ => always.extend((0..c).map(|i| [6 * l + k, i])),
                }
            }
            if mixing {
                let width = |name: &str| record["files"][format!("blocks.{l}.{name}")]["shape"][1].as_u64().unwrap() as usize;
                let (q, d) = (width("attn.q_proj"), width("mlp.c_fc"));
                always.push([6 * l, q]);
                groups.push((0..3).map(|i| [6 * l, i]).chain([[6 * l, q]]).collect());
                let chunks: Vec<Vec<usize>> = (0..d).collect::<Vec<_>>().chunks(3).filter(|c| c.len() > 1).map(<[usize]>::to_vec).collect();
                for (g, chunk) in chunks.iter().enumerate() {
                    mlp[chunk[0]].push([6 * l + 4, d + g]);
                    groups.push(chunk.iter().map(|&i| [6 * l + 4, i]).chain([[6 * l + 4, d + g]]).collect());
                }
                extra.extend([[6 * l, 1], [6 * l + 4, chunks.len()]]);
                groups.extend((0..width("mlp.down_proj")).collect::<Vec<_>>().chunks(4).filter(|c| c.len() > 1).map(|c| c.iter().map(|&i| [6 * l + 5, i]).collect()));
            }
            let carried = [[6, 1], [9, 1], [11, 0]];
            if cross.is_some() && l == 1 {
                always.retain(|s| !carried.contains(s));
                mlp.iter_mut().for_each(|c| c.retain(|s| !carried.contains(s)));
            }
            components.push(serde_json::json!({"read": {"own": [6 * l, 0]}, "tau": -1.0, "width": 1e-3, "slices": always}));
            if cross.is_some() && l == 0 {
                let d = record["files"]["blocks.0.mlp.c_fc"]["shape"][1].as_u64().unwrap() as usize;
                for (i, mut slices) in mlp.into_iter().enumerate() {
                    if i == 0 && cross == Some(false) {
                        continue;
                    }
                    if i == 0 {
                        slices.extend(carried);
                    }
                    let coefficients: Vec<f64> = (0..=d).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
                    components.push(serde_json::json!({"read": {"direction": {"site": 4, "coefficients": coefficients}}, "tau": 0.5, "width": 0.5, "slices": slices}));
                }
            } else {
                components.extend(mlp.into_iter().map(|slices| serde_json::json!({"read": {"own": slices[0]}, "tau": 0.5, "width": 0.5, "slices": slices})));
            }
        }
        std::fs::write(factors.join("export.json"), serde_json::json!({"config": {"sites": sites}, "files": files}).to_string()).unwrap();
        let start = factors.join("start.json");
        std::fs::write(&start, serde_json::json!([{"arm": "learned", "components": components, "mixing": groups, "extra": extra}]).to_string()).unwrap();
        let imported = crate::import::import_language_model(&dir, 6, 12).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        let native = crate::run_check::split_sites(&imported.program).unwrap();
        let layers = crate::run_check::layer_nodes(&native, 2).unwrap();
        let explanation = crate::library_vpd::explanation_with_gate(&native, &layers, &factors, &start, "learned", crate::library_vpd::Gate::Learned).unwrap();
        std::fs::remove_dir_all(&factors).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("a token slot") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        (native, explanation, sequences)
    }

    /// The budget's count on VPD-style gated slices with learned widths (`library_vpd`,
    /// `Gate::Learned`), each MLP input column of the tiny decoder its own component with an own gate
    /// at a threshold its reads cross (`τ` 0.5, width 0.5), the rest always on: the count's
    /// derivative in a threshold is nonzero and matches central differences of the count itself
    /// (`s² = w² + σ²` in both), and none of the budget's terms is a width's.
    #[test]
    fn a_learned_gates_count_answers_its_thresholds() {
        let (native, explanation, sequences) = learned_tiny("library_budget_learned");
        let settings = settings();
        let device = Device::host();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let key = training_key(settings.seed, 0, 0);
        let mut terms_at = |posterior: &Posterior| {
            let device_posterior = DevicePosterior::new(&device, &explanation, posterior, 72.0, None, 0).unwrap();
            complexity_terms(&mut scorer, &device_posterior, &explanation, &posterior.active, &batch, (key, false)).unwrap()
        };
        let (count, terms) = terms_at(&posterior);
        let name = |i: usize| explanation.artifact.program.operators[explanation.trainable[i]].name.clone();
        assert!(terms.iter().all(|(i, _, _)| !name(*i).ends_with(".width")), "a width takes the budget's pull");
        // The second layer's MLP thresholds (whose count no later layer's input carries).
        let (i, gradient) = terms.iter().map(|(i, m, _)| (*i, device.download(m).unwrap())).filter(|(i, _)| name(*i) == "library.l1.mlp.fc.threshold" || name(*i).starts_with("library.l1.mlp") && name(*i).ends_with("threshold")).last().expect("the last MLP's thresholds");
        let (r, c) = gradient.indexed_iter().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())).unwrap().0;
        assert!(count > 0.0 && gradient[[r, c]].abs() > 1e-3, "∂Ê/∂τ {} at a count of {count}", gradient[[r, c]]);
        let h = 1e-6;
        let moved = |delta: f64| {
            let mut p = posterior.clone();
            p.mean[i][[r, c]] += delta;
            p
        };
        let numeric = (terms_at(&moved(h)).0 - terms_at(&moved(-h)).0) / (2.0 * h);
        assert!((numeric - gradient[[r, c]]).abs() <= 1e-5 * (1.0 + numeric.abs()), "∂Ê/∂τ {} against the count's central difference {numeric}", gradient[[r, c]]);
    }

    /// The move's test counts the budget's parts on each side of the move at that side's own gate
    /// means (`complexity_terms` with `previous`): after a move of the thresholds alone, the old
    /// side's count is the count before the move, bit for bit, and the new side's differs.
    #[test]
    fn the_moves_old_side_counts_at_the_old_thresholds() {
        let (native, explanation, sequences) = learned_tiny("library_old_side");
        let (device, settings) = (Device::host(), settings());
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let key = training_key(settings.seed, 0, 0);
        let before = complexity_terms(&mut scorer, &device_posterior, &explanation, &posterior.active, &batch, (key, false)).unwrap().0;
        let mut moved = posterior.clone();
        for (i, op) in explanation.trainable.iter().enumerate() {
            if explanation.artifact.program.operators[*op].name.ends_with("threshold") {
                moved.mean[i].mapv_inplace(|t| t + 0.3);
            }
        }
        device_posterior.propose(&moved).unwrap();
        let old = complexity_terms(&mut scorer, &device_posterior, &explanation, &posterior.active, &batch, (key, true)).unwrap().0;
        let new = complexity_terms(&mut scorer, &device_posterior, &explanation, &posterior.active, &batch, (key, false)).unwrap().0;
        assert_eq!(old, before, "the old side's count, at the old thresholds");
        assert!((new - before).abs() > 1e-6, "a move of the thresholds moves the count: {before} to {new}");
    }

    /// The snapshot the best epoch is chosen by (`snapshot_estimates`, as the removal round's
    /// comparisons) scores F's data term with the all-on experiment: each batch's value is its
    /// experiments' bits plus its all-on experiment's, both through the hard gates at the batch's
    /// sample, and the all-on part is not zero.
    #[test]
    fn the_snapshot_scores_the_all_on_experiment() {
        let (native, explanation, sequences) = learned_tiny("library_snapshot_all_on");
        let (device, settings) = (Device::host(), settings());
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let evidence = Evidence { draws: &draws, sequences: &sequences, settings: &settings };
        let (_, evaluation) = snapshot_estimates(&mut scorer, &device_posterior, &mut posterior, &explanation, &evidence, None).unwrap();
        for (b, draw) in draws.iter().enumerate() {
            scorer.next_batch();
            let (values, batch) = (Values::Sample(noise_seed(settings.seed, 0, b)), draw.batch(&sequences).unwrap());
            let experiments = scorer.experiments(draw, &sequences).unwrap();
            let (experiments, targets) = part_targets(&scorer, &batch, experiments).unwrap();
            let alone = scorer.pass(&device_posterior, (&batch, &experiments, &targets), (values, Gates::Hard, false), (false, None)).unwrap();
            let on = scorer.all_on_pass(&device_posterior, (&batch, &experiments), (values, Gates::Hard), (false, None)).unwrap().expect("an all-on experiment");
            let (alone, on) = (alone.bits.iter().flatten().sum::<f64>(), on.bits.iter().flatten().sum::<f64>());
            assert!(on > 0.0, "batch {b}: the all-on experiment's bits");
            assert_eq!(evaluation.batches[b], (alone + on) * LN_2, "batch {b}");
        }
    }

    /// The explanation as a fit scores it (`Gates::Hard`: the held-out evaluation, the snapshot the
    /// best epoch is chosen by, the removal round) is the artifact it exports (`posterior_mean`),
    /// bit for bit, with gates near their thresholds, where the relaxed training law differs: the
    /// exported artifact's own law is the hard gate.
    #[test]
    fn the_scored_explanation_is_the_exported_artifact() {
        let (native, explanation, sequences) = learned_tiny("library_scored_export");
        let (device, settings) = (Device::host(), settings());
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let targets = scorer.experiments.targets(&batch, &experiments).unwrap();
        let scored = scorer.pass(&device_posterior, (&batch, &experiments, &targets), (Values::Mean, Gates::Hard, false), (false, None)).unwrap().bits;
        let relaxed = scorer.pass(&device_posterior, (&batch, &experiments, &targets), (Values::Mean, Gates::Relaxed, false), (false, None)).unwrap().bits;
        let gap = |a: &[Vec<f64>], b: &[Vec<f64>]| a.iter().flatten().zip(b.iter().flatten()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(gap(&scored, &relaxed) > 1e-6, "gates near their thresholds, where the relaxed law differs");
        let artifact = posterior_mean(&explanation, &posterior).unwrap();
        let sites: Vec<LayerNodes> = explanation.layers.iter().map(|l| l.sites.clone()).collect();
        let exported = Interchange::new(&device, &native, &sites, &artifact, &[], explanation.reads.clone(), settings.numeric_bytes, settings.head_tile_rows).unwrap();
        let bits = exported.evaluate_resident(&batch, &experiments, &targets, false).unwrap().bits;
        // Equal to the last bits of the passes' summation orders.
        assert!(gap(&bits, &scored) <= 1e-12, "the exported artifact scores as the fit scores its explanation: {bits:?} against {scored:?}");
    }

    /// An exact frame (`library_vpd`'s `mixing`, [`Mix`]) keeps an exact explanation exact: at random
    /// frames `A` (with extra slices and without) and random writes, pinned onto the frames' exact
    /// sets, the all-on experiment of slices summing to `M`'s maps is `M` (its KL under 1e-9 bits per
    /// token, as at the start), and the explanation with its gates differs from the start's: the
    /// parts move, their sum does not.
    #[test]
    fn a_frame_moves_the_parts_and_keeps_their_sum() {
        let (native, explanation, sequences) = learned_tiny_mixed("library_mixing_exact", true);
        assert!(explanation.mixes.iter().any(|m| m.slices.len() > m.base) && explanation.mixes.iter().any(|m| m.slices.len() == m.base));
        let (device, settings) = (Device::host(), settings());
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let targets = scorer.experiments.targets(&batch, &experiments).unwrap();
        let score = |scorer: &mut Scorer, device_posterior: &DevicePosterior| {
            let alone = scorer.pass(device_posterior, (&batch, &experiments, &targets), (Values::Mean, Gates::Hard, false), (false, None)).unwrap().bits;
            let on = scorer.all_on_pass(device_posterior, (&batch, &experiments), (Values::Mean, Gates::Hard), (false, None)).unwrap().unwrap().bits;
            (alone, on.iter().flatten().fold(0.0f64, |m, b| m.max(b.abs())))
        };
        scorer.pin_mixings(&mut device_posterior).unwrap();
        let (start, on) = score(&mut scorer, &device_posterior);
        assert!(on <= 1e-9, "the all-on experiment at the start frames: {on} bits");
        let mut rng = StdRng::seed_from_u64(5);
        for m in &mut scorer.mixings {
            m.frame.mapv_inplace(|a| a + rng.random_range(-0.5..0.5));
        }
        let writes: Vec<(usize, Array2<f64>)> = scorer.mixed_writes.iter().map(|(&op, w)| (scorer.at(op).unwrap(), w.mapv(|v| v + rng.random_range(-0.5..0.5)))).collect();
        device_posterior.pin(&writes).unwrap();
        scorer.pin_mixings(&mut device_posterior).unwrap();
        let (moved, on) = score(&mut scorer, &device_posterior);
        assert!(on <= 1e-9, "the all-on experiment at random frames: {on} bits");
        let gap = start.iter().flatten().zip(moved.iter().flatten()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(gap > 1e-6, "the gated parts moved by {gap}");
    }

    /// The gradient in a frame ([`Mixing::gather`], through the reads `V₀ Aᵀ` and the writes
    /// `U₀ A⁺ + W (I − A A⁺)`) is the relaxed pass's data term's, by central differences along each
    /// entry of a group's `A` with an extra slice (1e-5), its writes held at their values `W`.
    #[test]
    fn a_frames_gradient_is_the_central_difference_of_its_pass() {
        let (native, explanation, sequences) = learned_tiny_mixed("library_mixing_gradient", true);
        let (device, settings) = (Device::host(), settings());
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let targets = scorer.experiments.targets(&batch, &experiments).unwrap();
        scorer.sample_gates = false;
        // A group of the second layer's c_fc slices with its extra slice, at a frame away from its
        // start, its writes on the frame's exact set there.
        let k = scorer.mixings.iter().rposition(|m| m.mix.base == 3 && m.mix.slices.len() == 4).unwrap();
        let start = &scorer.mixings[k].frame + &Array2::from_shape_fn((4, 3), |(i, j)| 0.1 * ((3 * i + j) as f64 - 5.5) / 5.5);
        scorer.mixings[k].frame = start.clone();
        scorer.pin_mixings(&mut device_posterior).unwrap();
        let writes: Vec<(usize, Array2<f64>)> = scorer.mixed_writes.iter().map(|(&op, w)| (scorer.at(op).unwrap(), w.clone())).collect();
        let mut value = |scorer: &mut Scorer, frame: &Array2<f64>, gradient: bool| {
            device_posterior.pin(&writes).unwrap();
            scorer.mixings[k].frame = frame.clone();
            scorer.pin_mixings(&mut device_posterior).unwrap();
            scorer.clear_mixing_gradients();
            let mut evaluation = scorer.pass(&device_posterior, (&batch, &experiments, &targets), (Values::Mean, Gates::Relaxed, false), (gradient, None)).unwrap();
            if gradient {
                scorer.take_mixing_gradients(&mut evaluation.gradient).unwrap();
            }
            evaluation.bits.iter().flatten().sum::<f64>()
        };
        value(&mut scorer, &start, true);
        let gradient = scorer.mixings[k].gradient.clone();
        let h = 1e-6;
        for i in 0..4 {
            for j in 0..3 {
                let mut direction = Array2::<f64>::zeros((4, 3));
                direction[[i, j]] = h;
                let numeric = (value(&mut scorer, &(&start + &direction), false) - value(&mut scorer, &(&start - &direction), false)) / (2.0 * h);
                assert!(numeric.abs() > 1e-6, "the pass answers the frame's entry ({i}, {j})");
                assert!((gradient[[i, j]] - numeric).abs() <= 1e-5 * (1.0 + numeric.abs()), "∂/∂a_{i}{j} {} against the central difference {numeric}", gradient[[i, j]]);
            }
        }
    }

    /// A budget in bits (`Settings::budget_bits`) weighs each part's runs by its description: a
    /// group's bits add to the count its part's expected runs per token (the count's change when one
    /// group's bits go from 0 to 1, the same for a slice's read and its write, and positive), and the
    /// count is linear in the groups' bits. A fit's groups cost [`NUMBER_BITS`] per number.
    #[test]
    fn a_budget_in_bits_weighs_each_part_by_its_groups() {
        let (native, explanation, sequences) = learned_tiny("library_budget_bits");
        let (device, settings) = (Device::host(), settings());
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let fixed = Scorer::new(&device, &native, &explanation, &Settings { budget_bits: true, ..settings.clone() }).unwrap().group_bits.unwrap();
        let read = explanation.groups.iter().position(|g| g.name.starts_with("library.l1.mlp.") && g.name.ends_with(".read")).unwrap();
        let width = explanation.groups[read].cells[0].cols.len() as f64;
        assert_eq!(fixed[read], NUMBER_BITS * width, "a slice's read costs 16 bits per entry");
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let key = training_key(settings.seed, 0, 0);
        let groups = explanation.groups.len();
        let mut count = |bits: Vec<f64>| {
            scorer.group_bits = Some(bits);
            complexity_terms(&mut scorer, &device_posterior, &explanation, &posterior.active, &batch, (key, false)).unwrap().0
        };
        let unit = |g: usize| (0..groups).map(|i| if i == g { 1.0 } else { 0.0 }).collect::<Vec<f64>>();
        let zero = count(vec![0.0; groups]);
        // The last MLP's first c_fc slice: its read and its write.
        let write = explanation.groups.iter().position(|g| g.name == explanation.groups[read].name.replace(".read", ".write")).unwrap();
        let (by_read, by_write) = (count(unit(read)) - zero, count(unit(write)) - zero);
        assert!(by_read > 0.0, "a slice's read weighs its part's runs: {by_read}");
        assert!((by_read - by_write).abs() <= 1e-12, "the read's {by_read} and the write's {by_write}");
        let both: Vec<f64> = (0..groups).map(|i| if i == read || i == write { 1.0 } else { 0.0 }).collect();
        assert!((count(both) - zero - by_read - by_write).abs() <= 1e-12, "the count is linear in the bits");
    }

    /// A block across blocks (`library_vpd`: a component gated at the first MLP's input that also
    /// runs a q, an o and a down slice of the second layer, its gate recomputed in each of their
    /// rules): with every part on its slices still sum to `M` (the all-on experiment under 1e-9
    /// bits); with its gate shut it removes exactly its slices, the explanation scoring as one
    /// without the component and those slices (1e-10); and the fit counts its rank across the
    /// blocks it runs in.
    #[test]
    fn a_block_across_blocks_runs_its_slices_under_its_one_gate() {
        let (native, explanation, sequences) = cross_tiny("library_cross_block", true);
        let (device, settings) = (Device::host(), settings());
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batch = draws[0].batch(&sequences).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        let targets = scorer.experiments.targets(&batch, &experiments).unwrap();
        // The component's rank, counted at its home: its c_fc slice, its first layer's down slices
        // and the three it runs in the second layer.
        let program = &explanation.artifact.program;
        let threshold = index_of(program, "library.l0.mlp.fc.threshold").unwrap();
        let home = scorer.stages[0].iter().find(|s| s.threshold == threshold).unwrap();
        let downs = program.operators[index_of(program, "library.l0.mlp.dn_read").unwrap()].rows.width() / program.operators[threshold].rows.width();
        assert_eq!(home.ranks(&posterior.active)[0], (1 + downs + 3) as f64, "the block's rank across its blocks");
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let on = scorer.all_on_pass(&device_posterior, (&batch, &experiments), (Values::Mean, Gates::Hard), (false, None)).unwrap().unwrap().bits;
        assert!(on.iter().flatten().all(|b| b.abs() <= 1e-9), "every part on is M");
        // Its gate shut: z = g·x + c with c far below zero.
        let t = explanation.trainable.iter().position(|op| *op == threshold).unwrap();
        posterior.mean[t][[0, 0]] = -1e6;
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let shut = scorer.pass(&device_posterior, (&batch, &experiments, &targets), (Values::Mean, Gates::Hard, false), (false, None)).unwrap().bits;
        let (native_b, without, _) = cross_tiny("library_cross_block_without", false);
        let posterior_b = Posterior::new(&without, 72).unwrap();
        let device_b = DevicePosterior::new(&device, &without, &posterior_b, 72.0, None, 0).unwrap();
        let mut scorer_b = Scorer::new(&device, &native_b, &without, &settings).unwrap();
        let reference = scorer_b.pass(&device_b, (&batch, &experiments, &targets), (Values::Mean, Gates::Hard, false), (false, None)).unwrap().bits;
        let gap = shut.iter().flatten().zip(reference.iter().flatten()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(gap <= 1e-10, "the shut block against no block: {gap}");
    }

    /// A frame is priced as any parameter is ([`Mixing::entry_nats`]): each entry's
    /// `KL(N(a, σ²) ‖ N(a₀, v))` about its start `a₀` with `σ² = 1 / (N h + 1/v)`, zero at the start
    /// and `h = 0`.
    #[test]
    fn a_frame_is_priced_by_its_laplace_code() {
        let (native, explanation, _) = learned_tiny_mixed("library_mixing_price", true);
        let (device, settings) = (Device::host(), settings());
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        scorer.mix_tokens = 1000.0;
        for m in &mut scorer.mixings {
            m.frame = Array2::from_shape_fn(m.frame.dim(), |(i, j)| Mixing::start(i, j));
        }
        assert_eq!(scorer.mixing_nats(), 0.0, "a frame at its start, uncurved, costs nothing");
        let k = scorer.mixings.iter().position(|m| m.mix.base == 3 && m.mix.slices.len() == 4).unwrap();
        let offsets = Array2::from_shape_fn((4, 3), |(i, j)| 0.1 * (i as f64 + 1.0) - 0.07 * j as f64);
        let m = &mut scorer.mixings[k];
        m.frame = &m.frame + &offsets;
        m.curvature = Array2::from_elem((4, 3), 0.01);
        let variance = 1.0 / (1000.0 * 0.01 + 1.0 / FRAME_PRIOR);
        let expected: f64 = offsets.iter().map(|d| 0.5 * ((FRAME_PRIOR / variance).ln() + (d * d + variance) / FRAME_PRIOR - 1.0)).sum();
        assert!((scorer.mixing_nats() - expected).abs() <= 1e-12, "{} against {expected}", scorer.mixing_nats());
    }

    /// A block across blocks with every block on takes every drawn weight edit as `M` does (the gap
    /// `KL(M_e ‖ P_e)` under 1e-9 bits per token, `Interchange::weight_scores`): the carried gate's
    /// recompute reads its home's input under the edit as the home block does (compare's check).
    #[test]
    fn a_block_across_blocks_with_every_block_on_takes_weight_edits_as_m() {
        let (native, explanation, sequences) = cross_tiny("library_cross_weights", true);
        let (device, settings) = (Device::host(), settings());
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let screen = Batch::new(sequences[..2].to_vec(), sequences[..2].to_vec()).unwrap();
        let drawn = scorer.experiments.draw_weight_edits(&native, &screen, 3, (64, 16)).unwrap();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        device_posterior.mean_into(scorer.experiments.explanation_mut()).unwrap();
        for (t, rows) in scorer.thresholds.clone() {
            let on = device.upload(Array2::from_elem((rows, 1), 1e30).view()).unwrap();
            scorer.experiments.explanation_mut().replace_dense_parameter(t, on).unwrap();
        }
        scorer.experiments.explanation_mut().refresh_fused().unwrap();
        let scores = scorer.experiments.weight_scores(&screen, &(0..drawn.len()).collect::<Vec<_>>()).unwrap();
        let taken: Vec<&interchange::WeightScores> = scores.iter().flatten().collect();
        assert!(!taken.is_empty() && taken.iter().any(|s| s.effect > 1e-6), "edits that move M");
        for s in taken {
            assert!(s.gap.abs() <= 1e-9, "every block on: gap {} (effect {})", s.gap, s.effect);
        }
    }

    /// A budget that never binds (`K = ∞`, or `K` far above any count, where `λ` stays 0) leaves
    /// the fit bit for bit; a finite one records `K`, `Ê[k]` and `λ` in every epoch.
    #[test]
    fn a_budget_that_never_binds_leaves_the_fit_bit_for_bit() {
        let (native, layers, _, sequences) = tiny("library_budget_free", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let (train, held) = sequences.split_at(4);
        let run = |budget: Option<f64>| {
            let mut settings = settings();
            settings.epochs = Some(2);
            settings.budget = budget;
            fit(&Device::host(), &native, &explanation, train, held, &settings, "tiny", None, None).unwrap()
        };
        let free = run(None);
        for budget in [f64::INFINITY, 1e9] {
            let fitted = run(Some(budget));
            let same = fitted.posterior.mean.iter().zip(&free.posterior.mean).all(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()));
            assert!(same, "a budget of {budget} moved the posterior");
            assert_eq!(fitted.report.objective_bits.to_bits(), free.report.objective_bits.to_bits());
            if budget.is_finite() {
                for epoch in &fitted.report.epochs {
                    assert_eq!(epoch.budget, Some(budget));
                    assert_eq!(epoch.multiplier, Some(0.0));
                    assert!(epoch.expected_parts.is_some_and(|k| k > 0.0 && k < budget));
                }
            }
        }
    }

    /// A tiny gated library meets its budget, an inequality, once its passes outlast the momentum
    /// (`B` batches above `1 / (1 − β₁)`, as in a production fit, so the multiplier's horizon is one
    /// pass): on the tiny decoder with ReLU functions (gated parts), trained on 256 sequences of its
    /// tokens drawn uniformly (B = 128 batches of 2), with `K` at the heads plus half the gated share
    /// of the free fit's count, the expected parts per token over the last 6 of 30 epochs (each
    /// epoch's mean over its 128 steps) are at most `K` plus their standard error (from the epochs'
    /// own spread) and at least 0.8 `K`, so a collapse of the parts still fails. A closed ReLU
    /// function takes no data gradient, so the count reopens slowly once below `K`: no rule for λ
    /// held it within three standard errors of `K` on seeds 3–7 on both GHA's Linux host and the Mac
    /// (ba02aadbc1 and the variants it lists). With the fixture's 4 training sequences (B = 2 against
    /// the momentum's 100 steps) the count at a fixed λ of 20 wandered between 5 and 16 parts over 120
    /// epochs: no multiplier holds a mean of it at `K`.
    #[test]
    fn a_tiny_gated_library_meets_its_budget() {
        let (native, layers, _, sequences) = tiny("library_budget_binds", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let held = &sequences[4..];
        let mut symbols: Vec<u32> = sequences.iter().flatten().copied().collect();
        symbols.sort_unstable();
        symbols.dedup();
        let mut rng = StdRng::seed_from_u64(17);
        let train: Vec<Vec<u32>> = (0..256).map(|_| (0..sequences[0].len()).map(|_| symbols[rng.random_range(0..symbols.len())]).collect()).collect();
        let run = |budget: f64, epochs: usize| {
            let mut settings = settings();
            settings.epochs = Some(epochs);
            settings.budget = Some(budget);
            fit(&Device::host(), &native, &explanation, &train, held, &settings, "tiny", None, None).unwrap()
        };
        let free = run(1e9, 3);
        let free_parts = free.report.epochs.last().and_then(|e| e.expected_parts).unwrap();
        // The heads count whole; the budget asks for half of the rest.
        let heads: usize = explanation.layers.iter().map(|l| l.heads.len()).sum();
        let limit = heads as f64 + 0.5 * (free_parts - heads as f64);
        let bound = run(limit, 30);
        let tail: Vec<f64> = bound.report.epochs.iter().rev().take(6).map(|e| e.expected_parts.unwrap()).collect();
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        let spread = (tail.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (tail.len() - 1) as f64 / tail.len() as f64).sqrt();
        let last = bound.report.epochs.last().unwrap();
        // Every tenth epoch's count and multiplier, for a failure's message.
        let trace: Vec<String> = bound.report.epochs.iter().map(|e| format!("{:.2} at λ {:.3e}", e.expected_parts.unwrap_or(f64::NAN), e.multiplier.unwrap_or(f64::NAN))).collect();
        assert!(last.multiplier.is_some_and(|l| l > 0.0), "the multiplier stayed at zero over the budget: {trace:?}");
        assert!(mean <= limit + spread && mean >= 0.8 * limit, "the last 6 epochs' mean {mean} ± {spread} parts per token against the budget {limit} (free {free_parts}): {trace:?}");
    }

    /// The main line's budget (`library_vpd` with learned widths, its gates drawn on/off in training,
    /// the count projected to `K` after every step): on the learned tiny library trained on 256
    /// sequences of its tokens drawn uniformly (B = 128 batches of 2), with `K` three quarters of the
    /// free fit's count, the expected parts per token over the last 6 of 30 epochs are at most `K`
    /// plus their standard error and at least 0.8 `K` on seeds 3–7.
    #[test]
    fn a_learned_gate_library_meets_its_budget() {
        let (native, explanation, sequences) = learned_tiny("library_budget_learned_binds");
        let held = &sequences[..2];
        let mut symbols: Vec<u32> = sequences.iter().flatten().copied().collect();
        symbols.sort_unstable();
        symbols.dedup();
        let mut rng = StdRng::seed_from_u64(17);
        let train: Vec<Vec<u32>> = (0..256).map(|_| (0..sequences[0].len()).map(|_| symbols[rng.random_range(0..symbols.len())]).collect()).collect();
        let mut failures = Vec::new();
        for seed in [3u64, 4, 5, 6, 7] {
            let run = |budget: f64, epochs: usize| {
                let mut settings = settings();
                settings.seed = seed;
                settings.epochs = Some(epochs);
                settings.budget = Some(budget);
                fit(&Device::host(), &native, &explanation, &train, held, &settings, "tiny", None, None).unwrap()
            };
            let free = run(1e9, 3);
            let free_parts = free.report.epochs.last().and_then(|e| e.expected_parts).unwrap();
            let limit = 0.75 * free_parts;
            let bound = run(limit, 30);
            let tail: Vec<f64> = bound.report.epochs.iter().rev().take(6).map(|e| e.expected_parts.unwrap()).collect();
            let mean = tail.iter().sum::<f64>() / tail.len() as f64;
            let spread = (tail.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (tail.len() - 1) as f64 / tail.len() as f64).sqrt();
            let trace: Vec<String> = bound.report.epochs.iter().map(|e| format!("{:.2} at λ {:.3e}", e.expected_parts.unwrap_or(f64::NAN), e.multiplier.unwrap_or(f64::NAN))).collect();
            eprintln!("SEED {seed} mean {mean:.3} ± {spread:.3} K {limit:.3} free {free_parts:.2}: {}", trace.join(" | "));
            if mean > limit + spread || mean < 0.8 * limit {
                failures.push(format!("seed {seed}: the last 6 epochs' mean {mean} ± {spread} against K {limit} (free {free_parts}): {trace:?}"));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    fn settings() -> Settings {
        Settings {
            batch_sequences: 2,
            seed: 3,
            numeric_bytes: 1 << 26,
            head_tile_rows: 64,
            epochs: None,
            families: Vec::new(),
            budget: None,
            budget_bits: false,
        }
    }

    #[test]
    fn the_starting_library_preserves_native_activation_and_coefficients() {
        for law in ["relu", "gelu", "gelu_tanh", "silu"] {
            let (native, layers, family, _) = tiny(&format!("library_start_{law}"), law);
            let explanation = explanation(&native, &layers).unwrap();
            explanation.artifact.validate_coverage(&native).unwrap();
            assert_eq!(explanation.artifact.blocks.len(), 2 * (2 + 1), "two heads and one MLP per layer");
            let expected = native.execute(&family, false).unwrap().values[native.output].clone();
            let actual = explanation.artifact.execute(&family).unwrap().values[explanation.artifact.program.output].clone();
            let scale = expected.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            let difference = expected.iter().zip(&actual).fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
            assert!(difference <= 1e-12 * scale, "the starting library differs from the native model by {difference}");
            let posterior = Posterior::new(&explanation, 72).unwrap();
            let cells: usize = explanation.groups.iter().flat_map(|g| &g.cells).map(|c| c.rows.len() * c.cols.len()).sum();
            assert_eq!(cells, posterior.mean.iter().map(|m| m.len()).sum::<usize>(), "the groups partition the parameters");
            // `M`'s MLPs have no bias, so neither do the library's: a zero input still maps to zero.
            for l in 0..2 {
                assert!(operator_named(&explanation.artifact.program, &format!("library.l{l}.mlp.gate_bias")).is_none());
                let rule = explanation.artifact.program.rules.iter().find(|r| r.name == format!("library.l{l}.mlp")).unwrap();
                assert!(matches!(rule.nodes[1], Node::Affine { bias: None, .. }));
            }
            let indexed: usize = explanation
                .layers
                .iter()
                .map(|l| l.heads.iter().map(|(p, v)| p.len() + v.len()).sum::<usize>() + l.functions.iter().map(Vec::len).sum::<usize>())
                .sum();
            assert_eq!(indexed, explanation.groups.len(), "every group belongs to one head or one MLP function");
        }
    }

    #[test]
    fn a_scoped_explanation_charges_and_runs_its_blocks_alone() {
        let (native, layers, _, sequences) = tiny("library_scoped", "relu");
        let full = explanation(&native, &layers).unwrap();
        // Layer 1's MLP alone (block 3).
        let explanation = scoped(&full, &[3]).unwrap();
        let functions = &full.layers[1].functions;
        assert_eq!(explanation.groups.len(), functions.iter().map(Vec::len).sum::<usize>());
        assert!(explanation.groups.iter().all(|g| g.name.starts_with("library.l1.mlp.")));
        assert!(explanation.trainable.iter().all(|op| explanation.artifact.program.operators[*op].name.starts_with("library.l1.mlp.")));
        assert_eq!(scope(&explanation), vec![false, false, false, true]);
        assert_eq!(explanation.layers[1].functions.len(), functions.len());
        assert!(explanation.layers[0].functions.is_empty() && explanation.layers.iter().all(|l| l.heads.is_empty()));
        assert!(scoped(&full, &[4]).is_err() && scoped(&full, &[]).is_err());
        let settings = settings();
        let device = Device::host();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let experiments = scorer.experiments(&draws[0], &sequences).unwrap();
        assert!(experiments.iter().all(|e| e.explained == [false, false, false, true]));
        assert!(experiments.iter().any(|e| e.patch.is_some()));
        // At its start the scoped library is M in every experiment; weight noise in its block
        // alone costs data, and only its layer's functions are counted.
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let cycled: Vec<Vec<u32>> = sequences.iter().cycle().take(96).cloned().collect();
        let evaluation = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &cycled, &settings, 72, None).unwrap();
        for bits in evaluation.clean.iter().chain(&evaluation.patched).chain([&evaluation.read_patch]).flatten() {
            assert!(bits.abs() < 1e-10, "the scoped start diverges from the model by {bits} bits per token");
        }
        assert!(evaluation.data_bits_per_token > 0.0, "weight noise in the block costs data");
        assert_eq!(evaluation.layers[0].functions, 0);
        assert_eq!(evaluation.layers[1].functions, functions.len());
        // The scoped fit converges with the block's groups alone.
        let (train, held) = sequences.split_at(4);
        let fit = fit(&device, &native, &explanation, train, held, &settings, "tiny", None, None).unwrap();
        assert_eq!(fit.posterior.active.len(), explanation.groups.len());
        assert!(fit.report.objective_bits.is_finite());
    }

    #[test]
    fn the_starting_library_matches_the_model_in_every_experiment() {
        let (native, layers, _, sequences) = tiny("library_experiments", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        // Every cut and a read patch must be drawn: each held-out base draws one clean and one
        // patched experiment, and six bases leave some of the families out, so the six token
        // rows are cycled into 96 bases (a family drawn with chance p per experiment is missed
        // with chance (1 - p)^96).
        let sequences: Vec<Vec<u32>> = sequences.iter().cycle().take(96).cloned().collect();
        let evaluation = held_out(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings, 72, None).unwrap();
        // Every family the held-out sample drew (which hybrid sizes it holds depends on its seed).
        assert!(evaluation.read_patch.is_some() && evaluation.clean.iter().chain(&evaluation.patched).filter(|b| b.is_some()).count() >= 2);
        for bits in evaluation.clean.iter().chain(&evaluation.patched).chain([&evaluation.read_patch]).flatten() {
            assert!(bits.abs() < 1e-10, "the starting library diverges from the model by {bits} bits per token");
        }
        assert!(evaluation.data_bits_per_token > 0.0, "weight noise costs data");
        for (count, layer) in evaluation.layers.iter().zip(&explanation.layers) {
            assert_eq!(count.functions, layer.functions.len());
            assert_eq!(count.heads, layer.heads.len());
            assert!(count.nonzero_per_token > 0.0 && count.nonzero_per_token < layer.functions.len() as f64, "ReLU functions are exactly zero on some tokens");
            assert!(count.resolved_per_token <= count.nonzero_per_token);
        }
    }

    #[test]
    fn the_removal_estimate_predicts_a_small_removal_on_the_batches_samples() {
        // Two MLP output groups with their means and deviations scaled by ε: removing them zeroes a
        // step of size ε at each batch's weight sample, and the estimate measured at those samples,
        // Σ_b (−g_b,G · θ_b,G + ½ (u_b,G · θ_b,G)²), predicts the change of the data term that
        // acceptance scores on the same samples up to terms of relative size ε; for both groups
        // together with the cross terms between them (`Curvature::joint`).
        let (native, layers, _, sequences) = tiny("library_curvature", "gelu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let mut posterior = Posterior::new(&explanation, 1000).unwrap();
        let groups = [*explanation.layers[0].functions[1].last().unwrap(), *explanation.layers[0].functions[2].last().unwrap()];
        let epsilon: f64 = 1e-2;
        let position: BTreeMap<usize, usize> = explanation.trainable.iter().enumerate().map(|(i, op)| (*op, i)).collect();
        for group in groups {
            for cell in &explanation.groups[group].cells {
                let i = position[&cell.operator];
                for &r in &cell.rows {
                    for c in cell.cols.clone() {
                        posterior.mean[i][[r, c]] *= epsilon;
                        posterior.log_sd[i][[r, c]] += epsilon.ln();
                    }
                }
            }
        }
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let mut device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 1000.0, None, 0).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        // On the evaluation's own samples, where the estimate is of the change it scores.
        let curvature = removal_curvature(&mut scorer, &device_posterior, &draws, &sequences, &settings, 0, &[]).unwrap();
        assert_eq!(curvature.batches(), draws.len());
        let mut evaluate = |removed: &[usize]| expected_divergence(&mut scorer, &mut device_posterior, &posterior, &draws, &sequences, removed, &settings, None, None).unwrap().total();
        let kept = evaluate(&[]);
        let rise = posterior.removal_data(&curvature);
        for (removed, predicted) in [(vec![groups[0]], rise[groups[0]]), (groups.to_vec(), curvature.joint(&[(groups[0], 1.0), (groups[1], 1.0)]))] {
            let exact = evaluate(&removed) - kept;
            assert!(exact != 0.0 && (predicted - exact).abs() <= 0.1 * exact.abs(), "{removed:?}: predicted {predicted:e} nats, exact {exact:e}");
        }
        assert!((curvature.joint(&[(groups[0], 1.0)]) - rise[groups[0]]).abs() <= 1e-6 * rise[groups[0]].abs());
        assert!(curvature.spread()[groups[0]].is_finite());
    }

    #[test]
    fn the_training_noise_is_drawn_afresh_each_epoch() {
        // The training noise is drawn afresh every epoch: two epochs' steps scored at the same
        // posterior differ batch by batch, while an epoch's steps scored twice agree exactly (the
        // snapshots' common draws: `a_snapshot_is_f_on_the_removal_draws_and_leaves_the_posterior_as_it_was`).
        let (native, layers, _, sequences) = tiny("library_common_noise", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let (device, settings) = (Device::host(), settings());
        let tokens = 2 * sequences.len() * 12;
        let posterior = Posterior::new(&explanation, tokens).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, tokens as f64, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        assert!(draws.len() > 1 && training_key(settings.seed, 0, 0) != training_key(settings.seed, 0, 1), "one key for every batch");
        assert!((0..draws.len()).all(|b| training_key(settings.seed, 0, b) != training_key(settings.seed, 1, b)), "one key for every epoch");
        let mut epoch = |at: usize| -> Vec<f64> {
            let mut estimates = Vec::new();
            for (b, draw) in draws.iter().enumerate() {
                let batch = draw.batch(&sequences).unwrap();
                let experiments = scorer.experiments(draw, &sequences).unwrap();
                let bits = antithetic_step(&mut scorer, (&device, &device_posterior), &batch, experiments, training_key(settings.seed, at, b)).unwrap().1.bits;
                estimates.push(bits.iter().flatten().sum::<f64>());
            }
            estimates
        };
        let (first, again, second) = (epoch(0), epoch(0), epoch(1));
        assert_eq!(first, again, "one epoch's draws scored twice");
        assert!(first.iter().zip(&second).all(|(a, b)| a != b), "two epochs' draws at one posterior: {first:?} and {second:?}");
    }

    /// A snapshot (`snapshot_estimates`) is the removal comparisons' `F` of the posterior batch by
    /// batch, `B D_b` plus the rest of `F`, at the same draws every time: two snapshots of one
    /// posterior improve on each other by exactly zero. It only reads the device posterior: an
    /// iterate apart from its average, and the steps the average spans, are left as they were.
    #[test]
    fn a_snapshot_is_f_on_the_removal_draws_and_leaves_the_posterior_as_it_was() {
        let (native, layers, _, sequences) = tiny("library_snapshot", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let (device, settings) = (Device::host(), settings());
        let tokens = 2 * sequences.len() * 12;
        let mut posterior = Posterior::new(&explanation, tokens).unwrap();
        let mut device_posterior = DevicePosterior::new(&device, &explanation, &posterior, tokens as f64, None, 0).unwrap();
        // An iterate away from its average, which spans five steps.
        let means: Vec<Array2<f64>> = posterior.mean.iter().map(|m| (**m).clone()).collect();
        let iterate: Vec<Array2<f64>> = means.iter().map(|m| m * 1.1).collect();
        device_posterior.restore(&means, &iterate, 5).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let evidence = Evidence { draws: &draws, sequences: &sequences, settings: &settings };
        let (first, _) = snapshot_estimates(&mut scorer, &device_posterior, &mut posterior, &explanation, &evidence, None).unwrap();
        let (second, _) = snapshot_estimates(&mut scorer, &device_posterior, &mut posterior, &explanation, &evidence, None).unwrap();
        assert_eq!(first, second, "two snapshots of one posterior");
        assert_eq!(device_posterior.averaged(), 5);
        for (i, expected) in iterate.iter().enumerate() {
            assert_eq!(device_posterior.iterate(i).unwrap(), *expected, "operator {i}'s iterate");
        }
        // The removal comparisons' evaluation of the same posterior, which sets the device's values.
        let evaluation = expected_divergence(&mut scorer, &mut device_posterior, &posterior, &draws, &sequences, &[], &settings, None, None).unwrap();
        let rest = posterior.description() + explanation.fixed_nats;
        let count = draws.len() as f64;
        for (b, (snapshot, data)) in first.iter().zip(&evaluation.batches).enumerate() {
            let expected = count * data + rest;
            assert!((snapshot - expected).abs() <= 1e-12 * expected.abs(), "batch {b}: {snapshot} against {expected}");
        }
        let total = evaluation.total() + rest;
        assert!((first.iter().sum::<f64>() / count - total).abs() <= 1e-12 * total.abs());
    }

    /// Each base's source and experiments come from its own draws: the training collection, base by
    /// base, is the same in batches of one, two, four and seven bases.
    #[test]
    fn the_collection_does_not_depend_on_the_batch_size() {
        let (native, layers, _, sequences) = tiny("library_collection_packing", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let many: Vec<Vec<u32>> = sequences.iter().cycle().take(20).cloned().collect();
        // Per base its source and its two experiments, their indices into the batch made the
        // sequences'.
        let collection = |size: usize| -> Vec<(usize, usize, Vec<Experiment>)> {
            let mut out = Vec::new();
            for draw in draws(many.len(), size, settings.seed).unwrap() {
                let experiments = scorer.experiments(&draw, &many).unwrap();
                for (k, (&base, &source)) in draw.bases.iter().zip(&draw.sources).enumerate() {
                    let own: Vec<Experiment> = experiments.iter().filter(|e| e.base == k).cloned().collect();
                    assert!(own.len() == 2 && own.iter().all(|e| e.source == k), "base {base}: {own:?}");
                    out.push((base, source, own.into_iter().map(|e| Experiment { base, source, ..e }).collect()));
                }
            }
            out
        };
        let one = collection(1);
        assert_eq!(one.len(), many.len());
        assert!(one.iter().all(|(base, source, _)| base != source) && one.iter().any(|(_, _, own)| own[1].patch.is_some()));
        for size in [2, 4, 7] {
            assert_eq!(collection(size), one, "batches of {size}");
        }
    }

    /// The half of a batch whose Gauss–Newton factor the step takes is drawn from the batch's key:
    /// over 4096 training keys the first half is drawn within four standard errors of half the
    /// time, and a step's factor is the drawn half's, bit for bit, scored at that half's own sample
    /// (the first half at the key, the second at its antithetic twin).
    #[test]
    fn the_curvature_half_is_drawn_from_the_batch_key() {
        let keys = 4096;
        let first = (0..keys).filter(|b| factor_half(training_key(7, 0, *b))).count() as f64;
        assert!((first - 0.5 * keys as f64).abs() <= 4.0 * (0.25 * keys as f64).sqrt(), "the first half drawn {first} times of {keys}");
        let (native, layers, _, sequences) = tiny("library_factor_half", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let (device, settings) = (Device::host(), settings());
        let tokens = 2 * sequences.len() * 12;
        let posterior = Posterior::new(&explanation, tokens).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, tokens as f64, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let (batch, experiments) = (draws[0].batch(&sequences).unwrap(), scorer.experiments(&draws[0], &sequences).unwrap());
        let half = batch.base.len() / 2;
        assert!(half > 0, "a batch of two halves");
        for wanted in [true, false] {
            let key = (0..).map(|b| training_key(settings.seed, 0, b)).find(|k| factor_half(*k) == wanted).unwrap();
            let factor = antithetic_step(&mut scorer, (&device, &device_posterior), &batch, experiments.clone(), key).unwrap().1.factor.unwrap();
            let part: Vec<Experiment> = experiments.iter().filter(|e| (e.base < half) == wanted).cloned().collect();
            let targets = scorer.experiments.targets(&batch, &part).unwrap();
            let own = if wanted { key } else { key ^ gam_gpu::tensor::ANTITHETIC };
            let expected = scorer.pass(&device_posterior, (&batch, &part, &targets), (Values::Iterate(own), Gates::Relaxed, false), (true, Some(probe_key(own)))).unwrap().factor.unwrap();
            assert_eq!(factor.tokens, expected.tokens, "the first half drawn: {wanted}");
            assert!(!factor.gradient.is_empty() && factor.gradient.keys().eq(expected.gradient.keys()));
            for (op, u) in &factor.gradient {
                assert_eq!(device.download(u).unwrap(), device.download(&expected.gradient[op]).unwrap(), "operator {op}, the first half drawn: {wanted}");
            }
        }
    }

    /// The fit's held-out subset scored on its batches made once (`held_out_on`) is the evaluation
    /// that makes them afresh (`held_out`), bit for bit, however often it is scored.
    #[test]
    fn the_held_out_subset_with_targets_made_once_is_the_evaluation_made_afresh() {
        let (native, layers, _, sequences) = tiny("library_held_once", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let (device, settings) = (Device::host(), settings());
        let posterior = Posterior::new(&explanation, 72).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let subset = &sequences[..settings.batch_sequences.clamp(2, sequences.len())];
        let batches = held_batches(&scorer, subset, &settings).unwrap();
        let afresh = serde_json::to_value(held_out(&mut scorer, &explanation, (&posterior, &device_posterior), subset, &settings, 72, None).unwrap()).unwrap();
        for _ in 0..2 {
            let once = held_out_on(&mut scorer, &explanation, (&posterior, &device_posterior), (subset, &batches), &settings, 72, None).unwrap();
            assert_eq!(serde_json::to_value(once).unwrap(), afresh);
        }
    }

    #[test]
    fn removal_scores_the_fixed_collection() {
        // Removing every group changes neither the experiments nor M's targets: a trial is scored
        // on the collection drawn over M's reads, against M's targets.
        let (native, layers, _, sequences) = tiny("library_removal_evidence", "relu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let posterior = Posterior::new(&explanation, 2 * sequences.len() * 12).unwrap();
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let removed: Vec<usize> = (0..posterior.active.len()).collect();
        let mut trial = posterior.clone();
        trial.remove(&removed);
        // The reference, built apart from the fit's scoring.
        let (mut reference_bits, mut read_patches) = (0.0, 0);
        for (b, draw) in draws.iter().enumerate() {
            let batch = draw.batch(&sequences).unwrap();
            let experiments = scorer.experiments(draw, &sequences).unwrap();
            read_patches += experiments.iter().filter(|e| matches!(e.patch, Some(Patch::Read { .. }))).count();
            let theta = sample(&trial, noise_seed(settings.seed, 0, b));
            let targets = scorer.experiments.targets(&batch, &experiments).unwrap();
            scorer.experiments.load(&theta).unwrap();
            let reference = scorer.experiments.evaluate_resident(&batch, &experiments, &targets, false).unwrap();
            reference_bits += reference.bits.iter().flatten().sum::<f64>();
        }
        assert!(read_patches > 0, "the collection must hold read patches of removed functions");
        let mut device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 1.0, None, 0).unwrap();
        let actual = expected_divergence(&mut scorer, &mut device_posterior, &posterior, &draws, &sequences, &removed, &settings, None, None).unwrap().total();
        let reference = reference_bits * LN_2;
        assert!((actual - reference).abs() < 1e-10 * reference.abs().max(1.0), "removal must score the fixed collection");
    }

    #[test]
    fn description_derivatives_are_those_of_the_divergence() {
        let (native, layers, _, _) = tiny("library_derivatives", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        for log_sd in &mut posterior.log_sd {
            log_sd.mapv_inplace(|s| s + 0.6 * (rng.random::<f64>() - 0.5));
        }
        let zeros: Vec<Array2<f64>> = posterior.mean.iter().map(|m| Array2::zeros(m.dim())).collect();
        let derivatives = derivatives(&posterior, &zeros, &zeros);
        let last = posterior.mean.len() - 1;
        for (i, at) in [(0, (0, 0)), (1, (3, 5)), (2, (1, 2)), (last, (2, 7))] {
            let h = 1e-6;
            let central = |posterior: &Posterior, field: fn(&mut Posterior) -> &mut Vec<Shared>| {
                let (mut up, mut down) = (posterior.clone(), posterior.clone());
                field(&mut up)[i][at] += h;
                field(&mut down)[i][at] -= h;
                (up.description() - down.description()) / (2.0 * h)
            };
            let mean = central(&posterior, |p| &mut p.mean);
            let log_sd = central(&posterior, |p| &mut p.log_sd);
            let (analytic_mean, analytic_log_sd) = (derivatives[i].0[at], derivatives[i].1[at]);
            assert!((mean - analytic_mean).abs() <= 1e-5 * (1.0 + mean.abs()), "μ derivative {analytic_mean} against {mean}");
            assert!((log_sd - analytic_log_sd).abs() <= 1e-5 * (1.0 + log_sd.abs()), "ln σ derivative {analytic_log_sd} against {log_sd}");
        }
    }

    #[test]
    fn a_device_step_is_the_host_step() {
        let (native, layers, _, _) = tiny("library_device_step", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let device = Device::host();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        for log_sd in &mut posterior.log_sd {
            log_sd.mapv_inplace(|s| s + 0.6 * (rng.random::<f64>() - 0.5));
        }
        let tokens = 72.0;
        let mut device_posterior = DevicePosterior::new(&device, &explanation, &posterior, tokens, None, 0).unwrap();
        // A data gradient per operator, weighted by `scale`, and a Gauss–Newton factor, its square
        // weighted by `square`.
        let (scale, square) = (3.0, 0.2);
        let gradients: Vec<Array2<f64>> = posterior.mean.iter().map(|m| m.mapv(|_| rng.random::<f64>() - 0.5)).collect();
        let factors: Vec<Array2<f64>> = posterior.mean.iter().map(|m| m.mapv(|_| rng.random::<f64>() - 0.5)).collect();
        let up = |values: &[Array2<f64>]| -> BTreeMap<usize, Tensor> { explanation.trainable.iter().zip(values).map(|(op, g)| (*op, device.upload(g.view()).unwrap())).collect() };
        let ivon = Ivon { beta1: 0.9, beta2: 0.75 };
        device_posterior.step(&up(&gradients), scale, (&up(&factors), square), &BTreeMap::new(), &ivon).unwrap();
        let mut stepped = posterior.clone();
        let moments = device_posterior.download(&mut stepped).unwrap();
        // The host reference: IVON's first step from momentum zero and the curvature at which the
        // posterior's standard deviations are IVON's.
        let variance = posterior.variances();
        let mut reference = posterior.clone();
        let close = |what: &str, x: f64, y: f64| assert!((x - y).abs() <= 1e-12 * (1.0 + y.abs()), "the device step's {what} {x} against the host step's {y}");
        for i in 0..reference.mean.len() {
            for ((r, c), _) in posterior.mean[i].indexed_iter() {
                let delta = 1.0 / (tokens * variance[posterior.membership[i][[r, c]] as usize]);
                let sd = posterior.log_sd[i][[r, c]].exp();
                let h0 = (1.0 / (tokens * sd * sd) - delta).max(0.0);
                let d = square * factors[i][[r, c]] * factors[i][[r, c]] - h0;
                let h = h0 + (1.0 - ivon.beta2) * d;
                // The iterate holds until `RATIO_DRAWS` draws of the line step's ratios: the mean
                // stays, and the momentum holds the one gradient.
                reference.log_sd[i][[r, c]] = -0.5 * (tokens * (h + delta)).ln();
                close("momentum", moments[i][0][[r, c]], (1.0 - ivon.beta1) * scale * gradients[i][[r, c]]);
                close("curvature", moments[i][1][[r, c]], h);
            }
        }
        for (field, (device_values, host_values)) in [("μ", (&stepped.mean, &reference.mean)), ("ln σ", (&stepped.log_sd, &reference.log_sd))] {
            for (a, b) in device_values.iter().zip(host_values) {
                let gap = a.iter().zip(b.iter()).fold(0.0_f64, |m, (x, y)| m.max((x - y).abs() / (1.0 + y.abs())));
                assert!(gap < 1e-12, "the device step's {field} differs from the host step's by {gap}");
            }
        }
    }

    #[test]
    fn a_variance_scale_far_from_its_reference_has_a_finite_code() {
        // Finite variances whose ratio overflows still have a finite scale code, and the variance
        // stays where the divergence is least when the scale's code cannot shorten.
        let (variance, divergence, bits) = gam_gpu::tensor::group_prior(4.0, 4e300, 0.0, Some(1e-300));
        assert!(bits.is_finite() && divergence.is_finite() && variance == 1e300, "{variance} {divergence} {bits}");
        let at = |centre: f64| gam_gpu::tensor::group_prior(4.0, 4.0 * centre, 0.0, Some(1.0)).2;
        assert_eq!(at(1.0), at(1.2));
        assert_eq!(at(1.0), 1.0, "exponent 0 costs one bit");
    }

    /// The scale's closed-form code length is `codec`'s Elias δ length of the exponent's signed
    /// index, for every exponent a finite variance ratio can take.
    #[test]
    fn the_scale_code_is_the_codec_s_elias_delta_length() {
        for k in -2100_i64..=2100 {
            let codec = crate::codec::signed_codeword_argument(k).and_then(crate::codec::elias_delta_len_bits).unwrap();
            assert_eq!(gam_gpu::tensor::scale_code_bits(k), codec as f64, "exponent {k}");
        }
    }

    /// `v_G` minimizes the group's charge `KL(q_G ‖ N(0, v I)) + ln 2 · L_scale(v)`: a group of 40
    /// entries at `μ = 1`, reference `v⁰ = 1` (their mean square), with `σ² = 2^0.52 − 1`, so that
    /// `S / |G| = 2^0.52` sits just inside the bin of exponent 1 (4 bits) next to exponent 0's (1
    /// bit). The divergence rises by `½ |G| (e^(−w) − 1 + w) ≈ 2e-3` nats (`w = −0.02 ln 2`) at the
    /// bins' shared edge `2^0.5`, three bits less than the code saves: `v_G` is that edge, its code
    /// one bit, and the group's charge is that much below its charge at `S / |G|`.
    #[test]
    fn a_variance_just_inside_an_expensive_bin_takes_the_cheaper_neighbour() {
        let (n, tokens) = (40, 1000);
        let mut posterior = Posterior::from_parts(vec![Array2::from_elem((1, n), 1.0)], vec![Array2::zeros((1, n))], vec![1.0], tokens).unwrap();
        assert_eq!(posterior.references(), &[1.0]);
        let size = n as f64;
        // `KL(q_G ‖ N(0, v I))` of the group at `μ = 1` with every `σ² = sigma2`.
        let kl = |sigma2: f64, v: f64| 0.5 * (size * (1.0 + sigma2) / v + size * v.ln() - size - size * sigma2.ln());
        let precision = 0.5 * size.ln();
        let sigma2 = 2f64.powf(0.52) - 1.0;
        posterior.log_sd[0].fill(0.5 * sigma2.ln());
        let variance = posterior.variances()[0];
        assert!((variance - 2f64.sqrt()).abs() <= 1e-15, "v_G {variance} against the edge √2");
        let divergence = posterior.divergences()[0];
        assert!((divergence - kl(sigma2, variance)).abs() <= 1e-12, "the divergence is KL at v_G: {divergence} against {}", kl(sigma2, variance));
        let chosen = posterior.costs()[0];
        assert!((chosen - (kl(sigma2, variance) + precision + LN_2)).abs() <= 1e-12, "one bit of scale: {chosen}");
        let unclamped = kl(sigma2, 1.0 + sigma2) + precision + 4.0 * LN_2;
        assert!(unclamped - chosen > 2.0 * LN_2, "the cheaper bin saves nearly three bits: {chosen} against {unclamped}");
        // At the centre of exponent 2's bin (5 bits) the edge of exponent 1's (4 bits) costs
        // `½ |G| (e^(ln 2 / 2) − 1 − ln 2 / 2) ≈ 1.4` nats, more than the bit it saves: the variance
        // stays at `S / |G| = 4`.
        posterior.log_sd[0].fill(0.5 * 3f64.ln());
        assert!((posterior.variances()[0] - 4.0).abs() <= 1e-14, "{}", posterior.variances()[0]);
        assert!((posterior.costs()[0] - (kl(3.0, 4.0) + precision + 5.0 * LN_2)).abs() <= 1e-12);
    }


    #[test]
    fn a_group_starting_at_zero_has_a_reference_and_starts_from_it() {
        let (native, layers, _, _) = tiny("library_zero_group", "gelu_tanh");
        let mut explanation = explanation(&native, &layers).unwrap();
        // Zero the first MLP function's output column.
        let output = explanation.layers[0].functions[0].last().copied().unwrap();
        let cell = explanation.groups[output].cells[0].clone();
        let program = &mut explanation.artifact.program;
        let source = &program.operators[cell.operator];
        let mut values = source.matrix();
        values.column_mut(cell.cols.start).fill(0.0);
        program.operators[cell.operator] = Arc::new(Operator::dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values.clone(), exact_precision(values.iter().copied()).unwrap(), source.provenance.clone()).unwrap());
        // Made at these values, the zeroed group takes its operator's other entries' mean square.
        let others: Vec<f64> = values.columns().into_iter().enumerate().filter(|(c, _)| *c != cell.cols.start).flat_map(|(_, col)| col.to_vec()).collect();
        let inherited = others.iter().map(|v| v * v).sum::<f64>() / others.len() as f64;
        let made = mean_squares(&explanation.artifact.program, &explanation.groups);
        assert!((made[output] - inherited).abs() <= 1e-12 * inherited);
        assert!(made.iter().all(|r| *r > 0.0));
        // Started at zero, the group's deviations come from its reference, and it costs finitely.
        let posterior = Posterior::new(&explanation, 72).unwrap();
        assert_eq!(posterior.initial[output], explanation.reference[output]);
        assert!((posterior.log_sd[explanation.trainable.iter().position(|t| *t == cell.operator).unwrap()][[cell.rows[0], cell.cols.start]] - 0.5 * (explanation.reference[output] / 72.0).ln()).abs() < 1e-12);
        assert!(posterior.divergences()[output].is_finite());
    }


    #[test]
    fn the_device_rounds_the_posterior_as_the_host_does() {
        let (native, layers, _, _) = tiny("library_rounded", "gelu");
        let explanation = explanation(&native, &layers).unwrap();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        for log_sd in &mut posterior.log_sd {
            log_sd.mapv_inplace(|s| s + 6.0 * (rng.random::<f64>() - 0.5));
        }
        posterior.remove(&[0]);
        let device = Device::host();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 72.0, None, 0).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings()).unwrap();
        device_posterior.rounded_into(scorer.experiments.explanation_mut()).unwrap();
        let rounded = posterior.rounded();
        for (i, op) in explanation.trainable.iter().enumerate() {
            let held = device.download(scorer.experiments.explanation_mut().dense(*op).unwrap()).unwrap();
            assert_eq!(held, *rounded.mean[i], "operator {i}");
        }
    }

    #[test]
    fn the_device_variances_are_the_hosts() {
        let (native, layers, _, _) = tiny("library_code_length", "gelu");
        let explanation = explanation(&native, &layers).unwrap();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        // Spread deviations, so the variances' scales differ, and a removed group.
        let mut rng = StdRng::seed_from_u64(3);
        for log_sd in &mut posterior.log_sd {
            log_sd.mapv_inplace(|s| s + 4.0 * (rng.random::<f64>() - 0.5));
        }
        posterior.remove(&[1]);
        let device_posterior = DevicePosterior::new(&Device::host(), &explanation, &posterior, 72.0, None, 0).unwrap();
        // The device's variances are the host's (`Posterior::variances`).
        for (g, (device, host)) in device_posterior.variances().unwrap().iter().zip(posterior.variances()).enumerate() {
            assert!((device - host).abs() <= 1e-12 * host.abs(), "group {g}: v_G {device} on the device against {host}");
        }
    }

    #[test]
    fn a_group_at_its_prior_costs_only_its_variance() {
        let (native, layers, _, _) = tiny("library_prior", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let mut posterior = Posterior::new(&explanation, 72).unwrap();
        let group = &explanation.groups[0];
        let position = |op: usize| explanation.trainable.iter().position(|t| *t == op).unwrap();
        // The group at its prior at its reference variance, whose scale is exponent 0's one bit.
        let log_sd = 0.5 * posterior.references()[0].ln();
        for cell in &group.cells {
            let i = position(cell.operator);
            for &row in &cell.rows {
                for col in cell.cols.clone() {
                    posterior.mean[i][[row, col]] = 0.0;
                    posterior.log_sd[i][[row, col]] = log_sd;
                }
            }
        }
        let divergences = posterior.divergences();
        assert!(divergences[0].abs() < 1e-12, "an uninformed group at its prior costs {}", divergences[0]);
        assert!(divergences[1] > 0.0);
        let size: usize = group.cells.iter().map(|c| c.rows.len() * c.cols.len()).sum();
        let expected = 0.5 * (size as f64).ln() + LN_2;
        assert!((posterior.costs()[0] - expected).abs() < 1e-12, "its variance costs ½ ln |G| and its scale's one bit");
    }

    #[test]
    fn checkpoint_array_decoding_is_tiled_and_preserves_logical_order_and_bits() {
        let bits: Vec<u64> = (0..18_003)
            .map(|i| match i % 4 {
                0 => (-0.0_f64).to_bits(),
                1 => f64::INFINITY.to_bits(),
                2 => 0x7ff8_0000_0000_0123,
                _ => (i as f64 / 7.0).to_bits(),
            })
            .collect();
        let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut reader = std::io::Cursor::new(&bytes);
        let mut output = Array2::zeros((6001, 3)).reversed_axes();
        assert!(!output.is_standard_layout());
        read_checkpoint_array(&mut reader, &mut output, Precision::F64).unwrap();
        assert_eq!(reader.position(), bytes.len() as u64);
        assert_eq!(output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
        assert!(read_checkpoint_array(&mut &bytes[..bytes.len() - 1], &mut output, Precision::F64).is_err());
    }

    #[test]
    fn checkpoint_arrays_in_device_precision_round_trip_exactly_and_refuse_other_values() {
        // Values the device holds in each storage: f32 values (with -inf and a subnormal), and
        // bfloat16 values (f32s whose low 16 bits are zero).
        let f32s: Vec<f64> = (0..9001).map(|i| f64::from((i as f32 - 4500.0) / 7.0)).chain([f64::NEG_INFINITY, f64::from(f32::MIN_POSITIVE / 8.0)]).collect();
        let bf16s: Vec<f64> = f32s.iter().map(|v| f64::from(f32::from_bits((*v as f32).to_bits() & 0xffff_0000))).collect();
        for (values, precision) in [(&f32s, Precision::F32), (&bf16s, Precision::Bf16), (&f32s, Precision::F64)] {
            let array = Array2::from_shape_vec((values.len(), 1), values.clone()).unwrap();
            let mut bytes = Vec::new();
            write_checkpoint_array(&mut bytes, &array, precision).unwrap();
            assert_eq!(bytes.len(), values.len() * precision.bytes());
            let mut back = Array2::zeros(array.dim());
            read_checkpoint_array(&mut &bytes[..], &mut back, precision).unwrap();
            assert_eq!(back.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), array.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        let wide = Array2::from_elem((1, 1), 0.1);
        assert!(write_checkpoint_array(&mut Vec::new(), &wide, Precision::F32).is_err());
        assert!(write_checkpoint_array(&mut Vec::new(), &Array2::from_elem((1, 1), f64::from(1.0_f32 + f32::EPSILON)), Precision::Bf16).is_err());
    }

    /// A checkpoint of three operators (one empty) whose payload holds `precision`'s arrays per
    /// operator (none: the legacy layout, six float64 arrays), array `field` of operator `i`
    /// holding `100 i + 10 field + cell` (integers below 256, which every precision holds exactly).
    /// A legacy layout's header carries the momentum's one `(W, W2)` pair, as checkpoints written
    /// before the corrections were per operator did; the current layout's carries per operator
    /// corrections and the line step's slope averages.
    fn checkpoint_fixture(precision: Option<Vec<Precision>>) -> (Progress, Posterior, Vec<u8>) {
        let shapes = vec![(2, 3), (0, 2), (3, 1)];
        let posterior = Posterior {
            mean: shapes.iter().map(|&dim| Array2::from_elem(dim, -7.0).into()).collect(),
            log_sd: shapes.iter().map(|&dim| Array2::from_elem(dim, -8.0).into()).collect(),
            active: vec![true, true],
            membership: Arc::new(Vec::new()),
            spans: Vec::new(),
            initial: Vec::new(),
        };
        let legacy = precision.as_ref().is_none_or(|p| p.len() == LEGACY_ARRAYS);
        let progress = Progress {
            identity: Identity {
                export: "export".into(),
                tokens: "tokens".into(),
                program: "program".into(),
                groups: "groups".into(),
                sharing: "sharing".into(),
                definition: "definition".into(),
            },
            settings: settings(),
            tokens: 23,
            shapes,
            start: None,
            epoch: 4,
            step: 17,
            averaged: 3,
            ratio: (1.5, 2),
            slope: if legacy { (0.0, 0.0, 0) } else { (0.75, 1.25, 4) },
            weights: (!legacy).then(|| vec![0.5, 0.0, 0.875]),
            momentum_weights: legacy.then_some((0.25, 0.0625)),
            best: Some((7.0, 3)),
            epochs: Vec::new(),
            removals: Vec::new(),
            multiplier: 0.0,
            engaged: false,
            previous: Some(vec![1.0, 2.0]),
            collection: COLLECTION,
            active: vec![false, true],
            done: false,
            seconds: 5.0,
            training_seconds: 3.0,
            evaluation_seconds: 2.0,
            full_seconds: 1.0,
            prior: None,
            precision: precision.clone(),
            assignments: None,
            frames: None,
        };
        // The wire format: length-prefixed JSON, then each operator's arrays row-major in their
        // precisions. This fixture is independent of the streaming decoder and requires no GPU.
        let mut header = serde_json::to_value(&progress).unwrap();
        if let Some((w, w2)) = progress.momentum_weights {
            header["momentum_weights"] = serde_json::json!([w, w2]);
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        let layout = Precision::of_payload(precision.as_deref()).unwrap();
        for (i, &(rows, cols)) in progress.shapes.iter().enumerate() {
            for (field, &p) in layout.iter().enumerate() {
                let values = Array2::from_shape_fn((rows, cols), |(r, c)| (100 * i + 10 * field + r * cols + c) as f64);
                write_checkpoint_array(&mut bytes, &values, p).unwrap();
            }
        }
        (progress, posterior, bytes)
    }

    /// `load_checkpoint` of `checkpoint_fixture(precision)`: every restored array is the fixture's
    /// (the posterior's own arrays written in place), the iterate the payload's last array, and a
    /// legacy payload's gradient second moment dropped. Returns the loaded progress.
    fn restores_its_arrays(precision: Option<Vec<Precision>>, name: &str) -> Progress {
        let (expected, mut posterior, bytes) = checkpoint_fixture(precision.clone());
        let path = std::env::temp_dir().join(format!("library_checkpoint_{name}_{}.bin", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        let pointers: Vec<_> = posterior.mean.iter().chain(&posterior.log_sd).map(|array| array.as_ptr()).collect();
        let (progress, moments, _, iterates) = load_checkpoint(&path, &expected, &mut posterior).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(serde_json::to_value(&progress).unwrap(), serde_json::to_value(&expected).unwrap());
        assert_eq!(posterior.active, expected.active);
        assert_eq!(pointers, posterior.mean.iter().chain(&posterior.log_sd).map(|array| array.as_ptr()).collect::<Vec<_>>());
        let last = Precision::of_payload(precision.as_deref()).unwrap().len() - 1;
        for i in 0..progress.shapes.len() {
            let arrays = [(0, &*posterior.mean[i]), (1, &*posterior.log_sd[i]), (2, &moments[i][0]), (3, &moments[i][1]), (last, &iterates[i])];
            for (field, array) in arrays {
                for (cell, value) in array.iter().enumerate() {
                    assert_eq!(*value, (100 * i + 10 * field + cell) as f64, "{name}: operator {i}, array {field}");
                }
            }
        }
        progress
    }

    #[test]
    fn checkpoint_streaming_restores_the_legacy_payload_without_replacing_posterior_arrays() {
        // Six float64 arrays per operator, the gradient's second moment fifth, and the momentum's
        // one bias correction for every operator.
        let progress = restores_its_arrays(None, "legacy");
        assert_eq!(progress.momentum_weights, Some((0.25, 0.0625)));
        assert_eq!(saved_weights(progress.weights, progress.momentum_weights, 17, 3), vec![0.25; 3]);
        assert_eq!(saved_weights(None, None, 2, 2), vec![0.9 * (1.0 - 0.9) + (1.0 - 0.9); 2], "a checkpoint with neither: β₁ = 0.9's over its steps");
    }

    #[test]
    fn checkpoints_of_five_arrays_and_of_the_six_before_them_restore_in_their_precisions() {
        // The current layout in the f32 storage, and the one before it with its bfloat16 momentum.
        let current = restores_its_arrays(Some(vec![Precision::F32; CHECKPOINT_ARRAYS]), "current");
        assert_eq!(saved_weights(current.weights, current.momentum_weights, 17, 3), vec![0.5, 0.0, 0.875]);
        assert_eq!(current.slope, (0.75, 1.25, 4));
        let old = restores_its_arrays(Some(vec![Precision::F32, Precision::F32, Precision::Bf16, Precision::F32, Precision::F32, Precision::F32]), "bf16");
        assert_eq!(saved_weights(old.weights, old.momentum_weights, 17, 3), vec![0.25; 3]);
        assert!(Precision::of_payload(Some(&[Precision::F64; 4])).is_err(), "a payload of another layout");
    }

    #[test]
    fn checkpoint_streaming_refuses_truncation_trailing_bytes_and_overflow_before_mutating() {
        let (expected, posterior, bytes) = checkpoint_fixture(None);
        let path = std::env::temp_dir().join(format!("library_checkpoint_bad_{}.bin", std::process::id()));
        let mut trailing = bytes.clone();
        trailing.push(1);
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        for broken in [
            Vec::new(),
            bytes[..7].to_vec(),
            u64::MAX.to_le_bytes().to_vec(),
            bytes[..8 + header_len - 1].to_vec(),
            bytes[..bytes.len() - 1].to_vec(),
            trailing,
        ] {
            std::fs::write(&path, broken).unwrap();
            let mut unchanged = posterior.clone();
            assert!(load_checkpoint(&path, &expected, &mut unchanged).is_err());
            assert_eq!(unchanged.mean, posterior.mean);
            assert_eq!(unchanged.log_sd, posterior.log_sd);
            assert_eq!(unchanged.active, posterior.active);
        }
        std::fs::write(&path, &bytes).unwrap();
        let mut other = expected.clone();
        other.identity.groups.push('x');
        let message = load_checkpoint(&path, &other, &mut posterior.clone()).unwrap_err();
        assert!(message.contains("groups"), "{message}");
        other = expected.clone();
        other.tokens += 1;
        assert!(load_checkpoint(&path, &other, &mut posterior.clone()).unwrap_err().contains("another fit"));
        other = expected.clone();
        other.settings.seed += 1;
        assert!(load_checkpoint(&path, &other, &mut posterior.clone()).unwrap_err().contains("another fit"));
        // A checkpoint written before the collection's version was recorded drew other experiments:
        // refused, naming the collection.
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
        header.as_object_mut().unwrap().remove("collection").unwrap();
        let header = serde_json::to_vec(&header).unwrap();
        std::fs::write(&path, [&(header.len() as u64).to_le_bytes()[..], &header, &bytes[8 + header_len..]].concat()).unwrap();
        let message = load_checkpoint(&path, &expected, &mut posterior.clone()).unwrap_err();
        assert!(message.contains("experiment collection 0"), "{message}");
        let shapes = [(2, 3), (0, 2), (3, 1)];
        let (wide, device) = ([Precision::F64; CHECKPOINT_ARRAYS], [Precision::F32; CHECKPOINT_ARRAYS]);
        let legacy = [Precision::F32, Precision::F32, Precision::Bf16, Precision::F32, Precision::F32, Precision::F32];
        assert_eq!(checkpoint_payload_bytes(&shapes, &wide), Some(9 * 40));
        assert_eq!(checkpoint_payload_bytes(&shapes, &device), Some(9 * 20));
        assert_eq!(checkpoint_payload_bytes(&shapes, &legacy), Some(9 * 22));
        assert_eq!(checkpoint_payload_bytes(&[(usize::MAX, usize::MAX)], &wide), None);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn checkpoint_identity_reads_a_bounded_prefix_without_loading_the_payload() {
        let (expected, _, bytes) = checkpoint_fixture(None);
        let path = std::env::temp_dir().join(format!("library_checkpoint_identity_{}.bin", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        // A sparse payload makes the bytes read observable independently of posterior size.
        std::fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(8 * 1024 * 1024).unwrap();
        check_checkpoint(&path, &expected.identity).unwrap();
        let (progress, mut reader, _): (Progress, _, _) = checkpoint_header(&path).unwrap();
        assert_eq!(progress.identity, expected.identity);
        assert!(std::io::Seek::stream_position(reader.get_mut()).unwrap() <= 64 * 1024);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_fit_from_the_start_a_checkpoint_holds_is_the_fit_resumed_from_it() {
        let (native, layers, _, sequences) = tiny("library_fit_from", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let (train, held) = sequences.split_at(4);
        let finished_path = std::env::temp_dir().join(format!("library_fit_from_{}.bin", std::process::id()));
        let finished = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&finished_path), None).unwrap();
        // The finished fit's checkpoint reopened, so that it trains again from its posterior and
        // IVON's state, with convergence judged afresh.
        let bytes = std::fs::read(&finished_path).unwrap();
        let length = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
        header["done"] = serde_json::Value::Bool(false);
        header["previous"] = serde_json::Value::Null;
        let header = serde_json::to_vec(&header).unwrap();
        let path = std::env::temp_dir().join(format!("library_fit_from_reopened_{}.bin", std::process::id()));
        std::fs::write(&path, [&(header.len() as u64).to_le_bytes()[..], &header, &bytes[8 + length..]].concat()).unwrap();
        let start = checkpoint_start(&explanation, &path).unwrap();
        assert_eq!(start.epoch, finished.report.epochs.len());
        let resumed = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&path), None).unwrap();
        let started = fit_from(&device, &native, &explanation, train, held, &settings, "tiny", None, None, Some(start)).unwrap();
        for file in [&finished_path, &path] {
            for extension in ["bin", "json", "removals.jsonl"] {
                let written = file.with_extension(extension);
                if written.exists() {
                    std::fs::remove_file(written).unwrap();
                }
            }
        }
        let trained = &resumed.report.epochs[finished.report.epochs.len()..];
        assert!(!trained.is_empty(), "the reopened checkpoint trains");
        assert_eq!(trained.len(), started.report.epochs.len());
        for (a, b) in trained.iter().zip(&started.report.epochs) {
            assert_eq!(a.epoch, b.epoch);
            assert_eq!((a.data_bits.to_bits(), a.snapshot_bits.to_bits()), (b.data_bits.to_bits(), b.snapshot_bits.to_bits()), "epoch {}", a.epoch);
        }
        assert_eq!(started.posterior.mean, resumed.posterior.mean);
        assert_eq!(started.posterior.log_sd, resumed.posterior.log_sd);
        assert_eq!(started.posterior.active, resumed.posterior.active);
        // The same checkpoint marked as one of an older experiment collection, at the fit's own
        // path: it is the fit's start, not resumed, and stays beside as `.collection0.bin`.
        let mut older: serde_json::Value = serde_json::from_slice(&bytes[8..8 + length]).unwrap();
        older["done"] = serde_json::Value::Bool(false);
        older["previous"] = serde_json::Value::Null;
        older.as_object_mut().unwrap().remove("collection").unwrap();
        let older = serde_json::to_vec(&older).unwrap();
        let older = [&(older.len() as u64).to_le_bytes()[..], &older, &bytes[8 + length..]].concat();
        let path = std::env::temp_dir().join(format!("library_fit_from_older_{}.bin", std::process::id()));
        std::fs::write(&path, &older).unwrap();
        let warmed = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&path), None).unwrap();
        let aside = path.with_extension("collection0.bin");
        assert_eq!(std::fs::read(&aside).unwrap(), older, "the older checkpoint stays beside");
        for file in [&path, &aside] {
            for extension in ["bin", "json", "removals.jsonl", "collection0.bin"] {
                let written = file.with_extension(extension);
                if written.exists() {
                    std::fs::remove_file(written).unwrap();
                }
            }
        }
        assert_eq!(warmed.report.epochs.len(), started.report.epochs.len());
        for (a, b) in warmed.report.epochs.iter().zip(&started.report.epochs) {
            assert_eq!(a.epoch, b.epoch);
            assert_eq!((a.data_bits.to_bits(), a.snapshot_bits.to_bits()), (b.data_bits.to_bits(), b.snapshot_bits.to_bits()), "epoch {}", a.epoch);
        }
        assert_eq!(warmed.posterior.mean, started.posterior.mean);
        // A start of another explanation is refused.
        let other = Start { mean: Vec::new(), log_sd: Vec::new(), active: vec![true], state: None, epoch: 0 };
        assert!(fit_from(&device, &native, &explanation, train, held, &settings, "tiny", None, None, Some(other)).is_err());
    }

    /// A prior term that adds nothing to `F` over an explanation of `groups` groups and stops the
    /// fit at its `stop`-th training step, as a process killed there; its steps go into the
    /// checkpoint.
    struct Stop {
        groups: usize,
        steps: u64,
        stop: Option<u64>,
    }

    impl PriorTerm for Stop {
        fn operators(&self) -> Vec<usize> {
            Vec::new()
        }
        fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
            let groups = (explanation.groups.len(), posterior.active.len());
            if groups == (self.groups, self.groups) { Ok(()) } else { Err(format!("{groups:?} groups where {} were set", self.groups)) }
        }
        fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
            if !theta.is_empty() {
                return Err("a sample of operators it does not read".into());
            }
            let nats = self.cost(posterior)?;
            self.steps += u64::from(learn);
            if learn && Some(self.steps) == self.stop {
                return Err("stopped".into());
            }
            Ok((nats, BTreeMap::new()))
        }
        fn cost(&self, posterior: &Posterior) -> Result<f64, String> {
            if posterior.active.len() == self.groups { Ok(0.0) } else { Err("a posterior of another explanation".into()) }
        }
        fn save(&self) -> Result<serde_json::Value, String> {
            Ok(serde_json::json!({ "steps": self.steps }))
        }
        fn load(&mut self, value: &serde_json::Value) -> Result<(), String> {
            self.steps = value["steps"].as_u64().ok_or("a saved step count")?;
            Ok(())
        }
    }

    /// A prior term whose cost rises by `rise` nats at every epoch after the first (its structure
    /// chosen afresh, `PriorTerm::epoch`), so each epoch's snapshot of `F` is above the one before:
    /// the fit stops after its second epoch and goes back to the first. Its `stop`, which never
    /// stops the fit, checks the explanation, posterior and samples it is given.
    struct Rising {
        stop: Stop,
        epochs: u64,
        rise: f64,
    }

    impl PriorTerm for Rising {
        fn operators(&self) -> Vec<usize> {
            Vec::new()
        }
        fn epoch(&mut self, explanation: &Explanation, posterior: &Posterior) -> Result<(), String> {
            self.stop.epoch(explanation, posterior)?;
            self.epochs += 1;
            Ok(())
        }
        fn sample(&mut self, posterior: &Posterior, theta: &BTreeMap<usize, Array2<f64>>, learn: bool) -> Result<(f64, BTreeMap<usize, Array2<f64>>), String> {
            let (nats, gradients) = self.stop.sample(posterior, theta, learn)?;
            Ok((nats + self.rise * self.epochs as f64, gradients))
        }
        fn cost(&self, posterior: &Posterior) -> Result<f64, String> {
            Ok(self.stop.cost(posterior)? + self.rise * self.epochs as f64)
        }
        fn save(&self) -> Result<serde_json::Value, String> {
            Ok(serde_json::json!({ "epochs": self.epochs, "stop": self.stop.save()? }))
        }
        fn load(&mut self, value: &serde_json::Value) -> Result<(), String> {
            self.epochs = value["epochs"].as_u64().ok_or("a saved epoch count")?;
            self.stop.load(&value["stop"])
        }
    }

    /// Going back to the best epoch restores every state its snapshot was scored at, the prior
    /// term's with the posterior: with a prior whose cost rises at each epoch, the fit stops after
    /// its second epoch, and the removal round's `F` before any removal, scored at the restored
    /// state, is the first epoch's snapshot (the recorded best), not the second's.
    #[test]
    fn going_back_to_the_best_epoch_restores_the_prior_terms_state() {
        let (native, layers, _, sequences) = tiny("library_best_prior", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let (train, held) = sequences.split_at(4);
        let mut prior = Rising { stop: Stop { groups: explanation.groups.len(), steps: 0, stop: None }, epochs: 0, rise: 1e6 };
        let fitted = fit(&Device::host(), &native, &explanation, train, held, &settings(), "tiny", None, Some(&mut prior)).unwrap();
        let (epochs, removals) = (&fitted.report.epochs, &fitted.report.removals);
        assert!(epochs.len() >= 2 && !removals.is_empty(), "{} epochs, {} removal rounds", epochs.len(), removals.len());
        let rise = epochs[1].snapshot_bits - epochs[0].snapshot_bits;
        assert!(epochs[1].improvement_bits.is_some_and(|i| i < 0.0) && rise > 0.5e6 / LN_2, "the second epoch's snapshot rises by {rise} bits");
        let (best, restored) = (epochs[0].snapshot_bits, removals[0].before_bits);
        assert!((restored - best).abs() <= 1e-9 * best.abs(), "F at the restored state {restored} against the best epoch's {best} (the second's {})", epochs[1].snapshot_bits);
    }

    /// The per-token activity counts leave out the positions where a layer's block runs `M`'s MLP
    /// instead of its functions (its `Node::Select`, a transcoder layer's first token,
    /// 0e9b9c6213): on the tiny Qwen3 decoder with layer 1's MLP a transcoder's features, layer 1's
    /// nonzero functions per token are those of the host's pre-activations `x Wᵀ + c` over the
    /// tokens after each sequence's first, which differ from those over every token.
    #[test]
    fn activity_counts_leave_out_the_positions_a_block_runs_m_at() {
        let export = crate::test_support::tiny_qwen3_export("library_activity_select", 2);
        let imported = import_language_model(&export, 6, 12).unwrap();
        std::fs::remove_dir_all(&export).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let dir = std::env::temp_dir().join(format!("library_activity_select_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (full, kept_path) = (dir.join("layer_1.safetensors"), dir.join("kept_1.safetensors"));
        let features = 64;
        crate::test_support::transcoder_file(&full, features, 8, 3);
        crate::library_transcoder::Transcoder::open(&full).unwrap().write_kept(&(0..features).collect::<Vec<_>>(), &[0.0; 8], &kept_path).unwrap();
        let explanation = explanation_with(&native, &layers, &BTreeMap::from([(1, kept_path)])).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        // The host's count: layer 1 reads `M`'s input (layer 0 is `M`'s functions).
        let family = sequence_family(&sequences.iter().map(Vec::as_slice).collect::<Vec<_>>()).unwrap();
        let x = native.execute(&family, false).unwrap().values[layers[1].normed].clone();
        let program = &explanation.artifact.program;
        let operator = |name: &str| program.operators.iter().find(|op| op.name == name).unwrap().matrix();
        let pre = x.dot(&operator("library.l1.mlp.gate").t()) + &operator("library.l1.mlp.gate_bias").column(0);
        let positions = &family.layout.as_ref().unwrap().position;
        let firing = |rows: Vec<usize>| -> f64 { rows.iter().map(|&r| pre.row(r).iter().filter(|v| **v > 0.0).count()).sum::<usize>() as f64 / rows.len() as f64 };
        let later = firing((0..pre.nrows()).filter(|&r| positions[r] != 0).collect());
        assert_ne!(later, firing((0..pre.nrows()).collect()), "the first tokens fire differently");
        let settings = settings();
        let device = Device::host();
        let posterior = Posterior::new(&explanation, 1000).unwrap();
        let mut scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap();
        let device_posterior = DevicePosterior::new(&device, &explanation, &posterior, 1000.0, None, 0).unwrap();
        let counts = activity(&mut scorer, &explanation, (&posterior, &device_posterior), &sequences, &settings).unwrap();
        assert!((counts[1].nonzero_per_token - later).abs() <= 1e-12 * later, "layer 1 counts {} functions per token, the tokens after the first {later}", counts[1].nonzero_per_token);
    }

    /// With operation families (`Settings::families`), each base's patched experiment is a read
    /// patch or operations on shared sites, its family drawn from the batch's seed (the same
    /// collection on every call): on the tiny Qwen3 decoder with layer 1's MLP 64 transcoder
    /// features, both kinds appear, and a two-epoch fit on them ends with a finite objective.
    #[test]
    fn a_fit_samples_operations_among_its_families() {
        let export = crate::test_support::tiny_qwen3_export("library_fit_edits", 2);
        let imported = import_language_model(&export, 6, 12).unwrap();
        std::fs::remove_dir_all(&export).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let dir = std::env::temp_dir().join(format!("library_fit_edits_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (full, kept_path) = (dir.join("layer_1.safetensors"), dir.join("kept_1.safetensors"));
        crate::test_support::transcoder_file(&full, 64, 8, 3);
        crate::library_transcoder::Transcoder::open(&full).unwrap().write_kept(&(0..64).collect::<Vec<_>>(), &[0.0; 8], &kept_path).unwrap();
        let explanation = explanation_with(&native, &layers, &BTreeMap::from([(1, kept_path)])).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        use interchange::Family;
        let settings = Settings { families: vec![Family::Read, Family::Swap, Family::Zero, Family::Scale, Family::Push, Family::Cut], epochs: Some(2), ..settings() };
        let device = Device::host();
        let (train, held) = sequences.split_at(4);
        let scorer = Scorer::new(&device, &native, &explanation, &settings).unwrap().prepared(&native, train, &settings).unwrap();
        let mut kinds = (0, 0);
        for draw in draws(train.len(), settings.batch_sequences, settings.seed).unwrap() {
            let experiments = scorer.experiments(&draw, train).unwrap();
            assert_eq!(experiments, scorer.experiments(&draw, train).unwrap(), "one fixed collection");
            for e in &experiments {
                match &e.patch {
                    Some(Patch::Ops { ops, .. }) => {
                        kinds.1 += 1;
                        assert!(!ops.is_empty() && e.position < 12, "{e:?}");
                    }
                    Some(_) => kinds.0 += 1,
                    None => {}
                }
            }
        }
        assert!(kinds.0 > 0 && kinds.1 > 0, "read patches and edits: {kinds:?}");
        let fitted = fit(&device, &native, &explanation, train, held, &settings, "tiny", None, None).unwrap();
        assert!(fitted.report.objective_bits.is_finite(), "{}", fitted.report.objective_bits);
    }

    #[test]
    fn a_fit_stopped_in_the_middle_of_an_epoch_and_resumed_is_the_uninterrupted_fit() {
        let (native, layers, _, sequences) = tiny("library_fit_stopped", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let (train, held) = sequences.split_at(4);
        let path = |name: &str| std::env::temp_dir().join(format!("library_fit_{name}_{}.bin", std::process::id()));
        let (whole, stopped) = (path("whole"), path("stopped"));
        let groups = explanation.groups.len();
        let uninterrupted = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&whole), Some(&mut Stop { groups, steps: 0, stop: None })).unwrap();
        // The second epoch's second step, after its first: the checkpoint is the first epoch's.
        let batches = (train.len() / settings.batch_sequences) as u64;
        assert!(uninterrupted.report.epochs.len() >= 2, "the fit trains a second epoch");
        let Err(message) = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&stopped), Some(&mut Stop { groups, steps: 0, stop: Some(batches + 2) })) else {
            panic!("the fit ran past its stop");
        };
        assert!(message.contains("stopped"), "{message}");
        let resumed = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&stopped), Some(&mut Stop { groups, steps: 0, stop: None })).unwrap();
        // The optimizer's state the step's direction and length read: in this tiny fit the iterate
        // never moves (fewer than RATIO_DRAWS draws), so the means alone cannot show a resume that
        // lost it (the momentum's weights, 4d64ef9bd5's, were not saved).
        let state = |file: &std::path::PathBuf| -> serde_json::Value {
            let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(file.with_extension("json")).unwrap()).unwrap();
            serde_json::json!([saved["step"], saved["averaged"], saved["ratio"], saved["weights"], saved["slope"]])
        };
        let (resumed_state, uninterrupted_state) = (state(&stopped), state(&whole));
        assert!(uninterrupted_state[3].is_array(), "the checkpoint holds the momentum's bias corrections");
        assert_eq!(resumed_state, uninterrupted_state, "the resumed fit's optimizer state");
        for file in [&whole, &stopped] {
            for extension in ["bin", "json", "removals.jsonl"] {
                let written = file.with_extension(extension);
                if written.exists() {
                    std::fs::remove_file(written).unwrap();
                }
            }
        }
        assert_eq!(resumed.report.epochs.len(), uninterrupted.report.epochs.len());
        for (a, b) in resumed.report.epochs.iter().zip(&uninterrupted.report.epochs) {
            assert_eq!((a.data_bits.to_bits(), a.snapshot_bits.to_bits()), (b.data_bits.to_bits(), b.snapshot_bits.to_bits()), "epoch {}", a.epoch);
        }
        assert_eq!(resumed.report.removals.len(), uninterrupted.report.removals.len());
        assert_eq!(resumed.posterior.mean, uninterrupted.posterior.mean);
        assert_eq!(resumed.posterior.log_sd, uninterrupted.posterior.log_sd);
        assert_eq!(resumed.posterior.active, uninterrupted.posterior.active);
    }

    /// A resumed fit is the uninterrupted one bit for bit also where the step has moved IVON's
    /// iterate (the tiny fit above stops before it moves: fewer than `RATIO_DRAWS` draws) and on a
    /// transcoder block (ReLU features with fixed biases, `M`'s MLP at each sequence's first
    /// token through its `Node::Select`) whose experiments include edits of its parts (every
    /// family of `Settings::families`), on the host and on the accelerator when there is one:
    /// eight epochs of the tiny Qwen3 decoder with layer 1's MLP 64 transcoder features, stopped
    /// in the sixth epoch's second step after the iterate has moved, resume to the uninterrupted
    /// fit's final checkpoint (means, deviations, IVON's state, the iterate and the steps its
    /// average spans) and its epochs' objectives; the iterate moves both before the stop and after
    /// the resume, and the means after the resume.
    #[test]
    fn a_transcoder_fit_stopped_after_its_iterate_moved_resumes_bit_exactly() {
        let export = crate::test_support::tiny_qwen3_export("library_fit_moved", 2);
        let imported = import_language_model(&export, 6, 12).unwrap();
        std::fs::remove_dir_all(&export).unwrap();
        let native = split_sites(&imported.program).unwrap();
        let layers = layer_nodes(&native, 2).unwrap();
        let SlotValues::Tokens(tokens) = &imported.family.slots[0] else { panic!("tokens") };
        let sequences: Vec<Vec<u32>> = tokens.chunks(12).map(<[u32]>::to_vec).collect();
        let dir = std::env::temp_dir().join(format!("library_fit_moved_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (full, kept_path) = (dir.join("layer_1.safetensors"), dir.join("kept_1.safetensors"));
        crate::test_support::transcoder_file(&full, 64, 8, 3);
        crate::library_transcoder::Transcoder::open(&full).unwrap().write_kept(&(0..64).collect::<Vec<_>>(), &[0.0; 8], &kept_path).unwrap();
        let explanation = explanation_with(&native, &layers, &BTreeMap::from([(1, kept_path)])).unwrap();
        use interchange::Family;
        let settings = Settings { epochs: Some(8), families: vec![Family::Read, Family::Swap, Family::Zero, Family::Scale], ..settings() };
        let (train, held) = sequences.split_at(4);
        let batches = (train.len() / settings.batch_sequences) as u64;
        let groups = explanation.groups.len();
        let mut devices = vec![Device::host()];
        devices.extend(Device::accelerator(gam_gpu::GpuPolicy::Auto).expect("a device probe"));
        for (d, device) in devices.iter().enumerate() {
            let (whole, stopped) = (dir.join(format!("whole_{d}.bin")), dir.join(format!("stopped_{d}.bin")));
            let uninterrupted = fit(device, &native, &explanation, train, held, &settings, "tiny", Some(&whole), Some(&mut Stop { groups, steps: 0, stop: None })).unwrap();
            let Err(message) = fit(device, &native, &explanation, train, held, &settings, "tiny", Some(&stopped), Some(&mut Stop { groups, steps: 0, stop: Some(5 * batches + 2) })) else {
                panic!("device {d}: the fit ran past its stop");
            };
            assert!(message.contains("stopped"), "device {d}: {message}");
            // The checkpoint the resume reads: five epochs, ten steps.
            let middle = checkpoint_start(&explanation, &stopped).unwrap();
            assert_eq!(middle.epoch, 5);
            let resumed = fit(device, &native, &explanation, train, held, &settings, "tiny", Some(&stopped), Some(&mut Stop { groups, steps: 0, stop: None })).unwrap();
            let start = checkpoint_start(&explanation, &whole.with_extension("start.bin")).unwrap();
            let (a, b) = (checkpoint_start(&explanation, &stopped).unwrap(), checkpoint_start(&explanation, &whole).unwrap());
            let iterate = |s: &Start| s.state.as_ref().expect("a checkpoint holds IVON's state").iterate.clone();
            assert_ne!(iterate(&middle), start.mean, "device {d}: the iterate moved before the stop");
            assert_ne!(iterate(&b), iterate(&middle), "device {d}: the iterate moved after the resume");
            assert_ne!(b.mean, middle.mean, "device {d}: the means moved after the resume");
            let steps = |s: &Start| s.state.as_ref().map(|state| state.steps);
            assert_eq!((steps(&a), a.epoch), (steps(&b), b.epoch), "device {d}");
            assert_eq!(a.mean, b.mean, "device {d}: the means");
            assert_eq!(a.log_sd, b.log_sd, "device {d}: the deviations");
            assert_eq!(a.active, b.active, "device {d}");
            assert_eq!(a.state, b.state, "device {d}: IVON's state, the iterate and the steps its average spans");
            assert_eq!(resumed.report.epochs.len(), uninterrupted.report.epochs.len());
            for (x, y) in resumed.report.epochs.iter().zip(&uninterrupted.report.epochs) {
                assert_eq!((x.data_bits.to_bits(), x.snapshot_bits.to_bits()), (y.data_bits.to_bits(), y.snapshot_bits.to_bits()), "device {d}: epoch {}", x.epoch);
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A removal trial (a copy of the posterior with groups removed) shares every operator the
    /// removed groups do not touch with its source, and copies only those they do; the device's
    /// upload of the trial sends only those, and its values equal a full upload's bit for bit.
    #[test]
    fn a_removal_trial_shares_the_operators_it_does_not_change() {
        let (native, layers, _, _) = tiny("library_trial_sharing", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let posterior = Posterior::new(&explanation, 1000).unwrap();
        let group = explanation.groups.len() - 1;
        let mut trial = posterior.clone();
        trial.remove(&[group]);
        let touched: Vec<bool> = explanation.trainable.iter().map(|op| explanation.groups[group].cells.iter().any(|c| c.operator == *op)).collect();
        assert!(touched.iter().any(|t| *t) && touched.iter().any(|t| !*t));
        for (i, touched) in touched.iter().enumerate() {
            assert_eq!(!trial.mean[i].same(&posterior.mean[i]), *touched, "operator {i}'s mean");
            assert_eq!(!trial.log_sd[i].same(&posterior.log_sd[i]), *touched, "operator {i}'s deviations");
        }
        let device = Device::host();
        let mut sent = DevicePosterior::new(&device, &explanation, &posterior, 1000.0, None, 0).unwrap();
        sent.set_values(&posterior).unwrap();
        sent.set_values(&trial).unwrap();
        let fresh = DevicePosterior::new(&device, &explanation, &trial, 1000.0, None, 0).unwrap();
        for i in 0..explanation.trainable.len() {
            let ((a, b), (c, e)) = (sent.values(i).unwrap(), fresh.values(i).unwrap());
            assert!(a.iter().chain(b.iter()).zip(c.iter().chain(e.iter())).all(|(x, y)| x.to_bits() == y.to_bits()), "operator {i}");
        }
    }

    #[test]
    fn the_fit_converges_removes_and_reports_its_posterior_mean() {
        let (native, layers, _, sequences) = tiny("library_fit", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let device = Device::host();
        let (train, held) = sequences.split_at(4);
        let checkpoint = std::env::temp_dir().join(format!("library_fit_{}.bin", std::process::id()));
        let fit = fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&checkpoint), None).unwrap();
        // A finished fit's checkpoint resumes to the same posterior without another step.
        let resumed = super::fit(&device, &native, &explanation, train, held, &settings, "tiny", Some(&checkpoint), None).unwrap();
        // A checkpoint of another export, other sequences or another library is refused, and so
        // is a fit that would resume from it.
        let own = identity("tiny", &native, &explanation, train, held);
        check_checkpoint(&checkpoint, &own).unwrap();
        let mut groups = explanation.clone();
        groups.groups.swap(0, 1);
        let (mut swapped, mut shared) = (train.to_vec(), explanation.clone());
        swapped.swap(0, 1);
        shared.artifact.program.rules[0].name.push('x');
        // Head 1 of layer 0 reading head 0's key: the shared-parameter map changes.
        let mut tied = explanation.clone();
        let (key, _) = key_value(&tied.artifact.program, 0, 0);
        let rule = tied.artifact.program.rules.iter_mut().find(|r| r.name == "library.l0.h1").unwrap();
        rule.nodes[2] = Node::Affine { terms: vec![(0, key)], bias: None };
        // The same shape from other starting values: a warmed explanation is another fit.
        let mut warmed = explanation.clone();
        let op = warmed.trainable[0];
        let source = &warmed.artifact.program.operators[op];
        let values = source.matrix().mapv(|v| v * 1.5);
        warmed.artifact.program.operators[op] = Arc::new(Operator::dense(source.name.clone(), source.rows.clone(), source.cols.clone(), values.clone(), exact_precision(values.iter().copied()).unwrap(), source.provenance.clone()).unwrap());
        for (other, field) in [
            (identity("tiny", &native, &warmed, train, held), "definition"),
            (identity("another", &native, &explanation, train, held), "export"),
            (identity("tiny", &native, &explanation, &swapped, held), "tokens"),
            (identity("tiny", &native, &shared, train, held), "program"),
            (identity("tiny", &native, &groups, train, held), "groups"),
            (identity("tiny", &native, &tied, train, held), "sharing"),
        ] {
            let refusal = check_checkpoint(&checkpoint, &other).unwrap_err();
            assert!(refusal.contains(field), "{refusal}");
        }
        // The ownership map, the scale references and a tie's fixed operators define the
        // explanation as well: each alone changes only the definition.
        let mut owned = explanation.clone();
        owned.artifact.owners.pop();
        let mut referenced = explanation.clone();
        referenced.reference[0] *= 2.0;
        let column = crate::library_sharing::tie_column(&explanation, (1, 3), (0, 5), 0.7).unwrap();
        let mut scattered = column.clone();
        let at = scattered.artifact.program.operators.iter().position(|o| o.name.ends_with(".scatter")).unwrap();
        if let OperatorBody::Dense { values, .. } = &mut Arc::make_mut(&mut scattered.artifact.program.operators[at]).body {
            values.mapv_inplace(|v| 2.0 * v);
        }
        for (a, b) in [(&explanation, &owned), (&explanation, &referenced), (&column, &scattered)] {
            let (a, b) = (identity("tiny", &native, a, train, held), identity("tiny", &native, b, train, held));
            assert_eq!((&a.export, &a.tokens, &a.program, &a.groups, &a.sharing), (&b.export, &b.tokens, &b.program, &b.groups, &b.sharing));
            let refusal = check_checkpoint_identity(&checkpoint, &a, &b).unwrap_err();
            assert!(refusal.contains("definition"), "{refusal}");
        }
        assert!(super::fit(&device, &native, &explanation, train, held, &settings, "another", Some(&checkpoint), None).is_err());
        // The reader refuses a checkpoint of another explanation too, and reads its own exactly.
        assert!(checkpoint_posterior(&warmed, &checkpoint).unwrap_err().contains("definition"));
        assert_eq!(checkpoint_posterior(&explanation, &checkpoint).unwrap().means(), fit.posterior.means());
        // The running fit's explanation read from its checkpoint is the fit's posterior mean.
        assert!(!checkpoint.with_extension("artifact.bin").exists(), "a save writes the checkpoint alone");
        let literals = Literals::of(&device);
        assert_eq!(checkpoint_artifact(&explanation, &checkpoint, literals).unwrap(), literals.apply(posterior_mean(&explanation, &fit.posterior).unwrap()).unwrap());
        std::fs::remove_file(&checkpoint).unwrap();
        std::fs::remove_file(checkpoint.with_extension("json")).unwrap();
        assert_eq!(resumed.posterior.active, fit.posterior.active);
        assert_eq!(resumed.posterior.means(), fit.posterior.means());
        assert_eq!(resumed.report.epochs.len(), fit.report.epochs.len());
        let report = &fit.report;
        // Every clean experiment scores its 12 tokens, every patched one those from its position on.
        assert!(report.scored_tokens > 4 * 12 && report.scored_tokens <= 2 * 4 * 12, "{}", report.scored_tokens);
        assert_eq!(report.families.values().sum::<usize>(), 2 * 4, "two experiments per training base");
        assert_eq!(report.removals.last().unwrap().removed, 0, "the fit ends when no removal is accepted");
        assert!(report.removals.iter().all(|r| r.after_bits <= r.before_bits), "a removal never increases the objective");
        assert!(report.epochs.iter().all(|e| e.snapshot_bits.is_finite() && e.data_bits.is_finite() && e.held_out.objective_bits_per_token.is_finite()));
        let artifact = posterior_mean(&explanation, &fit.posterior).unwrap();
        artifact.validate_coverage(&native).unwrap();
        let (start, end) = (explanation.artifact.program.real_count(), artifact.program.real_count());
        let removed: usize = explanation
            .groups
            .iter()
            .zip(&fit.posterior.active)
            .filter(|(_, active)| !**active)
            .flat_map(|(g, _)| &g.cells)
            .filter(|c| explanation.artifact.program.operators[c.operator].rows.group_count() > 1)
            .map(|c| c.rows.len() * c.cols.len())
            .sum();
        assert!(end <= start && start - end >= removed.min(start), "removed groups are absent blocks: {start} → {end}");
    }

    #[test]
    fn the_scale_reference_survives_a_warm_start() {
        // A warm start from values three times M's keeps every group's reference at M's mean
        // square, so each active group's scale sends the exponent round(log2 9.009) = 3 and the
        // description grows by the difference of their Elias δ lengths; the start at M sends 0.
        // Exponent 2's bin codes in as many bits as 3's, and the edge of exponent 1's (one bit
        // shorter) lies `1.67 ln 2` below `log 9.009`, where the divergence of a group of at least
        // 8 entries has risen by more than 4 nats.
        let (native, layers, _, _) = tiny("library_scale_reference", "gelu_tanh");
        let start = explanation(&native, &layers).unwrap();
        let mut moved = start.artifact.clone();
        for &op in &start.trainable {
            let old = Arc::clone(&moved.program.operators[op]);
            let values = old.matrix() * 3.0;
            let precision = exact_precision(values.iter().copied()).unwrap();
            moved.program.operators[op] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).unwrap());
        }
        let warm = crate::library_sharing::warm(&start, &moved).unwrap();
        assert_eq!(warm.reference, start.reference);
        let (cold, hot) = (Posterior::new(&start, 1000).unwrap(), Posterior::new(&warm, 1000).unwrap());
        assert_eq!(hot.initial, cold.initial);
        let bits = gam_gpu::tensor::scale_code_bits;
        for (g, (c, h)) in cold.priors().iter().zip(hot.priors()).enumerate() {
            assert_eq!(c.bits, bits(0), "{}", start.groups[g].name);
            assert_eq!(h.bits, bits(3), "{}", start.groups[g].name);
        }
    }

    #[test]
    fn every_candidate_is_asked_the_native_questions() {
        // A warm start from moved values, and a tied candidate built from it, are asked the native
        // questions: bitwise the same experiments and targets as the start at M.
        let (native, layers, _, sequences) = tiny("library_protocol", "gelu_tanh");
        let start = explanation(&native, &layers).unwrap();
        let mut fitted = start.artifact.clone();
        for (k, &op) in start.trainable.iter().enumerate() {
            let old = Arc::clone(&fitted.program.operators[op]);
            let values = old.matrix().mapv(|v| v * (1.0 + 0.1 * (k as f64 + 7.0 * v).sin()));
            let precision = exact_precision(values.iter().copied()).unwrap();
            fitted.program.operators[op] = Arc::new(Operator::dense(old.name.clone(), old.rows.clone(), old.cols.clone(), values, precision, old.provenance.clone()).unwrap());
        }
        let warm = crate::library_sharing::warm(&start, &fitted).unwrap();
        let tied = crate::library_sharing::tie(&warm, &[crate::library_sharing::Tie { source: (0, 0), target: (1, 1), scale: 0.5 }]).unwrap();
        let settings = settings();
        let device = Device::host();
        let scorers: Vec<Scorer> = [&start, &warm, &tied].iter().map(|e| Scorer::new(&device, &native, e, &settings).unwrap()).collect();
        for (b, draw) in draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap().iter().enumerate() {
            let batch = draw.batch(&sequences).unwrap();
            let experiments: Vec<Vec<Experiment>> = scorers.iter().map(|s| s.experiments(draw, &sequences).unwrap()).collect();
            assert!(experiments.iter().all(|e| *e == experiments[0]), "batch {b}: the experiments");
            let made: Vec<Vec<(Array2<f64>, Vec<f64>)>> = scorers.iter().map(|s| s.experiments.targets(&batch, &experiments[0]).unwrap().host(&device).unwrap()).collect();
            assert!(made.iter().all(|m| *m == made[0]), "batch {b}: the targets");
        }
    }

    #[test]
    fn unequal_batches_weigh_every_scored_token_once() {
        // Six sequences in batches of four and two: the batches' data terms, weighed as the fit
        // weighs them, average to the collection's, and their per-token weights to the
        // collection's per token; weighing each batch by its own scored tokens would not.
        let (native, layers, _, sequences) = tiny("library_batch_weights", "gelu");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = Settings { batch_sequences: 4, ..settings() };
        let mut scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let shrunk: Vec<Array2<f64>> = Posterior::new(&explanation, 1000).unwrap().means().into_iter().map(|m| m * 0.9).collect();
        let draws = draws(sequences.len(), settings.batch_sequences, settings.seed).unwrap();
        let batches: Vec<(f64, usize)> = draws
            .iter()
            .map(|draw| {
                let experiments = scorer.experiments(draw, &sequences).unwrap();
                let (bits, _) = score(&mut scorer, &draw.batch(&sequences).unwrap(), &experiments, &shrunk, false).unwrap();
                (bits.iter().flatten().sum::<f64>() * LN_2, bits.iter().map(Vec::len).sum())
            })
            .collect();
        assert_eq!(batches.len(), 2);
        let tokens: usize = batches.iter().map(|b| b.1).sum();
        let collection: f64 = batches.iter().map(|b| b.0).sum();
        let (scale, weight) = batch_weights(draws.len(), tokens);
        let mean = |w: &dyn Fn(usize) -> f64| batches.iter().map(|&(d, n)| w(n) * d).sum::<f64>() / batches.len() as f64;
        assert!(collection > 0.0 && (mean(&|_| scale) - collection).abs() <= 1e-12 * collection);
        assert!((mean(&|_| weight) - collection / tokens as f64).abs() <= 1e-12 * collection / tokens as f64);
        let own = mean(&|n| tokens as f64 / n as f64);
        assert!((own - collection).abs() > 1e-3 * collection, "{own} against {collection}: the batches' shares of the tokens differ");
    }

    #[test]
    fn held_out_interventions_are_drawn_independently_of_the_training_batches() {
        // Training batch b and held-out batch b, one patched experiment per base: their positions
        // agree at the chance rate 1/12 of two independent uniform positions over 12 tokens, and
        // whole interventions (hybrid, patch and position) agree as often as those of training
        // batches b and b + 1, which are independent by construction.
        let (native, layers, _, sequences) = tiny("library_held_out_seed", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let many: Vec<Vec<u32>> = sequences.iter().cycle().take(2000).cloned().collect();
        let patched = |experiments: Vec<Experiment>| -> Vec<Experiment> { experiments.into_iter().filter(|e| e.patch.is_some()).collect() };
        let training: Vec<Vec<Experiment>> = draws(many.len(), settings.batch_sequences, settings.seed).unwrap().iter().map(|d| patched(scorer.experiments(d, &many).unwrap())).collect();
        let held: Vec<Vec<Experiment>> = held_out_experiments(&scorer, &many, &settings).unwrap().into_iter().map(|(_, e)| patched(e)).collect();
        let pairs = |a: &[Vec<Experiment>], b: &[Vec<Experiment>]| -> Vec<(Experiment, Experiment)> {
            a.iter().zip(b).flat_map(|(x, y)| x.iter().cloned().zip(y.iter().cloned())).collect()
        };
        let aligned = pairs(&training, &held);
        let shifted = pairs(&training, &training[1..]);
        let n = aligned.len() as f64;
        let positions = aligned.iter().filter(|(a, b)| a.position == b.position).count() as f64;
        let (mean, sd) = (n / 12.0, (n / 12.0 * 11.0 / 12.0).sqrt());
        assert!((positions - mean).abs() <= 5.0 * sd, "{positions} of {n} positions agree against {mean} ± {sd}");
        let same = |pairs: &[(Experiment, Experiment)]| pairs.iter().filter(|(a, b)| a.explained == b.explained && a.patch == b.patch && a.position == b.position).count() as f64;
        let (a, s) = (same(&aligned), same(&shifted));
        assert!((a - s).abs() <= 5.0 * (a + s + 1.0).sqrt(), "{a} held-out interventions repeat training ones, against {s} between training batches");
    }

    #[test]
    fn the_held_out_sample_follows_the_training_weights() {
        // Held-out experiments are a sample of the training distribution, not an enumeration of
        // hybrid sizes: one clean and one patched experiment per base, each family's count within
        // five standard deviations of its stated weight (four blocks: P alone 1/2 + 1/2 * 1/4;
        // attention and MLP single reads 1/4 each; joint reads 1/2).
        let (native, layers, _, sequences) = tiny("library_held_out_sample", "gelu_tanh");
        let explanation = explanation(&native, &layers).unwrap();
        let settings = settings();
        let scorer = Scorer::new(&Device::host(), &native, &explanation, &settings).unwrap();
        let held: Vec<Vec<u32>> = sequences.iter().cycle().take(2000).cloned().collect();
        let sampled = held_out_experiments(&scorer, &held, &settings).unwrap();
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for (draw, experiments) in &sampled {
            assert_eq!(experiments.len(), 2 * draw.bases.len());
            for (family, count) in interchange::census(experiments, scorer.experiments.variables()) {
                *counts.entry(family).or_default() += count;
            }
        }
        let n = held.len() as f64;
        let blocks = 2.0 * layers.len() as f64;
        for (family, p) in [("clean_alone", 0.5 + 0.5 / blocks), ("read_attention", 0.25), ("read_mlp", 0.25), ("read_joint", 0.5)] {
            let (mean, sd) = (n * p, (n * p * (1.0 - p)).sqrt());
            assert!((counts[family] as f64 - mean).abs() <= 5.0 * sd, "{family}: {} against {mean} ± {sd}", counts[family]);
        }
    }
}
