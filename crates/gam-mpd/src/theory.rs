//! The theory of the program decomposition (#2951): what its certificates prove, what its
//! objective is invariant to, when it identifies a planted program, how it reads as a causal
//! abstraction, why it needs no adversary, and which of its claims survive a change of code.
//!
//! Each theorem below is exercised numerically by `theory_tests`, and each names the owner files
//! whose behaviour it states. The objects are those of `contract`, `engine` and
//! `operator_program`; nothing here re-implements them. The only code in this file is the
//! bound on any adversary's distinguishing advantage ([`adversary_advantage_upper`], Theorem 5d),
//! the admissible family of integer codes ([`IntegerCode`]), the robustness criterion for a claim
//! across that family ([`claim_robustness`], Theorem 6), and the quotient-consistency check of an
//! internal interface ([`quotient_consistency`], Theorem 7).
//!
//! # 0. Setting
//!
//! * `F` is the native network, given as an `operator_program::OperatorProgram` over the
//!   contract's `operator_program::Declarations`, executed with forward-error bands.
//! * A contract (`contract::Contract`) declares a family of input rows `x_1..x_N` and,
//!   for a sampled family, the independent unit `u(i)` each row belongs to (a document, an
//!   episode). It declares no tolerance.
//! * A program `P` is a message `m ∈ {0,1}*` of the program code; `P = dec(m)` given the
//!   declarations. `L(P) = |m|` bits (`operator_program::OperatorProgram::code_bits`).
//! * The objective is the two-part code
//!   `J(P; F) = L(P) + D(P; F)`, `D(P; F) = Σ_i KL(F(x_i) ‖ P(x_i)) / ln 2`
//!   (`contract::Contract::score`). `J` is known only in an interval
//!   `[J⁻, J⁺]` from the bands; the engine accepts a candidate only when `J⁺(new) < J⁻(current)`
//!   (`contract::ProgramScore::proven_shorter_than`).
//! * `J*` denotes the exact-arithmetic objective: the exact program at its decoded (lattice)
//!   reals against the exact network. Every band below is a proven enclosure of `J*`.
//! * `ρ` is the native edit compiler (`compile`, owned by edit-compiler): a program-level
//!   intervention `α` compiles to a native parameter edit `ρ(α)` with `ρ(0) = θ`, and the
//!   requirement is `P_α(x) ≈ F_{ρ(α)}(x)` on the declared family.
//!
//! **Lemma 0 (monotone search).** Every accepted step of `engine::decompose` strictly
//! lowers `J*`: `J*(new) ≤ J⁺(new) < J⁻(cur) ≤ J*(cur)`. Since `J* ≥ L ≥ 0` and fewer than
//! `2^{b+1}` messages have length at most `b`, at most `2^{⌈J⁺(start)⌉+1}` steps are accepted, so
//! the search terminates even with an unbounded budget.
//!
//! # 1. Certificate soundness
//!
//! **Assumptions.** Each certificate holds under exactly these.
//!
//! * **(B) Bands.** IEEE-754 binary64 with round-to-nearest-even; `exp`, `sin`, `cos` within one
//!   ulp (the `attention` tests check this at their fixture points); gam-math's
//!   `normal_cdf_and_pdf` within its proven relative error and underflow floor. These give the
//!   forward-error radii of `operator_program` (module note there) and the KL-over-logit-box
//!   bound of [`super::bounds::kl_over_logit_boxes`]. A fused multiply-add only removes roundings,
//!   so it never invalidates a `γ_k` bound.
//! * **(D) Declarations fixed before data.** The declarations (domains, slots, declared cycles)
//!   and the code (the integer, subset and lattice codes of `codec` and `precision`) are fixed
//!   before any row is seen. Everything the search chooses from the data (supports, precisions,
//!   bases, recovered labellings) is written inside the message and paid for in `L`.
//! * **(U) Independent units.** For a sampled family, the units are drawn i.i.d. from the named
//!   population (or without replacement from a finite one; Hoeffding's bound holds for that
//!   too, by his Theorem 4). Rows inside one unit may depend on each other arbitrarily.
//! * **(CP) Beta quantile.** The Clopper–Pearson bound rests on statrs' regularized incomplete
//!   Beta function, whose accuracy is not proven. `contract::clopper_pearson_upper`
//!   raises the quantile until the computed CDF clears the confidence and steps one float up, so
//!   the only unproven step is the CDF evaluation. `theory_tests` checks the coverage of an
//!   independent Clopper–Pearson computed from log-space binomial tails.
//!
//! **Theorem 1a (exhaustive certificates).** For a `contract::FamilyKind::Complete`
//! family under (B), `total_kl`, `max_kl` and the argmax agreement of
//! `contract::Contract::evaluate` are proven statements about the exact decoded program
//! against the exact network on every row of the declared domain, and on nothing else.
//!
//! *Proof.* Per row, [`super::bounds::kl_over_logit_boxes`] encloses the exact KL between any two
//! logit vectors inside the two boxes (its module note), and the boxes contain the exact logits by
//! (B). [`super::verify::certified_argmax`] certifies an argmax only when its lower end clears
//! every other upper end, so a certified argmax is the exact one. The family maximum
//! ([`super::verify::exhaustive_supremum`]) and the sum (with its `γ_N` summation band) range over
//! every row. `contract::Contract::score` decodes the encoded message before it
//! executes, so the program certified is the one whose length is `L`. ∎
//!
//! **Lemma 1b (Kraft for the program code).** Under fixed declarations, the set `M` of messages
//! that `operator_program::OperatorProgram::decode` accepts is prefix-free, so
//! `Σ_{m∈M} 2^{−|m|} ≤ 1`.
//!
//! *Proof.* The decoder is sequential: each read (a prefix integer, a fixed index, a subset, a
//! lattice code, one bit) consumes bits whose number is determined by the bits already read, the
//! declarations and the objects already decoded. The only uses of the remaining length are
//! refusals ("counts beyond the message"), which can reject but never change how a bit is parsed.
//! Suppose `m ∈ M` and `m·s ∈ M` with `s` nonempty. Parsing `m·s` makes the same reads as parsing
//! `m` for as long as `m` lasts, because a refusal check that passes with fewer remaining bits
//! also passes with more. The parse of `m` ends exactly at its last bit (the decoder refuses
//! trailing bits), so the parse of `m·s` ends at the same point with `|s| > 0` bits left and is
//! refused. So no accepted message is a proper prefix of another, and Kraft's inequality holds.
//! The component codes are themselves prefix codes with Kraft sums: Elias ω, 1 in the unbounded
//! integers and below 1 for the `u64` range; a fixed index into `M` symbols, `M·2^{−⌈log₂M⌉} ≤ 1`;
//! the subset code `L_ω(k+1) + ⌈log₂ C(n,k)⌉`, `Σ_k 2^{−L_ω(k+1)} C(n,k) 2^{−⌈log₂C(n,k)⌉} ≤ 1`.
//! A program may have several accepting messages (a truncated label index, for instance); that
//! only adds prior mass and never breaks the inequality. ∎
//!
//! **Theorem 1c (fixed program, Clopper–Pearson).** Under (B), (U) and (CP), if the program was
//! fixed before the units were drawn (selected on a disjoint split), then with probability at
//! least `confidence` the population rate of units with an exact argmax disagreement is at most
//! `contract::PopulationBound::fixed_program_upper`.
//!
//! *Proof.* Let `ℓ*(u)` be 1 when some row of unit `u` has an exact argmax disagreement, and
//! `ℓ(u)` the computed indicator (a disagreement or an uncertified argmax). A certified agreement
//! is an exact agreement (1a), so `ℓ* ≤ ℓ` pointwise and the counts satisfy `k* ≤ k`. `k*` is
//! binomial `(n, R*)`, the Clopper–Pearson upper end `U(·)` covers `R*` at `k*` with probability at
//! least `confidence`, and `U` increases in `k`, so `U(k) ≥ U(k*)` covers too. ∎
//!
//! **Theorem 1d (selected program, Occam).** Under (B), (D) and (U), for any selection rule
//! whatever (the engine's search, its refits and precision choices included), with probability
//! at least `1 − δ` the selected program `P̂ = dec(m̂)` satisfies
//!
//! ```text
//! R*(P̂) ≤ k/n + √((|m̂| ln 2 + ln(1/δ)) / (2n)),
//! ```
//!
//! the value `contract::occam_upper` returns with `δ = 1 − confidence`, rounded outward.
//!
//! *Proof.* Fix `m ∈ M`. The `ℓ*(U_j)` are i.i.d. in `[0, 1]`, so Hoeffding gives
//! `Pr[R*(m) > k*_m/n + ε] ≤ e^{−2nε²}`. With `ε_m = √((|m| ln 2 + ln(1/δ))/(2n))` the right side is
//! `δ 2^{−|m|}`. A union bound over `M` and Lemma 1b give
//! `Pr[∃m ∈ M : R*(m) > k*_m/n + ε_m] ≤ δ Σ_m 2^{−|m|} ≤ δ`. Outside that event the inequality holds
//! for every message at once, so for `m̂`, however it depends on the sample, and `k*_m̂ ≤ k`. The
//! code is fixed by (D), which is what makes the prior `2^{−|m|}` independent of the sample. ∎
//!
//! **What is not certified.** The Clopper–Pearson bound applied to a program selected on the same
//! sample is not valid; `theory_tests` exhibits the undercoverage. Neither bound speaks about the
//! KL of a population row, only about the unit argmax-disagreement rate. An `Exhaustive` status
//! is about the declared domain alone and never extends to a population.
//!
//! # 2. Invariance
//!
//! **Theorem 2a (reference side).** `J(P; F)` and every certificate depend on `F` only through its
//! banded logits on the family. So if `g` is any function-preserving reparametrization of `F`
//! (a duplicated, rescaled or renamed component of the network, or an element of the declared
//! gauge group of `gauge`: norm-gain folds, SwiGLU unit scales, per-group `GL(r)` on OV, the
//! normed rotary QK scales), then in exact arithmetic `J*(P; F) = J*(P; F∘g)` for every `P`, and
//! the minimizers and verdicts coincide. In floating point, `|D(P; F) − D(P; F∘g)|` is at most the
//! sum over rows of [`super::bounds::kl_over_logit_boxes`]' band with the radii of both
//! executions, and a verdict certified against one is certified against the other whenever the
//! argmax margin clears both radii.
//!
//! *Proof.* `Contract::score` reads `F` only as the reference `BandedMatrix`. The exact logits of
//! `F∘g` equal those of `F`, so the exact KL of every row is unchanged; the enclosure widths are
//! the only difference. ∎
//!
//! This is the invariance an objective defined on the parameters lacks: a decomposition of `θ`
//! into parameter components changes when `θ` moves along a gauge orbit, while `J` sees only the
//! function.
//!
//! **Theorem 2b (renaming).** Operator names and provenance are not sent, and permuting the
//! operator list or the basis list (with every reference remapped) leaves every field's length
//! unchanged: a reference is a fixed index into a list whose length is unchanged. So
//! `L(πP) = L(P)` and `J(πP) = J(P)` exactly for every renaming `π`.
//!
//! **Theorem 2c (dyadic rescaling).** Scaling a dense operator by `2^k` and its reader by `2^{−k}`,
//! each moved to the lattice `k` bits coarser or finer, leaves every lattice index unchanged. `L`
//! changes by the two precision codewords alone, and the function by nothing (a power-of-two scale
//! is exact). A non-dyadic rescaling is a different program with the same function: the verdict
//! is unchanged, the code breaks the tie, and among the programs of one function orbit the
//! search keeps the shortest (the gauge ambiguity of Theorem 3).
//!
//! **Theorem 2d (duplication).** Let `P` read a dense operator `A` (lattice `q`, present reals
//! `E ≥ 1`) through one term of one affine node, and let `P_c`, `c = 2^k`, replace it by `c` equal
//! copies `A/c`, each on lattice `q + k` and read by its own term of the same node. Then `P_c`
//! computes the same function and `L(P_c) > L(P)`.
//!
//! *Proof.* Each copy's indices equal `A`'s, so each copy costs `A`'s code with the precision
//! codeword `ℓ(q)` replaced by `ℓ(q + k)`. The added terms, the larger operator count and the
//! wider operator references only add bits. Let `S ≥ 24` be `A`'s code without its precision
//! codeword and indices (two interfaces of at least 9 bits each, the 2-bit kind, a subset code of
//! at least 1 bit, a count of at least 3 bits). With `ℓ ≥ 1` and `Δ = max(0, ℓ(q) − ℓ(q + k))`,
//! `L(P_c) − L(P) ≥ (c − 1) S + c ℓ(q + k) − ℓ(q) ≥ 25(c − 1) − Δ`. For `q ≥ 0`, `Δ = 0` (the zigzag
//! argument grows with `q`). For `−k ≤ q < 0`, `Δ < ℓ(−k) = ω(2k) ≤ 13` for `k < 64`. For `q < −k`,
//! `zigzag(q)+1 = 2|q| ≤ (k + 1)(zigzag(q+k)+1)`, and one doubling of an Elias ω argument adds at most 5
//! bits in the `u64` range (the increment is `1 + Δ_ω(W)` with `Δ_ω(W) ≤ 4` for `W ≤ 64`), so
//! `Δ ≤ 5⌈log₂(k + 1)⌉ ≤ 30`. Since `25(2^k − 1) ≥ 25` exceeds `Δ` for `k = 1` (where `Δ ≤ 5`) and every
//! larger `k`, the difference is positive. ∎
//!
//! **Theorem 2e (splits need a subadditive index code).** Let `P′` replace `A` by parts
//! `A_1 + … + A_c = A` (`c ≥ 2`) on one lattice and with `A`'s present pattern, each read by its own
//! term of the same node. Then `P′` computes the same function, and `L(P′) > L(P)` whenever the index
//! code is subadditive, `ℓ(a + b) ≤ ℓ(a) + ℓ(b)`. The merged operator's structure code equals one part's,
//! so the parts' extra structure (at least 24 bits each, 2d) and extra terms are pure overhead, and its
//! indices cost `Σ_e ℓ(Σ_j a_{j,e}) ≤ Σ_j Σ_e ℓ(a_{j,e})`.
//!
//! The signed Elias δ code is strictly subadditive on nonzero pairs, and so is γ. With
//! `M = max(|a|, |b|) ≥ 1`: `zigzag(a+b)+1 ≤ 4M+1`; `ℓ_δ(4M+1) ≤ ℓ_δ(2M) + 3` (the leading term grows by
//! one and `⌊log₂(⌊log₂ n⌋+1)⌋` by at most one); `ℓ_δ(2M) ≤ ℓ_δ(zigzag(±M)+1)` by monotonicity; and
//! `ℓ_δ(zigzag(s)+1) ≥ ℓ_δ(2) = 4` for `s ≠ 0`. For γ the steps are `+2` and `≥ 3`.
//!
//! The signed Elias ω code is not subadditive: `ℓ(8) = 11 > ℓ(7) + ℓ(1) = 10`, and the excess reaches
//! 2 bits (`ℓ(−32768) = 28 > ℓ(−32767) + ℓ(−1) = 26`). Under ω, an operator of `k` entries equal to 8
//! beside one entry equal to 1 (which pins the lattice) costs `11k + 3` index bits and its split
//! `[7,…,7,1] + [1,…,1,0]` costs `10k + 4`, so for large `k` the structure-free split was strictly
//! shorter. On this theorem the program code (`operator_program`) writes lattice indices in signed
//! δ; the standalone [`super::precision::LatticeCode`] still writes ω, and `theory_tests` keeps both
//! the ω counterexample and the δ subadditivity. Counts and subset cardinalities are still written in ω, whose excess is at most
//! 2 bits per row group; a split with a different present pattern per part is therefore covered
//! only when that excess is outweighed by the saved operator overhead. A split across lattices
//! (a coarse part plus a fine correction) is not a duplicate: it is multiresolution structure
//! and may be shorter.
//!
//! **Theorem 2f (edit certificate).** `sup{uᵀKu : uᵀGu ≤ 1}` and its refusal `ker G ⊄ ker K` are
//! invariant under every surjective linear reparametrization of the control vector: renaming
//! (a permutation `S`), rescaling (an invertible diagonal `S`) and duplication (`T` summing the
//! copies). With `G′ = TᵀGT`, `K′ = TᵀKT` and `T` onto, `{Tu′ : u′ᵀG′u′ ≤ 1} = {v : vᵀGv ≤ 1}` and
//! `u′ᵀK′u′ = (Tu′)ᵀK(Tu′)`, so the suprema agree, and `ker G′ = T⁻¹ ker G ⊆ T⁻¹ ker K = ker K′` iff
//! `ker G ⊆ ker K`. The certificate is therefore a property of the realized edits, not of how the
//! explanation names or scales its controls. It is stated on the resolved quotient: a direction
//! whose edit is below the eigenvalue band of `G` counts as no edit.
//!
//! # 3. Identification within the IR class
//!
//! Write `D_μ(P) = E_{x∼μ} KL(F(x) ‖ P(x)) / ln 2` for the per-row divergence under the row law
//! `μ` (for a complete family repeated `r` times, `μ` is uniform on the domain `X` and `N = r|X|`),
//! and `P ~ Q` for the equivalence generated by: renaming (2b); the gauge moves of `P`'s reals that
//! map lattice points to lattice points and keep the function (signed permutations, the affine
//! automorphisms `x ↦ ax + b` of a recovered cycle); and binding equivalence (which instance of a
//! reused rule is the template, when the alternatives cost the same).
//!
//! **Theorem 3 (MDL identification).** Assume
//!
//! 1. (Kraft) Lemma 1b under fixed declarations;
//! 2. (realizability) some `P*` of the IR class has `KL(F(x) ‖ P*(x)) = 0` for `μ`-almost every `x`;
//! 3. (uniqueness) every `P` with `L(P) ≤ L(P*)` and `D_μ(P) = 0` has `P ~ P*`;
//! 4. (bounded loss) every `P` with `L(P) < L(P*)` has `KL(F(x) ‖ P(x)) ≤ B` for every `x`;
//! 5. (reachability) the search returns a minimizer of `J` over a set `R` of programs that contains
//!    `P*`, and the bands separate `J` at the minimizer.
//!
//! Let `S = {P : L(P) < L(P*), D_μ(P) > 0}`; it has fewer than `2^{L(P*)}` members. Then:
//!
//! * (complete family) the minimizers of `J` lie in `[P*]` iff
//!   `r > r₀ = max_{P∈S∩R} (L(P*) − L(P)) / (|X| D_μ(P))`, and for `r < r₀` some shorter program
//!   beats `P*`;
//! * (sampled rows, `x_i` i.i.d. `μ`) `Pr[argmin J ⊄ [P*]] ≤ Σ_{P∈S∩R} exp(−2N (D_μ(P) − (L(P*) − L(P))/N)² ln²2 / B²)`
//!   for `N > max_{S∩R} (L(P*) − L(P))/D_μ(P)`, which tends to 0 exponentially.
//!
//! *Proof.* `J(P*) = L(P*)` for every sample, since every row KL vanishes. A program with
//! `L(P) ≥ L(P*)` has `J(P) ≥ L(P) ≥ J(P*)`, with equality only when `D = 0`, hence (by 3) only
//! when `P ~ P*`. A program with `L(P) < L(P*)` and `D_μ(P) = 0` is `~ P*` by 3. Only `S` remains, and
//! `J(P) − J(P*) = N D̂_N(P) − (L(P*) − L(P))`. For a complete family `D̂_N = D_μ` exactly, which gives
//! the sharp threshold. For sampled rows, `D̂_N(P)` is a mean of `N` i.i.d. values in `[0, B/ln 2]`,
//! and Hoeffding bounds `Pr[D̂_N(P) ≤ (L(P*) − L(P))/N]` by the displayed term; a union over the
//! finite `S ∩ R` finishes. ∎
//!
//! This is two-part MDL consistency in the realizable case (Barron & Cover 1991, index of
//! resolvability `min_P L(P)/N + D_μ(P)`, which here equals `L(P*)/N` and tends to 0). The finiteness of
//! `S` is the MDL fact that does the work: only finitely many programs are shorter than `P*`, and
//! each one that is wrong pays a divergence that grows linearly in `N`.
//!
//! **Reachability.** The engine is a proven-descent search (Lemma 0), so it returns a minimizer
//! over `R` in either of two situations:
//!
//! * (class) some primitive proposes every member of a finite class `H ∋ P*` at every program. At
//!   termination no member is proven shorter, so the result minimizes `J` over `H` up to the bands;
//! * (descent) every program `Q ≁ P*` the search can visit has a proposal whose `J` is proven lower,
//!   and `P*` has none. The search then cannot stop anywhere else, and it stops (Lemma 0).
//!
//! The discovery primitives supply reachability pattern by pattern. A declared or recovered cycle
//! reaches the character basis when the planted table is a sum of character planes (the
//! `PlaneBasis` / declared-characters primitives);
//! `DropBlocks` and `Coarsen` reach any sub-support and coarser lattice of the current program. A
//! factor group reused up to change of basis is reachable by rule discovery only when its instances
//! are exactly related (the ARD evidence and anti-unification propose, they do not certify).
//!
//! **Remaining ambiguities.** Identification is of `[P*]`, not of `P*`: the gauge moves above
//! (`engine_tests`' recovered cycle is the planted one up to `x ↦ ax + b`), binding equivalence, the
//! listing order, and exact code-length ties inside the class, broken by search order.
//!
//! # 4. The causal-abstraction correspondence
//!
//! In the terms of Geiger et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic
//! Interpretability* (JMLR 26, 2025):
//!
//! * **Low level `L`.** The causal model of `F`: activation variables at every native node, input
//!   variables, and parameter variables `Θ` with constant mechanisms (the trained weights).
//! * **High level `H`.** The causal model of `P`: one variable per program node, whose mechanism is the
//!   node's law (affine, bilinear, softmax, mix, pointwise, readout), with the operators' reals as
//!   constant parameter variables. Unlike an input-output model, `H` supplies `F_H` for every
//!   intermediate variable.
//! * **Interventionals on `H`.** The program edits of `engine::Edit` and the gain settings of
//!   blocks, planes and laws, as set-type atoms: each sets its target to a value relative to the base
//!   program. Atoms on distinct targets commute, and a second setting of one target annihilates the
//!   first (Def. 16), so their closure is an intervention algebra (Thm. 21), and every history of
//!   edits has the normal form `Sort(Collapse(·))` of Def. 18 / Remark 22. An atom must be read off
//!   the base program: a lattice move applied to already-rounded reals is a double rounding, which
//!   is not left-annihilative (Remark 22's failure), so the algebra's precision atom sets the lattice
//!   of the base reals.
//! * **Interventionals on `L`.** Native parameter edits `ρ(α)`: hard interventions on `Θ`
//!   (`compile::NativeEditPlan`), composed with input primings.
//! * **`ω` and `τ`.** `ω(x ∘ ρ(α)) = x ∘ α` on `Domain(ω) = {x ∘ ρ(α) : x ∈ family, α ∈ A}`, so `ρ` is a
//!   section of `ω`. `τ` sends a low-level solution to its inputs, its outputs and, for each certified
//!   internal interface (a node reached from native activations by an exact rewrite), that node's
//!   value. Where the rewrite chain is a basis change or a composition of exact rewrites, the
//!   alignment is the bijective translation of Def. 28, and `ρ` is its canonical pullback (Remark 29):
//!   an edit of a high-level coordinate is the native edit `τ⁻¹ ∘ (set coordinate) ∘ τ`.
//!
//! **Theorem 4.** Let `P` be returned with its rewrite chain, `A` a declared finite family of
//! program interventions with `ρ(0) = θ`, and `X` the contract's family. Then:
//!
//! 1. `H` is an approximate transformation of `L` (Def. 41) under `(τ, ω)` with `Sim` the KL of output
//!    distributions, `P` uniform on `X × A` and `S` the maximum, and its grade is the exhaustive
//!    certificate `ε = max_{x∈X, α∈A} KL(F_{ρ(α)}(x) ‖ P_α(x))`, proven up to the bands (Theorem 1a).
//!    This is the `d_max` approximate abstraction of Beckers et al. (Remark 43) with `Domain(ω)`
//!    enumerated, not sampled.
//! 2. The alignment factors as Prop. 40 does, one certified step each: bijective translations (the
//!    exact rewrites: basis changes, composition, constant folding; Thm. 30); marginalization (dropped
//!    blocks and pruned nodes, `Π_⊥`); variable merge (one operator shared by every node that reads it,
//!    and one rule shared by its bound instances); value merge (rounding reals to a declared lattice,
//!    `δ = round`). By Remark 27 the exact steps compose, and the approximate ones are graded by 1.
//! 3. If `ε = 0` within the bands and `ρ` is the pullback of a bijective translation on `A`, then `H` is a
//!    constructive abstraction of `L` (Def. 33) on `Domain(ω) = ρ(A)`, and the square of Eq. 3 also
//!    commutes for interventions that target a certified internal interface.
//! 4. For a sampled family, the grade is a population statement about units: with probability at
//!    least `1 − δ`, the rate of units on which some row's argmax differs is at most the Occam bound
//!    (Theorem 1d), with `S` the unit-level disagreement probability.
//!
//! *Proof.* 1 is Def. 41 with the stated `(Sim, P, S)`, evaluated by enumeration. 2 is Prop. 40 applied
//! to the alignment the rewrite chain induces, with Thm. 30 for each translation. For 3, a vanishing
//! grade makes Eq. 3 hold on every element of `Domain(ω)`, and Remark 29's construction makes `ω`
//! order-preserving (an isomorphism onto the hard interventions of the translated signature). 4 is
//! Theorem 1d. ∎
//!
//! **Complexity and strength.** The claim a run makes is the triple (the program `P`, its realization
//! `(rewrite chain, ρ)`, the declared input and intervention families). `L(P)` measures complexity in
//! bits and `D` measures infidelity in bits, so `J = L + D` trades them at one bit per bit, a rate fixed
//! by the code rather than a hand-set coefficient. A smaller `Domain(ω)` is a weaker claim, which is why
//! every certificate is stated with its domain.
//!
//! **Comparison with VPD.** The comparison uses only VPD's public method (the Goodfire paper and its
//! repository `goodfire-ai/param-decomp`): each weight matrix is written as a sum of rank-one
//! subcomponents plus a residual, a causal-importance network maps each input to importances
//! `g_c(x) ∈ [0, 1]`, masks are drawn in `[g_c(x), 1]`, and training penalizes reconstruction error under
//! stochastic and adversarially chosen masks together with an importance penalty. Read through the
//! definitions above:
//!
//! * The square VPD checks is Eq. 3 with the high-level model reduced to inputs and outputs: `ω` sends every
//!   permitted masking at `x` to the unintervened input `x`. The decomposition is of parameters and names
//!   no high-level mechanism for an intermediate variable, so commutation under interventions that target
//!   an internal cell is not part of the claim. Here every node of `P` is a high-level variable with its own
//!   mechanism, and Theorem 4.3 covers interventions on certified internal interfaces.
//! * Its interventionals (Def. 11) are maskings of rank-one subcomponents; set-type masking atoms commute
//!   and left-annihilate, so they form an intervention algebra (Thm. 21's argument), but they are the only
//!   atoms. Here the atoms are typed edits of blocks, laws, bases and rule bindings, and `ρ` compiles each
//!   to a native edit or reports it coupled or infeasible.
//! * Its `Domain(ω)` at `x` is fixed by the learned importances `g(x)`, so the domain of the claim is itself
//!   an output of a trained network. Here `Domain(ω)` is declared before the search and enumerated.
//! * Its grade is a Def. 41 statistic over sampled or gradient-searched masks, which lower-bounds the
//!   supremum over the domain and grows with the attack budget. Here the grade is a certified maximum
//!   (finite families), an exact generalized eigenvalue (linear edit families), or an Occam bound
//!   (sampled families): Section 5.
//! * An importance strictly inside `(0, 1)` makes a subcomponent partly ablatable, which an alignment's
//!   binary partition (Def. 31) cannot express. `P`'s partition is binary; a real-valued edit is a
//!   coordinate of a translated signature (Def. 28), not a graded cell membership.
//! * Its simplicity terms are penalties with hand-set coefficients. Here complexity is a code length that
//!   satisfies Kraft's inequality, which is what lets it enter the Occam bound.
//!
//! # 5. Why no adversary is needed
//!
//! VPD's requirement that "any combination of ablations of the unimportant components keeps the
//! output" is a universally quantified constraint over per-input partial masks,
//! `∀x ∈ X, ∀m with m_c ∈ [g_c(x), 1]: d(x, m) ≤ ε`, where `g` is the trained importance network. A search
//! over `m` (PGD) only produces lower bounds on `sup_m d`: a violation it finds refutes the claim, and a
//! search that fails proves nothing. The guarantee our method gives for each kind of claim is:
//!
//! | claim | guarantee | conditions |
//! |---|---|---|
//! | joint program fidelity on a finite domain `X × A` | **exact**: the maximum row KL, the summed KL and the argmax agreement, exhaustively | (B); `X`, `A` declared and finite |
//! | joint program fidelity on a sampled domain | **statistical**: Clopper–Pearson for a program fixed before the sample, Occam for the selected program, on the unit argmax-disagreement rate and on the mean unit total variation (5d) | (B), (D), (U); only i.i.d. units count; no worst-case KL over the population |
//! | linear edit family `u ∈ ℝ^m` | **exact**: `‖e(u)‖² ≤ λ* ‖ΔD(u)‖²_G` for every `u`, `λ*` attained, or a refusal with a witness | error linear in `u`; `G`, `K` with assembly bands; stated on the resolved quotient (2f) |
//! | nonlinear edit family with an enclosure | **sound bound** over every mask box, interiors included | the `secant` and `bounds` enclosures; an unresolved box reports its gap |
//! | nonlinear edit family without an enclosure | **search-survived**: no certificate, reported with the search that failed to refute it | never promoted to a bound |
//!
//! **Theorem 5 (the guarantees are suprema, not search results).**
//!
//! 1. (finite) For finite `X × A` the exhaustive maximum (Theorem 1a) is the supremum: the outcome is
//!    certified, or refuted with the maximizing `(x, α)` as its witness.
//! 2. (sampled) Theorems 1c and 1d.
//! 3. (linear) If the prediction error is linear in the control change `u` and the realized edit is
//!    `ΔD(u) = Σ_j u_j D_j`, then `‖e(u)‖² ≤ λ* ‖ΔD(u)‖²_G` for every `u ∈ ℝ^m` at once, where
//!    `λ* = sup{uᵀKu : uᵀGu ≤ 1}` is the top generalized eigenvalue on `range(G)`
//!    ([`super::compile::null::physically_null_supremum`]), attained at its witness, and the certificate is
//!    refused when `ker G ⊄ ker K`. Every mask vector `m ∈ [0, 1]^C` is one `u = 1 − m`, so every combination
//!    of partial ablations is covered by one number.
//! 4. (nonlinear with an enclosure) The secant operators of `secant` enclose the exact finite change of
//!    softmax and bilinear maps between two endpoints, and
//!    [`super::bounds::kl_supremum_over_logit_boxes`] bounds `sup KL` over every logit pair in two boxes, so a
//!    box enclosure over a mask box bounds every mask inside it.
//!
//! *Proof.* 1 by enumeration. 2 is Section 1.
//! 3: for `u ∈ range(G)` scale to `uᵀGu = 1`; for `u ∈ ker G`, `Ku = 0` unless refused, and `K` is positive
//! semidefinite, so the cross terms vanish. 4: the enclosures are proven over their whole boxes (`secant`,
//! `bounds` module notes). ∎
//!
//! **Corollary (adversaries are dominated).** Any adversary returns some point of the declared domain, so its
//! value is at most the certified supremum; when the status is `Exact` with a witness, the witness attains it.
//! `theory_tests` runs random and ascending searches against the linear certificate and sampled points
//! against the box bound, and checks both facts.
//!
//! **Theorem 5b (a global joint claim has no per-input quantifier).** A program `P` asserts
//! `∀x ∈ X, ∀α ∈ A: KL(F_{ρ(α)}(x) ‖ P_α(x)) ≤ ε`, with `A` a declared set of global edits that does not depend on
//! `x` or on any trained function. Its truth depends only on `(F, P, ρ, X, A)`.
//!
//! *Proof.* The quantifier domain is `X × A`, fixed before the search. `P` states no per-input importance: a
//! dropped block, a coarsened lattice or a removed plane is removed for every input, so the clean-run
//! fidelity on `X` already evaluates every input's use of every retained and every removed piece, jointly,
//! in one program. Whatever `P` computes per input (a softmax route, a gate) is computed by `P` itself and is
//! part of `P_α(x)`, not a separate assertion. So there is no per-input object (an importance, a permitted
//! mask box) whose choice an adversary could exploit: an adversary can only choose `(x, α) ∈ X × A`, and
//! Theorem 5 covers every such choice. VPD's claim instead ranges over `x` and over masks inside `[g(x), 1]`,
//! a box set by a trained network, so the claim's domain is itself a learned object, and its supremum is not
//! computed. ∎
//!
//! **Theorem 5c (counterexample search as a verifier, CEGAR).** Let a search (any heuristic, PGD included)
//! propose `(x, α)` outside the current contract's family, and let every proposal be evaluated exactly. The
//! loop is: the engine minimizes `J` on the contract `(X_k, A_k)`; the verifier checks the returned `P_k`
//! on the declared `(X, A)`; a refuting `(x, α)` is added as a row, `X_{k+1} = X_k ∪ {x}`, `A_{k+1} = A_k ∪ {α}`.
//!
//! 1. A counterexample enters `J` as one more row with the weight of every row. It never enters as a loss
//!    weighted by an attack budget, and nothing is differentiated through the search.
//! 2. For finite `(X, A)` the loop ends after at most `|X| + |A|` refinements, and its final statement is the
//!    exhaustive certificate over all of `(X, A)` (Theorem 5.1), which does not depend on the search.
//! 3. Where the verifier is a search without an enclosure, the final statement is "search-survived", with
//!    the search named, and never a bound.
//! 4. Rows added by the search are not i.i.d. units, so they are excluded from the unit count of Theorems 1c
//!    and 1d; the population bounds use only the i.i.d. units, or a fresh held-out split.
//!
//! *Proof.* 1 and 3 are the loop's definition. 2: each refinement adds a new element of `X` or `A`, so it
//! cannot repeat; acceptance is by the certificate over the declared domain. 4: Hoeffding's bound needs the
//! units to be drawn independently of the selection, and an added row is chosen by the search. ∎
//!
//! This is why the method does not overfit to the attack. Adversarial training makes the attack's
//! objective part of the fitted objective, so the fitted decomposition is selected to defeat that attack
//! at that budget, and a stronger attack finds more (the reported robustness grows worse with the number
//! of attack steps). Here the attack only chooses which rows are added, and acceptance never consults it:
//! beating the search does not pass the certificate, and a stronger search can only add rows.
//!
//! **Theorem 5d (the adversary's game, won without an adversary).** Fix a declared finite edit family `A`
//! and a population `μ` of units. In the game `G_k(A)` an adversary plays `k` rounds; in round `j` a unit
//! `U_j ∼ μ` is drawn fresh, and the adversary, seeing `U_j` and the whole transcript so far, knowing `F`, `P`,
//! `ρ` and every certificate, with unbounded computation and its own randomness, picks a row `x` of `U_j` and an
//! edit `α ∈ A`. It then receives one output `y_j`, drawn from `F_{ρ(α)}(x)` in world 0 and from `P_α(x)` in
//! world 1 (greedy decoding is the case of point masses at the argmax). After `k` rounds it outputs a bit; its
//! advantage is `|Pr_0[1] − Pr_1[1]|`. Choosing a mask at an input to break the explanation, which is what
//! VPD's adversary does, is one move of round 1.
//!
//! Let `Z*(u) = max_{x ∈ u, α ∈ A} TV(F_{ρ(α)}(x), P_α(x))` and let `Ẑ(u) ≥ Z*(u)` be its certified upper end: the
//! exhaustive maximum (Theorem 1a) over the unit's rows and `A` of
//! [`super::bounds::total_variation_over_logit_boxes`], or, for greedy decoding, the unit's disagreement
//! indicator `ℓ(u)` of 1c (the total variation of two point masses is `1[argmax differs]`). For a continuous
//! family the per-row maximum comes from 5.3 or 5.4 instead of enumeration. Then:
//!
//! 1. (composition) every adversary's advantage in `G_k(A)` is at most `k E_μ[Z*] ≤ k E_μ[Ẑ]`;
//! 2. (fixed program) with probability at least `1 − δ` over `n` i.i.d. units,
//!    `E_μ[Ẑ] ≤ Ẑ̄ + √(ln(1/δ)/(2n))`, `Ẑ̄` the sample mean; for greedy decoding the Clopper–Pearson end of 1c
//!    replaces it;
//! 3. (selected program) with probability at least `1 − δ`, for the selected `P̂ = dec(m̂)`,
//!    `E_μ[Ẑ_{P̂}] ≤ Ẑ̄ + √((|m̂| ln 2 + ln(1/δ))/(2n))`;
//! 4. (the guarantee) so, with probability at least `1 − δ`, no adversary of `G_k(A)`, adaptive and unbounded,
//!    has advantage above `min(1, k(Ẑ̄ + √((|m̂| ln 2 + ln(1/δ))/(2n))))`, the value
//!    [`adversary_advantage_upper`] returns (`|m̂| = 0` for 2). Certificates compose in the same currency: failure
//!    probabilities of separate statistical statements add (a union bound), and total variations along a chain
//!    of approximations add (the triangle inequality: the native model, its compiled edit, the program).
//!
//! *Proof.* 1: let `H_j` answer the first `j` rounds from `P` and the rest from `F`, so `H_0` is world 0 and
//! `H_k` world 1. `H_{j−1}` and `H_j` differ only in the kernel that answers round `j`; everything after it
//! (later units, the adversary's later choices, later answers, the output bit) is one Markov kernel applied to
//! both, and total variation does not increase under a kernel. So `TV(H_{j−1}, H_j)` is at most the expected
//! total variation of round `j`'s answer given the past, `E[TV(F_{ρ(α_j)}(x_j), P_{α_j}(x_j))]`. `U_j` is drawn
//! independently of the past and `(x_j, α_j)` lies in `U_j × A`, so that is at most `E_μ[Z*]`. The triangle
//! inequality over the `k` hybrids gives `k E_μ[Z*]`, and a bit's advantage is at most the total variation of
//! the transcripts. 2: Hoeffding's inequality for the i.i.d. `[0, 1]`-valued `Ẑ(U_i)`. 3: the union argument of
//! 1d with Lemma 1b; it needs only values in `[0, 1]`, not indicators. 4: 1 on the event of 2 or 3. ∎
//!
//! The computed per-unit maximum over `A` is what makes an adversary unnecessary: its best move in each round
//! is already enumerated (or bounded, for a continuous family), and the population average is certified by a
//! bound that holds for every selection rule. `theory_tests` plays random adaptive strategies against the
//! exact transcript laws and checks 1, checks that the greedy one-round strategy attains `E_μ[Z*]`, and checks
//! the coverage of 2 and 3 over repeated samples.
//!
//! **Theorem 5e (where an adversary still adds power).** Exactly where the game changes:
//!
//! 1. (inputs chosen, not drawn) 5d averages over `μ`. For any `t > 0`, `μ{u : Z*(u) > t} ≤ E_μ[Ẑ]/t` (Markov),
//!    so a search that draws candidates from `μ` finds a unit beyond `t` with probability at most `E_μ[Ẑ]/t`
//!    per draw and needs at least `t/E_μ[Ẑ]` draws in expectation. A search that optimizes the input itself
//!    (off `μ`, or onto a set of small `μ`-measure) asks for `sup_x`, which no finite sample certifies.
//! 2. (edits outside `A`, or a continuous family with neither linear structure nor an enclosure) the
//!    per-unit maximum is not computed.
//!
//! In both cases the search is a verifier in the loop of 5c: its finds become rows, the statement for the
//! uncovered part stays "search-survived", and nothing else changes. *Proof.* 1 is Markov's inequality applied
//! to `Z* ≤ Ẑ`, and a geometric count of draws. 2 is the absence of the statements 5.1, 5.3, 5.4 use. ∎
//!
//! **What is not guaranteed.** Interventions outside the declared family (other inputs, other edit kinds,
//! use-specific rather than global edits) are not covered. For a sampled family only unit averages are
//! bounded (the disagreement rate and the mean unit total variation), at confidence `1 − δ`, and not the
//! population's worst-case KL (5e.1). The linear certificate is relative to the chosen edit metric `G` and to
//! the resolved quotient of 2f, and it covers a nonlinear network's error only to first order unless an
//! enclosure bounds the remainder. An unresolved box reports its gap and claims nothing inside it. Every
//! statement assumes (B), with `ln` within one ulp and `sqrt` correctly rounded for 5d's bound.
//!
//! # 6. Code sensitivity and robust claims
//!
//! The code decides what counts as short, so it is the method's main inductive bias. A claim `φ` (a
//! predicate on programs: "these heads share rule R", "behaviour B is explained by subprogram Q", "the
//! table keeps plane 2 alone") is evaluated against a finite set `C` of certified candidates (the programs
//! the searches returned and certified) under a family `𝒦` of admissible codes.
//!
//! **Admissible codes.** A code of the program IR is admissible when every field is written with a prefix
//! code (Kraft), and the codes of lattice indices and cardinalities are nondecreasing in the magnitude
//! and subadditive (Theorem 2e). The family [`IntegerCode::FAMILY`] holds Elias γ, δ and ω, whose tails grow as
//! `2 log n`, `log n + 2 log log n` and `log n + log log n + …`. Only lengths matter: by the converse of Kraft's
//! inequality a prefix code with any lengths satisfying `Σ 2^{−ℓ} ≤ 1` exists, and both MDL and the Occam bound
//! read only lengths. Elias ω fails subadditivity (2e) and is kept in the family as the incumbent.
//!
//! **Definition.** Under a code `c`, `φ` is *proven* on `C` when `min_{P∈C, φ(P)} J_c⁺(P) < min_{P∈C, ¬φ(P)} J_c⁻(P)`, and
//! *refuted* when the same holds with `φ` and `¬φ` exchanged. It is *robust* on `(C, 𝒦)` when it is proven under
//! every `c ∈ 𝒦`, *robustly refuted* when refuted under every `c`, and *code-dependent* when it is proven under
//! one code and refuted under another ([`claim_robustness`]). The data term does not depend on the code, so
//! `J_c(P) = L_c(P) + D(P)` and only the program bits vary.
//!
//! **Theorem 6.**
//!
//! 1. (invariances make most alternatives equivalent) Two admissible codes that differ only on fields whose
//!    values are the same multiset in every candidate change every `J_c` by one constant, so they give the
//!    same verdict. Renaming (2b) and exact gauge moves (2a, 2c) change no field's multiset of values; duplication
//!    is penalized by every admissible code (2d, 2e). So the family need only vary the codes of the fields on which
//!    candidates differ: lattice indices, counts and cardinalities, precision exponents, binding indices.
//! 2. (mixture closure) For codes `c_1..c_K ∈ 𝒦`, `L_mix(P) = min_k L_{c_k}(P) + ⌈log₂ K⌉` is a prefix code (a
//!    `⌈log₂ K⌉`-bit selector, then the chosen code). A claim robust over `{c_k}` is proven under `L_mix`.
//! 3. (margin) A claim robust with margin `μ` (the least over codes of `min_{¬φ} J⁻ − min_φ J⁺`) is proven under every
//!    code `c′` with `|L_{c′}(P) − L_c(P)| < μ/2` for some `c` and every candidate `P`.
//! 4. (code dependence is a finite-sample effect) Under Theorem 3's hypotheses for each `c ∈ 𝒦` (with the threshold
//!    `r₀(c)`), every identification claim about `[P*]` is robust for `r > max_c r₀(c)`.
//!
//! *Proof.* 1: `J_c − J_{c′}` is a sum of the two codes' length differences over fields whose value multisets
//! agree, hence a constant. 2: the selector makes `L_mix` prefix-free, and with `k′` attaining
//! `min_k min_{¬φ} J⁻_{c_k}`, `min_φ J⁺_mix − ⌈log₂K⌉ ≤ min_φ J⁺_{c_{k′}} < min_{¬φ} J⁻_{c_{k′}} = min_{¬φ} J⁻_mix − ⌈log₂K⌉`.
//! 3: each side moves by less than `μ/2`. 4: Theorem 3 for each of the finitely many codes. ∎
//!
//! A run therefore reports a claim as robust, robustly refuted, or code-dependent with the codes on each side;
//! a code-dependent claim is a statement about the prior, not about the network.
//!
//! **Theorem 6b (what the code decides).** Over the admissible codes:
//!
//! *Code-free* (the statement does not read `L`, or holds under every prefix code fixed before the data):
//!
//! * every certificate of Sections 1a, 5, 5d.1 and 7: they read `F`, `P`, the bands and the declared family, never `L`;
//! * the validity of 1c, 1d and 5d.2–4 under any prefix code fixed before the sample. Only the value moves: codes
//!   `c`, `c′` change the Occam term by at most `√(|L_c(P̂) − L_{c′}(P̂)| ln 2 / (2n))`, since `√(a + b) ≤ √a + √b`;
//! * the invariances 2a, 2b, 2f and the function side of 2c;
//! * the duplication penalty 2d, under each code of the family: the proof needs only
//!   `Δ = max(0, ℓ(q) − ℓ(q + k)) < 25(2^k − 1)`, and `theory_tests` checks it for γ, δ and ω over every
//!   `|q| ≤ 2^16`, `k < 64`, with the doubling step `ℓ(2w) ≤ ℓ(w) + 2` (γ), `+ 3` (δ), `+ 5` (ω) closing the tail;
//! * identification in the limit (6.4) and every claim with margin above twice the largest per-candidate length
//!   gap between the codes (6.3).
//!
//! *Code-dependent* (the verdict can flip between admissible codes):
//!
//! * the split penalty 2e: it holds under γ and δ and fails under ω, whose worst subadditivity excess is 2 bits;
//! * any finite-sample selection with margin below the codes' gap, which includes Theorem 3's regime `r ≤ max_c r₀(c)`;
//! * every number that is a length: `L`, `J`, the value of the Occam term, `r₀`, Lemma 0's step count;
//! * precision and magnitude trade-offs. With `f = ⌊log₂ n⌋`, `ℓ_γ(n) − ℓ_δ(n) = f − 2⌊log₂(f + 1)⌋`, which is
//!   unbounded, so a candidate that writes a few large indices (a fine lattice) against one that writes many small
//!   ones (a coarse lattice with a residual) can be proven under one code and refuted under another.
//!
//! Unlike Kolmogorov complexity, no invariance theorem bounds the gap between two admissible codes by a constant:
//! they are not universal machines, and their difference grows with the magnitudes written. So the code-free list
//! is exactly what a run may state without naming a code. *Proof.* The first list is Sections 1, 2, 5, 7 read for
//! their use of `L`, 2d's proof with each code's doubling step, and 6.3–6.4. The second: 2e's counterexample, 6.3's
//! converse, the definitions, and `ℓ_γ = 2f + 1`, `ℓ_δ = f + 2⌊log₂(f + 1)⌋ + 1`. ∎
//!
//! # 7. Quotient consistency of an internal interface
//!
//! An internal interface `v` of `P` summarizes the native state `s` at a cut of `F` by
//! `α(s)`, the node's value under its alignment (Section 4). Downstream of the cut the network
//! moves the state by `T(s, u)`, where `u` ranges over the declared interventions that act after the
//! cut, the identity included, and the next interface (the next node, or the output) reads
//! `α′(T(s, u))`.
//!
//! **Theorem 7 (bisimulation).** On a family of states `S`, a deterministic abstract update
//! `f` with `α′(T(s, u)) = f(α(s), u)` for every `s ∈ S` and every declared `u` exists iff
//!
//! ```text
//! α(s₁) = α(s₂)  ⇒  α′(T(s₁, u)) = α′(T(s₂, u))      for all s₁, s₂ ∈ S and every declared u.
//! ```
//!
//! *Proof.* If `f` exists, both sides equal `f(α(s₁), u)`. Conversely, define `f(a, u)` as
//! `α′(T(s, u))` for any `s` with `α(s) = a`; the condition makes the choice irrelevant. ∎
//!
//! This is the existence of `H`'s mechanism at the next variable, which a constructive abstraction
//! requires (Def. 33 needs `F_H` to exist before Eq. 3 can commute), and it is decided without any
//! proposed `f`: checking a given `f` (`E′∘T = g∘E`) is weaker than this, which decides
//! whether any `f` exists. The quantifier over `u` is load-bearing: a summary can be consistent on the
//! clean run and inconsistent under a declared intervention, which then has no abstract counterpart.
//!
//! **Checkable form.** [`quotient_consistency`] takes the `α`-class of each row (from
//! [`exact_classes`] when the interface value is exact, as a dyadic program's node is, or from a
//! declared value merge) and the banded next state of each row under each intervention. Over the
//! finite family it returns the exhaustive `ε = max_u max_{same class} ‖α′(T(s₁, u)) − α′(T(s₂, u))‖∞`
//! with its band. A pair whose next states differ by more than their bands is a counterexample (no
//! deterministic update exists, witnessed by the pair and `u`). Otherwise the family admits an
//! `ε`-bisimulation: choosing any representative per class gives an update within `ε` of every row,
//! and `ε` is exactly zero in exact arithmetic when the bands are all that separates the rows.
//! Real-valued states computed along different native paths are never bitwise equal, so exact
//! bisimulation is certified only as `ε ≤` the band, never as an identity.

use super::codec::{CodecError, prefix_integer_len_bits, signed_codeword_argument};
use super::secant::BandedMatrix;
use super::supports::{EvidenceStatus, EvidenceStatusError, ExactBasis};
use std::collections::BTreeMap;
use std::fmt;

/// Theorem 5d.4: with probability at least `confidence` over the i.i.d. units, no adversary of the
/// `rounds`-round game has advantage above `min(1, rounds (Ẑ̄ + √((code_bits ln 2 + ln(1/δ))/(2n))))`, where
/// `unit_upper[i]` is unit `i`'s certified `Ẑ ∈ [0, 1]`, `δ = 1 − confidence`, and `code_bits` is the selected
/// program's message length (`0` for a program fixed before the units were drawn). Every operation is rounded
/// up; `ln` is taken within one ulp and then stepped up twice.
pub fn adversary_advantage_upper(unit_upper: &[f64], code_bits: u64, confidence: f64, rounds: u64) -> Result<f64, TheoryError> {
    if unit_upper.is_empty() || rounds == 0 {
        return Err(TheoryError::InvalidInput("the bound needs at least one unit and one round".to_string()));
    }
    if !(confidence > 0.0 && confidence < 1.0) {
        return Err(TheoryError::InvalidInput(format!("a confidence lies in (0, 1); got {confidence}")));
    }
    if let Some(index) = unit_upper.iter().position(|value| !(0.0..=1.0).contains(value)) {
        return Err(TheoryError::InvalidInput(format!("unit {index} has total variation {}", unit_upper[index])));
    }
    let n = unit_upper.len() as f64;
    let total = unit_upper.iter().fold(0.0_f64, |sum, value| (sum + value).next_up());
    let mean = (total / n).next_up();
    // `δ` is stepped down, which only raises `ln(1/δ)`; `ln δ < 0`, so its negation is stepped up.
    let delta = (1.0 - confidence).next_down();
    let log_inverse_delta = (-delta.ln()).next_up().next_up();
    let program = (code_bits as f64 * std::f64::consts::LN_2.next_up()).next_up();
    let radius = (((program + log_inverse_delta).next_up() / (2.0 * n)).next_up().sqrt()).next_up();
    Ok(((mean + radius).next_up() * rounds as f64).next_up().min(1.0))
}

/// An integer code of the admissible family of Theorem 6, known by its codeword lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntegerCode {
    /// `2⌊log₂ n⌋ + 1` bits.
    EliasGamma,
    /// `⌊log₂ n⌋ + 2⌊log₂(⌊log₂ n⌋ + 1)⌋ + 1` bits.
    EliasDelta,
    /// The codec's prefix integer code ([`super::codec::encode_prefix_integer`]).
    EliasOmega,
}

impl IntegerCode {
    /// The family Theorem 6 varies over.
    pub const FAMILY: [IntegerCode; 3] = [IntegerCode::EliasGamma, IntegerCode::EliasDelta, IntegerCode::EliasOmega];

    /// The codeword length of `n ≥ 1`.
    pub fn len_bits(self, n: u64) -> Result<u64, CodecError> {
        if n == 0 {
            return Err(CodecError::InvalidInput("the integer codes cover integers from 1".to_string()));
        }
        let floor_log = u64::from(u64::BITS - 1 - n.leading_zeros());
        Ok(match self {
            Self::EliasGamma => 2 * floor_log + 1,
            Self::EliasDelta => floor_log + 2 * u64::from(u64::BITS - 1 - (floor_log + 1).leading_zeros()) + 1,
            Self::EliasOmega => prefix_integer_len_bits(n)?,
        })
    }

    /// The codeword length of a signed integer through the codec's signed map `zigzag(v) + 1`.
    pub fn signed_len_bits(self, value: i64) -> Result<u64, CodecError> {
        self.len_bits(signed_codeword_argument(value)?)
    }
}

/// One candidate's two-part score under every code of a family: its program bits per code, and
/// its code-independent data bits with their proven ends.
#[derive(Clone, Debug, PartialEq)]
pub struct CodedCandidate {
    pub program_bits: Vec<u64>,
    pub data_bits_lower: f64,
    pub data_bits_upper: f64,
}

/// A claim's standing across a family of codes (module note, section 6).
#[derive(Clone, Debug, PartialEq)]
pub enum ClaimVerdict {
    /// Proven under every code; `margin_bits` is the least proven gap.
    Robust { margin_bits: f64 },
    /// Refuted under every code; `margin_bits` is the least proven gap.
    RobustlyRefuted { margin_bits: f64 },
    /// Proven under the `supporting` codes and refuted under the `refuting` ones (indices into the
    /// family); the `undecided` codes leave the comparison inside the bands.
    CodeDependent { supporting: Vec<usize>, refuting: Vec<usize>, undecided: Vec<usize> },
    /// No code refutes and some code leaves the comparison inside the bands, or the reverse.
    Unresolved { supporting: Vec<usize>, refuting: Vec<usize>, undecided: Vec<usize> },
}

/// A refused robustness evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum TheoryError {
    /// The candidates, the claim and the codes do not pair up.
    Shape(String),
    /// An argument outside the domain of a bound.
    InvalidInput(String),
    /// A data-bit interval that is not an ordered pair of nonnegative numbers.
    Interval { candidate: usize, lower: f64, upper: f64 },
    /// Classes read off a banded state whose band is not zero at `row`.
    InexactState { row: usize },
    Evidence(EvidenceStatusError),
}

impl From<EvidenceStatusError> for TheoryError {
    fn from(error: EvidenceStatusError) -> Self {
        Self::Evidence(error)
    }
}

impl fmt::Display for TheoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(message) => write!(f, "claim robustness: {message}"),
            Self::InvalidInput(message) => write!(f, "adversary bound: {message}"),
            Self::Interval { candidate, lower, upper } => {
                write!(f, "claim robustness: candidate {candidate} has data bits [{lower}, {upper}]")
            }
            Self::InexactState { row } => write!(f, "quotient classes: row {row} of the state is not exact"),
            Self::Evidence(error) => write!(f, "theory: {error}"),
        }
    }
}

impl std::error::Error for TheoryError {}

/// The least proven upper and lower total over the candidates on one side of a claim, under one
/// code: `(min J⁺, min J⁻)`, both `+∞` when the side is empty.
fn side_extremes(candidates: &[CodedCandidate], claim: &[bool], side: bool, code: usize) -> (f64, f64) {
    candidates.iter().zip(claim).filter(|(_, holds)| **holds == side).fold(
        (f64::INFINITY, f64::INFINITY),
        |(upper, lower), (candidate, _)| {
            let bits = candidate.program_bits[code] as f64;
            (upper.min((bits + candidate.data_bits_upper).next_up()), lower.min((bits + candidate.data_bits_lower).next_down()))
        },
    )
}

/// Whether `claim` (one flag per candidate) is robust over every code of the family (module note,
/// section 6). Each candidate lists its program bits under every code, in one order.
pub fn claim_robustness(candidates: &[CodedCandidate], claim: &[bool]) -> Result<ClaimVerdict, TheoryError> {
    if candidates.is_empty() || candidates.len() != claim.len() {
        return Err(TheoryError::Shape(format!("{} candidates and {} claim flags", candidates.len(), claim.len())));
    }
    let codes = candidates[0].program_bits.len();
    if codes == 0 || candidates.iter().any(|candidate| candidate.program_bits.len() != codes) {
        return Err(TheoryError::Shape("every candidate needs its bits under the same nonempty family".to_string()));
    }
    for (index, candidate) in candidates.iter().enumerate() {
        let (lower, upper) = (candidate.data_bits_lower, candidate.data_bits_upper);
        if !(lower >= 0.0 && lower <= upper) {
            return Err(TheoryError::Interval { candidate: index, lower, upper });
        }
    }
    let (mut supporting, mut refuting, mut undecided) = (Vec::new(), Vec::new(), Vec::new());
    let (mut support_margin, mut refute_margin) = (f64::INFINITY, f64::INFINITY);
    for code in 0..codes {
        let (holds_upper, holds_lower) = side_extremes(candidates, claim, true, code);
        let (fails_upper, fails_lower) = side_extremes(candidates, claim, false, code);
        if holds_upper < fails_lower {
            supporting.push(code);
            support_margin = support_margin.min(fails_lower - holds_upper);
        } else if fails_upper < holds_lower {
            refuting.push(code);
            refute_margin = refute_margin.min(holds_lower - fails_upper);
        } else {
            undecided.push(code);
        }
    }
    Ok(if supporting.len() == codes {
        ClaimVerdict::Robust { margin_bits: support_margin }
    } else if refuting.len() == codes {
        ClaimVerdict::RobustlyRefuted { margin_bits: refute_margin }
    } else if !supporting.is_empty() && !refuting.is_empty() {
        ClaimVerdict::CodeDependent { supporting, refuting, undecided }
    } else {
        ClaimVerdict::Unresolved { supporting, refuting, undecided }
    })
}

/// The family a quotient-consistency status ranges over (module note, section 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuotientDomain {
    pub rows: usize,
    pub classes: usize,
    pub interventions: usize,
}

/// Two rows of one class, the intervention and the coordinate at which their next states are
/// farthest apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuotientWitness {
    pub intervention: usize,
    pub rows: (usize, usize),
    pub coordinate: usize,
}

/// The classes of an exact state: rows with equal values share a class, numbered in order of first
/// appearance. Refused when any band is nonzero, since a banded value does not prove equality.
pub fn exact_classes(state: &BandedMatrix) -> Result<Vec<usize>, TheoryError> {
    if state.values.dim() != state.bands.dim() {
        return Err(TheoryError::Shape(format!("values {:?} and bands {:?}", state.values.dim(), state.bands.dim())));
    }
    let mut seen: BTreeMap<Vec<u64>, usize> = BTreeMap::new();
    let mut classes = Vec::with_capacity(state.values.nrows());
    for (row, (values, bands)) in state.values.outer_iter().zip(state.bands.outer_iter()).enumerate() {
        if bands.iter().any(|band| *band != 0.0) {
            return Err(TheoryError::InexactState { row });
        }
        // `+ 0.0` identifies the two zeros.
        let key: Vec<u64> = values.iter().map(|v| (v + 0.0).to_bits()).collect();
        let next = seen.len();
        classes.push(*seen.entry(key).or_insert(next));
    }
    Ok(classes)
}

/// Theorem 7's criterion over a finite family: `classes[row]` is the `α`-class of each row and
/// `next[u]` the banded next state of every row under declared intervention `u`. Returns the
/// exhaustive `ε = max_u max_{same class} ‖next_u(s₁) − next_u(s₂)‖∞` with its band, or a counterexample
/// (threshold 0) at the pair whose difference exceeds its band by the most.
pub fn quotient_consistency(
    classes: &[usize],
    next: &[BandedMatrix],
) -> Result<EvidenceStatus<QuotientWitness, QuotientDomain>, TheoryError> {
    let rows = classes.len();
    if rows == 0 || next.is_empty() {
        return Err(TheoryError::Shape("a quotient check needs rows and at least one intervention".to_string()));
    }
    let width = next[0].values.ncols();
    if next.iter().any(|n| n.values.dim() != (rows, width) || n.bands.dim() != (rows, width)) {
        return Err(TheoryError::Shape(format!("every next state must be {rows} × {width} with its bands")));
    }
    let class_count = classes.iter().copied().max().map_or(0, |largest| largest + 1);
    let domain = QuotientDomain { rows, classes: class_count, interventions: next.len() };
    let (mut spread, mut error, mut witness) = (0.0_f64, 0.0_f64, None);
    let mut violation: Option<(f64, f64, QuotientWitness)> = None;
    for (intervention, state) in next.iter().enumerate() {
        for coordinate in 0..width {
            // Per class: the rows holding the largest and the smallest value, and the widest band.
            let mut extremes: BTreeMap<usize, (usize, usize, f64)> = BTreeMap::new();
            for (row, &class) in classes.iter().enumerate() {
                let value = state.values[[row, coordinate]];
                let entry = extremes.entry(class).or_insert((row, row, 0.0));
                if value > state.values[[entry.0, coordinate]] {
                    entry.0 = row;
                }
                if value < state.values[[entry.1, coordinate]] {
                    entry.1 = row;
                }
                entry.2 = entry.2.max(state.bands[[row, coordinate]]);
            }
            for &(high, low, widest) in extremes.values() {
                let difference = state.values[[high, coordinate]] - state.values[[low, coordinate]];
                let rounding = f64::EPSILON * difference;
                // The extremal pair's own bands decide a violation; any pair's exact difference is
                // within twice the class's widest band of the computed spread.
                let band = ((state.bands[[high, coordinate]] + state.bands[[low, coordinate]]).next_up() + rounding).next_up();
                let pair = QuotientWitness { intervention, rows: (low, high), coordinate };
                if difference > spread {
                    spread = difference;
                    witness = Some(pair);
                }
                error = error.max(((2.0 * widest).next_up() + rounding).next_up());
                let excess = difference - band;
                if excess > 0.0 && violation.as_ref().is_none_or(|(value, value_band, _)| excess > value - value_band) {
                    violation = Some((difference, band, pair));
                }
            }
        }
    }
    if let Some((value, band, pair)) = violation {
        return Ok(EvidenceStatus::counterexample(value, band, 0.0, pair)?);
    }
    let cardinality = (rows * next.len()) as u64;
    Ok(EvidenceStatus::exact(spread, error, ExactBasis::Exhaustive { cardinality }, witness, domain)?)
}
