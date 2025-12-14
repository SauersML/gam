/*
----------------------------------------------------------------------------------------------------
// 1. Executive Summary
----------------------------------------------------------------------------------------------------
//
//  Problem:
//      We have a linear prediction model built from thousands of binary variables. Some of these
//      variables are imputed, and we have a confidence score (a probability `p`) for each
//      imputation. We must decide how to incorporate these uncertain variables into the final
//      patient score.
//
//  Two competing strategies:
//      A) The Filtering Method (`S_F`): Set a confidence threshold. Discard all variables below
//         the threshold, treating their contribution as zero.
//      B) The Expectation Method (`S_E`): Use all variables, but weight the effect size of each
//         imputed variable by its probability of being in the "active" state (1).
//
//      We must implement the Expectation Method (`S_E`). It is mathematically guaranteed to
//      produce predictions with a lower or equal Mean Squared Error (MSE) than any filtering
//      method. It is the most accurate predictor that can be constructed from our model and data.
//
----------------------------------------------------------------------------------------------------
// 2. Formal Problem Definition
----------------------------------------------------------------------------------------------------
//
//  Objective:
//      To construct a predictor, `S`, that is the most accurate possible estimate of a patient's
//      true, underlying value, `Y_true`. Accuracy is formally defined by minimizing the
//      Mean Squared Error (MSE).
//
//      Minimize: E[(S - Y_true)²]
//
//  Model Definition:
//      The true value `Y_true` is determined by a linear combination of binary variables.
//
//      Y_true = β₀ + Σᵢ βᵢxᵢ
//
//      Where:
//          * `Y_true`: The true, unobservable value for the patient (e.g., true genetic risk).
//          * `β₀`: The model intercept or baseline value.
//          * `βᵢ`: The known, pre-trained weight (effect size) for variable `i`.
//          * `xᵢ`: The true, unknown binary state (0 or 1) of variable `i`.
//
//  Available Data:
//      For each variable `i`, we do not know `xᵢ` with certainty. Instead, we have a
//      probability derived from our imputation model:
//
//      pᵢ = P(xᵢ = 1 | Data)
//
//      This `pᵢ` is the posterior probability that the true state is 1, given all available
//      data for the patient. For non-imputed variables, `pᵢ` is either 0 or 1.
//
----------------------------------------------------------------------------------------------------
// 3. Mathematical Definition of the Competing Methods
----------------------------------------------------------------------------------------------------
//
//  Method E: The Expectation Score (S_E)
//      This score is the mathematical expectation of `Y_true`, conditioned on the available data.
//      S_E = E[Y_true | Data] = E[β₀ + Σᵢ βᵢxᵢ | Data] = β₀ + Σᵢ βᵢ * E[xᵢ | Data]
//      Since E[xᵢ | Data] = pᵢ, the formula is:
//
//      S_E = β₀ + Σᵢ βᵢpᵢ
//
//  Method F: The Filtering Score (S_F)
//      This score is based on a confidence threshold `τ`. We partition all variables `I` into
//      a high-confidence set `I_H` and a low-confidence set `I_L`, which is discarded.
//
//      I_L = {i ∈ I | 1-τ < pᵢ < τ}  // The set of discarded variables.
//
//      The score is constructed by assuming the contribution of all variables in `I_L` is zero.
//      (For simplicity, we assume the hard-imputed value `x̂ᵢ` is used for the kept set `I_H`).
//
//      S_F = β₀ + Σ_{i ∈ I_H} βᵢx̂ᵢ
//
----------------------------------------------------------------------------------------------------
// 4. Minimization of Mean Squared Error
----------------------------------------------------------------------------------------------------
//
//  We will now prove that MSE(S_E) ≤ MSE(S_F) for any choice of threshold `τ`.
//
//  A. MSE Analysis of the Expectation Score (S_E)
//  ==============================================
//
//  The error of S_E is:
//      Error(S_E) = S_E - Y_true = (β₀ + Σᵢ βᵢpᵢ) - (β₀ + Σᵢ βᵢxᵢ) = Σᵢ βᵢ(pᵢ - xᵢ)
//
//  The MSE is the expected squared error. Assuming uncorrelated errors between variables:
//      MSE(S_E) = E[ (Σᵢ βᵢ(pᵢ - xᵢ))² ] = Σᵢ βᵢ² * E[(pᵢ - xᵢ)²]
//
//  By definition, `pᵢ` is the conditional mean of `xᵢ`, so the error term `(pᵢ - xᵢ)` has a
//  mean of zero. Thus, `E[(pᵢ - xᵢ)²]` is the conditional variance of `xᵢ`.
//  The variance of a Bernoulli variable `xᵢ` with P(xᵢ=1)=pᵢ is `pᵢ(1 - pᵢ)`.
//
//      MSE(S_E) = Σᵢ βᵢ² * pᵢ(1 - pᵢ)
//
//
//  B. MSE Analysis of the Filtering Score (S_F)
//  ============================================
//
//  The error of S_F is the sum of contributions from all discarded variables:
//      Error(S_F) = S_F - Y_true = (β₀ + Σ_{i ∈ I_H} βᵢxᵢ) - (β₀ + Σ_{i ∈ I_H} βᵢxᵢ + Σ_{i ∈ I_L} βᵢxᵢ)
//      Error(S_F) = - Σ_{i ∈ I_L} βᵢxᵢ
//
//  The MSE of any estimator can be decomposed into `Bias² + Variance`.
//
//      1. Bias of S_F:
//         Bias(S_F) = E[S_F - Y_true] = E[-Σ_{i ∈ I_L} βᵢxᵢ] = -Σ_{i ∈ I_L} βᵢpᵢ
//         S_F is a BIASED estimator. It systematically underestimates or overestimates the true score.
//
//      2. Variance of the Error of S_F:
//         Var(Error(S_F)) = Var(-Σ_{i ∈ I_L} βᵢxᵢ) = Σ_{i ∈ I_L} βᵢ² * Var(xᵢ) = Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ)
//
//  Combining these:
//      MSE(S_F) = (Bias(S_F))² + Var(Error(S_F))
//      MSE(S_F) = (-Σ_{i ∈ I_L} βᵢpᵢ)² + Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ)
//      MSE(S_F) = (Σ_{i ∈ I_L} βᵢpᵢ)² + Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ)
//
//
//  C. Direct Comparison
//  ====================
//
//  Let's compare the MSE of the two methods. Note that for the high-confidence variables
//  kept by S_F, their `pᵢ` is effectively 0 or 1, making their `pᵢ(1-pᵢ)` term zero.
//  Therefore, the error for S_E comes only from the low-confidence set `I_L`.
//
//      MSE(S_E) = Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ)
//
//      MSE(S_F) = Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ) + (Σ_{i ∈ I_L} βᵢpᵢ)²
//
//  The difference is immediately obvious:
//
//      MSE(S_F) - MSE(S_E) = (Σ_{i ∈ I_L} βᵢpᵢ)²
//
//  This term is a square and is therefore ALWAYS greater than or equal to zero.
//
//      => MSE(S_f) ≥ MSE(S_e)
//
//  Quod erat demonstrandum.
//
----------------------------------------------------------------------------------------------------
// 5. Refutation of Common Counter-Arguments
----------------------------------------------------------------------------------------------------
//
//  FALLACY 1: "The Expectation Score is too 'timid' and has low variance. A score needs
//              high variance to discriminate between high and low-risk patients."
//
//  CORRECTION: This confuses score variance with predictive power. The formal measure of a
//  predictor's power is its R-squared (`R²`), where `R² = 1 - (MSE / Var(Y_true))`. To maximize
//  `R²`, one must MINIMIZE MSE. Our proof shows that S_E minimizes MSE, therefore S_E maximizes
//  the predictive power (`R²`). The "boldness" of S_F is not signal; it is amplified noise,
//  which increases error and lowers predictive power.
//
//  FALLACY 2: "The proof only shows S_E is the best estimator of the genetic score `G`, not
//              the best predictor of the final clinical phenotype `Y_phenotype`."
//
//  CORRECTION: This is false. The optimal predictor for `Y_phenotype` is, by definition, its
//  conditional expectation `E[Y_phenotype | Data]`.
//
//      S_optimal = E[Y_phenotype | Data] = E[G + ε | Data]
//                = E[G | Data] + E[ε | Data]
//
//  Since the non-genetic component `ε` is independent of the genetic `Data`, `E[ε | Data] = E[ε] = 0`.
//  This leaves `S_optimal = E[G | Data]`, which is precisely the formula for `S_E`. The Expectation
//  Score is the optimal predictor for both the genetic score and the final clinical outcome.
//
----------------------------------------------------------------------------------------------------
// 6. Final Implementation Directive
----------------------------------------------------------------------------------------------------
//
//  The correct, optimal, and required implementation is the Expectation-Based Method.
//
//  For each patient, the score MUST be calculated as follows:
//
//      let score = intercept + effects_and_probabilities.iter().map(|(beta, p)| beta * p).sum();
//
//  Where:
//      * `intercept` corresponds to `β₀`.
//      * `beta` corresponds to the effect size `βᵢ`.
//      * `p` corresponds to the probability `pᵢ` that the variable's true state is 1.
//
//  This approach is not a heuristic. It is the direct implementation of the provably optimal
//  predictor that minimizes prediction error and maximizes predictive power.
//
====================================================================================================
*/

/*
====================================================================================================
                                  VCF EXECUTION ENGINE
                             DESIGN & IMPLEMENTATION GUIDE
====================================================================================================

1.  DESIGN PHILOSOPHY & ARCHITECTURAL SUMMARY
----------------------------------------------------------------------------------------------------

This document specifies the design for a high-performance execution engine for single-sample,
unindexed, bgzip-compressed VCF files. The architecture is engineered to achieve the theoretical
maximum throughput for streaming input by adhering to three foundational principles.

1.  **Single-Pass I/O Model:** The VCF, as the largest input and primary I/O bottleneck, is
    read from disk in a single, forward-only pass. This eliminates redundant I/O operations
    inherent to multi-pass (e.g., prepare-then-run) algorithms.

2.  **Specialized, Zero-Allocation Parsing:** A purpose-built VCF stream parser will be
    implemented. By targeting the specific, expected input format and operating on reusable
    byte buffers, it avoids the overhead of general-purpose VCF libraries. The parser will
    extract only the essential fields (`CHROM`, `POS`, `REF`, `ALT`, and sample dosage),
    minimizing both CPU cycles and heap allocations per variant.

3.  **Stateless Streaming Core:** The core engine operates without large, intermediate state.
    It does not materialize a collection of required variants in memory. The memory complexity
    is O(S), where S is the number of score files, not O(V), where V is the number of variants.
    All variant data is processed and discarded on the fly.

To realize these principles, the conventional separation of a "prepare" stage and a "run" stage
is deliberately fused into a monolithic, unified engine. This design trades modular separation
for a significant gain in performance by combining data reconciliation and computation into a
single, continuous pass.


2.  HIGH-LEVEL ARCHITECTURE & PROCESS FLOW
----------------------------------------------------------------------------------------------------

The VCF workflow is managed by a new, self-contained `vcf_engine` module, which functions as
a unified streaming processor. Its operation is a live, three-way merge-join between two input
data streams and one mutable state object.

1.  **The VCF Data Stream:** Produced by the specialized, high-performance VCF parser.
2.  **The Score Rule Stream:** Produced by the `KWayMergeIterator`, which reads from all
    score files simultaneously to yield a single, sorted stream of score rules.
3.  **The Final Accumulator State:** Represented by the `final_scores` and `missing_counts`
    vectors, which are mutated in place throughout the process.

The sequence of operations is as follows:

1.  **Pre-Flight:** A minimal, one-time setup phase in `main.rs` establishes the dimensions of
    the final output and pre-computes a global baseline score.
2.  **Dispatch:** `main.rs` invokes the `vcf_engine::run` function, passing it all necessary
    context and mutable references to the final accumulator state.
3.  **Unified Stream Processing:** The `vcf_engine` instantiates the two input iterators and
    begins the main merge-join loop. It continuously compares the variant keys at the head of
    each stream, calculates score adjustments on-the-fly, and applies them directly to the
    final accumulator state.
4.  **Finalization:** Upon completion of the single pass, control returns to `main.rs`, which
    writes the fully populated accumulator vectors to the output file.


3.  DETAILED IMPLEMENTATION PLAN
----------------------------------------------------------------------------------------------------

#### PHASE 1: PRE-FLIGHT OPERATIONS

This logic is executed once when a VCF input is detected, before the main engine is invoked.

1.  **Parse Score Headers:** Concurrently read the headers from all specified score files to
    build a global, sorted, and unique list of all score names.

2.  **Establish Global Score Metadata:** From the complete list of names, construct the global
    `score_name_to_col_index: AHashMap<String, ScoreColumnIndex>` lookup map.

3.  **Global Baseline Score Calculation:** A single, fast pass is performed over all score
    files to compute a global baseline score vector. For every variant rule in every score
    file, if the effect allele is the reference allele ("flipped" variant), its contribution
    (`2.0 * weight`) is added to a baseline accumulator. This pass is essential for
    correctly handling missing variants.

4.  **Allocate Final State:** The `final_scores: Vec<f64>` vector is initialized with the
    values from the computed global baseline vector. The `missing_counts: Vec<u32>` vector is
    initialized to zero. Both are sized for one sample and the total number of unique scores.

5.  **Dispatch:** Invoke `vcf_engine::run`, passing all necessary context.

#### PHASE 2: THE SPECIALIZED VCF STREAM PARSER

This is a private component within the `vcf_engine` module.

1.  **Iterator Definition:** The parser will be implemented as an iterator that wraps a `bgzip`
    decoder stream and is compatible with the `peekable()` adapter.

2.  **Dosage Parsing Policy:** The parser must correctly handle both `DS` (dosage) and `GP`
    (genotype probability) `FORMAT` fields.
    *   If `DS` is present, its value is used directly.
    *   If `GP` is present, the expected dosage must be calculated as:
        `Dosage = P(0/1) + (2 * P(1/1))`.

3.  **Zero-Allocation Parsing Model:** The iterator will own a single, reusable line buffer.
    On each call to `next()`, it will read a line into this buffer, then perform optimized,
    byte-slice-based parsing. The `Item` yielded will contain slices that borrow from the
    iterator's internal buffer, avoiding heap allocations for every variant record.

#### PHASE 3: THE UNIFIED STREAMING ENGINE (in `vcf_engine.rs`)

This is the core implementation of the VCF processing logic.

1.  **Iterator Instantiation:** Create `peekable()` instances of the new VCF parser and the
    `KWayMergeIterator`.

2.  **Main Merge-Join Loop:** Implement the primary `while` loop that `peek()`s at the head of
    both iterators and compares their `VariantKey`s.
    *   **If VCF key < Score key:** The VCF contains a variant not required by any score.
        Consume and discard the VCF record.
    *   **If Score key < VCF key:** A required variant is missing from the VCF. This logic is
        critical for correctness. It will consume the score rule, increment the corresponding
        `missing_counts` accumulator, and apply a correction to the `final_scores` accumulator
        if the variant was a flipped variant (by subtracting `2.0 * weight`).
    *   **If keys are equal:** A match is found. This triggers the hot path logic.

#### PHASE 4: THE HOT PATH (MATCHED VARIANT PROCESSING)

This logic is executed within the `Ordering::Equal` branch of the main loop.

1.  **Consume VCF Record:** Consume the single matched record from the VCF iterator. Its
    parsed data (dosage, ref/alt alleles) is now available.

2.  **Handle Missing Genotype:** If the parsed dosage indicates a missing genotype (`./.`),
    the logic will consume the entire group of matching score rules, incrementing the
    `missing_counts` for each, and applying baseline corrections before returning to the main loop.

3.  **Process Score Rule Group:** Loop to consume all score rules from the `KWayMergeIterator`
    that share the same variant key. For each score rule:
    *   **Complex Locus Detection:** A flag from the `KWayMergeIterator` will indicate if
        this is a simple (one rule per locus) or complex (multiple rules) variant.
    *   **Simple Variant Path:**
        1.  Verify the score's effect allele matches either the VCF REF or ALT allele. If not,
            the rule is inapplicable; continue.
        2.  Determine the flip status by comparing the effect allele to the VCF REF allele.
        3.  Calculate the baseline dosage (2.0 for flipped, 0.0 for non-flipped).
        4.  Calculate the final score adjustment: `weight * (observed_dosage - baseline_dosage)`.
            This calculates the delta from the pre-computed baseline state.
        5.  Add this adjustment directly to the `final_scores` accumulator at the index
            provided by the score rule.
    *   **Complex Variant Path:** If a complex locus is detected, switch to a dedicated handler.
        This handler will buffer the single VCF record's data and gather all subsequent
        score rules for that locus. Once the group is collected, it will perform resolution
        logic to apply only the rules compatible with the observed VCF alleles.

#### PHASE 5: CORRECTNESS GUARANTEES & SCENARIO HANDLING

The architecture is designed for arithmetic and logical correctness across all required cases.

*   **Effect Allele as Reference:** Accounted for by the comprehensive baseline-and-adjustment
    system, which correctly handles both initialization and missing data for flipped variants.
*   **Multiple Scores at one Position:** Addressed by the `KWayMergeIterator` and the logic that
    processes the entire group of matching score rules for a given position.
*   **Different Effect Alleles Across Scores:** Addressed because each score rule is evaluated
    independently against the single observed VCF genotype.
*   **Indels and Complex Alleles:** Handled as the allele matching logic relies on byte-wise
    equality, which is agnostic to allele length or composition.
*   **Multiallelic Sites:** Addressed robustly by the "Complex Variant Path." By collecting all
    rules for a site and comparing them against the single, true genotype from the VCF, the
    engine correctly applies only the scores relevant to the alleles the individual actually
    possesses.

*/
