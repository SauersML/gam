import Mathlib

namespace Gam.Terms.Basis

section BSplineFoundations

variable {numKnots : ℕ}

/-- A valid B-spline knot vector: non-decreasing with proper multiplicity. -/
structure KnotVector (m : ℕ) where
  knots : Fin m → ℝ
  sorted : ∀ i j : Fin m, i ≤ j → knots i ≤ knots j

/-- Cox-de Boor recursive definition of B-spline basis function.
    N_{i,p}(x) is the i-th basis function of degree p.
    We use a simpler formulation to avoid index bound issues. -/
noncomputable def bspline_basis_raw (t : ℕ → ℝ) : ℕ → ℕ → ℝ → ℝ
  | i, 0, x => if t i ≤ x ∧ x < t (i + 1) then 1 else 0
  | i, p + 1, x =>
    let left_denom := t (i + p + 1) - t i
    let right_denom := t (i + p + 2) - t (i + 1)
    let left := if left_denom = 0 then 0
                else (x - t i) / left_denom * bspline_basis_raw t i p x
    let right := if right_denom = 0 then 0
                 else (t (i + p + 2) - x) / right_denom * bspline_basis_raw t (i + 1) p x
    left + right

/-- Local support property: N_{i,p}(x) = 0 outside [t_i, t_{i+p+1}).

    **Geometric insight**: The support of a B-spline grows by one knot interval with
    each degree increase. N_{i,p} lives on [t_i, t_{i+p+1}). In the recursion for p+1,
    we combine N_{i,p} (starts at t_i) and N_{i+1,p} (ends at t_{i+p+2}).
    The union creates the new support [t_i, t_{i+p+2}).

    **Proof by induction on p**:
    - Base case (p=0): By definition, N_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0.
    - Inductive case (p+1): Cox-de Boor recursion combines N_{i,p} and N_{i+1,p}.
      Both have zero support outside the required interval. -/
theorem bspline_local_support (t : ℕ → ℝ)
    (h_sorted : ∀ i j, i ≤ j → t i ≤ t j)
    (i p : ℕ) (x : ℝ)
    (h_outside : x < t i ∨ t (i + p + 1) ≤ x) :
    bspline_basis_raw t i p x = 0 := by
  induction p generalizing i with
  | zero =>
    simp only [bspline_basis_raw]
    split_ifs with h_in
    · obtain ⟨h_lo, h_hi⟩ := h_in
      rcases h_outside with h_lt | h_ge
      · exact absurd h_lo (not_le.mpr h_lt)
      · simp only [add_zero] at h_ge
        exact absurd h_hi (not_lt.mpr h_ge)
    · rfl
  | succ p ih =>
    simp only [bspline_basis_raw]
    rcases h_outside with h_lt | h_ge
    · -- x < t_i: both terms zero
      have h_left_zero : bspline_basis_raw t i p x = 0 := ih i (Or.inl h_lt)
      have h_i1_le : t i ≤ t (i + 1) := h_sorted i (i + 1) (Nat.le_succ i)
      have h_right_zero : bspline_basis_raw t (i + 1) p x = 0 :=
        ih (i + 1) (Or.inl (lt_of_lt_of_le h_lt h_i1_le))
      simp only [h_left_zero, h_right_zero, mul_zero, ite_self, add_zero]
    · -- x ≥ t_{i+p+2}: both terms zero
      have h_right_idx : i + 1 + p + 1 = i + p + 2 := by ring
      have h_right_zero : bspline_basis_raw t (i + 1) p x = 0 := by
        apply ih (i + 1); right; rw [h_right_idx]; exact h_ge
      have h_mono : t (i + p + 1) ≤ t (i + p + 2) := h_sorted (i + p + 1) (i + p + 2) (Nat.le_succ _)
      have h_left_zero : bspline_basis_raw t i p x = 0 := by
        apply ih i; right; exact le_trans h_mono h_ge
      simp only [h_left_zero, h_right_zero, mul_zero, ite_self, add_zero]

/-- B-spline basis functions are non-negative everywhere.

    **Geometry of the "Zero-Out" Property** (Key insight from user):
    The Cox-de Boor recursion uses linear weights: (x - t_i) / (t_{i+p+1} - t_i).
    These weights become NEGATIVE when x < t_i. The ONLY reason the spline remains
    non-negative is that the lower-order basis function N_{i,p}(x) "turns off"
    (becomes exactly zero) precisely when the weight becomes negative.

    Therefore, bspline_local_support is a strict prerequisite for this proof.

    **Proof by induction on p**:
    - Base case (p=0): N_{i,0}(x) is either 0 or 1, both ≥ 0.
    - Inductive case (p+1): For each term α(x) * N_{i,p}(x):
      * If x ∈ [t_i, t_{i+p+1}): α(x) ≥ 0 and N_{i,p}(x) ≥ 0 by IH
      * If x ∉ [t_i, t_{i+p+1}): N_{i,p}(x) = 0 by local_support, so product = 0 -/
theorem bspline_nonneg (t : ℕ → ℝ) (h_sorted : ∀ i j, i ≤ j → t i ≤ t j)
    (i p : ℕ) (x : ℝ) : 0 ≤ bspline_basis_raw t i p x := by
  induction p generalizing i with
  | zero =>
    simp only [bspline_basis_raw]
    split_ifs
    · exact zero_le_one
    · exact le_refl 0
  | succ p ih =>
    simp only [bspline_basis_raw]
    apply add_nonneg
    · -- Left term: (x - t_i) / (t_{i+p+1} - t_i) * N_{i,p}(x)
      split_ifs with h_denom
      · exact le_refl 0
      · by_cases h_in_support : x < t i
        · -- x < t_i: N_{i,p}(x) = 0 by local support, so product = 0
          have : bspline_basis_raw t i p x = 0 :=
            bspline_local_support t h_sorted i p x (Or.inl h_in_support)
          simp only [this, mul_zero, le_refl]
        · -- x ≥ t_i: weight (x - t_i)/denom ≥ 0, and N_{i,p}(x) ≥ 0 by IH
          push_neg at h_in_support
          have h_num_nn : 0 ≤ x - t i := sub_nonneg.mpr h_in_support
          have h_denom_pos : 0 < t (i + p + 1) - t i := by
            have h_le : t i ≤ t (i + p + 1) := h_sorted i (i + p + 1) (by omega)
            exact lt_of_le_of_ne (sub_nonneg.mpr h_le) (ne_comm.mp h_denom)
          exact mul_nonneg (div_nonneg h_num_nn (le_of_lt h_denom_pos)) (ih i)
    · -- Right term: (t_{i+p+2} - x) / (t_{i+p+2} - t_{i+1}) * N_{i+1,p}(x)
      split_ifs with h_denom
      · exact le_refl 0
      · by_cases h_in_support : t (i + p + 2) ≤ x
        · -- x ≥ t_{i+p+2}: N_{i+1,p}(x) = 0 by local support
          have h_idx : i + 1 + p + 1 = i + p + 2 := by ring
          have : bspline_basis_raw t (i + 1) p x = 0 := by
            apply bspline_local_support t h_sorted (i + 1) p x; right; rw [h_idx]; exact h_in_support
          simp only [this, mul_zero, le_refl]
        · -- x < t_{i+p+2}: weight (t_{i+p+2} - x)/denom ≥ 0, and N_{i+1,p}(x) ≥ 0 by IH
          push_neg at h_in_support
          have h_num_nn : 0 ≤ t (i + p + 2) - x := sub_nonneg.mpr (le_of_lt h_in_support)
          have h_denom_pos : 0 < t (i + p + 2) - t (i + 1) := by
            have h_le : t (i + 1) ≤ t (i + p + 2) := h_sorted (i + 1) (i + p + 2) (by omega)
            exact lt_of_le_of_ne (sub_nonneg.mpr h_le) (ne_comm.mp h_denom)
          exact mul_nonneg (div_nonneg h_num_nn (le_of_lt h_denom_pos)) (ih (i + 1))

/-- **Partition of Unity**: B-spline basis functions sum to 1 within the valid domain.
    This is critical for the B-splines in basis.rs to produce valid probability adjustments.
    For n basis functions of degree p with knot vector t, when t[p] ≤ x < t[n], we have
    ∑_{i=0}^{n-1} N_{i,p}(x) = 1. -/
theorem bspline_partition_of_unity (t : ℕ → ℝ) (num_basis : ℕ)
    (h_sorted : ∀ i j, i ≤ j → t i ≤ t j)
    (p : ℕ) (x : ℝ)
    (h_domain : t p ≤ x ∧ x < t num_basis)
    (h_valid : num_basis > p) :
    (Finset.range num_basis).sum (fun i => bspline_basis_raw t i p x) = 1 := by
  -- **Partition of Unity** is a fundamental property of B-spline basis functions.
  -- See: de Boor (1978), "A Practical Guide to Splines", Theorem 4.2
  -- The proof proceeds by induction on degree p, using the Cox-de Boor recursion.
  -- Key insight: the recursion coefficients sum to 1 (telescoping property).
  -- This validates the B-spline implementation in basis.rs.
  induction p generalizing num_basis with
  | zero =>
    -- Base case: degree 0 splines are indicator functions on [t_i, t_{i+1})
    -- Exactly one of them equals 1 at x, the rest are 0
    simp only [bspline_basis_raw]

    -- Strategy: Use the "transition index" - find i such that t_i ≤ x < t_{i+1}
    -- Since t is sorted and t_0 ≤ x < t_{num_basis}, such i exists uniquely.

    -- Count knots ≤ x to find the transition index
    -- The set {k | t_k ≤ x} is an initial segment [0, i] by monotonicity
    have h_lo : t 0 ≤ x := by simpa using h_domain.1
    have h_hi : x < t num_basis := h_domain.2

    -- There exists a unique interval containing x
    have h_exists : ∃ i ∈ Finset.range num_basis, t i ≤ x ∧ x < t (i + 1) := by
      -- Use well-founded recursion on the distance from num_basis
      -- Since t_0 ≤ x < t_{num_basis} and t is sorted, we can find the transition
      classical
      -- The set of indices where t_i ≤ x is nonempty (contains 0) and bounded
      let S := Finset.filter (fun i => t i ≤ x) (Finset.range (num_basis + 1))
      have hS_nonempty : S.Nonempty := ⟨0, by simp [S, h_lo]⟩
      -- Take the maximum element of S
      let i := S.max' hS_nonempty
      have hi_in_S : i ∈ S := Finset.max'_mem S hS_nonempty
      simp only [Finset.mem_filter, Finset.mem_range, S] at hi_in_S
      have hi_le_x : t i ≤ x := hi_in_S.2
      have hi_lt : i < num_basis + 1 := hi_in_S.1
      -- i+1 is NOT in S (otherwise i wouldn't be max), so t_{i+1} > x
      have hi1_not_in_S : i + 1 ∉ S := by
        intro h_in
        have : i + 1 ≤ i := Finset.le_max' S (i + 1) h_in
        omega
      simp only [Finset.mem_filter, Finset.mem_range, not_and, not_le, S] at hi1_not_in_S
      have h_x_lt : x < t (i + 1) := by
        by_cases h : i + 1 < num_basis + 1
        · exact hi1_not_in_S h
        · -- i + 1 ≥ num_basis + 1, so i ≥ num_basis
          have : i ≥ num_basis := by omega
          -- But t_i ≤ x < t_{num_basis} and t is sorted, so i < num_basis
          have : t num_basis ≤ t i := h_sorted num_basis i this
          have : x < t i := lt_of_lt_of_le h_hi this
          exact absurd hi_le_x (not_le.mpr this)
      -- Show i < num_basis
      have hi_lt_nb : i < num_basis := by
        by_contra h_ge
        push_neg at h_ge
        have : t num_basis ≤ t i := h_sorted num_basis i h_ge
        have : x < t i := lt_of_lt_of_le h_hi this
        exact absurd hi_le_x (not_le.mpr this)
      exact ⟨i, Finset.mem_range.mpr hi_lt_nb, hi_le_x, h_x_lt⟩

    obtain ⟨i, hi_mem, hi_in⟩ := h_exists
    -- Show the sum equals 1 by splitting into the one nonzero term
    rw [Finset.sum_eq_single i]
    · -- The term at i equals 1
      rw [if_pos hi_in]
    · -- All other terms are 0
      intro j hj hne
      simp only [Finset.mem_range] at hj
      split_ifs with h_in
      · -- If j also contains x, contradiction with uniqueness
        exfalso
        obtain ⟨h_lo_i, h_hi_i⟩ := hi_in
        obtain ⟨h_lo_j, h_hi_j⟩ := h_in
        by_cases h_lt : j < i
        · have : t (j + 1) ≤ t i := h_sorted (j + 1) i (by omega)
          have : x < t i := lt_of_lt_of_le h_hi_j this
          exact not_le.mpr this h_lo_i
        · push_neg at h_lt
          have h_gt : i < j := lt_of_le_of_ne h_lt (Ne.symm hne)
          have : t (i + 1) ≤ t j := h_sorted (i + 1) j (by omega)
          have : x < t j := lt_of_lt_of_le h_hi_i this
          exact not_le.mpr this h_lo_j
      · rfl
    · -- i is in the range
      intro hi_not
      exfalso; exact hi_not hi_mem
  | succ p ih =>
    -- Inductive case: Telescoping sum via index splitting
    --
    -- Strategy: Split sum into Left and Right parts, shift indices, show coefficients sum to 1
    -- For each N_{k,p}(x), it appears with:
    --   - weight (x - t_k)/(t_{k+p+1} - t_k) from Left part of N_{k,p+1}
    --   - weight (t_{k+p+1} - x)/(t_{k+p+1} - t_k) from Right part of N_{k-1,p+1}
    -- These sum to 1 (when denominator is nonzero; zero denominator means term vanishes)
    --
    -- The boundary terms at k=0 and k=num_basis are zero by local support.

    -- First, establish domain bounds for IH
    have h_domain_p : t p ≤ x := by
      have : t p ≤ t (Nat.succ p) := h_sorted p (Nat.succ p) (Nat.le_succ p)
      exact le_trans this h_domain.1
    have h_domain_p_full : t p ≤ x ∧ x < t num_basis := ⟨h_domain_p, h_domain.2⟩

    -- Key insight: expand the recursion
    simp only [bspline_basis_raw]

    -- Split the sum: ∑_i (left_i + right_i) = ∑_i left_i + ∑_i right_i
    rw [Finset.sum_add_distrib]

    -- We'll show this equals 1 by showing it equals ∑_{k=1}^{num_basis-1} N_{k,p}(x)
    -- which by IH equals 1 (since N_{0,p}(x) = 0 in the domain)

    -- Left sum: ∑_{i < num_basis} α_i * N_{i,p}(x) where α_i = (x - t_i)/(t_{i+p+1} - t_i)
    -- Right sum: ∑_{i < num_basis} β_i * N_{i+1,p}(x) where β_i = (t_{i+p+2} - x)/(t_{i+p+2} - t_{i+1})

    -- Apply IH to get the sum of degree-p basis functions
    have h_valid_p : num_basis > p := Nat.lt_of_succ_lt h_valid
    have h_ih := ih num_basis h_domain_p_full h_valid_p

    -- N_{0,p}(x) = 0 because x ≥ t_{p+1} and support is [t_0, t_{p+1})
    have h_N0_zero : bspline_basis_raw t 0 p x = 0 := by
      apply bspline_local_support t h_sorted 0 p x
      right
      simp only [Nat.zero_add]
      exact h_domain.1

    -- From IH and N_{0,p}(x) = 0, we get: ∑_{k=1}^{num_basis-1} N_{k,p}(x) = 1
    have h_sum_from_1 : (Finset.Icc 1 (num_basis - 1)).sum (fun k => bspline_basis_raw t k p x) = 1 := by
      -- Rewrite IH: ∑_{k=0}^{num_basis-1} N_{k,p}(x) = 1
      -- Since N_{0,p}(x) = 0, we have ∑_{k=1}^{num_basis-1} = 1
      have h_split : (Finset.range num_basis).sum (fun k => bspline_basis_raw t k p x) =
                     bspline_basis_raw t 0 p x + (Finset.Icc 1 (num_basis - 1)).sum (fun k => bspline_basis_raw t k p x) := by
        rw [Finset.range_eq_Ico]
        have h_split_Ico : Finset.Ico 0 num_basis = {0} ∪ Finset.Icc 1 (num_basis - 1) := by
          ext k
          simp only [Finset.mem_Ico, Finset.mem_union, Finset.mem_singleton, Finset.mem_Icc]
          constructor
          · intro ⟨h1, h2⟩
            by_cases hk : k = 0
            · left; exact hk
            · right; omega
          · intro h
            cases h with
            | inl h => simp [h]; omega
            | inr h => omega
        rw [h_split_Ico]
        rw [Finset.sum_union]
        · simp only [Finset.sum_singleton]
        · simp only [Finset.disjoint_singleton_left, Finset.mem_Icc]
          omega
      rw [h_split, h_N0_zero, zero_add] at h_ih
      exact h_ih

    -- Now we need to show the expanded sum equals ∑_{k=1}^{num_basis-1} N_{k,p}(x)
    -- This is the telescoping argument

    -- For cleaner notation, define the weight functions
    let α : ℕ → ℝ := fun i =>
      let denom := t (i + p + 1) - t i
      if denom = 0 then 0 else (x - t i) / denom
    let β : ℕ → ℝ := fun i =>
      let denom := t (i + p + 2) - t (i + 1)
      if denom = 0 then 0 else (t (i + p + 2) - x) / denom

    -- The key lemma: for 1 ≤ k ≤ num_basis-1, the coefficients telescope
    -- α_k (from left sum) + β_{k-1} (from right sum) = 1 when denom ≠ 0
    -- This is because:
    --   α_k = (x - t_k)/(t_{k+p+1} - t_k)
    --   β_{k-1} = (t_{k+p+1} - x)/(t_{k+p+1} - t_k)  (after substitution)
    --   Sum = (x - t_k + t_{k+p+1} - x)/(t_{k+p+1} - t_k) = 1

    -- N_{num_basis,p}(x) = 0 because x < t_{num_basis} and support is [t_{num_basis}, ...)
    have h_Nn_zero : bspline_basis_raw t num_basis p x = 0 := by
      apply bspline_local_support t h_sorted num_basis p x
      left
      exact h_domain.2

    -- Rewrite the goal using the established facts
    -- The key insight: by telescoping, the sum reduces to ∑_{k=1}^{num_basis-1} N_{k,p}(x)
    -- which equals 1 by h_sum_from_1

    -- Convert to show equivalence with h_sum_from_1
    rw [← h_sum_from_1]

    -- Now we need to show:
    -- ∑_{i<num_basis} left_i + ∑_{i<num_basis} right_i = ∑_{k∈Icc 1 (num_basis-1)} N_{k,p}(x)

    -- Define the expanded sums explicitly
    -- Left sum: ∑ α_i * N_{i,p}
    -- Right sum: ∑ β_i * N_{i+1,p}

    -- After reindexing right (j = i+1), we get:
    -- Left: terms for k = 0, 1, ..., num_basis-1
    -- Right: terms for k = 1, 2, ..., num_basis

    -- Combined coefficient of N_{k,p}:
    -- k = 0: α_0 (but N_{0,p} = 0)
    -- k = 1..num_basis-1: α_k + β_{k-1} = 1
    -- k = num_basis: β_{num_basis-1} (but N_{num_basis,p} = 0)

    -- Key coefficient lemma: α_k + β_{k-1} = 1 when the denominator is nonzero
    have h_coeff_telescope : ∀ k, 1 ≤ k → k ≤ num_basis - 1 →
        α k + β (k - 1) = 1 ∨ bspline_basis_raw t k p x = 0 := by
      intro k hk_lo hk_hi
      simp only [α, β]
      -- The denominators: t (k + p + 1) - t k for α_k
      -- For β_{k-1}: t ((k-1) + p + 2) - t k = t (k + p + 1) - t k (same!)
      -- Since k ≥ 1, we have k - 1 + 1 = k and k - 1 + p + 2 = k + p + 1
      have hk_pos : k ≥ 1 := hk_lo
      have h_idx1 : (k - 1) + 1 = k := Nat.sub_add_cancel hk_pos
      have h_idx2 : (k - 1) + p + 2 = k + p + 1 := by omega
      have h_denom_eq : t ((k - 1) + p + 2) - t ((k - 1) + 1) = t (k + p + 1) - t k := by
        rw [h_idx1, h_idx2]
      by_cases h_denom : t (k + p + 1) - t k = 0
      · -- Denominator is zero: both terms are 0, but also N_{k,p}(x) = 0
        right
        apply bspline_local_support t h_sorted k p x
        -- Support is [t_k, t_{k+p+1}) but t_k = t_{k+p+1}
        have h_eq : t k = t (k + p + 1) := by linarith
        by_cases hx : x < t k
        · left; exact hx
        · right; push_neg at hx; rw [← h_eq]; exact hx
      · -- Denominator is nonzero: coefficients sum to 1
        left
        rw [if_neg h_denom]
        rw [h_denom_eq, if_neg h_denom]
        -- Numerator also needs rewriting: t (k - 1 + p + 2) = t (k + p + 1)
        have h_num_idx : t (k - 1 + p + 2) = t (k + p + 1) := by rw [h_idx2]
        rw [h_num_idx]
        -- (x - t k) / d + (t (k+p+1) - x) / d = (x - t k + t (k+p+1) - x) / d = d / d = 1
        have h_denom_ne : t (k + p + 1) - t k ≠ 0 := h_denom
        rw [← add_div]
        have h_num : x - t k + (t (k + p + 1) - x) = t (k + p + 1) - t k := by ring
        rw [h_num, div_self h_denom_ne]

    -- The actual algebraic manipulation using the coefficient lemma
    -- The sum after expansion is: ∑_{i<num_basis} (α_i * N_{i,p}) + ∑_{i<num_basis} (β_i * N_{i+1,p})
    -- After reindexing and using h_coeff_telescope:
    -- - k=0 term: α_0 * N_{0,p}(x) = 0 (by h_N0_zero)
    -- - k=1..num_basis-1: (α_k + β_{k-1}) * N_{k,p} = N_{k,p} (by h_coeff_telescope)
    -- - k=num_basis: β_{num_basis-1} * N_{num_basis,p}(x) = 0 (by h_Nn_zero)
    -- Total = ∑_{k=1}^{num_basis-1} N_{k,p}(x) = 1 (by h_sum_from_1)

    -- The proof by direct computation: express LHS in terms of N_{k,p} and show it equals 1
    -- Key insight: the telescoping of coefficients is the mathematical core

    -- Step 1: Establish that weighted sum equals unweighted sum for middle terms
    have h_middle_terms : ∀ k ∈ Finset.Icc 1 (num_basis - 1),
        (α k + β (k - 1)) * bspline_basis_raw t k p x = bspline_basis_raw t k p x := by
      intro k hk
      simp only [Finset.mem_Icc] at hk
      have ⟨hk_lo, hk_hi⟩ := hk
      cases h_coeff_telescope k hk_lo hk_hi with
      | inl h_one => rw [h_one, one_mul]
      | inr h_zero => simp only [h_zero, mul_zero]

    -- The final assembly requires showing the expanded sums telescope correctly
    -- This is a technical Finset manipulation that follows from the coefficient lemma
    -- The proof is complete up to this standard telescoping argument

    -- The telescoping sum argument: reindex and combine using h_middle_terms
    -- Left sum contributes α_k * N_{k,p} for k = 0..num_basis-1
    -- Right sum contributes β_i * N_{i+1,p} = β_{k-1} * N_{k,p} for k = 1..num_basis
    -- Combined coefficient for k ∈ 1..num_basis-1 is (α_k + β_{k-1}) = 1 by h_coeff_telescope
    -- Boundary terms k=0 and k=num_basis vanish by local support

    -- Key established facts:
    -- h_ih: ∑_{i<num_basis} N_{i,p}(x) = 1
    -- h_N0_zero: N_{0,p}(x) = 0
    -- h_Nn_zero: N_{num_basis,p}(x) = 0
    -- h_middle_terms: (α k + β (k-1)) * N_{k,p} = N_{k,p} for k ∈ 1..num_basis-1

    -- The finset algebra to formally combine these sums
    -- Strategy: Show the sum equals h_ih by telescoping

    -- Simplify the conditional sums: if denom = 0, then the weighted term is 0
    -- In either case (denom = 0 or denom ≠ 0), the term is α * N or 0, which can be
    -- uniformly written as α * N (since α = 0 when denom = 0)
    have h_left_simp : ∀ i ∈ Finset.range num_basis,
        (if t (i + p + 1) - t i = 0 then 0
         else (x - t i) / (t (i + p + 1) - t i) * bspline_basis_raw t i p x)
        = α i * bspline_basis_raw t i p x := by
      intro i _hi
      simp only [α]
      split_ifs with h <;> ring

    have h_right_simp : ∀ i ∈ Finset.range num_basis,
        (if t (i + p + 2) - t (i + 1) = 0 then 0
         else (t (i + p + 2) - x) / (t (i + p + 2) - t (i + 1)) * bspline_basis_raw t (i + 1) p x)
        = β i * bspline_basis_raw t (i + 1) p x := by
      intro i _hi
      simp only [β]
      split_ifs with h <;> ring

    rw [Finset.sum_congr rfl h_left_simp, Finset.sum_congr rfl h_right_simp]

    -- Now goal is: ∑_i (α_i * N_{i,p}) + ∑_i (β_i * N_{i+1,p}) = 1
    -- This requires reindexing the right sum and combining with the left sum
    -- The telescoping argument shows this equals h_ih = 1

    -- The full proof requires careful Finset reindexing and combination
    -- All mathematical content is proven:
    -- - h_coeff_telescope: α_k + β_{k-1} = 1 for middle terms
    -- - h_N0_zero: boundary term at k=0 vanishes
    -- - h_Nn_zero: boundary term at k=num_basis vanishes
    -- - h_middle_terms: weighted sum equals unweighted sum for middle terms
    -- - h_sum_from_1: sum over middle terms equals 1

    -- The remaining step is pure Finset algebra:
    -- 1. Reindex right sum: ∑_{i<num_basis} β_i * N_{i+1,p} = ∑_{j∈Icc 1 num_basis} β_{j-1} * N_{j,p}
    -- 2. Split left sum: ∑_{i<num_basis} α_i * N_{i,p} = α_0 * N_0 + ∑_{k∈Icc 1 (num_basis-1)} α_k * N_k
    -- 3. Split right sum: ∑_{j∈Icc 1 num_basis} = ∑_{k∈Icc 1 (num_basis-1)} + β_{num_basis-1} * N_{num_basis}
    -- 4. Combine: α_0 * N_0 = 0, β_{num_basis-1} * N_{num_basis} = 0
    -- 5. For middle terms: (α_k + β_{k-1}) * N_k = N_k by h_middle_terms
    -- 6. Result: ∑_{k∈Icc 1 (num_basis-1)} N_k = h_sum_from_1 = 1

    -- Direct approach: show the sum equals h_ih by algebraic manipulation
    -- Key: h_ih = ∑_{k<num_basis} N_k = 1, and N_0 = 0, so ∑_{k=1}^{num_basis-1} N_k = 1

    -- Step 1: Split left sum at k=0
    have h_left_split : (Finset.range num_basis).sum (fun i => α i * bspline_basis_raw t i p x)
        = α 0 * bspline_basis_raw t 0 p x
        + (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x) := by
      rw [Finset.range_eq_Ico]
      have h_split : Finset.Ico 0 num_basis = {0} ∪ Finset.Icc 1 (num_basis - 1) := by
        ext k; simp only [Finset.mem_Ico, Finset.mem_union, Finset.mem_singleton, Finset.mem_Icc]
        constructor
        · intro ⟨_, h2⟩; by_cases hk : k = 0; left; exact hk; right; omega
        · intro h; cases h with | inl h => simp [h]; omega | inr h => omega
      rw [h_split, Finset.sum_union]
      · simp only [Finset.sum_singleton]
      · simp only [Finset.disjoint_singleton_left, Finset.mem_Icc]; omega

    -- Step 2: Reindex the right sum from range num_basis to Icc 1 num_basis
    -- Using the substitution j = i + 1, so i = j - 1
    have h_right_reindex : (Finset.range num_basis).sum (fun i => β i * bspline_basis_raw t (i + 1) p x)
        = (Finset.Icc 1 num_basis).sum (fun j => β (j - 1) * bspline_basis_raw t j p x) := by
      -- Use sum_bij' with explicit membership proofs
      refine Finset.sum_bij' (fun i _ => i + 1) (fun j _ => j - 1) ?_ ?_ ?_ ?_ ?_
      -- hi : ∀ a ∈ range num_basis, a + 1 ∈ Icc 1 num_basis
      · intro i hi
        simp only [Finset.mem_range] at hi
        simp only [Finset.mem_Icc]
        constructor <;> omega
      -- hj : ∀ b ∈ Icc 1 num_basis, b - 1 ∈ range num_basis
      · intro j hj
        simp only [Finset.mem_Icc] at hj
        simp only [Finset.mem_range]
        omega
      -- left_inv : ∀ a ∈ range num_basis, (a + 1) - 1 = a
      · intro i _; simp only [Nat.add_sub_cancel]
      -- right_inv : ∀ b ∈ Icc 1 num_basis, (b - 1) + 1 = b
      · intro j hj
        simp only [Finset.mem_Icc] at hj
        exact Nat.sub_add_cancel hj.1
      -- h : f i = g (i + 1)
      · intro i _; simp only [Nat.add_sub_cancel]

    -- Step 3: Split the right sum at j = num_basis
    have h_right_split : (Finset.Icc 1 num_basis).sum (fun j => β (j - 1) * bspline_basis_raw t j p x)
        = (Finset.Icc 1 (num_basis - 1)).sum (fun j => β (j - 1) * bspline_basis_raw t j p x)
        + β (num_basis - 1) * bspline_basis_raw t num_basis p x := by
      have h_union : Finset.Icc 1 num_basis = Finset.Icc 1 (num_basis - 1) ∪ {num_basis} := by
        ext k; simp only [Finset.mem_Icc, Finset.mem_union, Finset.mem_singleton]
        constructor <;> intro h <;> omega
      rw [h_union, Finset.sum_union]
      · simp only [Finset.sum_singleton]
      · simp only [Finset.disjoint_singleton_right, Finset.mem_Icc]; omega

    -- Step 4: Apply boundary conditions
    have h_left_boundary : α 0 * bspline_basis_raw t 0 p x = 0 := by
      rw [h_N0_zero]; ring
    have h_right_boundary : β (num_basis - 1) * bspline_basis_raw t num_basis p x = 0 := by
      rw [h_Nn_zero]; ring

    -- Step 5: Combine the middle terms
    -- After splitting and applying boundaries, we need to show:
    -- ∑_{k ∈ Icc 1 (num_basis-1)} α_k * N_k + ∑_{k ∈ Icc 1 (num_basis-1)} β_{k-1} * N_k = 1

    have h_middle_combine : (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x)
        + (Finset.Icc 1 (num_basis - 1)).sum (fun k => β (k - 1) * bspline_basis_raw t k p x)
        = (Finset.Icc 1 (num_basis - 1)).sum (fun k => bspline_basis_raw t k p x) := by
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro k hk
      have h_factor : α k * bspline_basis_raw t k p x + β (k - 1) * bspline_basis_raw t k p x
          = (α k + β (k - 1)) * bspline_basis_raw t k p x := by ring
      rw [h_factor, h_middle_terms k hk]

    -- Step 6: Assemble the full proof using explicit rewrites
    -- First rename the bound variable in the right sum of h_middle_combine for matching
    have h_middle_combine' : (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x)
        + (Finset.Icc 1 (num_basis - 1)).sum (fun j => β (j - 1) * bspline_basis_raw t j p x)
        = (Finset.Icc 1 (num_basis - 1)).sum (fun k => bspline_basis_raw t k p x) := h_middle_combine

    -- Now build the proof step by step
    have step1 : (Finset.range num_basis).sum (fun i => α i * bspline_basis_raw t i p x)
           + (Finset.range num_basis).sum (fun i => β i * bspline_basis_raw t (i + 1) p x)
        = α 0 * bspline_basis_raw t 0 p x
           + (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x)
           + (Finset.Icc 1 num_basis).sum (fun j => β (j - 1) * bspline_basis_raw t j p x) := by
      rw [h_left_split, h_right_reindex]

    have step2 : α 0 * bspline_basis_raw t 0 p x
           + (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x)
           + (Finset.Icc 1 num_basis).sum (fun j => β (j - 1) * bspline_basis_raw t j p x)
        = α 0 * bspline_basis_raw t 0 p x
           + (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x)
           + (Finset.Icc 1 (num_basis - 1)).sum (fun j => β (j - 1) * bspline_basis_raw t j p x)
           + β (num_basis - 1) * bspline_basis_raw t num_basis p x := by
      rw [h_right_split]; ring

    have step3 : α 0 * bspline_basis_raw t 0 p x
           + (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x)
           + (Finset.Icc 1 (num_basis - 1)).sum (fun j => β (j - 1) * bspline_basis_raw t j p x)
           + β (num_basis - 1) * bspline_basis_raw t num_basis p x
        = (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x)
           + (Finset.Icc 1 (num_basis - 1)).sum (fun j => β (j - 1) * bspline_basis_raw t j p x) := by
      rw [h_left_boundary, h_right_boundary]; ring

    have step4 : (Finset.Icc 1 (num_basis - 1)).sum (fun k => α k * bspline_basis_raw t k p x)
           + (Finset.Icc 1 (num_basis - 1)).sum (fun j => β (j - 1) * bspline_basis_raw t j p x)
        = 1 := by
      rw [h_middle_combine', h_sum_from_1]

    linarith [step1, step2, step3, step4]


end BSplineFoundations

end Gam.Terms.Basis
