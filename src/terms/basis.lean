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

end BSplineFoundations

end Gam.Terms.Basis
