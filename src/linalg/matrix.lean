import Mathlib

namespace Gam.Linalg.Matrix

section WeightedOrthogonality

/-!
### Weighted Orthogonality Constraints

The calibration code applies sum-to-zero and polynomial orthogonality constraints
via nullspace projection. These theorems formalize that the projection is correct.
-/

set_option linter.unusedSectionVars false

variable {n m k : ℕ} [Fintype (Fin n)] [Fintype (Fin m)] [Fintype (Fin k)]
variable [DecidableEq (Fin n)] [DecidableEq (Fin m)] [DecidableEq (Fin k)]

/-- A diagonal weight matrix constructed from a weight vector. -/
def diagonalWeight (w : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal w

/-- Two column spaces are weighted-orthogonal if their weighted inner product is zero.
    Uses explicit transpose to avoid parsing issues. -/
def IsWeightedOrthogonal (A : Matrix (Fin n) (Fin m) ℝ)
    (B : Matrix (Fin n) (Fin k) ℝ) (W : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  Matrix.transpose A * W * B = 0

/-- A matrix Z spans the nullspace of M if MZ = 0 and Z has maximal rank. -/
def SpansNullspace (Z : Matrix (Fin m) (Fin (m - k)) ℝ)
    (M : Matrix (Fin k) (Fin m) ℝ) : Prop :=
  M * Z = 0 ∧ Matrix.rank Z = m - k

/-- **Constraint Projection Correctness**: If Z spans the nullspace of BᵀWC,
    then B' = BZ is weighted-orthogonal to C.
    This validates `apply_weighted_orthogonality_constraint` in basis.rs.

    **Proof**:
    (BZ)ᵀ W C = Zᵀ (Bᵀ W C) = 0 because Z is in the nullspace of (Bᵀ W C)ᵀ.

    More precisely:
    - SpansNullspace Z M means M * Z = 0
    - Here M = (Bᵀ W C)ᵀ = Cᵀ Wᵀ B = Cᵀ W B (if W is symmetric, which diagonal matrices are)
    - We want: (BZ)ᵀ W C = Zᵀ Bᵀ W C
    - By associativity: Zᵀ Bᵀ W C = (Bᵀ W C)ᵀ · Z = M · Z = 0 (by h_spans.1)

    Wait, transpose swap: (Zᵀ (Bᵀ W C))ᵀ = (Bᵀ W C)ᵀ Z
    Actually: Zᵀ · (Bᵀ W C) has shape (m-k) × k, while M · Z = 0 where M = (Bᵀ W C)ᵀ

    The key relation is: Zᵀ · A = (Aᵀ · Z)ᵀ, so if Aᵀ · Z = 0, then Zᵀ · A = 0. -/
theorem constraint_projection_correctness
    (B : Matrix (Fin n) (Fin m) ℝ)
    (C : Matrix (Fin n) (Fin k) ℝ)
    (W : Matrix (Fin n) (Fin n) ℝ)
    (Z : Matrix (Fin m) (Fin (m - k)) ℝ)
    (h_spans : SpansNullspace Z (Matrix.transpose (Matrix.transpose B * W * C))) :
    IsWeightedOrthogonal (B * Z) C W := by
  unfold IsWeightedOrthogonal
  -- Goal: Matrix.transpose (B * Z) * W * C = 0
  -- Expand: (BZ)ᵀ W C = Zᵀ Bᵀ W C
  have h1 : Matrix.transpose (B * Z) = Matrix.transpose Z * Matrix.transpose B := by
    exact Matrix.transpose_mul B Z
  rw [h1]
  -- Now: Zᵀ Bᵀ W C
  -- We need to show: Zᵀ * (Bᵀ W C) = 0
  -- From h_spans: (Bᵀ W C)ᵀ * Z = 0
  -- Taking transpose: Zᵀ * (Bᵀ W C) = ((Bᵀ W C)ᵀ * Z)ᵀ
  -- If (Bᵀ W C)ᵀ * Z = 0, then Zᵀ * (Bᵀ W C) = 0ᵀ = 0
  have h2 : Matrix.transpose Z * Matrix.transpose B * W * C =
            Matrix.transpose Z * (Matrix.transpose B * W * C) := by
    simp only [Matrix.mul_assoc]
  rw [h2]
  -- Now use the nullspace condition
  have h3 : Matrix.transpose (Matrix.transpose B * W * C) * Z = 0 := h_spans.1
  -- Taking transpose of both sides: Zᵀ * (Bᵀ W C) = 0
  have h4 : Matrix.transpose Z * (Matrix.transpose B * W * C) =
            Matrix.transpose (Matrix.transpose (Matrix.transpose B * W * C) * Z) := by
    rw [Matrix.transpose_mul]
    simp only [Matrix.transpose_transpose]
  rw [h4, h3]
  simp only [Matrix.transpose_zero]

/-- The constrained basis preserves the column space spanned by valid coefficients. -/
theorem constrained_basis_spans_subspace
    (B : Matrix (Fin n) (Fin m) ℝ)
    (Z : Matrix (Fin m) (Fin (m - k)) ℝ)
    (β : Fin (m - k) → ℝ) :
    ∃ (β' : Fin m → ℝ), (B * Z).mulVec β = B.mulVec β' := by
  use Z.mulVec β
  rw [Matrix.mulVec_mulVec]

/-- Sum-to-zero constraint: the constraint matrix C is a column of ones. -/
def sumToZeroConstraint (n : ℕ) : Matrix (Fin n) (Fin 1) ℝ :=
  fun _ _ => 1

/-- After applying sum-to-zero constraint, basis evaluations sum to zero at data points.
    Note: This theorem uses a specialized constraint for k=1. -/
theorem sum_to_zero_after_projection
    (B : Matrix (Fin n) (Fin m) ℝ)
    (W : Matrix (Fin n) (Fin n) ℝ) (hW_diag : W = Matrix.diagonal (fun i => W i i))
    (Z : Matrix (Fin m) (Fin (m - 1)) ℝ)
    (h_constraint : SpansNullspace Z (Matrix.transpose (Matrix.transpose B * W * sumToZeroConstraint n)))
    (β : Fin (m - 1) → ℝ) :
    Finset.univ.sum (fun i : Fin n => ((B * Z).mulVec β) i * W i i) = 0 := by
  -- Use constraint_projection_correctness to get weighted orthogonality
  have h_orth : IsWeightedOrthogonal (B * Z) (sumToZeroConstraint n) W :=
    constraint_projection_correctness B (sumToZeroConstraint n) W Z h_constraint
  -- IsWeightedOrthogonal (B * Z) C W means: (BZ)ᵀ * W * C = 0
  -- For C = sumToZeroConstraint n (all ones), the (i,0) entry of (BZ)ᵀ * W * C is:
  --   Σⱼ ((BZ)ᵀ * W)_{i,j} * C_{j,0} = Σⱼ ((BZ)ᵀ * W)_{i,j} * 1 = Σⱼ ((BZ)ᵀ * W)_{i,j}
  -- When we sum over the "first column" being all zeros, we get the constraint.
  -- More directly: the (0,0) entry of Cᵀ * (BZ)ᵀ * W * C = 0
  -- which expands to: Σᵢ Σⱼ C_{i,0} * ((BZ)ᵀ * W)_{j,i} * C_{j,0}
  --                 = Σᵢ Σⱼ 1 * ((BZ)ᵀ * W)_{j,i} * 1
  -- For a diagonal W, ((BZ)ᵀ * W)_{j,i} = (BZ)_{i,j} * W_{i,i}
  --
  -- Actually the goal is: Σᵢ (BZ · β)ᵢ * Wᵢᵢ = 0
  -- This is related to the weighted orthogonality by:
  --   (sumToZeroConstraint n)ᵀ * diag(W) * (BZ · β)
  -- where we interpret W as having diagonal form.
  --
  -- The proof uses that (BZ)ᵀ * W * C = 0 implies the weighted inner product
  -- of any column of BZ with the ones vector is zero.
  unfold IsWeightedOrthogonal at h_orth
  -- h_orth : Matrix.transpose (B * Z) * W * sumToZeroConstraint n = 0
  -- For any column j of (BZ), we have: Σᵢ (BZ)ᵢⱼ * (W * 1)ᵢ = 0
  -- where 1 is the all-ones vector.
  -- The goal is: Σᵢ (Σⱼ (BZ)ᵢⱼ * βⱼ) * Wᵢᵢ = 0
  --            = Σⱼ βⱼ * (Σᵢ (BZ)ᵢⱼ * Wᵢᵢ)
  -- Each inner sum Σᵢ (BZ)ᵢⱼ * Wᵢᵢ corresponds to a column of (BZ)ᵀ * W * 1
  -- Since (BZ)ᵀ * W * C = 0 where C is all ones, each entry is 0.
  -- Therefore the entire sum is 0.

  -- Step 1: Expand mulVec and rewrite the goal as a double sum
  simp only [Matrix.mulVec, dotProduct]
  -- Goal: Σᵢ (Σⱼ (B*Z)ᵢⱼ * βⱼ) * Wᵢᵢ = 0

  -- Step 2: Use diagonal form of W to simplify
  rw [hW_diag]
  simp

  -- Step 3: Swap the order of summation
  -- Σᵢ (Σⱼ aᵢⱼ * βⱼ) * wᵢ = Σⱼ βⱼ * (Σᵢ aᵢⱼ * wᵢ)
  classical
  have h_swap :
      ∑ x, (∑ x_1, (B * Z) x x_1 * β x_1) * W x x
        = ∑ x, ∑ x_1, (B * Z) x x_1 * β x_1 * W x x := by
    refine Finset.sum_congr rfl ?_
    intro x _
    calc
      (∑ x_1, (B * Z) x x_1 * β x_1) * W x x
          = W x x * ∑ x_1, (B * Z) x x_1 * β x_1 := by ring
      _ = ∑ x_1, W x x * ((B * Z) x x_1 * β x_1) := by
          simpa [Finset.mul_sum]
      _ = ∑ x_1, (B * Z) x x_1 * β x_1 * W x x := by
          refine Finset.sum_congr rfl ?_
          intro x_1 _
          ring
  rw [h_swap]
  rw [Finset.sum_comm]

  -- After swap: Σⱼ Σᵢ (B*Z)ᵢⱼ * βⱼ * Wᵢᵢ = Σⱼ βⱼ * (Σᵢ (B*Z)ᵢⱼ * Wᵢᵢ)
  have h_factor :
      ∀ y, ∑ x, (B * Z) x y * β y * W x x = β y * ∑ x, (B * Z) x y * W x x := by
    intro y
    calc
      ∑ x, (B * Z) x y * β y * W x x
          = ∑ x, β y * ((B * Z) x y * W x x) := by
              refine Finset.sum_congr rfl ?_
              intro x _
              ring
      _ = β y * ∑ x, (B * Z) x y * W x x := by
              simpa [Finset.mul_sum]
  simp [h_factor]
  -- Now: Σⱼ βⱼ * (Σᵢ (B*Z)ᵢⱼ * Wᵢᵢ)

  -- Step 4: Show each inner sum Σᵢ (B*Z)ᵢⱼ * Wᵢᵢ = 0 using h_orth
  -- The (j, 0) entry of (BZ)ᵀ * W * C is: Σᵢ (BZ)ᵀⱼᵢ * (W * C)ᵢ₀
  --                                      = Σᵢ (BZ)ᵢⱼ * (Σₖ Wᵢₖ * Cₖ₀)
  -- For diagonal W and C = all ones:    = Σᵢ (BZ)ᵢⱼ * Wᵢᵢ * 1
  --                                      = Σᵢ (BZ)ᵢⱼ * Wᵢᵢ
  -- Since h_orth says the whole matrix is 0, entry (j, 0) = 0.

  apply Finset.sum_eq_zero
  intro j _
  -- Show βⱼ * (Σᵢ (B*Z)ᵢⱼ * Wᵢᵢ) = 0
  -- Suffices to show Σᵢ (B*Z)ᵢⱼ * Wᵢᵢ = 0
  suffices h_inner : Finset.univ.sum (fun i => (B * Z) i j * W i i) = 0 by
    simp [h_inner]

  -- Extract from h_orth: the (j, 0) entry of (BZ)ᵀ * W * C = 0
  have h_entry : (Matrix.transpose (B * Z) * W * sumToZeroConstraint n) j 0 = 0 := by
    rw [h_orth]
    rfl

  -- Expand this entry
  simp only [Matrix.mul_apply, Matrix.transpose_apply, sumToZeroConstraint] at h_entry
  -- (BZ)ᵀ * W * C at (j, 0) = Σₖ ((BZ)ᵀ * W)ⱼₖ * Cₖ₀ = Σₖ ((BZ)ᵀ * W)ⱼₖ * 1
  -- = Σₖ (Σᵢ (BZ)ᵀⱼᵢ * Wᵢₖ) = Σₖ (Σᵢ (BZ)ᵢⱼ * Wᵢₖ)

  -- For diagonal W, Wᵢₖ = 0 unless i = k, so:
  -- = Σᵢ (BZ)ᵢⱼ * Wᵢᵢ (the i=k diagonal terms)

  -- The entry expansion gives us what we need
  convert h_entry using 1
  -- Need to show the sum forms are equal

  -- Expand both sides more carefully
  simp only [Matrix.mul_apply]
  -- LHS: Σᵢ (B*Z)ᵢⱼ * Wᵢᵢ
  -- RHS: Σₖ (Σᵢ (B*Z)ᵢⱼ * Wᵢₖ) * 1

  -- Use diagonal structure: Wᵢₖ = W i i if i = k, else 0
  rw [hW_diag]
  simp [Matrix.diagonal_apply]

  -- Inner sum: Σᵢ (B*Z)ᵢⱼ * (if i = k then W i i else 0)
  -- = (B*Z)ₖⱼ * W k k (only i=k term survives)

end WeightedOrthogonality

end Gam.Linalg.Matrix
