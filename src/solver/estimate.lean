import Mathlib.Tactic
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.ToLin

namespace Gam.Solver.Estimate

section WoodReparameterization

/-!
### Wood's Stable Reparameterization

The PIRLS solver in estimate.rs uses Wood (2011)'s reparameterization to
avoid numerical instability. This section proves the algebraic equivalence.
-/

variable {n p : ℕ} [Fintype (Fin n)] [Fintype (Fin p)]

/-- Quadratic form: βᵀSβ computed as dot product. -/
noncomputable def quadForm (S : Matrix (Fin p) (Fin p) ℝ) (β : Fin p → ℝ) : ℝ :=
  Finset.univ.sum (fun i => β i * (S.mulVec β) i)

/-- Penalized least squares objective: ‖y - Xβ‖² + βᵀSβ -/
noncomputable def penalized_objective
    (X : Matrix (Fin n) (Fin p) ℝ) (y : Fin n → ℝ)
    (S : Matrix (Fin p) (Fin p) ℝ) (β : Fin p → ℝ) : ℝ :=
  ‖y - X.mulVec β‖^2 + quadForm S β

/-- A matrix Q is orthogonal if QQᵀ = I. Uses explicit transpose. -/
def IsOrthogonal (Q : Matrix (Fin p) (Fin p) ℝ) : Prop :=
  Q * Matrix.transpose Q = 1 ∧ Matrix.transpose Q * Q = 1

/-- Transpose-dot identity: (Au) ⬝ v = u ⬝ (Aᵀv).
    This is the key algebraic identity for bilinear form transformations. -/
lemma sum_mulVec_mul_eq_sum_mul_transpose_mulVec
    (A : Matrix (Fin p) (Fin p) ℝ) (u v : Fin p → ℝ) :
    ∑ i, (A.mulVec u) i * v i = ∑ i, u i * ((Matrix.transpose A).mulVec v) i := by
  -- Unfold mulVec and dotProduct to get explicit sums
  simp only [Matrix.mulVec, dotProduct, Matrix.transpose_apply]
  -- LHS: ∑ i, (∑ j, A i j * u j) * v i
  -- RHS: ∑ i, u i * (∑ j, A j i * v j)
  -- Distribute the outer multiplication into the inner sums
  simp only [Finset.sum_mul, Finset.mul_sum]
  -- LHS: ∑ i, ∑ j, A i j * u j * v i
  -- RHS: ∑ i, ∑ j, u i * A j i * v j
  -- Convert to sums over Fin p × Fin p using sum_product'
  simp only [← Finset.sum_product']
  -- Now both sides are sums over univ ×ˢ univ
  -- Use Finset.sum_equiv with Equiv.prodComm to swap indices
  refine Finset.sum_equiv (Equiv.prodComm (Fin p) (Fin p)) ?_ ?_
  · intro _; simp
  · intro ⟨i, j⟩ _
    simp only [Equiv.prodComm_apply, Prod.swap_prod_mk]
    ring

/-- The penalty transforms as a congruence under reparameterization.

    **Proof**: (Qβ')ᵀ S (Qβ') = β'ᵀ Qᵀ S Q β' = β'ᵀ (QᵀSQ) β'
    This is just associativity of matrix-vector multiplication.

    This is a key step in Wood's (2011) stable reparameterization for GAMs,
    as it shows how the penalty matrix S transforms under an orthogonal change
    of basis Q. By choosing Q to be the eigenvectors of S, the transformed
    penalty matrix QᵀSQ becomes diagonal, simplifying the optimization problem. -/
theorem penalty_congruence
    (S : Matrix (Fin p) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (β' : Fin p → ℝ) (_h_orth : IsOrthogonal Q) :
    quadForm S (Q.mulVec β') = quadForm (Matrix.transpose Q * S * Q) β' := by
  -- quadForm S (Qβ') = Σᵢ (Qβ')ᵢ * (S(Qβ'))ᵢ = (Qβ')ᵀ S (Qβ')
  -- = β'ᵀ Qᵀ S Q β' = β'ᵀ (QᵀSQ) β' = quadForm (QᵀSQ) β'
  unfold quadForm
  -- LHS: Σᵢ (Q.mulVec β') i * (S.mulVec (Q.mulVec β')) i
  -- RHS: Σᵢ β' i * ((QᵀSQ).mulVec β') i

  -- Step 1: Simplify RHS using mulVec_mulVec
  have h_rhs : (Matrix.transpose Q * S * Q).mulVec β' =
               (Matrix.transpose Q).mulVec (S.mulVec (Q.mulVec β')) := by
    simp only [Matrix.mul_assoc, Matrix.mulVec_mulVec]

  rw [h_rhs]
  -- Now need: Σᵢ (Qβ')ᵢ * (S(Qβ'))ᵢ = Σᵢ β'ᵢ * (Qᵀ(S(Qβ')))ᵢ

  -- Step 2: Apply transpose-dot identity
  -- Let w = Q.mulVec β' and u = S.mulVec w
  -- LHS = Σᵢ w i * u i
  -- RHS = Σᵢ β' i * (Qᵀ.mulVec u) i
  -- By sum_mulVec_mul_eq_sum_mul_transpose_mulVec with A = Q:
  --   Σᵢ (Q.mulVec β') i * u i = Σᵢ β' i * (Qᵀ.mulVec u) i
  exact sum_mulVec_mul_eq_sum_mul_transpose_mulVec Q β' (S.mulVec (Q.mulVec β'))

/-- **Reparameterization Equivalence**: Under orthogonal change of variables β = Qβ',
    the penalized objective transforms covariantly.
    This validates `stable_reparameterization` in estimate.rs.

    **Proof Sketch (Isometry)**:
    1. Residual: y - X(Qβ') = y - (XQ)β', so ‖residual‖² depends only on XQ, not Q separately
    2. Penalty: (Qβ')ᵀS(Qβ') = β'ᵀ(QᵀSQ)β' by associativity of matrix multiplication

    This shows minimizing over β = Qβ' is equivalent to minimizing over β' with transformed design/penalty. -/
theorem reparameterization_equivalence
    (X : Matrix (Fin n) (Fin p) ℝ) (y : Fin n → ℝ)
    (S : Matrix (Fin p) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (β' : Fin p → ℝ) (h_orth : IsOrthogonal Q) :
    penalized_objective X y S (Q.mulVec β') =
    penalized_objective (X * Q) y (Matrix.transpose Q * S * Q) β' := by
  unfold penalized_objective
  -- Step 1: Show the residual norms are equal
  -- X(Qβ') = (XQ)β' by Matrix.mulVec_mulVec
  have h_residual : y - X.mulVec (Q.mulVec β') = y - (X * Q).mulVec β' := by
    rw [Matrix.mulVec_mulVec]
  rw [h_residual]

  -- Step 2: Show the penalty terms are equal
  -- quadForm S (Qβ') = quadForm (QᵀSQ) β'
  have h_penalty : quadForm S (Q.mulVec β') = quadForm (Matrix.transpose Q * S * Q) β' := by
    exact penalty_congruence S Q β' h_orth

  rw [h_penalty]

omit [Fintype (Fin n)] in
/-- The fitted values are invariant under reparameterization. -/
theorem fitted_values_invariant
    (X : Matrix (Fin n) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (β : Fin p → ℝ) (_h_orth : IsOrthogonal Q)
    (β' : Fin p → ℝ) (h_relation : β = Q.mulVec β') :
    X.mulVec β = (X * Q).mulVec β' := by
  rw [h_relation]
  rw [Matrix.mulVec_mulVec]

/-- Eigenvalue structure is preserved: if S = QΛQᵀ, then QᵀSQ = Λ.
    This is the key insight that makes the reparameterization numerically stable.

    **Proof**: QᵀSQ = Qᵀ(QΛQᵀ)Q = (QᵀQ)Λ(QᵀQ) = IΛI = Λ by orthogonality of Q. -/
theorem eigendecomposition_diagonalizes
    (S : Matrix (Fin p) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (Λ : Matrix (Fin p) (Fin p) ℝ)
    (h_orth : IsOrthogonal Q)
    (h_decomp : S = Q * Λ * Matrix.transpose Q)
    (_h_diag : ∀ i j : Fin p, i ≠ j → Λ i j = 0) :
    Matrix.transpose Q * S * Q = Λ := by
  rw [h_decomp]
  -- Qᵀ(QΛQᵀ)Q = (QᵀQ)Λ(QᵀQ) = IΛI = Λ
  have h_assoc : Matrix.transpose Q * (Q * Λ * Matrix.transpose Q) * Q
                = Matrix.transpose Q * Q * Λ * (Matrix.transpose Q * Q) := by
    -- Use associativity of matrix multiplication
    simp only [Matrix.mul_assoc]
  rw [h_assoc]
  -- By orthogonality: QᵀQ = I
  rw [h_orth.2]
  simp only [Matrix.one_mul, Matrix.mul_one]

/-- The optimal β under the reparameterized system transforms back correctly. -/
theorem optimal_solution_transforms
    (X : Matrix (Fin n) (Fin p) ℝ) (y : Fin n → ℝ)
    (S : Matrix (Fin p) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (h_orth : IsOrthogonal Q) (β_opt : Fin p → ℝ) (β'_opt : Fin p → ℝ)
    (h_opt : ∀ β, penalized_objective X y S β_opt ≤ penalized_objective X y S β)
    (h_opt'_unique :
      ∀ β',
        penalized_objective (X * Q) y (Matrix.transpose Q * S * Q) β' ≤
            penalized_objective (X * Q) y (Matrix.transpose Q * S * Q) β'_opt ↔
          β' = β'_opt) :
    X.mulVec β_opt = (X * Q).mulVec β'_opt := by
  -- Let `g` be the reparameterized objective function
  let g := penalized_objective (X * Q) y (Matrix.transpose Q * S * Q)
  -- Let `β'_test` be the transformed original optimal solution
  let β'_test := (Matrix.transpose Q).mulVec β_opt
  -- We show that `β'_test` is a minimizer for `g`. `h_opt` shows `β_opt` minimizes the original objective `f`.
  -- By `reparameterization_equivalence`, `f(Qβ') = g(β')`.
  -- So `g(β'_test) = f(Qβ'_test) = f(β_opt)`. For any other `β'`, `g(β') = f(Qβ')`.
  -- Since `f(β_opt) ≤ f(Qβ')`, we have `g(β'_test) ≤ g(β')`.
  have h_test_is_opt : ∀ β', g β'_test ≤ g β' := by
    intro β'
    let f := penalized_objective X y S
    have h_g_eq_f : ∀ b, g b = f (Q.mulVec b) :=
      fun b => (reparameterization_equivalence X y S Q b h_orth).symm
    rw [h_g_eq_f, h_g_eq_f]
    have h_simplify : Q.mulVec β'_test = β_opt := by
      simp only [β'_test, Matrix.mulVec_mulVec, h_orth.1, Matrix.one_mulVec]
    rw [h_simplify]
    exact h_opt (Q.mulVec β')
  -- From `h_test_is_opt`, `g(β'_test) ≤ g(β'_opt)`. By uniqueness `h_opt'_unique`, this implies `β'_test = β'_opt`.
  have h_beta_eq : β'_test = β'_opt := (h_opt'_unique β'_test).mp (h_test_is_opt β'_opt)
  -- The final goal `X.mulVec β_opt = (X * Q).mulVec β'_opt` follows by substituting this equality.
  rw [← h_beta_eq]
  simp only [β'_test, Matrix.mulVec_mulVec, Matrix.mul_assoc, h_orth.1, Matrix.mul_one]

end WoodReparameterization

end Gam.Solver.Estimate
