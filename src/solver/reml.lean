import Mathlib

namespace Gam.Solver.Reml

section GradientDescentVerification

open Matrix

variable {n p k : ℕ} [Fintype (Fin n)] [Fintype (Fin p)] [Fintype (Fin k)]

/-!
### Matrix Calculus: Log-Determinant Derivatives

We define `H(rho) = A + exp(rho) * B` and prove that the derivative of `log(det(H(rho)))`
with respect to `rho` is `exp(rho) * trace(H(rho)⁻¹ * B)`. This uses Jacobi's formula
for the derivative of the determinant.
-/

variable {m : Type*} [Fintype m] [DecidableEq m]

/-- Matrix function H(ρ) = A + exp(ρ) * B. -/
noncomputable def H_matrix (A B : Matrix m m ℝ) (rho : ℝ) : Matrix m m ℝ := A + Real.exp rho • B

/-- The log-determinant function f(ρ) = log(det(H(ρ))). -/
noncomputable def log_det_H (A B : Matrix m m ℝ) (rho : ℝ) := Real.log (H_matrix A B rho).det

/-- The derivative of log(det(H(ρ))) = log(det(A + exp(ρ)B)) with respect to ρ
    is exp(ρ) * trace(H(ρ)⁻¹ * B). This is derived using Jacobi's formula. -/
theorem derivative_log_det_H_matrix (A B : Matrix m m ℝ)
    (_hB : B.IsSymm)
    (rho : ℝ) (h_pos : (H_matrix A B rho).PosDef) :
    deriv (log_det_H A B) rho = Real.exp rho * ((H_matrix A B rho)⁻¹ * B).trace := by
  have h_inv : (H_matrix A B rho).det ≠ 0 := h_pos.det_pos.ne'
  have h_det : deriv (fun rho => Real.log (Matrix.det (A + Real.exp rho • B))) rho = Real.exp rho * Matrix.trace ((A + Real.exp rho • B)⁻¹ * B) := by
    have h_det_step1 : deriv (fun rho => Matrix.det (A + Real.exp rho • B)) rho = Matrix.det (A + Real.exp rho • B) * Matrix.trace ((A + Real.exp rho • B)⁻¹ * B) * Real.exp rho := by
      have h_jacobi : deriv (fun rho => Matrix.det (A + Real.exp rho • B)) rho = Matrix.trace (Matrix.adjugate (A + Real.exp rho • B) * deriv (fun rho => A + Real.exp rho • B) rho) := by
        have h_jacobi : ∀ (M : ℝ → Matrix m m ℝ), DifferentiableAt ℝ M rho → deriv (fun rho => Matrix.det (M rho)) rho = Matrix.trace (Matrix.adjugate (M rho) * deriv M rho) := by
          intro M hM_diff
          have h_jacobi : deriv (fun rho => Matrix.det (M rho)) rho = ∑ i, ∑ j, (Matrix.adjugate (M rho)) i j * deriv (fun rho => (M rho) j i) rho := by
            simp +decide [ Matrix.det_apply', Matrix.adjugate_apply, Matrix.mul_apply ]
            have h_jacobi : deriv (fun rho => ∑ σ : Equiv.Perm m, (↑(↑((Equiv.Perm.sign : Equiv.Perm m → ℤˣ) σ) : ℤ) : ℝ) * ∏ i : m, M rho ((σ : m → m) i) i) rho = ∑ σ : Equiv.Perm m, (↑(↑((Equiv.Perm.sign : Equiv.Perm m → ℤˣ) σ) : ℤ) : ℝ) * ∑ i : m, (∏ j ∈ Finset.univ.erase i, M rho ((σ : m → m) j) j) * deriv (fun rho => M rho ((σ : m → m) i) i) rho := by
              have h_jacobi : ∀ σ : Equiv.Perm m, deriv (fun rho => ∏ i : m, M rho ((σ : m → m) i) i) rho = ∑ i : m, (∏ j ∈ Finset.univ.erase i, M rho ((σ : m → m) j) j) * deriv (fun rho => M rho ((σ : m → m) i) i) rho := by
                intro σ
                have h_prod_rule : ∀ (f : m → ℝ → ℝ), (∀ i, DifferentiableAt ℝ (f i) rho) → deriv (fun rho => ∏ i, f i rho) rho = ∑ i, (∏ j ∈ Finset.univ.erase i, f j rho) * deriv (f i) rho := by
                  intro f hf
                  convert deriv_finset_prod (u := Finset.univ) (f := f) (x := rho) (fun i _ => hf i)
                  simp
                apply h_prod_rule
                intro i
                exact DifferentiableAt.comp rho ( differentiableAt_pi.1 ( differentiableAt_pi.1 hM_diff _ ) _ ) differentiableAt_id
              have h_deriv_sum : deriv (fun rho => ∑ σ : Equiv.Perm m, (↑(↑((Equiv.Perm.sign : Equiv.Perm m → ℤˣ) σ) : ℤ) : ℝ) * ∏ i : m, M rho ((σ : m → m) i) i) rho = ∑ σ : Equiv.Perm m, (↑(↑((Equiv.Perm.sign : Equiv.Perm m → ℤˣ) σ) : ℤ) : ℝ) * deriv (fun rho => ∏ i : m, M rho ((σ : m → m) i) i) rho := by
                have h_diff : ∀ σ : Equiv.Perm m, DifferentiableAt ℝ (fun rho => ∏ i : m, M rho ((σ : m → m) i) i) rho := by
                  intro σ
                  have h_diff : ∀ i : m, DifferentiableAt ℝ (fun rho => M rho ((σ : m → m) i) i) rho := by
                    intro i
                    exact DifferentiableAt.comp rho ( differentiableAt_pi.1 ( differentiableAt_pi.1 hM_diff _ ) _ ) differentiableAt_id
                  convert DifferentiableAt.finset_prod (u := Finset.univ) (f := fun i rho => M rho ((σ : m → m) i) i) (x := rho) (fun i _ => h_diff i)
                  simp
                norm_num [ h_diff ]
              simpa only [ h_jacobi ] using h_deriv_sum
            simp +decide only [h_jacobi, Finset.mul_sum _ _ _]
            simp +decide [ Finset.sum_mul _ _ _, Matrix.updateRow_apply ]
            rw [ Finset.sum_comm ]
            refine' Finset.sum_congr rfl fun i hi => _
            rw [ Finset.sum_comm, Finset.sum_congr rfl ] ; intros ; simp +decide [ Finset.prod_ite, Finset.filter_ne', Finset.filter_eq' ] ; ring
            rw [ Finset.sum_eq_single ( ( ‹Equiv.Perm m› : m → m ) i ) ] <;> simp +decide [ Finset.prod_ite, Finset.filter_ne', Finset.filter_eq' ] ; ring
            intro j hj; simp +decide [ Pi.single_apply, hj ]
            rw [ Finset.prod_eq_zero_iff.mpr ] <;> simp +decide [ hj ]
            exact ⟨ ( ‹Equiv.Perm m›.symm j ), by simp +decide, by simpa [ Equiv.symm_apply_eq ] using hj ⟩
          rw [ h_jacobi, Matrix.trace ]
          rw [ deriv_pi ]
          · simp +decide [ Matrix.mul_apply, Finset.mul_sum _ _ _ ]
            refine' Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => _
            rw [ deriv_pi ]
            intro i; exact (by
            exact DifferentiableAt.comp rho ( differentiableAt_pi.1 ( differentiableAt_pi.1 hM_diff j ) i ) differentiableAt_id)
          · exact fun i => DifferentiableAt.comp rho ( differentiableAt_pi.1 hM_diff i ) differentiableAt_id
        apply h_jacobi
        exact differentiableAt_pi.2 fun i => differentiableAt_pi.2 fun j => DifferentiableAt.add ( differentiableAt_const _ ) ( DifferentiableAt.smul ( Real.differentiableAt_exp ) ( differentiableAt_const _ ) )
      simp_all +decide [ Matrix.inv_def, mul_assoc, mul_left_comm, mul_comm, Matrix.trace_mul_comm ( Matrix.adjugate _ ) ]
      rw [ show deriv ( fun rho => A + Real.exp rho • B ) rho = Real.exp rho • B from ?_ ]
      · by_cases h : Matrix.det ( A + Real.exp rho • B ) = 0 <;> simp_all +decide [ Matrix.trace_smul, mul_assoc, mul_comm, mul_left_comm ]
        exact False.elim <| h_inv h
      · rw [ deriv_pi ] <;> norm_num [ Real.differentiableAt_exp, mul_comm ]
        ext i; rw [ deriv_pi ] <;> norm_num [ Real.differentiableAt_exp, mul_comm ]
    by_cases h_det : DifferentiableAt ℝ ( fun rho => Matrix.det ( A + Real.exp rho • B ) ) rho <;> simp_all +decide [ Real.exp_ne_zero, mul_assoc, mul_comm, mul_left_comm ]
    · convert HasDerivAt.deriv ( HasDerivAt.log ( h_det.hasDerivAt ) h_inv ) using 1 ; ring!
      exact eq_div_of_mul_eq ( by aesop ) ( by linear_combination' h_det_step1.symm )
    · contrapose! h_det
      simp +decide [ Matrix.det_apply' ]
      fun_prop (disch := norm_num)
  exact h_det

-- 1. Model Functions
noncomputable def S_lambda_fn (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (rho : Fin k → ℝ) : Matrix (Fin p) (Fin p) ℝ :=
  ∑ i, (Real.exp (rho i) • S_basis i)

noncomputable def L_pen_fn (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ) (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (rho : Fin k → ℝ) (beta : Matrix (Fin p) (Fin 1) ℝ) : ℝ :=
  - (log_lik beta) + 0.5 * (beta.transpose * (S_lambda_fn S_basis rho) * beta).trace

noncomputable def Hessian_fn (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (rho : Fin k → ℝ) (beta : Matrix (Fin p) (Fin 1) ℝ) : Matrix (Fin p) (Fin p) ℝ :=
  X.transpose * (W beta) * X + S_lambda_fn S_basis rho

/-- Algebraic matrix inverse via Cramer's rule. Over `ℝ` this is definitionally
    equal to `M⁻¹`, but it avoids carrying inverse-specific structure in later
    definitions that are easier to normalize as polynomial expressions. -/
noncomputable def matrixInvAlg {α : Type*} [Fintype α] [DecidableEq α] (M : Matrix α α ℝ) : Matrix α α ℝ :=
  (M.det)⁻¹ • M.adjugate

theorem matrixInvAlg_eq_inv {α : Type*} [Fintype α] [DecidableEq α] (M : Matrix α α ℝ) :
    matrixInvAlg M = M⁻¹ := by
  by_cases h_det : M.det = 0
  · simp [matrixInvAlg, Matrix.inv_def, h_det]
  · simp [matrixInvAlg, Matrix.inv_def, h_det]

theorem inv_mul_self_of_det_ne_zero {α : Type*} [Fintype α] [DecidableEq α]
    (M : Matrix α α ℝ) (h_det : M.det ≠ 0) : M⁻¹ * M = 1 := by
  simp [Matrix.inv_def, Matrix.adjugate_mul, h_det]

noncomputable def LAML_explicit (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ) (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (rho : Fin k → ℝ) (beta : Matrix (Fin p) (Fin 1) ℝ) : ℝ :=
  let H := Hessian_fn S_basis X W rho beta
  L_pen_fn log_lik S_basis rho beta + 0.5 * Real.log (H.det) - 0.5 * Real.log ((S_lambda_fn S_basis rho).det)

noncomputable def LAML_fn (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ) (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ) (rho : Fin k → ℝ) : ℝ :=
  LAML_explicit log_lik S_basis X W rho (beta_hat rho)

noncomputable def LAML_fixed_beta_fn (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ) (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (b : Matrix (Fin p) (Fin 1) ℝ) (rho : Fin k → ℝ) : ℝ :=
  LAML_explicit log_lik S_basis X W rho b

-- 2. Rust Code Components
noncomputable def rust_delta_fn (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ) (rho : Fin k → ℝ) (i : Fin k) : Matrix (Fin p) (Fin 1) ℝ :=
  let b := beta_hat rho
  let H := Hessian_fn S_basis X W rho b
  let H_inv := matrixInvAlg H
  let lambda := Real.exp (rho i)
  let dS := lambda • S_basis i
  (-H_inv) * (dS * b)

noncomputable def rust_correction_fn (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ) (grad_op : (Matrix (Fin p) (Fin 1) ℝ → ℝ) → Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin p) (Fin 1) ℝ) (rho : Fin k → ℝ) (i : Fin k) : ℝ :=
  let b := beta_hat rho
  let delta := rust_delta_fn S_basis X W beta_hat rho i
  let dV_dbeta := (fun b_val => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W rho b_val)))
  ((grad_op dV_dbeta b).transpose * delta).trace

noncomputable def rust_direct_gradient_fn (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ) (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ) (rho : Fin k → ℝ) (i : Fin k) : ℝ :=
  let b := beta_hat rho
  let H := Hessian_fn S_basis X W rho b
  let H_inv := matrixInvAlg H
  let S := S_lambda_fn S_basis rho
  let S_inv := matrixInvAlg S
  let lambda := Real.exp (rho i)
  let Si := S_basis i
  0.5 * lambda * (b.transpose * Si * b).trace +
  0.5 * lambda * (H_inv * Si).trace -
  0.5 * lambda * (S_inv * Si).trace

-- 3. Verification Theorem

/-- Gradient definition for matrix-to-real functions. -/
def HasGradientAt (f : Matrix (Fin p) (Fin 1) ℝ → ℝ) (g : Matrix (Fin p) (Fin 1) ℝ) (x : Matrix (Fin p) (Fin 1) ℝ) :=
  ∃ (L : Matrix (Fin p) (Fin 1) ℝ →L[ℝ] ℝ),
    (∀ h, L h = (g.transpose * h).trace) ∧ HasFDerivAt f L x

noncomputable def laml_u (rho : Fin k → ℝ) (i : Fin k) (r : ℝ) := Function.update rho i r

noncomputable def laml_L1 (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ) (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ) (rho : Fin k → ℝ) (i : Fin k) (r : ℝ) : ℝ :=
  L_pen_fn log_lik S_basis (laml_u rho i r) (beta_hat (laml_u rho i r))

noncomputable def laml_L2 (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ) (rho : Fin k → ℝ) (i : Fin k) (r : ℝ) : ℝ :=
  0.5 * Real.log ((Hessian_fn S_basis X W (laml_u rho i r) (beta_hat (laml_u rho i r))).det)

noncomputable def laml_L3 (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (rho : Fin k → ℝ) (i : Fin k) (r : ℝ) : ℝ :=
  0.5 * Real.log ((S_lambda_fn S_basis (laml_u rho i r)).det)

/-- Rigorous compositional verification of the LAML gradient assembly.
    This packages the sum/subtraction rule argument once the three scalar
    component derivatives are established. -/
theorem laml_gradient_composition_verification
    (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)
    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)
    (X : Matrix (Fin n) (Fin p) ℝ)
    (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ)
    (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ)
    (grad_op : (Matrix (Fin p) (Fin 1) ℝ → ℝ) → Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin p) (Fin 1) ℝ)
    (rho : Fin k → ℝ) (i : Fin k)
    (h_deriv_L1 : deriv (laml_L1 log_lik S_basis beta_hat rho i) (rho i) =
      0.5 * Real.exp (rho i) * trace ((beta_hat rho).transpose * (S_basis i) * (beta_hat rho)))
    (h_deriv_L2 : deriv (laml_L2 S_basis X W beta_hat rho i) (rho i) =
      0.5 * Real.exp (rho i) * trace ((Hessian_fn S_basis X W rho (beta_hat rho))⁻¹ * (S_basis i)) +
      trace ((grad_op (fun b_val => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W rho b_val))) (beta_hat rho)).transpose * rust_delta_fn S_basis X W beta_hat rho i))
    (h_deriv_L3 : deriv (laml_L3 S_basis rho i) (rho i) =
      0.5 * Real.exp (rho i) * trace ((S_lambda_fn S_basis rho)⁻¹ * (S_basis i)))
    (h_diff_L1 : DifferentiableAt ℝ (laml_L1 log_lik S_basis beta_hat rho i) (rho i))
    (h_diff_L2 : DifferentiableAt ℝ (laml_L2 S_basis X W beta_hat rho i) (rho i))
    (h_diff_L3 : DifferentiableAt ℝ (laml_L3 S_basis rho i) (rho i)) :
    deriv (fun r => LAML_fn log_lik S_basis X W beta_hat (laml_u rho i r)) (rho i) =
      rust_direct_gradient_fn S_basis X W beta_hat log_lik rho i +
      rust_correction_fn S_basis X W beta_hat grad_op rho i := by
  let L1 := laml_L1 log_lik S_basis beta_hat rho i
  let L2 := laml_L2 S_basis X W beta_hat rho i
  let L3 := laml_L3 S_basis rho i

  have h_diff_L1' : DifferentiableAt ℝ L1 (rho i) := h_diff_L1
  have h_diff_L2' : DifferentiableAt ℝ L2 (rho i) := h_diff_L2
  have h_diff_L3' : DifferentiableAt ℝ L3 (rho i) := h_diff_L3

  have h_split : ∀ r, LAML_fn log_lik S_basis X W beta_hat (laml_u rho i r) = L1 r + L2 r - L3 r := by
    intro r
    unfold LAML_fn
    rfl

  rw [show (fun r => LAML_fn log_lik S_basis X W beta_hat (laml_u rho i r)) = fun r => L1 r + L2 r - L3 r by
    funext r
    exact h_split r]
  change deriv ((fun r => L1 r + L2 r) - L3) (rho i) = _

  have h_diff_sum : DifferentiableAt ℝ (fun r => L1 r + L2 r) (rho i) := by
    exact DifferentiableAt.add h_diff_L1' h_diff_L2'
  have h_deriv_sum :
      deriv (fun r => L1 r + L2 r) (rho i) = deriv L1 (rho i) + deriv L2 (rho i) := by
    exact deriv_add h_diff_L1' h_diff_L2'

  rw [deriv_sub h_diff_sum h_diff_L3']
  rw [h_deriv_sum]
  rw [h_deriv_L1, h_deriv_L2, h_deriv_L3]
  unfold rust_direct_gradient_fn rust_correction_fn
  simp [matrixInvAlg_eq_inv]
  ring_nf

/-- Fixed-`β` verification: the explicit derivative of the LAML objective with
    respect to `rho_i` matches the Rust direct-gradient assembly. -/
theorem laml_fixed_beta_gradient_is_exact
    (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)
    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)
    (X : Matrix (Fin n) (Fin p) ℝ)
    (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ)
    (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ)
    (rho : Fin k → ℝ) (i : Fin k)
    (b : Matrix (Fin p) (Fin 1) ℝ)
    (h_b : b = beta_hat rho)
    (h_diff_pen : DifferentiableAt ℝ (fun r => L_pen_fn log_lik S_basis (Function.update rho i r) b) (rho i))
    (h_diff_log_H : DifferentiableAt ℝ (fun r => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b))) (rho i))
    (h_diff_log_S : DifferentiableAt ℝ (fun r => -0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r)))) (rho i))
    (h_deriv_pen : deriv (fun r => L_pen_fn log_lik S_basis (Function.update rho i r) b) (rho i) =
      0.5 * Real.exp (rho i) * trace (b.transpose * S_basis i * b))
    (h_deriv_log_H : deriv (fun r => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b))) (rho i) =
      0.5 * Real.exp (rho i) * trace ((Hessian_fn S_basis X W rho b)⁻¹ * S_basis i))
    (h_deriv_log_S : deriv (fun r => -0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r)))) (rho i) =
      -0.5 * Real.exp (rho i) * trace ((S_lambda_fn S_basis rho)⁻¹ * S_basis i)) :
  deriv (fun r => LAML_fixed_beta_fn log_lik S_basis X W b (Function.update rho i r)) (rho i) =
  rust_direct_gradient_fn S_basis X W beta_hat log_lik rho i := by
  change deriv (fun r =>
      L_pen_fn log_lik S_basis (Function.update rho i r) b +
      0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b)) -
      0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r)))) (rho i) = _
  dsimp only [rust_direct_gradient_fn]
  have h_add1 : deriv (fun r => L_pen_fn log_lik S_basis (Function.update rho i r) b +
      (0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b)) +
      -0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r))))) (rho i) =
    deriv (fun r => L_pen_fn log_lik S_basis (Function.update rho i r) b) (rho i) +
    deriv (fun r => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b)) +
      -0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r)))) (rho i) := by
    apply deriv_add h_diff_pen
    exact DifferentiableAt.add h_diff_log_H h_diff_log_S
  have h_add2 : deriv (fun r => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b)) +
      -0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r)))) (rho i) =
    deriv (fun r => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b))) (rho i) +
    deriv (fun r => -0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r)))) (rho i) := by
    exact deriv_add h_diff_log_H h_diff_log_S
  have h_sub_to_add : (fun r => L_pen_fn log_lik S_basis (Function.update rho i r) b +
      0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b)) -
      0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r)))) =
    (fun r => L_pen_fn log_lik S_basis (Function.update rho i r) b +
      (0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W (Function.update rho i r) b)) +
      -0.5 * Real.log (Matrix.det (S_lambda_fn S_basis (Function.update rho i r))))) := by
    ext r
    ring
  rw [h_sub_to_add, h_add1, h_add2, h_deriv_pen, h_deriv_log_H, h_deriv_log_S]
  simp [matrixInvAlg_eq_inv]
  rw [← h_b]
  ring_nf

/-- Structural verification: `rust_delta_fn` implements the correct implicit derivative formula.

    If `grad(L_pen) = 0`, then differentiation gives `H * dbeta + dS * beta = 0`,
    so `dbeta = -H^-1 * dS * beta`.
    This theorem verifies that `rust_delta_fn` computes exactly this quantity. -/
theorem rust_delta_correctness
    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)
    (X : Matrix (Fin n) (Fin p) ℝ)
    (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ)
    (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ)
    (rho : Fin k → ℝ) (i : Fin k) :
    rust_delta_fn S_basis X W beta_hat rho i =
    -(Hessian_fn S_basis X W rho (beta_hat rho))⁻¹ *
    ((Real.exp (rho i) • S_basis i) * beta_hat rho) := by
  unfold rust_delta_fn
  simp [matrixInvAlg_eq_inv, neg_mul, Matrix.smul_mul]

/-- Structural verification: `laml_gradient_validity`

    This theorem proves that the total derivative of `LAML_fn` is correctly assembled
    from its partial derivatives and the implicit derivative of `beta`.

    It relies on structural hypotheses:
    1. Chain rule: d(LAML)/d(rho_i) = ∂(LAML)/∂(rho_i) + <∇_beta(LAML), d(beta)/d(rho_i)>
    2. Partial rho: ∂(LAML)/∂(rho_i) matches `rust_direct_gradient_fn`
    3. Partial beta: ∇_beta(LAML) matches the gradient term in `rust_correction_fn`
    4. Implicit beta: the differentiated optimality condition gives the linear system
       `H * d(beta)/d(rho_i) = -dS * beta`, which is then solved to recover `rust_delta_fn`

    This replaces the previous vacuous verification with a rigorous assembly proof. -/
theorem laml_gradient_validity
    (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)
    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)
    (X : Matrix (Fin n) (Fin p) ℝ)
    (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ)
    (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ)
    (grad_op : (Matrix (Fin p) (Fin 1) ℝ → ℝ) → Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin p) (Fin 1) ℝ)
    (rho : Fin k → ℝ) (i : Fin k)
    -- 1. Hessian solvability at the evaluation point
    (h_hess_pos : (Hessian_fn S_basis X W rho (beta_hat rho)).PosDef)
    -- 2. Implicit differentiation of the optimality condition, stated without inversion
    (h_implicit : Hessian_fn S_basis X W rho (beta_hat rho) *
                  deriv (fun r => beta_hat (Function.update rho i r)) (rho i) =
                  - (Real.exp (rho i) • S_basis i) * (beta_hat rho))
    -- 2. Partial derivative wrt rho matches rust_direct_gradient_fn
    (h_partial_rho : deriv (fun r => LAML_fn log_lik S_basis X W (fun _ => beta_hat rho) (Function.update rho i r)) (rho i) =
                     rust_direct_gradient_fn S_basis X W beta_hat log_lik rho i)
    -- 3. Gradient wrt beta matches the term used in rust_correction_fn
    --    Note: rust_correction_fn uses `grad_op dV_dbeta`.
    --    Optimality of beta implies grad(L_pen) = 0, so grad(LAML) = grad(0.5 log det H).
    (h_grad_beta : HasGradientAt (fun b => LAML_fn log_lik S_basis X W (fun _ => b) rho)
                                 (grad_op (fun b_val => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W rho b_val))) (beta_hat rho))
                                 (beta_hat rho))
    -- 4. Chain rule holds for the total derivative
    (h_chain : deriv (fun r => LAML_fn log_lik S_basis X W beta_hat (Function.update rho i r)) (rho i) =
               deriv (fun r => LAML_fn log_lik S_basis X W (fun _ => beta_hat rho) (Function.update rho i r)) (rho i) +
               ( (grad_op (fun b_val => 0.5 * Real.log (Matrix.det (Hessian_fn S_basis X W rho b_val))) (beta_hat rho)).transpose *
                 deriv (fun r => beta_hat (Function.update rho i r)) (rho i) ).trace) :
  deriv (fun r => LAML_fn log_lik S_basis X W beta_hat (Function.update rho i r)) (rho i) =
  rust_direct_gradient_fn S_basis X W beta_hat log_lik rho i +
  rust_correction_fn S_basis X W beta_hat grad_op rho i :=
by
  have h_hess_det : (Hessian_fn S_basis X W rho (beta_hat rho)).det ≠ 0 := h_hess_pos.det_pos.ne'
  have h_deriv_beta : deriv (fun r => beta_hat (Function.update rho i r)) (rho i) =
      rust_delta_fn S_basis X W beta_hat rho i := by
    let H := Hessian_fn S_basis X W rho (beta_hat rho)
    let dbeta := deriv (fun r => beta_hat (Function.update rho i r)) (rho i)
    have h_solved :
        dbeta = -H⁻¹ * ((Real.exp (rho i) • S_basis i) * (beta_hat rho)) := by
      have h_mul := congrArg (fun M => H⁻¹ * M) h_implicit
      have h_left : H⁻¹ * (H * dbeta) = dbeta := by
        rw [← Matrix.mul_assoc, inv_mul_self_of_det_ne_zero H h_hess_det, Matrix.one_mul]
      calc
        dbeta = H⁻¹ * (H * dbeta) := by
          symm
          exact h_left
        _ = H⁻¹ * (- (Real.exp (rho i) • S_basis i) * beta_hat rho) := by
          simpa [H] using h_mul
        _ = -H⁻¹ * ((Real.exp (rho i) • S_basis i) * beta_hat rho) := by
          simp [Matrix.mul_assoc, neg_mul]
    calc
      deriv (fun r => beta_hat (Function.update rho i r)) (rho i)
          = -H⁻¹ * ((Real.exp (rho i) • S_basis i) * beta_hat rho) := h_solved
      _ = rust_delta_fn S_basis X W beta_hat rho i := by
        symm
        simpa [H] using rust_delta_correctness S_basis X W beta_hat rho i
  rw [h_chain, h_partial_rho, h_deriv_beta]
  unfold rust_correction_fn
  rfl

end GradientDescentVerification

end Gam.Solver.Reml
