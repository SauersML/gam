use super::*;

// FD-OK: this entire module is the outer-gradient finite-difference AUDIT
// oracle. It deliberately computes central finite differences of the outer
// REML/LAML criterion to verify the analytic gradient against — it is a
// diagnostic that runs only behind the `outer_fd_audit_eligible` gates in
// spatial_optimization.rs / custom_family/fit.rs, never on the fit math path.
// The `fd`-named struct fields and fn names below are intrinsic to that audit
// role, so the no-production-finite-differences guard treats this file as a
// sanctioned audit region closed at the bottom of the module.

/// Per-θ component of an outer-gradient finite-difference audit.
#[derive(Clone, Debug)]
pub struct OuterGradientFdComponent {
    /// Human label for the block this coordinate belongs to (e.g. "timewiggle").
    pub block: String,
    /// Flat θ index.
    pub index: usize,
    /// Analytic ∂V/∂θ_i returned by the family evaluator.
    pub analytic: f64,
    /// Second-order central finite-difference of the outer criterion in θ_i at
    /// the coarse step `h`. Its leading error is `O(h²)·V'''`, which is large
    /// on a steeply-curved coordinate (e.g. the Matérn log-κ axis, whose
    /// operator penalty scales like κ^{2m}, m = ν + d/2).
    pub fd: f64,
    /// Richardson 4th-order refinement of `fd`, combining the `h` and `h/2`
    /// central differences to cancel the leading `O(h²)` truncation term
    /// (residual `O(h⁴)`). Populated ONLY when the coarse `fd` gap looks like it
    /// might trip the DESYNC band, so clean coordinates stay cheap (2 evals);
    /// suspicious ones pay 2 extra evals to tell genuine derivative error
    /// (`h`-independent gap) apart from pure FD truncation (`h²`-shrinking gap).
    pub fd_refined: Option<f64>,
}

impl OuterGradientFdComponent {
    /// The FD estimate that best matches the analytic gradient — the
    /// Richardson-refined value when it is closer, else the coarse `fd`.
    ///
    /// Picking the closer of the two is deliberately CONSERVATIVE for the
    /// desync verdict: a true derivative bug makes the analytic gradient differ
    /// from the *true* derivative by an `h`-independent amount, so BOTH the
    /// coarse and the refined FD (each converging to that true derivative)
    /// stay far from it — the min gap remains large and DESYNC still fires.
    /// Only a truncation-dominated gap (where refinement recovers the true
    /// derivative) collapses, which is exactly the case that must NOT be
    /// flagged. So `abs_gap` can only shrink vs the coarse-only value: no
    /// previously-passing audit can regress.
    pub fn best_fd(&self) -> f64 {
        match self.fd_refined {
            Some(r) if (self.analytic - r).abs() < (self.analytic - self.fd).abs() => r,
            _ => self.fd,
        }
    }
    /// Absolute analytic−FD gap against the best available FD estimate.
    pub fn abs_gap(&self) -> f64 {
        (self.analytic - self.best_fd()).abs()
    }
    /// analytic/fd ratio (None when fd≈0). A clean −1 signals a sign
    /// convention; a stable constant ≠1 signals a dropped/extra additive term.
    pub fn ratio(&self) -> Option<f64> {
        let fd = self.best_fd();
        if fd.abs() > 1e-12 {
            Some(self.analytic / fd)
        } else {
            None
        }
    }
}

/// Result of a component-by-component finite-difference audit of an outer
/// REML/LAML gradient at a fixed θ, plus the outer-Hessian eigenvalues.
///
/// This is the discriminating diagnostic that forks the two failure modes of a
/// non-terminating outer loop: an **objective↔gradient desync** (analytic ≠ FD
/// on some component → the trust region chases a phantom descent direction
/// forever) versus **weak identifiability** (analytic ≈ FD everywhere but a
/// near-zero outer-Hessian eigenvalue → a genuinely flat valley the optimizer
/// crawls along). It is family-agnostic: any path that exposes an outer
/// evaluator closure `θ ↦ (V, ∇V, H)` can call it.
#[derive(Clone, Debug)]
pub struct OuterGradientFdAudit {
    /// Outer criterion value at θ₀.
    pub value: f64,
    /// Per-coordinate analytic-vs-FD comparison.
    pub components: Vec<OuterGradientFdComponent>,
    /// Eigenvalues of the (symmetrized) outer Hessian at θ₀, ascending. Empty
    /// when no analytic/operator Hessian was available.
    pub hessian_eigenvalues: Vec<f64>,
}

impl OuterGradientFdAudit {
    /// Per-block L2 norm of the analytic gradient.
    pub fn analytic_block_norms(&self) -> Vec<(String, f64)> {
        let mut order: Vec<String> = Vec::new();
        let mut acc: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for c in &self.components {
            if !acc.contains_key(&c.block) {
                order.push(c.block.clone());
            }
            *acc.entry(c.block.clone()).or_insert(0.0) += c.analytic * c.analytic;
        }
        order
            .into_iter()
            .map(|b| {
                let v = acc.get(&b).copied().unwrap_or(0.0).sqrt();
                (b, v)
            })
            .collect()
    }

    /// Worst per-coordinate analytic−FD gap and its component.
    pub fn worst_component(&self) -> Option<&OuterGradientFdComponent> {
        self.components.iter().max_by(|a, b| {
            a.abs_gap()
                .partial_cmp(&b.abs_gap())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Smallest-magnitude outer-Hessian eigenvalue (flatness proxy).
    pub fn min_abs_eigenvalue(&self) -> Option<f64> {
        self.hessian_eigenvalues
            .iter()
            .map(|e| e.abs())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Emit a single human-readable verdict block to the log.
    pub fn log_verdict(&self, context: &str) {
        log::warn!("[OUTER-FD-AUDIT/{context}] value={:.6e}", self.value);
        for (block, norm) in self.analytic_block_norms() {
            log::warn!("[OUTER-FD-AUDIT/{context}] block={block} |g_analytic|={norm:.6e}");
        }
        for c in &self.components {
            let ratio = c
                .ratio()
                .map(|r| format!("{r:.4}"))
                .unwrap_or_else(|| "n/a".to_string());
            // Report the FD estimate the verdict actually uses (`best_fd`); when
            // a Richardson refinement was taken, surface both so the log shows
            // the O(h²)→O(h⁴) truncation collapse explicitly.
            let refined = match c.fd_refined {
                Some(r) => format!(" fd_coarse={:.6e} fd_richardson={:.6e}", c.fd, r),
                None => String::new(),
            };
            log::warn!(
                "[OUTER-FD-AUDIT/{context}] block={} i={} analytic={:.6e} fd={:.6e} gap={:.3e} ratio={}{}",
                c.block,
                c.index,
                c.analytic,
                c.best_fd(),
                c.abs_gap(),
                ratio,
                refined,
            );
        }
        if !self.hessian_eigenvalues.is_empty() {
            let evs: Vec<String> = self
                .hessian_eigenvalues
                .iter()
                .map(|e| format!("{e:.4e}"))
                .collect();
            log::warn!(
                "[OUTER-FD-AUDIT/{context}] hessian_eigenvalues=[{}] min_abs={:.4e}",
                evs.join(", "),
                self.min_abs_eigenvalue().unwrap_or(f64::NAN)
            );
        }
        match self.worst_component() {
            Some(w) if w.abs_gap() > 1e-3 && w.abs_gap() > 1e-3 * w.fd.abs().max(1.0) => {
                log::warn!(
                    "[OUTER-FD-AUDIT/{context}] VERDICT=DESYNC worst_block={} worst_i={} gap={:.3e} (analytic gradient disagrees with FD of the criterion: fix the derivative)",
                    w.block,
                    w.index,
                    w.abs_gap()
                );
            }
            _ => {
                let flat = self.min_abs_eigenvalue().map(|m| m < 1e-6).unwrap_or(false);
                if flat {
                    log::warn!(
                        "[OUTER-FD-AUDIT/{context}] VERDICT=FLATNESS min_abs_eig={:.3e} (analytic≈FD but the outer Hessian is near-singular: weak identifiability, fix termination not the gradient)",
                        self.min_abs_eigenvalue().unwrap_or(f64::NAN)
                    );
                } else {
                    log::warn!(
                        "[OUTER-FD-AUDIT/{context}] VERDICT=CLEAN analytic≈FD and outer Hessian well-conditioned at this θ"
                    );
                }
            }
        }
    }
}

/// Run a component-by-component central finite-difference audit of an outer
/// REML/LAML gradient at a fixed θ₀.
///
/// `eval` is the family's outer evaluator: `θ, mode ↦ (V, ∇V, H)` where the
/// gradient is honored at `ValueAndGradient`/`ValueGradientHessian` and `H` at
/// `ValueGradientHessian`. `block_for_index` labels each flat θ coordinate
/// (used only to group the report). `h` is the FD step.
///
/// Cost: one `ValueGradientHessian` eval at θ₀ plus `2·len(θ)` `ValueOnly`
/// evals. The caller is responsible for only invoking this on a
/// diagnostic-sized problem (it is not part of the production hot loop).
pub fn outer_gradient_fd_audit<EvalF>(
    // fd-ok: FD-audit certificate, not in math path
    theta0: &Array1<f64>,
    h: f64,
    block_for_index: impl Fn(usize) -> String,
    mut eval: EvalF,
) -> Result<OuterGradientFdAudit, String>
where
    EvalF: FnMut(
        &Array1<f64>,
        crate::estimate::reml::reml_outer_engine::EvalMode,
    ) -> Result<(f64, Array1<f64>, HessianResult), String>,
{
    use crate::estimate::reml::reml_outer_engine::EvalMode;
    let (value, analytic_grad, hess) = eval(theta0, EvalMode::ValueGradientHessian)?;
    if analytic_grad.len() != theta0.len() {
        return Err(format!(
            "outer_gradient_fd_audit: analytic gradient length {} != theta length {}",
            analytic_grad.len(),
            theta0.len()
        ));
    }
    let mut components = Vec::with_capacity(theta0.len());
    for i in 0..theta0.len() {
        let mut tp = theta0.clone();
        tp[i] += h;
        let mut tm = theta0.clone();
        tm[i] -= h;
        let (vp, _, _) = eval(&tp, EvalMode::ValueOnly)?;
        let (vm, _, _) = eval(&tm, EvalMode::ValueOnly)?;
        let fd = (vp - vm) / (2.0 * h);

        // Cheap-by-default: only a coordinate whose coarse gap could trip the
        // DESYNC band earns a Richardson refinement. The leading central-FD
        // error is `O(h²)·V'''`; on a steeply-curved coordinate (the Matérn
        // log-κ axis, whose operator penalty scales like κ^{2m}) that
        // truncation alone can exceed the band even though the analytic
        // gradient is exact. Combining the `h` and `h/2` central differences as
        // `D_R = (4·D_{h/2} − D_h)/3` cancels the `O(h²)` term, leaving `O(h⁴)`
        // — enough to separate truncation (gap collapses) from a real
        // derivative bug (gap is `h`-independent, so `D_R` stays as far from the
        // analytic value as `D_h` was). The clean-coordinate path stays at 2
        // evals; the rare suspicious coordinate pays 2 more.
        let coarse_gap = (analytic_grad[i] - fd).abs();
        let desync_band = (1e-3_f64).max(1e-3 * fd.abs().max(1.0));
        let fd_refined = if coarse_gap > desync_band {
            let h2 = 0.5 * h;
            let mut tp2 = theta0.clone();
            tp2[i] += h2;
            let mut tm2 = theta0.clone();
            tm2[i] -= h2;
            let (vp2, _, _) = eval(&tp2, EvalMode::ValueOnly)?;
            let (vm2, _, _) = eval(&tm2, EvalMode::ValueOnly)?;
            let fd_half = (vp2 - vm2) / (2.0 * h2);
            Some((4.0 * fd_half - fd) / 3.0)
        } else {
            None
        };
        components.push(OuterGradientFdComponent {
            block: block_for_index(i),
            index: i,
            analytic: analytic_grad[i],
            fd,
            fd_refined,
        });
    }
    let hessian_eigenvalues = match hess.materialize_dense() {
        Ok(Some(mut hmat)) => {
            // Symmetrize defensively before the self-adjoint solve.
            let n = hmat.nrows();
            if n == hmat.ncols() && n > 0 {
                for r in 0..n {
                    for c in (r + 1)..n {
                        let avg = 0.5 * (hmat[[r, c]] + hmat[[c, r]]);
                        hmat[[r, c]] = avg;
                        hmat[[c, r]] = avg;
                    }
                }
                match gam_linalg::faer_ndarray::FaerEigh::eigh(&hmat, faer::Side::Lower) {
                    Ok((vals, _)) => {
                        let mut v: Vec<f64> = vals.to_vec();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        v
                    }
                    Err(_) => Vec::new(),
                }
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    };
    Ok(OuterGradientFdAudit {
        value,
        components,
        hessian_eigenvalues,
    })
}
// END-FD-OK
