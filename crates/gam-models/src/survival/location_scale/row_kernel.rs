use super::*;

use gam_math::jet_scalar::{JetScalar, OneSeed, Order2, TwoSeed};

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurvivalExactRowKernel {
    pub(crate) w: f64,
    pub(crate) d: f64,
    pub(crate) log_s0: f64,
    pub(crate) r0: f64,
    pub(crate) dr0: f64,
    pub(crate) ddr0: f64,
    pub(crate) dddr0: f64,
    pub(crate) log_s1: f64,
    pub(crate) r1: f64,
    pub(crate) dr1: f64,
    pub(crate) ddr1: f64,
    pub(crate) dddr1: f64,
    pub(crate) logphi1: f64,
    pub(crate) dlogphi1: f64,
    pub(crate) d2logphi1: f64,
    pub(crate) d3logphi1: f64,
    pub(crate) d4logphi1: f64,
    pub(crate) log_g: f64,
    pub(crate) d_log_g: f64,
    pub(crate) d2_log_g: f64,
    pub(crate) d3_log_g: f64,
    pub(crate) d4_log_g: f64,
}

/// Mix event and censored contributions, avoiding `0 * Inf = NaN` when
/// `d ∈ {0, 1}` and one branch is non-finite.
#[inline]
pub(crate) fn event_mix(d: f64, event_val: f64, censored_val: f64) -> f64 {
    if d == 1.0 {
        event_val
    } else if d == 0.0 {
        censored_val
    } else {
        d * event_val + (1.0 - d) * censored_val
    }
}

impl SurvivalExactRowKernel {
    #[inline]
    pub(crate) fn log_likelihood(self) -> f64 {
        self.w * (event_mix(self.d, self.logphi1 + self.log_g, self.log_s1) - self.log_s0)
    }

    #[inline]
    pub(crate) fn nll_index_tower(self) -> gam_math::jet_tower::Tower3<3> {
        use gam_math::jet_tower::Tower3;

        // The inner-Newton consumer (`row_derivatives_rescaled`) reads only the
        // value/gradient/Hessian and the diagonal THIRD derivatives of this
        // index NLL — never a fourth-order tensor entry. Building a full
        // `Tower4<3>` here computed the entire `K⁴ = 81`-entry fourth tensor (the
        // dominant per-row Faà-di-Bruno cost) on every row of every inner cycle
        // and discarded all of it. A `Tower3<3>` carries exactly the channels the
        // consumer reads and is bit-identical to `Tower4<3>` on them (the order-3
        // Faà-di-Bruno terms never touch the f⁗ derivative slot), so this is a
        // strict cost truncation, not an approximation. The separate outer
        // joint-Hessian directional-derivative path keeps its own fourth-order
        // stack (`dddr*`, `d4*`) and is unaffected.
        let u0 = Tower3::<3>::variable(0.0, 0);
        let u1 = Tower3::<3>::variable(0.0, 1);
        let g = Tower3::<3>::variable(0.0, 2);
        let mut nll = u0
            .compose_unary([self.log_s0, -self.r0, -self.dr0, -self.ddr0])
            .scale(self.w);

        let censored_weight = self.w * (1.0 - self.d);
        if censored_weight != 0.0 {
            nll = nll
                + u1.compose_unary([self.log_s1, -self.r1, -self.dr1, -self.ddr1])
                    .scale(-censored_weight);
        }

        let event_weight = self.w * self.d;
        if event_weight != 0.0 {
            nll = nll
                + u1.compose_unary([
                    self.logphi1,
                    self.dlogphi1,
                    self.d2logphi1,
                    self.d3logphi1,
                ])
                .scale(-event_weight)
                + g.compose_unary([self.log_g, self.d_log_g, self.d2_log_g, self.d3_log_g])
                    .scale(-event_weight);
        }

        nll
    }
}

pub(crate) struct SurvivalJointQuantities {
    /// Entry-only derivatives of ell w.r.t. q0.
    pub(crate) d1_q0: Array1<f64>,
    pub(crate) d2_q0: Array1<f64>,
    pub(crate) d3_q0: Array1<f64>,
    /// Exit-only derivatives of ell w.r.t. q1.
    pub(crate) d1_q1: Array1<f64>,
    pub(crate) d2_q1: Array1<f64>,
    pub(crate) d3_q1: Array1<f64>,
    /// Exit-only derivatives of ell w.r.t. qdot1 = dq/dt.
    pub(crate) d1_qdot1: Array1<f64>,
    pub(crate) d2_qdot1: Array1<f64>,
    pub(crate) h_time_h0: Array1<f64>,
    pub(crate) h_time_h1: Array1<f64>,
    pub(crate) h_time_d: Array1<f64>,
    /// Exit-side dq/d(eta_t) = -exp(-eta_ls_exit).
    pub(crate) dq_t: Array1<f64>,
    /// Exit-side dq/d(eta_ls).
    pub(crate) dq_ls: Array1<f64>,
    pub(crate) d2q_tls: Array1<f64>,
    pub(crate) d2q_ls: Array1<f64>,
    pub(crate) d3q_tls_ls: Array1<f64>,
    pub(crate) d3q_ls: Array1<f64>,
    /// Entry-side dq0/d(eta_t_entry) = -exp(-eta_ls_entry) (only for time-varying).
    pub(crate) dq_t_entry: Option<Array1<f64>>,
    /// Entry-side q-chain derivatives at entry (only for time-varying sigma).
    pub(crate) dq_ls_entry: Option<Array1<f64>>,
    pub(crate) d2q_tls_entry: Option<Array1<f64>>,
    pub(crate) d2q_ls_entry: Option<Array1<f64>>,
    pub(crate) d3q_tls_ls_entry: Option<Array1<f64>>,
    pub(crate) d3q_ls_entry: Option<Array1<f64>>,
    pub(crate) dqdot_t: Array1<f64>,
    pub(crate) dqdot_ls: Array1<f64>,
    pub(crate) dqdot_td: Array1<f64>,
    pub(crate) dqdot_lsd: Array1<f64>,
    pub(crate) d2qdot_tt: Array1<f64>,
    pub(crate) d2qdot_tls: Array1<f64>,
    pub(crate) d2qdot_ttd: Array1<f64>,
    pub(crate) d2qdot_tlsd: Array1<f64>,
    pub(crate) d2qdot_ls: Array1<f64>,
    pub(crate) d2qdot_lstd: Array1<f64>,
    pub(crate) d2qdot_lslsd: Array1<f64>,
    // NOTE: the only consumer of the 3rd-order qdot maps (d3qdot_tls_ls /
    // _tls_lsd / _td_ls_ls / _ls_ls_ls / _ls_ls_lsd) is the dense `Tower4<9>`
    // location-scale directional path, which `row_kernel_directional_supported`
    // reports as unsupported (returns false). With no live reader, this kernel
    // and `SurvivalDynamicGeometry` carry no 3rd-order qdot state.
}

/// Per-row negative-log-likelihood **curvatures** of the three functionally
/// independent time-channel indices `(h0, h1, d_raw)` — i.e. the diagonal of
/// the row NLL Hessian in time-channel space.
///
/// The stored `SurvivalJointQuantities` fields (`h_time_h0`, `h_time_h1`,
/// `h_time_d`) all hold the *log-likelihood* second derivatives `+∂²ℓ/∂·²`
/// (they are `-tower.h[i][i]` of the NLL jet, double-negated). The NLL Hessian
/// negates each **uniformly** — `H = -∂²ℓ`. Historically each assembly site
/// hand-applied that minus per channel, and one site drifted to `+h_time_d`,
/// flipping the event-Jacobian (`g`) self-term and every `g`-coupled
/// cross-block term (gam#1396). This type makes the sign live in exactly one
/// place: the three channels are negated together at construction, so a
/// per-channel sign skew is structurally unrepresentable.
pub(crate) struct TimeChannelNllCurvatures {
    /// `-∂²ℓ/∂h0²` (entry-survival channel).
    pub(crate) h0: Array1<f64>,
    /// `-∂²ℓ/∂h1²` (exit-survival/event-density channel).
    pub(crate) h1: Array1<f64>,
    /// `-∂²ℓ/∂d_raw²` (event-Jacobian `g = d_raw + qdot` channel).
    pub(crate) d: Array1<f64>,
}

impl SurvivalJointQuantities {
    /// Build the time-channel NLL curvature triple, applying the `H = -∂²ℓ`
    /// negation once and uniformly across `(h0, h1, d_raw)`. Every diagonal
    /// time-channel Hessian assembly site (block-diagonal time-time, full-joint
    /// time-time, and the `g`-coupled time×{threshold,log_sigma,wiggle} cross
    /// blocks) consumes this so the three channels can never disagree on sign.
    pub(crate) fn time_channel_nll_curvatures(&self) -> TimeChannelNllCurvatures {
        TimeChannelNllCurvatures {
            h0: -&self.h_time_h0,
            h1: -&self.h_time_h1,
            d: -&self.h_time_d,
        }
    }
}

pub(crate) struct SurvivalJointPsiDirection {
    pub(crate) x_t_exit_psi: Option<Array2<f64>>,
    pub(crate) x_t_entry_psi: Option<Array2<f64>>,
    pub(crate) x_ls_exit_psi: Option<Array2<f64>>,
    pub(crate) x_ls_entry_psi: Option<Array2<f64>>,
    pub(crate) z_t_exit_psi: Array1<f64>,
    pub(crate) z_t_entry_psi: Array1<f64>,
    pub(crate) z_ls_exit_psi: Array1<f64>,
    pub(crate) z_ls_entry_psi: Array1<f64>,
    pub(crate) x_t_exit_action: Option<CustomFamilyPsiDesignAction>,
    pub(crate) x_t_entry_action: Option<CustomFamilyPsiDesignAction>,
    pub(crate) x_ls_exit_action: Option<CustomFamilyPsiDesignAction>,
    pub(crate) x_ls_entry_action: Option<CustomFamilyPsiDesignAction>,
}

pub(crate) fn split_survival_psi_design(
    x_psi: &Array2<f64>,
    n: usize,
    time_varying: bool,
    label: &str,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if time_varying {
        if x_psi.nrows() != 2 * n && x_psi.nrows() != 3 * n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label} stacked psi design row mismatch: got {}, expected {} or {}",
                    x_psi.nrows(),
                    2 * n,
                    3 * n,
                ),
            }
            .into());
        }
        Ok((
            x_psi.slice(s![0..n, ..]).to_owned(),
            x_psi.slice(s![n..2 * n, ..]).to_owned(),
        ))
    } else {
        if x_psi.nrows() != n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label} psi design row mismatch: got {}, expected {}",
                    x_psi.nrows(),
                    n
                ),
            }
            .into());
        }
        Ok((x_psi.clone(), x_psi.clone()))
    }
}

/// Number of linear-predictor primary channels for the survival
/// location-scale row kernel (non-wiggle configurations).
///
/// The row likelihood `ell = w[d(log f(u1)+log g) + (1-d)log S(u1) - log S(u0)]`
/// depends on three indices `(u0, u1, g)`, each an **affine** function of the
/// model's linear predictors. We make those linear predictors the primary
/// space so the row Jacobian is fixed (the `RowKernel` framework requires
/// this), and fold the nonlinear scale map `q = -eta_t·exp(-eta_ls)` into the
/// per-row kernel. The nine channels are:
///
/// | idx | predictor       | design                              | feeds |
/// |-----|-----------------|-------------------------------------|-------|
/// | 0   | h0  (time entry)| `time_jac_entry`                    | u0    |
/// | 1   | h1  (time exit) | `time_jac_exit`                     | u1    |
/// | 2   | d_raw (time dot)| `time_jac_deriv`                    | g     |
/// | 3   | eta_t_exit      | `x_threshold`                       | u1, g |
/// | 4   | eta_t_entry     | `x_threshold_entry` (or threshold)  | u0    |
/// | 5   | eta_t_deriv     | `x_threshold_deriv` (or none)       | g     |
/// | 6   | eta_ls_exit     | `x_log_sigma`                       | u1, g |
/// | 7   | eta_ls_entry    | `x_log_sigma_entry` (or log_sigma)  | u0    |
/// | 8   | eta_ls_deriv    | `x_log_sigma_deriv` (or none)       | g     |
///
/// `H[a][b] = -Σ_i (ell_ii·D_i[a]·D_i[b] + ell_i·D2_i[a][b])` reproduces
/// `assemble_joint_hessian_from_quantities` term-for-term (verified by the
/// equivalence test). Indices `i ∈ {u0,u1,g}` are functionally independent so
/// the index-space derivative tensors are diagonal in `i`.
pub(crate) const SLS_ROW_K: usize = 9;

/// `RowKernel<9>` adapter for the survival location-scale joint likelihood
/// (non-wiggle path). Holds the per-β quantities already computed by
/// [`SurvivalLocationScaleFamily::collect_joint_quantities_rescaled`] and
/// [`SurvivalLocationScaleFamily::build_dynamic_geometry`]; every trait method
/// is a pure repackaging of those scalars into linear-predictor primary space,
/// so the math is identical to the bespoke assembly by construction.
pub(crate) struct SurvivalLsRowKernel<'a> {
    pub(crate) family: &'a SurvivalLocationScaleFamily,
    pub(crate) q: &'a SurvivalJointQuantities,
    pub(crate) dynamic: &'a SurvivalDynamicGeometry,
    pub(crate) deriv_log_scale: f64,
    /// Joint block offsets `[0, p_time, p_time+p_thr, p_total]` (3 blocks).
    pub(crate) offsets: Vec<usize>,
}

impl SurvivalLsRowKernel<'_> {
    /// Resolve the design for a threshold/log-sigma channel, falling back to the
    /// exit design when the entry/derivative variant is absent (time-invariant).
    #[inline]
    pub(crate) fn entry_design<'b>(
        opt: &'b Option<DesignMatrix>,
        fallback: &'b DesignMatrix,
    ) -> &'b DesignMatrix {
        opt.as_ref().unwrap_or(fallback)
    }

    /// Per-row dense design row for each channel within its coefficient block:
    /// returns `(block_index, row_vector)` for channels `0..9`. Used by the
    /// pullback / diagonal assembly. Channels with an absent derivative design
    /// (time-invariant derivative channels) return `None` and contribute
    /// nothing.
    pub(crate) fn channel_block(&self, ch: usize) -> Option<usize> {
        match ch {
            0 | 1 | 2 => Some(Self::THRESHOLD_BLOCK_TIME),
            3 | 4 | 5 => Some(Self::THRESHOLD_BLOCK_THR),
            6 | 7 | 8 => Some(Self::THRESHOLD_BLOCK_LS),
            _ => None,
        }
    }
    pub(crate) const THRESHOLD_BLOCK_TIME: usize = 0;
    pub(crate) const THRESHOLD_BLOCK_THR: usize = 1;
    pub(crate) const THRESHOLD_BLOCK_LS: usize = 2;

    /// Dense per-row design vector for `channel` (length = its block width), or
    /// `None` when the channel's design is absent (time-invariant deriv channel,
    /// which carries no coefficients of its own).
    pub(crate) fn channel_row(&self, ch: usize, row: usize) -> Option<Array1<f64>> {
        let fam = self.family;
        match ch {
            0 => Some(self.dynamic.time_jac_entry.row(row).to_owned()),
            1 => Some(self.dynamic.time_jac_exit.row(row).to_owned()),
            2 => Some(self.dynamic.time_jac_deriv.row(row).to_owned()),
            3 => Some(design_dense_row(&fam.x_threshold, row)),
            4 => Some(design_dense_row(
                Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold),
                row,
            )),
            5 => fam
                .x_threshold_deriv
                .as_ref()
                .map(|d| design_dense_row(d, row)),
            6 => Some(design_dense_row(&fam.x_log_sigma, row)),
            7 => Some(design_dense_row(
                Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma),
                row,
            )),
            8 => fam
                .x_log_sigma_deriv
                .as_ref()
                .map(|d| design_dense_row(d, row)),
            _ => None,
        }
    }

    pub(crate) fn row_primary_values(&self, row: usize) -> [f64; SLS_ROW_K] {
        let inv_sigma_exit = self.dynamic.inv_sigma_exit[row];
        let eta_t_exit = -self.dynamic.q_exit[row] / inv_sigma_exit;
        let eta_ls_deriv = self.q.dqdot_t[row] / inv_sigma_exit;
        let eta_t_deriv = eta_t_exit * eta_ls_deriv - self.dynamic.qdot_exit[row] / inv_sigma_exit;
        [
            self.dynamic.h_entry[row],
            self.dynamic.h_exit[row],
            self.dynamic.hdot_exit[row],
            eta_t_exit,
            -self.dynamic.q_entry[row] / self.dynamic.inv_sigma_entry[row],
            eta_t_deriv,
            self.dynamic.eta_ls_exit[row],
            self.dynamic.eta_ls_entry[row],
            eta_ls_deriv,
        ]
    }

    /// The row's exact f64 derivative-stack kernel and the nine primary values
    /// `p` — the scalar-independent inputs the generic row NLL
    /// ([`sls_row_nll`]) consumes. Computed once per row; reused across every
    /// `JetScalar` instantiation (value/grad/Hessian, contracted third/fourth).
    fn row_nll_inputs(
        &self,
        row: usize,
    ) -> Result<([f64; SLS_ROW_K], SurvivalExactRowKernel), String> {
        let p = self.row_primary_values(row);
        let state = self.family.row_predictor_state(
            self.dynamic.h_entry[row],
            self.dynamic.h_exit[row],
            self.dynamic.hdot_exit[row],
            self.dynamic.q_entry[row],
            self.dynamic.q_exit[row],
            self.dynamic.qdot_exit[row],
        );
        let kernel = self
            .family
            .exact_row_kernel_rescaled(row, state, self.deriv_log_scale)?
            .ok_or_else(|| format!("survival location-scale row {row} has no exact kernel"))?;
        Ok((p, kernel))
    }
}

/// The survival location-scale row negative log-likelihood, written ONCE over a
/// generic [`JetScalar<9>`] so a single expression yields every derivative
/// channel a consumer needs:
///
/// * `S = Tower4<9>` → full `(v, g, H, t3, t4)` (the all-channels oracle path,
///   [`SurvivalLsRowKernel::row_nll_tower`]),
/// * `S = OneSeed<9>` → the contracted third `Σ_c ℓ_{abc} dir_c` (1.46 KiB/row,
///   the `RowKernel::row_third_contracted` directional path),
/// * `S = TwoSeed<9>` → the contracted fourth `Σ_{cd} ℓ_{abcd} u_c v_d`
///   (2.8 KiB/row, the `RowKernel::row_fourth_contracted` path).
///
/// (The packed `Order2<9>` value/grad/Hessian scalar — 728 B — is the base each
/// of `OneSeed`/`TwoSeed` is built on; it is the channel a future joint-Hessian
/// cutover would instantiate here once `row_kernel_joint_hessian_supported` is
/// enabled.)
///
/// The nine primary channels are `(h_entry, h_exit, hdot_exit, eta_t_exit,
/// eta_t_entry, eta_t_deriv, eta_ls_exit, eta_ls_entry, eta_ls_deriv)` — see
/// [`SurvivalLsRowKernel::row_primary_values`]. From them the survival index
/// quantities are
///   `u0 = h_entry − eta_t_entry·e^{−eta_ls_entry}`  (entry / left-truncation),
///   `u1 = h_exit  − eta_t_exit ·e^{−eta_ls_exit}`   (exit),
///   `g  = hdot_exit + e^{−eta_ls_exit}·(eta_t_exit·eta_ls_deriv − eta_t_deriv)`
/// (the event log-density's Jacobian factor), and the NLL is
///   `w[ logS0(u0) − (1−d)·logS1(u1) − d·(logφ1(u1) + log g(g)) ]`,
/// each residual-distribution stack `logS/logφ/log g` supplied as a hand-certified
/// `[f64; 5]` derivative stack on the kernel and entered through
/// [`JetScalar::compose_unary`]. There is exactly one source for value and every
/// derivative order (the #736/#932 single-source contract).
pub(crate) fn sls_row_nll<S: JetScalar<SLS_ROW_K>>(
    vars: &[S; SLS_ROW_K],
    kernel: &SurvivalExactRowKernel,
) -> Result<S, String> {
    let inv_sigma_entry = vars[7].neg().exp();
    let u0 = vars[0].sub(&vars[4].mul(&inv_sigma_entry));
    let inv_sigma_exit = vars[6].neg().exp();
    let u1 = vars[1].sub(&vars[3].mul(&inv_sigma_exit));
    let g = vars[2].add(&inv_sigma_exit.mul(&vars[3].mul(&vars[8]).sub(&vars[5])));

    let mut nll = u0
        .compose_unary([
            kernel.log_s0,
            -kernel.r0,
            -kernel.dr0,
            -kernel.ddr0,
            -kernel.dddr0,
        ])
        .scale(kernel.w);
    let censored_weight = kernel.w * (1.0 - kernel.d);
    if censored_weight != 0.0 {
        nll = nll.add(
            &u1.compose_unary([
                kernel.log_s1,
                -kernel.r1,
                -kernel.dr1,
                -kernel.ddr1,
                -kernel.dddr1,
            ])
            .scale(-censored_weight),
        );
    }
    let event_weight = kernel.w * kernel.d;
    if event_weight != 0.0 {
        nll = nll
            .add(
                &u1.compose_unary([
                    kernel.logphi1,
                    kernel.dlogphi1,
                    kernel.d2logphi1,
                    kernel.d3logphi1,
                    kernel.d4logphi1,
                ])
                .scale(-event_weight),
            )
            .add(
                &g.compose_unary([
                    kernel.log_g,
                    kernel.d_log_g,
                    kernel.d2_log_g,
                    kernel.d3_log_g,
                    kernel.d4_log_g,
                ])
                .scale(-event_weight),
            );
    }
    Ok(nll)
}

/// Materialize `X[row, :]` as a dense length-`ncols` vector (no sparse-aware
/// fast path — used only by the dense-Hessian / diagonal assembly, never the
/// hot matvec inner loop).
pub(crate) fn design_dense_row(d: &DesignMatrix, row: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(d.ncols());
    d.axpy_row_into(row, 1.0, &mut out.view_mut())
        .expect("design_dense_row: ncols-sized buffer matches design width");
    out
}

/// Accumulate `alpha * jac[row, :]` into the coefficient slice `out` for a dense
/// time Jacobian (the survival time block is materialized densely as
/// `time_jac_*`, so it has no sparse axpy primitive).
#[inline]
pub(crate) fn axpy_dense_row_into(jac: &Array2<f64>, row: usize, alpha: f64, out: &mut [f64]) {
    if alpha == 0.0 {
        return;
    }
    let jr = jac.row(row);
    for (o, &j) in out.iter_mut().zip(jr.iter()) {
        *o += alpha * j;
    }
}

pub(crate) fn row_set_from_survival_mask(
    row_mask: Option<&Array1<f64>>,
    n: usize,
) -> crate::row_kernel::RowSet {
    let Some(mask) = row_mask else {
        return crate::row_kernel::RowSet::All;
    };
    let rows = mask
        .iter()
        .enumerate()
        .filter_map(|(index, &weight)| {
            (weight != 0.0).then_some(crate::outer_subsample::WeightedOuterRow {
                index,
                weight,
                stratum: 0,
            })
        })
        .collect::<Vec<_>>();
    crate::row_kernel::RowSet::Subsample {
        rows: Arc::new(rows),
        n_full: n,
    }
}

/// #932 link-wiggle: the survival-LS row NLL extended with the link warp
/// `q = q0 + Σ_j βw_j·B_j(q0)` and the qdot coupling `g = m1·g0`,
/// `m1 = 1 + Σ_j βw_j·B'_j(q0_exit)`, written ONCE over a generic jet scalar
/// (`KW = SLS_ROW_K + pw`). `vars[0..9]` are the base channels (exactly
/// [`sls_row_nll`]); `vars[9..9+pw]` are the wiggle amplitudes βw. The per-row
/// basis stacks are evaluated at the BASE indices (the warp composes the basis
/// onto the index jet): `b{0,1,2}e` = `B/B'/B''` at `q0_entry`, `b{0,1,2,3}x` =
/// `B/B'/B''/B'''` at `q0_exit`. Bit-identical (modulo association) to the nested
/// witness in `survival_ls_wiggle_joint_hessian_matches_assembler_932`.
/// Per-row warp basis stacks evaluated at the BASE indices. `b_u0` holds
/// `[B, B', B'']` at `q0_entry` (entry warp `u0w`); `b_u1` holds
/// `[B, B', B'', B''']` at `q0_exit` (exit warp `u1w` and the qdot slope `m1`).
/// Each inner slice has one entry per wiggle column (`pw` long). Bundling the
/// seven slices keeps [`sls_row_nll_wiggle`] within the argument budget.
pub(crate) struct SlsWiggleRowBasis<'b> {
    pub(crate) b_u0: [&'b [f64]; 3],
    pub(crate) b_u1: [&'b [f64]; 4],
}

pub(crate) fn sls_row_nll_wiggle<const KW: usize, S: JetScalar<KW>>(
    vars: &[S; KW],
    kernel: &SurvivalExactRowKernel,
    pw: usize,
    basis: &SlsWiggleRowBasis<'_>,
) -> S {
    let [b0e, b1e, b2e] = basis.b_u0;
    let [b0x, b1x, b2x, b3x] = basis.b_u1;
    let inv_sigma_entry = vars[7].neg().exp();
    let u0 = vars[0].sub(&vars[4].mul(&inv_sigma_entry));
    let inv_sigma_exit = vars[6].neg().exp();
    let u1 = vars[1].sub(&vars[3].mul(&inv_sigma_exit));
    let g0 = vars[2].add(&inv_sigma_exit.mul(&vars[3].mul(&vars[8]).sub(&vars[5])));
    let mut u0w = u0;
    let mut u1w = u1;
    let mut m1 = S::constant(1.0);
    for j in 0..pw {
        let bw = vars[SLS_ROW_K + j];
        u0w = u0w.add(&bw.mul(&u0.compose_unary([b0e[j], b1e[j], b2e[j], 0.0, 0.0])));
        u1w = u1w.add(&bw.mul(&u1.compose_unary([b0x[j], b1x[j], b2x[j], b3x[j], 0.0])));
        m1 = m1.add(&bw.mul(&u1.compose_unary([b1x[j], b2x[j], b3x[j], 0.0, 0.0])));
    }
    let g = m1.mul(&g0);
    let mut nll = u0w
        .compose_unary([
            kernel.log_s0,
            -kernel.r0,
            -kernel.dr0,
            -kernel.ddr0,
            -kernel.dddr0,
        ])
        .scale(kernel.w);
    let censored_weight = kernel.w * (1.0 - kernel.d);
    if censored_weight != 0.0 {
        nll = nll.add(
            &u1w.compose_unary([
                kernel.log_s1,
                -kernel.r1,
                -kernel.dr1,
                -kernel.ddr1,
                -kernel.dddr1,
            ])
            .scale(-censored_weight),
        );
    }
    let event_weight = kernel.w * kernel.d;
    if event_weight != 0.0 {
        nll = nll
            .add(
                &u1w.compose_unary([
                    kernel.logphi1,
                    kernel.dlogphi1,
                    kernel.d2logphi1,
                    kernel.d3logphi1,
                    kernel.d4logphi1,
                ])
                .scale(-event_weight),
            )
            .add(
                &g.compose_unary([
                    kernel.log_g,
                    kernel.d_log_g,
                    kernel.d2_log_g,
                    kernel.d3_log_g,
                    kernel.d4_log_g,
                ])
                .scale(-event_weight),
            );
    }
    nll
}

/// #932 link-wiggle joint-Hessian production kernel: routes the survival-LS
/// joint Hessian for link-wiggle rows through the single-source §13 warp
/// ([`sls_row_nll_wiggle`]) instead of the bespoke `assemble_h_wiggle`. The base
/// 9 channels reuse the existing [`SurvivalLsRowKernel`] designs; the βw
/// amplitudes are an IDENTITY map into a wiggle coefficient block appended last.
/// `KW = SLS_ROW_K + pw`.
pub(crate) struct SurvivalLsWiggleRowKernel<'a, const KW: usize> {
    base: SurvivalLsRowKernel<'a>,
    pw: usize,
    wiggle_off: usize,
    betaw: Vec<f64>,
    b_u0_0: Array2<f64>,
    b_u0_1: Array2<f64>,
    b_u0_2: Array2<f64>,
    b_u1_0: Array2<f64>,
    b_u1_1: Array2<f64>,
    b_u1_2: Array2<f64>,
    b_u1_3: Array2<f64>,
}

impl<'a, const KW: usize> SurvivalLsWiggleRowKernel<'a, KW> {
    pub(crate) fn new(
        family: &'a SurvivalLocationScaleFamily,
        q: &'a SurvivalJointQuantities,
        dynamic: &'a SurvivalDynamicGeometry,
        deriv_log_scale: f64,
    ) -> Result<Self, String> {
        let base = SurvivalLsRowKernel {
            family,
            q,
            dynamic,
            deriv_log_scale,
            offsets: family.joint_block_offsets(),
        };
        let knots = family
            .wiggle_knots
            .as_ref()
            .ok_or("link-wiggle kernel: missing wiggle knots")?;
        let degree = family
            .wiggle_degree
            .ok_or("link-wiggle kernel: missing wiggle degree")?;
        // Base exit/entry indices where the warp basis is evaluated.
        let q_exit = dynamic.q_exit.clone();
        let q_entry = dynamic.q_entry.clone();
        let b_u0_0 = survival_wiggle_basis_with_options(
            q_entry.view(),
            knots,
            degree,
            BasisOptions::value(),
        )?;
        let b_u0_1 = survival_wiggle_basis_with_options(
            q_entry.view(),
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let b_u0_2 = survival_wiggle_basis_with_options(
            q_entry.view(),
            knots,
            degree,
            BasisOptions::second_derivative(),
        )?;
        let b_u1_0 = survival_wiggle_basis_with_options(
            q_exit.view(),
            knots,
            degree,
            BasisOptions::value(),
        )?;
        let b_u1_1 = survival_wiggle_basis_with_options(
            q_exit.view(),
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let b_u1_2 = survival_wiggle_basis_with_options(
            q_exit.view(),
            knots,
            degree,
            BasisOptions::second_derivative(),
        )?;
        let b_u1_3 = survival_wiggle_third_basis(q_exit.view(), knots, degree)?;
        let pw = b_u1_0.ncols();
        if SLS_ROW_K + pw != KW {
            return Err(format!(
                "link-wiggle kernel: KW={KW} but SLS_ROW_K+pw={}",
                SLS_ROW_K + pw
            ));
        }
        // joint_block_offsets() appends the wiggle block last, so its start is
        // the second-to-last offset (offsets = [0, time, +thr, +ls, +wiggle]).
        let wiggle_off = base.offsets[base.offsets.len() - 2];
        // βw is carried on the dynamic geometry (populated from the wiggle block
        // in `build_dynamic_geometry`), so this kernel needs no `block_states` —
        // letting the workspace `_from_parts` directional entry points build it.
        let betaw = dynamic
            .wiggle_beta
            .as_ref()
            .map(|b| b.to_vec())
            .ok_or("link-wiggle kernel: missing wiggle_beta on dynamic geometry")?;
        Ok(Self {
            base,
            pw,
            wiggle_off,
            betaw,
            b_u0_0,
            b_u0_1,
            b_u0_2,
            b_u1_0,
            b_u1_1,
            b_u1_2,
            b_u1_3,
        })
    }

    #[inline]
    fn row_vars<S: JetScalar<KW>>(&self, row: usize, seed: impl Fn(f64, usize) -> S) -> [S; KW] {
        let p = self.base.row_primary_values(row);
        std::array::from_fn(|a| {
            if a < SLS_ROW_K {
                seed(p[a], a)
            } else {
                seed(self.betaw[a - SLS_ROW_K], a)
            }
        })
    }

    #[inline]
    fn eval<S: JetScalar<KW>>(&self, row: usize, vars: &[S; KW]) -> Result<S, String> {
        let kernel = self.base.row_nll_inputs(row)?.1;
        let r_u0_0 = self.b_u0_0.row(row).to_vec();
        let r_u0_1 = self.b_u0_1.row(row).to_vec();
        let r_u0_2 = self.b_u0_2.row(row).to_vec();
        let r_u1_0 = self.b_u1_0.row(row).to_vec();
        let r_u1_1 = self.b_u1_1.row(row).to_vec();
        let r_u1_2 = self.b_u1_2.row(row).to_vec();
        let r_u1_3 = self.b_u1_3.row(row).to_vec();
        let basis = SlsWiggleRowBasis {
            b_u0: [&r_u0_0, &r_u0_1, &r_u0_2],
            b_u1: [&r_u1_0, &r_u1_1, &r_u1_2, &r_u1_3],
        };
        Ok(sls_row_nll_wiggle(vars, &kernel, self.pw, &basis))
    }

    /// Per-(channel, row) coefficient-block + dense design row, length KW.
    /// Base channels delegate to [`SurvivalLsRowKernel`]; βw channels map to the
    /// wiggle block with a unit (identity) row `e_b`.
    fn jrow(&self, ch: usize, row: usize) -> Option<(usize, Array1<f64>)> {
        if ch < SLS_ROW_K {
            let blk = self.base.channel_block(ch)?;
            let r = self.base.channel_row(ch, row)?;
            Some((self.base.offsets[blk], r))
        } else {
            let b = ch - SLS_ROW_K;
            let mut e = Array1::<f64>::zeros(self.pw);
            e[b] = 1.0;
            Some((self.wiggle_off, e))
        }
    }
}

impl<const KW: usize> crate::row_kernel::RowKernel<KW>
    for SurvivalLsWiggleRowKernel<'_, KW>
{
    fn n_rows(&self) -> usize {
        self.base.family.n
    }

    fn n_coefficients(&self) -> usize {
        self.wiggle_off + self.pw
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; KW], [[f64; KW]; KW]), String> {
        let vars: [Order2<KW>; KW] = self.row_vars(row, |x, a| Order2::variable(x, a));
        let out = self.eval(row, &vars)?;
        Ok((JetScalar::value(&out), out.g(), out.h()))
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; KW] {
        let base = self.base.jacobian_action(row, &d_beta[..self.wiggle_off]);
        std::array::from_fn(|a| {
            if a < SLS_ROW_K {
                base[a]
            } else {
                d_beta[self.wiggle_off + (a - SLS_ROW_K)]
            }
        })
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; KW], out: &mut [f64]) {
        let vbase: [f64; SLS_ROW_K] = std::array::from_fn(|a| v[a]);
        self.base
            .jacobian_transpose_action(row, &vbase, &mut out[..self.wiggle_off]);
        for b in 0..self.pw {
            out[self.wiggle_off + b] += v[SLS_ROW_K + b];
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; KW]; KW], target: &mut Array2<f64>) {
        let rows: Vec<Option<(usize, Array1<f64>)>> =
            (0..KW).map(|ch| self.jrow(ch, row)).collect();
        for a in 0..KW {
            let Some((off_a, ra)) = rows[a].as_ref() else {
                continue;
            };
            for b in 0..KW {
                let hab = h[a][b];
                if hab == 0.0 {
                    continue;
                }
                let Some((off_b, rb)) = rows[b].as_ref() else {
                    continue;
                };
                for (ia, &va) in ra.iter().enumerate() {
                    if va == 0.0 {
                        continue;
                    }
                    let w = hab * va;
                    let mut trow = target.row_mut(off_a + ia);
                    for (ib, &vb) in rb.iter().enumerate() {
                        trow[off_b + ib] += w * vb;
                    }
                }
            }
        }
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; KW]; KW], diag: &mut [f64]) {
        let rows: Vec<Option<(usize, Array1<f64>)>> =
            (0..KW).map(|ch| self.jrow(ch, row)).collect();
        for a in 0..KW {
            let Some((off_a, ra)) = rows[a].as_ref() else {
                continue;
            };
            for b in 0..KW {
                let hab = h[a][b];
                if hab == 0.0 {
                    continue;
                }
                let Some((off_b, rb)) = rows[b].as_ref() else {
                    continue;
                };
                if off_a != off_b {
                    continue;
                }
                for (k, (&va, &vb)) in ra.iter().zip(rb.iter()).enumerate() {
                    diag[off_a + k] += hab * va * vb;
                }
            }
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; KW]) -> Result<[[f64; KW]; KW], String> {
        let vars: [OneSeed<KW>; KW] =
            self.row_vars(row, |x, a| OneSeed::seed_direction(x, a, dir[a]));
        Ok(self.eval(row, &vars)?.contracted_third())
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; KW],
        dir_v: &[f64; KW],
    ) -> Result<[[f64; KW]; KW], String> {
        let vars: [TwoSeed<KW>; KW] =
            self.row_vars(row, |x, a| TwoSeed::seed(x, a, dir_u[a], dir_v[a]));
        Ok(self.eval(row, &vars)?.contracted_fourth())
    }
}

/// Dispatch the runtime wiggle width `pw` to a concrete `KW = SLS_ROW_K + pw`
/// and assemble the link-wiggle joint Hessian through the single-source kernel.
pub(crate) fn survival_ls_wiggle_joint_hessian_dense(
    family: &SurvivalLocationScaleFamily,
    q: &SurvivalJointQuantities,
    dynamic: &SurvivalDynamicGeometry,
    deriv_log_scale: f64,
) -> Result<Array2<f64>, String> {
    use crate::row_kernel::{RowSet, build_row_kernel_cache, row_kernel_hessian_dense};
    let pw = family
        .x_link_wiggle
        .as_ref()
        .map(|d| d.ncols())
        .ok_or("wiggle joint Hessian: no link-wiggle design")?;
    macro_rules! run {
        ($kw:literal) => {{
            let kernel =
                SurvivalLsWiggleRowKernel::<$kw>::new(family, q, dynamic, deriv_log_scale)?;
            let rows = RowSet::All;
            let cache = build_row_kernel_cache(&kernel, &rows)?;
            Ok(row_kernel_hessian_dense(&kernel, &cache, &rows))
        }};
    }
    match SLS_ROW_K + pw {
        10 => run!(10),
        11 => run!(11),
        12 => run!(12),
        13 => run!(13),
        14 => run!(14),
        15 => run!(15),
        16 => run!(16),
        17 => run!(17),
        18 => run!(18),
        19 => run!(19),
        20 => run!(20),
        other => Err(format!("link-wiggle joint Hessian: unsupported KW={other}")),
    }
}

/// Dispatch the runtime wiggle width `pw` to a concrete `KW` and assemble the
/// single-source link-wiggle FIRST directional derivative `Σ_c ℓ_{abc} dir_c =
/// (D_dir H)[a][b]` — the ε-Hessian channel of the §13 warp row NLL at the
/// packed `OneSeed<KW>` directional scalar, pulled back into coefficient space
/// by the SAME `JᵀHJ` the joint-Hessian path uses. Replaces the bespoke hand
/// assembly the `_from_parts_masked` wiggle fall-through previously ran (the
/// #736/#932 hand-derivative genus the single-source contract removes). The
/// convention matches the non-wiggle base path, which routes its directional
/// through the identical `row_kernel_directional_derivative` free function.
pub(crate) fn survival_ls_wiggle_directional_derivative_dense(
    family: &SurvivalLocationScaleFamily,
    q: &SurvivalJointQuantities,
    dynamic: &SurvivalDynamicGeometry,
    deriv_log_scale: f64,
    rows: &crate::row_kernel::RowSet,
    d_beta: &[f64],
) -> Result<Array2<f64>, String> {
    use crate::row_kernel::row_kernel_directional_derivative;
    let pw = family
        .x_link_wiggle
        .as_ref()
        .map(|d| d.ncols())
        .ok_or("wiggle directional derivative: no link-wiggle design")?;
    macro_rules! run {
        ($kw:literal) => {{
            let kernel =
                SurvivalLsWiggleRowKernel::<$kw>::new(family, q, dynamic, deriv_log_scale)?;
            row_kernel_directional_derivative(&kernel, rows, d_beta)
        }};
    }
    match SLS_ROW_K + pw {
        10 => run!(10),
        11 => run!(11),
        12 => run!(12),
        13 => run!(13),
        14 => run!(14),
        15 => run!(15),
        16 => run!(16),
        17 => run!(17),
        18 => run!(18),
        19 => run!(19),
        20 => run!(20),
        other => Err(format!(
            "link-wiggle directional derivative: unsupported KW={other}"
        )),
    }
}

/// Dispatch the runtime wiggle width `pw` to a concrete `KW` and assemble the
/// single-source link-wiggle SECOND directional derivative `Σ_cd ℓ_{abcd} u_c
/// v_d` — the ε,δ-Hessian channel of the §13 warp row NLL at the packed
/// `TwoSeed<KW>` bidirectional scalar. Replaces the previous wiggle carve-out
/// that returned `None` (no second-directional curvature for wiggle rows).
pub(crate) fn survival_ls_wiggle_second_directional_derivative_dense(
    family: &SurvivalLocationScaleFamily,
    q: &SurvivalJointQuantities,
    dynamic: &SurvivalDynamicGeometry,
    deriv_log_scale: f64,
    rows: &crate::row_kernel::RowSet,
    d_beta_u: &[f64],
    d_beta_v: &[f64],
) -> Result<Array2<f64>, String> {
    use crate::row_kernel::row_kernel_second_directional_derivative;
    let pw = family
        .x_link_wiggle
        .as_ref()
        .map(|d| d.ncols())
        .ok_or("wiggle second directional derivative: no link-wiggle design")?;
    macro_rules! run {
        ($kw:literal) => {{
            let kernel =
                SurvivalLsWiggleRowKernel::<$kw>::new(family, q, dynamic, deriv_log_scale)?;
            row_kernel_second_directional_derivative(&kernel, rows, d_beta_u, d_beta_v)
        }};
    }
    match SLS_ROW_K + pw {
        10 => run!(10),
        11 => run!(11),
        12 => run!(12),
        13 => run!(13),
        14 => run!(14),
        15 => run!(15),
        16 => run!(16),
        17 => run!(17),
        18 => run!(18),
        19 => run!(19),
        20 => run!(20),
        other => Err(format!(
            "link-wiggle second directional derivative: unsupported KW={other}"
        )),
    }
}

impl crate::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_> {
    fn n_rows(&self) -> usize {
        self.family.n
    }

    fn n_coefficients(&self) -> usize {
        *self.offsets.last().expect("offsets has block bounds")
    }

    fn row_kernel(
        &self,
        row: usize,
    ) -> Result<(f64, [f64; SLS_ROW_K], [[f64; SLS_ROW_K]; SLS_ROW_K]), String> {
        // #932: value, gradient and Hessian derive from the SAME single-sourced
        // row NLL (`sls_row_nll`) instantiated at the packed `Order2<9>` scalar —
        // the exact order-≤2 truncation of the Leibniz / Faà di Bruno rules (728
        // B/row). There is no longer a hand-assembled coefficient-space chain rule
        // here: the `(v, g, H)` channels are the order-≤2 part of the very
        // expression whose order-3/4 directional contractions `row_third_contracted`
        // / `row_fourth_contracted` already evaluate, so all four channels share one
        // mathematical definition (the #736/#932 single-source contract). Identity
        // seeding (`Order2::variable`) makes the ε/δ-free tower carry ∂/∂p_a and
        // ∂²/∂p_a∂p_b directly. Bit-identical to `row_nll_tower(row)?` value/grad/
        // Hessian by the `survival_ls_joint_row_kernel_agrees_with_jet_tower_program_all_channels`
        // oracle (≤ 1e-9).
        let (p, kernel) = self.row_nll_inputs(row)?;
        let vars: [Order2<SLS_ROW_K>; SLS_ROW_K] =
            std::array::from_fn(|a| Order2::variable(p[a], a));
        let out = sls_row_nll(&vars, &kernel)?;
        Ok((out.value(), out.g(), out.h()))
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; SLS_ROW_K] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.offsets[0]..self.offsets[1]]);
        let d_thr = d_beta.slice(s![self.offsets[1]..self.offsets[2]]);
        let d_ls = d_beta.slice(s![self.offsets[2]..self.offsets[3]]);
        let fam = self.family;
        let t_entry = Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold);
        let ls_entry = Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma);
        let ch5 = fam
            .x_threshold_deriv
            .as_ref()
            .map_or(0.0, |d| d.dot_row_view(row, d_thr));
        let ch8 = fam
            .x_log_sigma_deriv
            .as_ref()
            .map_or(0.0, |d| d.dot_row_view(row, d_ls));
        [
            self.dynamic.time_jac_entry.row(row).dot(&d_time),
            self.dynamic.time_jac_exit.row(row).dot(&d_time),
            self.dynamic.time_jac_deriv.row(row).dot(&d_time),
            fam.x_threshold.dot_row_view(row, d_thr),
            t_entry.dot_row_view(row, d_thr),
            ch5,
            fam.x_log_sigma.dot_row_view(row, d_ls),
            ls_entry.dot_row_view(row, d_ls),
            ch8,
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; SLS_ROW_K], out: &mut [f64]) {
        let fam = self.family;
        // Time block: channels 0,1,2 via the dense time Jacobians.
        {
            let time = &mut out[self.offsets[0]..self.offsets[1]];
            axpy_dense_row_into(&self.dynamic.time_jac_entry, row, v[0], time);
            axpy_dense_row_into(&self.dynamic.time_jac_exit, row, v[1], time);
            axpy_dense_row_into(&self.dynamic.time_jac_deriv, row, v[2], time);
        }
        // Threshold block: channels 3 (exit), 4 (entry), 5 (deriv).
        {
            let mut thr = ndarray::ArrayViewMut1::from(&mut out[self.offsets[1]..self.offsets[2]]);
            fam.x_threshold
                .axpy_row_into(row, v[3], &mut thr)
                .expect("threshold exit axpy");
            Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold)
                .axpy_row_into(row, v[4], &mut thr)
                .expect("threshold entry axpy");
            if let Some(d) = fam.x_threshold_deriv.as_ref() {
                d.axpy_row_into(row, v[5], &mut thr)
                    .expect("threshold deriv axpy");
            }
        }
        // Log-sigma block: channels 6 (exit), 7 (entry), 8 (deriv).
        {
            let mut ls = ndarray::ArrayViewMut1::from(&mut out[self.offsets[2]..self.offsets[3]]);
            fam.x_log_sigma
                .axpy_row_into(row, v[6], &mut ls)
                .expect("log_sigma exit axpy");
            Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma)
                .axpy_row_into(row, v[7], &mut ls)
                .expect("log_sigma entry axpy");
            if let Some(d) = fam.x_log_sigma_deriv.as_ref() {
                d.axpy_row_into(row, v[8], &mut ls)
                    .expect("log_sigma deriv axpy");
            }
        }
    }

    fn add_pullback_hessian(
        &self,
        row: usize,
        h: &[[f64; SLS_ROW_K]; SLS_ROW_K],
        target: &mut Array2<f64>,
    ) {
        // Materialize each channel's dense block row once, then accumulate
        // h[a][b]·(row_a ⊗ row_b) into the (block_a, block_b) sub-block.
        let rows: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
            .map(|ch| self.channel_block(ch).zip(self.channel_row(ch, row)))
            .collect();
        for a in 0..SLS_ROW_K {
            let Some((ba, ra)) = rows[a].as_ref() else {
                continue;
            };
            let off_a = self.offsets[*ba];
            for b in 0..SLS_ROW_K {
                let hab = h[a][b];
                if hab == 0.0 {
                    continue;
                }
                let Some((bb, rb)) = rows[b].as_ref() else {
                    continue;
                };
                let off_b = self.offsets[*bb];
                for (ia, &va) in ra.iter().enumerate() {
                    if va == 0.0 {
                        continue;
                    }
                    let w = hab * va;
                    let mut trow = target.row_mut(off_a + ia);
                    for (ib, &vb) in rb.iter().enumerate() {
                        trow[off_b + ib] += w * vb;
                    }
                }
            }
        }
    }

    fn add_diagonal_quadratic(
        &self,
        row: usize,
        h: &[[f64; SLS_ROW_K]; SLS_ROW_K],
        diag: &mut [f64],
    ) {
        // diag[c] += Σ_{a,b ∈ block(c)} h[a][b]·row_a[c]·row_b[c]. Only
        // same-block channel pairs touch a given coefficient's diagonal slot.
        let rows: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
            .map(|ch| self.channel_block(ch).zip(self.channel_row(ch, row)))
            .collect();
        for a in 0..SLS_ROW_K {
            let Some((ba, ra)) = rows[a].as_ref() else {
                continue;
            };
            for b in 0..SLS_ROW_K {
                let hab = h[a][b];
                if hab == 0.0 {
                    continue;
                }
                let Some((bb, rb)) = rows[b].as_ref() else {
                    continue;
                };
                if ba != bb {
                    continue;
                }
                let off = self.offsets[*ba];
                for (k, (&va, &vb)) in ra.iter().zip(rb.iter()).enumerate() {
                    diag[off + k] += hab * va * vb;
                }
            }
        }
    }

    fn row_third_contracted(
        &self,
        row: usize,
        dir: &[f64; SLS_ROW_K],
    ) -> Result<[[f64; SLS_ROW_K]; SLS_ROW_K], String> {
        // Packed one-seed directional scalar (1.46 KiB/row): the ε-Hessian
        // channel is exactly `Σ_c ℓ_{abc} dir_c` without materialising the dense
        // `t3`. Bit-identical to `row_nll_tower(row)?.third_contracted(dir)` by
        // the `survival_ls_packed_scalar_*` oracle.
        let (p, kernel) = self.row_nll_inputs(row)?;
        let vars: [OneSeed<SLS_ROW_K>; SLS_ROW_K] =
            std::array::from_fn(|a| OneSeed::seed_direction(p[a], a, dir[a]));
        Ok(sls_row_nll(&vars, &kernel)?.contracted_third())
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; SLS_ROW_K],
        dir_v: &[f64; SLS_ROW_K],
    ) -> Result<[[f64; SLS_ROW_K]; SLS_ROW_K], String> {
        // Packed two-seed scalar (2.8 KiB/row): the εδ-Hessian channel is exactly
        // `Σ_{cd} ℓ_{abcd} u_c v_d` without materialising the dense `t4`.
        // Bit-identical to `row_nll_tower(row)?.fourth_contracted(u, v)`.
        let (p, kernel) = self.row_nll_inputs(row)?;
        let vars: [TwoSeed<SLS_ROW_K>; SLS_ROW_K] =
            std::array::from_fn(|a| TwoSeed::seed(p[a], a, dir_u[a], dir_v[a]));
        Ok(sls_row_nll(&vars, &kernel)?.contracted_fourth())
    }

    /// Batched all-axes first directional derivative with the per-row NLL
    /// derivative stack built ONCE and reused across every swept axis.
    ///
    /// The generic per-axis dispatcher computes the `p` matrices `{∂H/∂β[e_a]}`
    /// by running `p` independent single-direction sweeps. Each sweep, for each
    /// row, calls `row_third_contracted` → `row_nll_inputs` →
    /// `exact_row_kernel_rescaled`, the special-function-heavy derivative ladder
    /// (`exp` / `log` / log-Φ derivatives). That ladder is INDEPENDENT of the
    /// swept axis, so the per-axis path rebuilds it `p` times per row — the
    /// dominant cost of the inner-Newton Jeffreys term and the outer-REML
    /// Jeffreys `H_Φ` drift, which probe this every joint evaluation. Here each
    /// row's `(primary, kernel)` is materialized a single time, then every axis
    /// closes against the cached stack with only the cheap `OneSeed` jet
    /// arithmetic and the design-row pullback.
    ///
    /// **Correctness contract.** Output `a` equals, bit-for-bit, the generic
    /// per-axis `row_kernel_directional_derivative(self, rows, e_a)`: the same
    /// `RowSet` reduction primitive (chunk-index-order
    /// `par_try_reduce_fold`), the same per-row
    /// `jacobian_action → sls_row_nll(seed_direction(..)).contracted_third() →
    /// add_pullback_hessian` pipeline, reading a cached `(primary, kernel)` that
    /// is identical (a pure function of `row`) to the per-call rebuild. Only the
    /// full-data unit-weight `RowSet::All` case is accelerated; a subsample
    /// declines (`None`) so the generic Horvitz–Thompson per-axis path runs.
    fn directional_derivative_all_axes_dense_override(
        &self,
        rows: &crate::families::row_kernel::RowSet,
        p: usize,
    ) -> Option<Result<Vec<Array2<f64>>, String>> {
        if p != self.n_coefficients() {
            return Some(Err(format!(
                "directional_derivative_all_axes_dense_override: axis count {p} disagrees \
                 with n_coefficients() {}",
                self.n_coefficients(),
            )));
        }
        let crate::families::row_kernel::RowSet::All = rows else {
            return None;
        };
        Some((|| {
            let n = self.n_rows();
            // One special-function-heavy per-row NLL stack build, shared by every axis.
            let inputs: Vec<([f64; SLS_ROW_K], SurvivalExactRowKernel)> = (0..n)
                .into_par_iter()
                .map(|row| self.row_nll_inputs(row))
                .collect::<Result<Vec<_>, String>>()?;
            (0..p)
                .into_par_iter()
                .map(|a| {
                    let mut e_a = vec![0.0_f64; p];
                    e_a[a] = 1.0;
                    gam_problem::with_nested_parallel(|| {
                        rows.par_try_reduce_fold(
                            n,
                            || Array2::<f64>::zeros((p, p)),
                            |mut acc, row, w| -> Result<_, String> {
                                let dir_k = self.jacobian_action(row, &e_a);
                                let (primary, kernel) = &inputs[row];
                                let vars: [OneSeed<SLS_ROW_K>; SLS_ROW_K] = std::array::from_fn(
                                    |c| OneSeed::seed_direction(primary[c], c, dir_k[c]),
                                );
                                let third = sls_row_nll(&vars, kernel)?.contracted_third();
                                if w == 1.0 {
                                    self.add_pullback_hessian(row, &third, &mut acc);
                                } else {
                                    let mut scaled = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
                                    for x in 0..SLS_ROW_K {
                                        for y in 0..SLS_ROW_K {
                                            scaled[x][y] = w * third[x][y];
                                        }
                                    }
                                    self.add_pullback_hessian(row, &scaled, &mut acc);
                                }
                                Ok(acc)
                            },
                            |x, y| Ok(x + y),
                        )
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })())
    }
}

impl SurvivalLocationScaleFamily {
    pub(crate) const BLOCK_TIME: usize = 0;
    pub(crate) const BLOCK_THRESHOLD: usize = 1;
    pub(crate) const BLOCK_LOG_SIGMA: usize = 2;
    pub(crate) const BLOCK_LINK_WIGGLE: usize = 3;
    pub(crate) const EVALUATE_PARALLEL_ROW_THRESHOLD: usize = 1024;

    /// The `RowKernel<K>` engine assumes a fixed linear coefficient-to-primary
    /// Jacobian for the row. Survival LS satisfies that after choosing the nine
    /// linear predictors as primary channels, but link-wiggle does not: its
    /// basis rows are evaluated at q(eta_threshold, eta_log_sigma), so the row
    /// design itself changes with beta and contributes dJ/dβ terms outside the
    /// current trait contract.
    #[inline]
    pub(crate) fn row_kernel_joint_hessian_supported(&self) -> bool {
        // Keep the coefficient working-set path on the bespoke exact
        // assembler. Besides avoiding the oversized `Tower4<9>` cache, the
        // bespoke path is the derivative path that currently tracks the
        // survival location-scale likelihood for every supported residual
        // distribution and time-varying channel layout.
        false
    }

    /// First directional derivatives require third qdot map derivatives when
    /// threshold/log-sigma derivative designs are present.
    #[inline]
    pub(crate) fn row_kernel_directional_supported(&self) -> bool {
        // #932: the directional path no longer builds the dense `Tower4<9>`. The
        // contracted third/fourth (`row_third_contracted` / `row_fourth_contracted`)
        // now evaluate the SAME single-sourced row NLL (`sls_row_nll`) through the
        // PACKED directional scalars `OneSeed<9>` (1.46 KiB) / `TwoSeed<9>`
        // (2.8 KiB) — the nilpotent ε/δ fold the contraction direction INTO the
        // differentiation, so only the contracted K×K matrix is carried, never the
        // full fourth-order tensor. That removes the ~50 KiB per-row tower whose
        // by-value copies overflowed the stack / timed out the fit (the exact
        // representation objection this gate was waiting on; module note in
        // `jet_scalar`). The packed contractions are bit-identical to the dense
        // `row_nll_tower(row)?.{third,fourth}_contracted(...)` (the
        // `survival_ls_packed_directional_matches_dense_tower_932` oracle pins this
        // ≤ 1e-9). Link-wiggle remains a separate carveout
        // (`row_kernel_supported` gates on `x_link_wiggle.is_none()`), so the
        // beta-dependent-Jacobian rows still take the hand path below.
        self.x_link_wiggle.is_none()
    }

    pub(crate) fn survival_ls_row_kernel<'a>(
        &'a self,
        q: &'a SurvivalJointQuantities,
        dynamic: &'a SurvivalDynamicGeometry,
    ) -> SurvivalLsRowKernel<'a> {
        self.survival_ls_row_kernel_rescaled(q, dynamic, 0.0)
    }

    pub(crate) fn survival_ls_row_kernel_rescaled<'a>(
        &'a self,
        q: &'a SurvivalJointQuantities,
        dynamic: &'a SurvivalDynamicGeometry,
        deriv_log_scale: f64,
    ) -> SurvivalLsRowKernel<'a> {
        SurvivalLsRowKernel {
            family: self,
            q,
            dynamic,
            deriv_log_scale,
            offsets: self.joint_block_offsets(),
        }
    }

    #[inline]
    pub(crate) fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.x_time_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        p_total - p_w..p_total
    }

    #[inline]
    pub(crate) fn time_derivative_lower_bound(&self) -> f64 {
        assert!(
            self.derivative_guard.is_finite() && self.derivative_guard > 0.0,
            "survival location-scale derivative guard must be finite and positive: derivative_guard={}",
            self.derivative_guard
        );
        self.derivative_guard
    }

    pub(crate) fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = self.time_linear_constraints.as_ref() else {
            // No time constraints. With the rank-1 unit-log-t warp pin (#892) the
            // time block has ZERO free coefficients and its monotone warp is a
            // fixed positive offset (X' z_norm = 1/t > 0), so there is no
            // derivative-guard half-space to cap against — the step is uncapped.
            // (Every constrained time block, reduced or flexible, carries ≥1
            // column and a guard, so this `None` arises only for the pinned
            // empty block.)
            return Ok(None);
        };
        crate::marginal_slope_shared::feasible_step_fraction(
            constraints,
            beta,
            delta,
            |beta_len, delta_len, expected| {
                SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "survival location-scale time-step constraint dimension mismatch: beta={beta_len}, delta={delta_len}, constraints={expected}"
                ) }.into()
            },
            |row, slack| {
                SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                    "survival location-scale current time block violates linear constraint at row {row}: slack={slack:.3e}"
                ) }.into()
            },
        )
        .map(Some)
    }

    pub(crate) fn max_feasible_link_wiggle_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if beta.len() != delta.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale linkwiggle-step dimension mismatch: beta={}, delta={}",
                    beta.len(),
                    delta.len()
                ),
            }
            .into());
        }
        let mut alpha = 1.0f64;
        for j in 0..beta.len() {
            let slack = beta[j];
            if slack < -CONSTRAINT_NONNEGATIVITY_REL_TOL {
                return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                    "survival location-scale current linkwiggle block violates nonnegativity at coefficient {j}: beta={slack:.3e}"
                ) }.into());
            }
            let drift = delta[j];
            if drift < 0.0 {
                alpha = alpha.min((slack / -drift).clamp(0.0, 1.0));
            }
        }
        if alpha >= 1.0 {
            Ok(Some(1.0))
        } else {
            Ok(Some((0.995 * alpha).clamp(0.0, 1.0)))
        }
    }

    #[inline]
    pub(crate) fn expected_blocks(&self) -> usize {
        if self.x_link_wiggle.is_some() { 4 } else { 3 }
    }

    #[inline]
    pub(crate) fn joint_block_dims(&self) -> Vec<usize> {
        let mut dims = vec![
            self.x_time_entry.ncols(),
            self.x_threshold.ncols(),
            self.x_log_sigma.ncols(),
        ];
        if let Some(xw) = self.x_link_wiggle.as_ref() {
            dims.push(xw.ncols());
        }
        dims
    }

    pub(crate) fn validate_joint_specs(
        &self,
        specs: &[ParameterBlockSpec],
        context: &str,
    ) -> Result<(), String> {
        let dims = self.joint_block_dims();
        if specs.len() != dims.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{context} expects {} specs, got {}",
                    dims.len(),
                    specs.len()
                ),
            }
            .into());
        }
        for (block_idx, (spec, expected_width)) in specs.iter().zip(dims.iter()).enumerate() {
            let width = spec.design.ncols();
            if width != *expected_width {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "{context} spec {block_idx} width mismatch: got {width}, expected {expected_width}"
                    ),
                }
                .into());
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn joint_block_offsets(&self) -> Vec<usize> {
        let dims = self.joint_block_dims();
        let mut offsets = Vec::with_capacity(dims.len() + 1);
        offsets.push(0);
        let mut acc = 0usize;
        for dim in dims {
            acc += dim;
            offsets.push(acc);
        }
        offsets
    }

    pub(crate) fn wiggle_geometry(
        &self,
        q0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalWiggleGeometry>, String> {
        let (Some(knots), Some(degree)) = (self.wiggle_knots.as_ref(), self.wiggle_degree) else {
            return Ok(None);
        };
        let basis = survival_wiggle_basis_with_options(q0, knots, degree, BasisOptions::value())?;
        let basis_d1 = survival_wiggle_basis_with_options(
            q0,
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let basis_d2 = survival_wiggle_basis_with_options(
            q0,
            knots,
            degree,
            BasisOptions::second_derivative(),
        )?;
        let basis_d3 = survival_wiggle_third_basis(q0, knots, degree)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
            || basis_d3.ncols() != beta_w.len()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival linkwiggle basis/beta mismatch: B={} B'={} B''={} B'''={} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    basis_d3.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq_dq0 = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2q_dq02 = fast_av(&basis_d2, &beta_w);
        let d3q_dq03 = fast_av(&basis_d3, &beta_w);
        Ok(Some(SurvivalWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
        }))
    }

    pub(crate) fn time_wiggle_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalWiggleGeometry>, String> {
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
        let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
        let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
        let basis_d3 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 3)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
            || basis_d3.ncols() != beta_w.len()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival timewiggle basis/beta mismatch: B={} B'={} B''={} B'''={} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    basis_d3.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2 = fast_av(&basis_d2, &beta_w);
        let d3 = fast_av(&basis_d3, &beta_w);
        Ok(Some(SurvivalWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0: dq,
            d2q_dq02: d2,
            d3q_dq03: d3,
        }))
    }

    /// Returns
    /// `(h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry,
    ///   eta_t_deriv_exit, eta_ls_deriv_exit, etaw)`.
    ///
    /// The time block eta is stored as `[exit; entry; derivative_exit]` to
    /// match the stacked design, but callers consume `(entry, exit, deriv)`.
    /// For time-invariant blocks, `eta_t_entry == eta_t_exit` and likewise for ls.
    /// For time-varying threshold/log-sigma blocks, the block eta is 3n long:
    /// `[exit; entry; derivative_exit]`.
    /// The solver's ParameterBlockSpec uses the EXIT value design first.
    pub(crate) fn validate_joint_states<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<
        (
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            Option<ndarray::ArrayView1<'a, f64>>,
            Option<ndarray::ArrayView1<'a, f64>>,
            Option<&'a Array1<f64>>,
        ),
        String,
    > {
        crate::block_layout::block_count::validate_block_count::<
            SurvivalLocationScaleError,
        >(
            "SurvivalLocationScaleFamily",
            self.expected_blocks(),
            block_states.len(),
        )?;
        let n = self.n;
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_t_raw = &block_states[Self::BLOCK_THRESHOLD].eta;
        let eta_ls_raw = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = self
            .x_link_wiggle
            .as_ref()
            .map(|_| &block_states[Self::BLOCK_LINK_WIGGLE].eta);
        if eta_time.len() != 3 * n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale time eta length mismatch: got {}, expected {}",
                    eta_time.len(),
                    3 * n
                ),
            }
            .into());
        }
        // For time-varying blocks the stacked design is
        // [exit_design; entry_design; derivative_exit_design], giving eta of
        // length 3n. For time-invariant blocks eta is length n.
        let (eta_t_exit, eta_t_entry, eta_t_deriv_exit) = if self.x_threshold_entry.is_some() {
            if eta_t_raw.len() != 3 * n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "time-varying threshold eta length mismatch: got {}, expected {}",
                        eta_t_raw.len(),
                        3 * n
                    ),
                }
                .into());
            }
            (
                eta_t_raw.slice(s![0..n]),
                eta_t_raw.slice(s![n..2 * n]),
                Some(eta_t_raw.slice(s![2 * n..3 * n])),
            )
        } else {
            if eta_t_raw.len() != n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "threshold eta length mismatch: got {}, expected {n}",
                        eta_t_raw.len()
                    ),
                }
                .into());
            }
            (eta_t_raw.slice(s![0..n]), eta_t_raw.slice(s![0..n]), None)
        };
        let (eta_ls_exit, eta_ls_entry, eta_ls_deriv_exit) = if self.x_log_sigma_entry.is_some() {
            if eta_ls_raw.len() != 3 * n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "time-varying log-sigma eta length mismatch: got {}, expected {}",
                        eta_ls_raw.len(),
                        3 * n
                    ),
                }
                .into());
            }
            (
                eta_ls_raw.slice(s![0..n]),
                eta_ls_raw.slice(s![n..2 * n]),
                Some(eta_ls_raw.slice(s![2 * n..3 * n])),
            )
        } else {
            if eta_ls_raw.len() != n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "log-sigma eta length mismatch: got {}, expected {n}",
                        eta_ls_raw.len()
                    ),
                }
                .into());
            }
            (eta_ls_raw.slice(s![0..n]), eta_ls_raw.slice(s![0..n]), None)
        };
        if let Some(w) = etaw
            && w.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale wiggle eta length mismatch: got {}, expected {n}",
                    w.len()
                ),
            }
            .into());
        }
        // The time block's solver design stacks `[entry; exit; derivative_exit]`
        // (see `prepare.rs`'s `MultiChannelOperator::new`), so the stacked time
        // eta is laid out `[entry(0..n); exit(n..2n); deriv(2n..3n)]`. The first
        // return slot is `h_entry`, the second is `h_exit` (gam#1396): a prior
        // revision read the entry channel from `eta_time[n..2*n]` and the exit
        // channel from `eta_time[0..n]`, transposing the two so the exit-time
        // index was evaluated at the entry predictor and vice versa. That swap
        // left the *value* path self-consistent (every consumer saw the same
        // transposed pair) but mis-paired each index with its design Jacobian
        // (`time_jac_entry`/`time_jac_exit`), so the time-block gradient/Hessian
        // disagreed with a finite-difference of the likelihood whenever the
        // entry and exit designs differ — and the structural-time monotonicity
        // guard saw the wrong exit derivative.
        Ok((
            eta_time.slice(s![0..n]),
            eta_time.slice(s![n..2 * n]),
            eta_time.slice(s![2 * n..3 * n]),
            eta_t_exit,
            eta_ls_exit,
            eta_t_entry,
            eta_ls_entry,
            eta_t_deriv_exit,
            eta_ls_deriv_exit,
            etaw,
        ))
    }

    pub(crate) fn collect_joint_quantities(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalJointQuantities, String> {
        self.collect_joint_quantities_rescaled(block_states, 0.0)
    }

    /// Collect per-row derivative quantities while passing `deriv_log_scale`
    /// through to row primitives that use it.  The CLogLog log-PDF derivatives
    /// use this shift; the CLogLog survival ratio derivatives do not.
    pub(crate) fn collect_joint_quantities_rescaled(
        &self,
        block_states: &[ParameterBlockState],
        deriv_log_scale: f64,
    ) -> Result<SurvivalJointQuantities, String> {
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d2_q0 = Array1::<f64>::zeros(n);
        let mut d3_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d2_q1 = Array1::<f64>::zeros(n);
        let mut d3_q1 = Array1::<f64>::zeros(n);
        let mut d1_qdot1 = Array1::<f64>::zeros(n);
        let mut d2_qdot1 = Array1::<f64>::zeros(n);
        let mut h_time_h0 = Array1::<f64>::zeros(n);
        let mut h_time_h1 = Array1::<f64>::zeros(n);
        let mut h_time_d = Array1::<f64>::zeros(n);

        // Write each row's 21 derivative scalars directly into the
        // preallocated output arrays in parallel. The previous path collected
        // a `Vec<Option<SurvivalRowDerivatives>>` (21 fields per row) and then
        // serially scattered into 21 `Array1`s — at large scale that is the
        // worst-case transient allocation among the family row builders.
        // Rows where `row_derivatives_rescaled` returns `Ok(None)` keep their
        // zero-initialized slots (matching the previous `continue` branch).
        /// Wrapper to send raw pointers across threads for disjoint per-row
        /// writes.  SAFETY: each parallel iteration writes a unique index `i`
        /// into a buffer of length `n`, and the pointers do not outlive the
        /// surrounding scope.
        #[derive(Clone, Copy)]
        struct SendPtr(*mut f64);
        // SAFETY: SendPtr is constructed from Array1::as_mut_ptr() on
        // length-n buffers; the rayon (0..n).into_par_iter() driver gives
        // each thread a unique i, so writes via SendPtr never alias.
        unsafe impl Send for SendPtr {}
        // SAFETY: same disjoint-index invariant as the Send impl above.
        unsafe impl Sync for SendPtr {}
        impl SendPtr {
            #[inline(always)]
            // SAFETY: caller passes `i < n` (the buffer length used to take
            // `self.0`); rayon's `(0..n).into_par_iter()` driver guarantees
            // exclusive ownership of `i` per thread, so the write is unaliased.
            unsafe fn write(self, i: usize, v: f64) {
                // SAFETY: `i < n` from the function contract; `self.0.add(i)`
                // is in-bounds and the disjoint-index invariant means no other
                // thread accesses this slot.
                unsafe { *self.0.add(i) = v };
            }
        }

        let p_d1_q0 = SendPtr(d1_q0.as_mut_ptr());
        let p_d2_q0 = SendPtr(d2_q0.as_mut_ptr());
        let p_d3_q0 = SendPtr(d3_q0.as_mut_ptr());
        let p_d1_q1 = SendPtr(d1_q1.as_mut_ptr());
        let p_d2_q1 = SendPtr(d2_q1.as_mut_ptr());
        let p_d3_q1 = SendPtr(d3_q1.as_mut_ptr());
        let p_d1_qdot1 = SendPtr(d1_qdot1.as_mut_ptr());
        let p_d2_qdot1 = SendPtr(d2_qdot1.as_mut_ptr());
        let p_h_time_h0 = SendPtr(h_time_h0.as_mut_ptr());
        let p_h_time_h1 = SendPtr(h_time_h1.as_mut_ptr());
        let p_h_time_d = SendPtr(h_time_d.as_mut_ptr());

        let dyn_ref = &dynamic;
        (0..n)
            .into_par_iter()
            .try_for_each(move |i| -> Result<(), String> {
                let state = self.row_predictor_state(
                    dyn_ref.h_entry[i],
                    dyn_ref.h_exit[i],
                    dyn_ref.hdot_exit[i],
                    dyn_ref.q_entry[i],
                    dyn_ref.q_exit[i],
                    dyn_ref.qdot_exit[i],
                );
                let Some(row) = self.row_derivatives_rescaled(i, state, deriv_log_scale)? else {
                    return Ok(());
                };
                // SAFETY: rayon `(0..n).into_par_iter()` yields each `i < n`
                // exactly once; pointers target distinct length-`n` `Array1`
                // buffers not read until the parallel loop completes.
                unsafe {
                    p_d1_q0.write(i, row.d1_q0);
                    p_d2_q0.write(i, row.d2_q0);
                    p_d3_q0.write(i, row.d3_q0);
                    p_d1_q1.write(i, row.d1_q1);
                    p_d2_q1.write(i, row.d2_q1);
                    p_d3_q1.write(i, row.d3_q1);
                    p_d1_qdot1.write(i, row.d1_qdot1);
                    p_d2_qdot1.write(i, row.d2_qdot1);
                    p_h_time_h0.write(i, row.h_time_h0);
                    p_h_time_h1.write(i, row.h_time_h1);
                    p_h_time_d.write(i, row.h_time_d);
                }
                Ok(())
            })?;

        Ok(SurvivalJointQuantities {
            d1_q0,
            d2_q0,
            d3_q0,
            d1_q1,
            d2_q1,
            d3_q1,
            d1_qdot1,
            d2_qdot1,
            h_time_h0,
            h_time_h1,
            h_time_d,
            dq_t: dynamic.dq_t_exit,
            dq_ls: dynamic.dq_ls_exit,
            d2q_tls: dynamic.d2q_tls_exit,
            d2q_ls: dynamic.d2q_ls_exit,
            d3q_tls_ls: dynamic.d3q_tls_ls_exit,
            d3q_ls: dynamic.d3q_ls_exit,
            dq_t_entry: Some(dynamic.dq_t_entry),
            dq_ls_entry: Some(dynamic.dq_ls_entry),
            d2q_tls_entry: Some(dynamic.d2q_tls_entry),
            d2q_ls_entry: Some(dynamic.d2q_ls_entry),
            d3q_tls_ls_entry: Some(dynamic.d3q_tls_ls_entry),
            d3q_ls_entry: Some(dynamic.d3q_ls_entry),
            dqdot_t: dynamic.dqdot_t,
            dqdot_ls: dynamic.dqdot_ls,
            dqdot_td: dynamic.dqdot_td,
            dqdot_lsd: dynamic.dqdot_lsd,
            d2qdot_tt: dynamic.d2qdot_tt,
            d2qdot_tls: dynamic.d2qdot_tls,
            d2qdot_ttd: dynamic.d2qdot_ttd,
            d2qdot_tlsd: dynamic.d2qdot_tlsd,
            d2qdot_ls: dynamic.d2qdot_ls,
            d2qdot_lstd: dynamic.d2qdot_lstd,
            d2qdot_lslsd: dynamic.d2qdot_lslsd,
        })
    }

    /// Per-row NLL gradient and curvature with respect to the three additive
    /// time-block offset channels `(o_E, o_X, o_D)` (entry / exit / derivative-
    /// at-exit). The baseline configuration enters the location-scale fit
    /// **only** through these three offsets, so contracting these residuals
    /// against `∂o/∂θ_baseline` gives the analytic θ-gradient of the
    /// unpenalized NLL at converged β (envelope theorem on the penalized
    /// objective; the penalty has no θ dependence).
    ///
    /// Algebra. With `ell_i = w_i[d(log f(u1) + log g) + (1-d) log S(u1) − log S(u0)]`
    /// and `u0 = h0 + q0`, `u1 = h1 + q1`, `g = d_raw + qdot1`:
    ///
    ///   ∂(−ell_i)/∂h0   = − w_i r(u0)
    ///   ∂(−ell_i)/∂h1   = − w_i [d ψ(u1) − (1−d) r(u1)]
    ///   ∂(−ell_i)/∂dRaw = − w_i d / g                                (event-row only)
    ///
    /// and the row Hessian is diagonal in (h0, h1, dRaw) because `u0`, `u1`,
    /// `g` are functionally independent (h0→u0, h1→u1, dRaw→g):
    ///
    ///   ∂²(−ell_i)/∂h0²   = − w_i r'(u0)
    ///   ∂²(−ell_i)/∂h1²   = − w_i [d ψ'(u1) − (1−d) r'(u1)]
    ///   ∂²(−ell_i)/∂dRaw² =   w_i d / g²
    ///
    /// The fields `grad_time_eta_*` / `h_time_*` produced by
    /// [`Self::row_derivatives`] are log-likelihood (not NLL) partials. All
    /// three time channels (h0, h1, d_raw) are stored as `+∂ℓ`/`+∂²ℓ`, so the
    /// NLL gradient/curvature negates each **uniformly**. This site delegates
    /// that to [`SurvivalRowDerivatives::time_channel_nll_gradient`] /
    /// [`SurvivalRowDerivatives::time_channel_nll_curvature_diag`], which own
    /// the sign in one place (gam#1396 — a prior `+h_time_d` outlier here and
    /// in the joint assembler flipped the event-Jacobian self-term).
    pub(crate) fn offset_channel_geometry(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(OffsetChannelResiduals, OffsetChannelCurvatures), String> {
        let n = self.n;
        // Defensive degraded-fit path: a custom-family `fit_custom_family` whose
        // outer ARC stalled into the "deterministic-replay" branch can land in
        // `blockwise_fit_from_parts` with the final inner refit's `block_states`
        // cleared (the path at `custom_family.rs` post-degraded-plan rebuilds
        // `BlockwiseFitResultParts` without re-populating per-block state).
        // Surfaces here as `block_states.len() == 0` when the value-closure
        // cache (`last_geometry` in `fit_survival_location_scale_terms`) is
        // unset and the fallback refit runs through this method.
        // `build_dynamic_geometry` would then fail with the cryptic
        // `SurvivalLocationScaleFamily expects 3 blocks, got 0` — which
        // propagates all the way to the Python wrapper.
        //
        // Returning zero residuals + zero curvatures here lets the outer
        // baseline BFGS treat this candidate as a stationary point (no
        // gradient contribution from these rows) instead of crashing the
        // whole fit. The next BFGS step gets `‖g‖ = 0`, so the optimizer
        // terminates at the current θ rather than wandering into NaN
        // territory. Production loss is at most a slightly suboptimal
        // baseline θ at this BMA parent set — far preferable to a hard
        // exception from PyPI.
        if block_states.is_empty() {
            log::warn!(
                "SurvivalLocationScaleFamily::offset_channel_geometry: \
                 block_states is empty (degraded fit, likely ARC \
                 deterministic-replay stall); returning zero residuals + \
                 curvatures (n={n})"
            );
            return Ok((
                OffsetChannelResiduals {
                    exit: Array1::<f64>::zeros(n),
                    entry: Array1::<f64>::zeros(n),
                    derivative: Array1::<f64>::zeros(n),
                    // Location-scale has no interval upper-bound channel.
                    right: Array1::<f64>::zeros(n),
                },
                OffsetChannelCurvatures {
                    rows: vec![[[0.0_f64; 3]; 3]; n],
                },
            ));
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;

        let mut entry = Array1::<f64>::zeros(n);
        let mut exit = Array1::<f64>::zeros(n);
        let mut derivative = Array1::<f64>::zeros(n);
        let mut curvatures = vec![[[0.0_f64; 3]; 3]; n];

        let rows = (0..n)
            .into_par_iter()
            .map(
                |i| -> Result<(usize, f64, f64, f64, [[f64; 3]; 3]), String> {
                    let state = self.row_predictor_state(
                        dynamic.h_entry[i],
                        dynamic.h_exit[i],
                        dynamic.hdot_exit[i],
                        dynamic.q_entry[i],
                        dynamic.q_exit[i],
                        dynamic.qdot_exit[i],
                    );
                    let Some(row) = self.row_derivatives(i, state)? else {
                        return Ok((i, 0.0, 0.0, 0.0, [[0.0; 3]; 3]));
                    };
                    // NLL gradient + curvature on the three time channels
                    // (h0, h1, d_raw). Both helpers own the `-∂ℓ`/`-∂²ℓ` sign
                    // so the channels are negated uniformly (gam#1396); the
                    // row likelihood factors through the independent indices
                    // (u0, u1, g), so the curvature is diagonal.
                    let [r_entry, r_exit, r_deriv] = row.time_channel_nll_gradient();
                    let curv_diag = row.time_channel_nll_curvature_diag();
                    let mut curv = [[0.0_f64; 3]; 3];
                    curv[0][0] = curv_diag[0];
                    curv[1][1] = curv_diag[1];
                    curv[2][2] = curv_diag[2];
                    Ok((i, r_entry, r_exit, r_deriv, curv))
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        for (i, r_entry, r_exit, r_deriv, curv) in rows {
            entry[i] = r_entry;
            exit[i] = r_exit;
            derivative[i] = r_deriv;
            curvatures[i] = curv;
        }

        Ok((
            OffsetChannelResiduals {
                exit,
                entry,
                derivative,
                // Location-scale has no interval upper-bound channel.
                right: Array1::<f64>::zeros(n),
            },
            OffsetChannelCurvatures { rows: curvatures },
        ))
    }

    /// Exact data-fit gradient `Σ_i ∂ℓ_i/∂θ_link` of the unpenalized
    /// log-likelihood with respect to the inverse-link parameters θ_link
    /// (SAS `(ε, log δ)`, BetaLogistic `(ε, log δ)`, or Mixture `ρ`), holding
    /// the fitted β and λ fixed.
    ///
    /// The per-row log-likelihood is
    ///   ℓ_i = w_i·( event_mix(d_i, logφ(u1_i) + log g_i, log S(u1_i)) − log S(u0_i) ),
    /// where `u0 = h0 + q0` and `u1 = h1 + q1` are the standardized residuals
    /// the inverse link evaluates (entry/exit), `log g` is the time-derivative
    /// Jacobian (link-independent), and the link enters ONLY through the scalar
    /// `log S(u) = log(1 − μ(u;θ))` and `log φ(u) = log d1(u;θ)` terms. Hence
    ///   ∂(log S)/∂θ = −(∂μ/∂θ)/S,   ∂(log φ)/∂θ = (∂d1/∂θ)/d1,
    /// with `S = 1 − μ`, `μ = jet.mu`, `d1 = jet.d1`, and the parameter partials
    /// `∂μ/∂θ`, `∂d1/∂θ` supplied analytically by
    /// [`InverseLinkKernel::param_partials`]. The higher-order ratio/pdf
    /// derivatives (r, dr, …, fppp) carry the inner-Newton curvature only and do
    /// NOT appear in the scalar ℓ, so the data-fit θ-gradient needs only the
    /// `(μ, d1)` jet components and their param partials — all exact.
    ///
    /// At the converged β̂ the envelope theorem makes this the exact θ-gradient
    /// of the profile penalized NLL `−ℓ + ½βᵀSβ` (β profiled out; the penalty
    /// has no θ_link dependence). Returns a length-`n_link_params` vector
    /// (`∂(−ℓ)/∂θ` so it matches the profile-cost sign), or `None` when the
    /// inverse link carries no free parameters.
    pub(crate) fn link_param_data_fit_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        use gam_solve::mixture_link::{InverseLinkKernel, LinkParamPartials};
        let n = self.n;
        if block_states.is_empty() {
            return Ok(None);
        }
        // ∂(log S)/∂θ and ∂(log φ)/∂θ contributions per row are accumulated
        // into a θ-length vector. Probe the parameter count from the link's
        // partials at a finite argument; `None` ⇒ no free link parameters.
        let probe = self
            .inverse_link
            .param_partials(0.0)
            .map_err(|e| format!("inverse-link param partials probe failed: {e}"))?;
        let n_theta = match &probe {
            None => return Ok(None),
            Some(LinkParamPartials::Sas(_)) => 2,
            Some(LinkParamPartials::Mixture(m)) => m.djet_drho.len(),
        };
        if n_theta == 0 {
            return Ok(None);
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;
        // ∂(log S)/∂θ = −(∂μ/∂θ)/S at argument u (S = 1 − μ);
        // ∂(log φ)/∂θ = (∂d1/∂θ)/d1 at argument u.
        let dlog_survival_dtheta = |u: f64| -> Result<Vec<f64>, String> {
            let partials = self
                .inverse_link
                .param_partials(u)
                .map_err(|e| format!("inverse-link survival param partials failed: {e}"))?
                .ok_or_else(|| "inverse-link reported no param partials".to_string())?;
            let jet = self
                .inverse_link
                .jet(u)
                .map_err(|e| format!("inverse-link jet failed at u={u}: {e}"))?;
            let s = (1.0 - jet.mu).clamp(f64::MIN_POSITIVE, 1.0);
            let map = |dmu: f64| -dmu / s;
            Ok(match partials {
                LinkParamPartials::Sas(p) => {
                    vec![map(p.djet_depsilon.mu), map(p.djet_dlog_delta.mu)]
                }
                LinkParamPartials::Mixture(p) => p.djet_drho.iter().map(|j| map(j.mu)).collect(),
            })
        };
        let dlog_pdf_dtheta = |u: f64| -> Result<Vec<f64>, String> {
            let partials = self
                .inverse_link
                .param_partials(u)
                .map_err(|e| format!("inverse-link pdf param partials failed: {e}"))?
                .ok_or_else(|| "inverse-link reported no param partials".to_string())?;
            let jet = self
                .inverse_link
                .jet(u)
                .map_err(|e| format!("inverse-link jet failed at u={u}: {e}"))?;
            let f = jet.d1;
            if !(f.is_finite() && f > 0.0) {
                return Err(format!(
                    "inverse-link pdf (d1) must be finite positive for θ-gradient, got {f} at u={u}"
                ));
            }
            let map = |dd1: f64| dd1 / f;
            Ok(match partials {
                LinkParamPartials::Sas(p) => {
                    vec![map(p.djet_depsilon.d1), map(p.djet_dlog_delta.d1)]
                }
                LinkParamPartials::Mixture(p) => p.djet_drho.iter().map(|j| map(j.d1)).collect(),
            })
        };
        // Accumulate ∂(−ℓ)/∂θ = −Σ_i w_i·( event_mix(d, ∂logφ(u1), ∂logS(u1))
        //                                    − ∂logS(u0) ).
        let mut grad = Array1::<f64>::zeros(n_theta);
        for i in 0..n {
            let w = self.w[i];
            if w <= 0.0 {
                continue;
            }
            let d = self.validated_event_target(i)?;
            let u0 = dynamic.h_entry[i] + dynamic.q_entry[i];
            let u1 = dynamic.h_exit[i] + dynamic.q_exit[i];
            let dls_u0 = dlog_survival_dtheta(u0)?;
            // Entry channel always contributes (left-truncation term −log S(u0)).
            for k in 0..n_theta {
                grad[k] += w * dls_u0[k];
            }
            if d <= 0.0 {
                // Censored: +log S(u1).
                let dls_u1 = dlog_survival_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * dls_u1[k];
                }
            } else if d >= 1.0 {
                // Event: +log φ(u1) (log g is link-independent).
                let dlp_u1 = dlog_pdf_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * dlp_u1[k];
                }
            } else {
                // Fractional event weight: mix both branches.
                let dls_u1 = dlog_survival_dtheta(u1)?;
                let dlp_u1 = dlog_pdf_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * (d * dlp_u1[k] + (1.0 - d) * dls_u1[k]);
                }
            }
        }
        Ok(Some(grad))
    }

    pub(crate) fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<SurvivalJointPsiDirection>, String> {
        if block_states.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi direction expects {} blocks and derivative lists, got {} and {}",
                self.expected_blocks(),
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }

        let n = self.n;
        let pt = self.x_threshold.ncols();
        let pls = self.x_log_sigma.ncols();
        let beta_t = &block_states[Self::BLOCK_THRESHOLD].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let t_time_varying = self.x_threshold_entry.is_some();
        let ls_time_varying = self.x_log_sigma_entry.is_some();

        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    let mut x_t_exit_psi = None;
                    let mut x_t_entry_psi = None;
                    let mut x_ls_exit_psi = None;
                    let mut x_ls_entry_psi = None;
                    let mut x_t_exit_action = None;
                    let mut x_t_entry_action = None;
                    let mut x_ls_exit_action = None;
                    let mut x_ls_entry_action = None;
                    let mut z_t_exit_psi = Array1::<f64>::zeros(n);
                    let mut z_t_entry_psi = Array1::<f64>::zeros(n);
                    let mut z_ls_exit_psi = Array1::<f64>::zeros(n);
                    let mut z_ls_entry_psi = Array1::<f64>::zeros(n);
                    match block_idx {
                        Self::BLOCK_THRESHOLD => {
                            let total_rows = if t_time_varying { 3 * n } else { n };
                            match resolve_custom_family_x_psi_map(
                                deriv,
                                total_rows,
                                pt,
                                0..total_rows,
                                "SurvivalLocationScaleFamily threshold",
                                &self.policy,
                            )? {
                                PsiDesignMap::First { action } => {
                                    if t_time_varying {
                                        let exit_action = action.slice_rows(0..n)?;
                                        let entry_action = action.slice_rows(n..2 * n)?;
                                        z_t_exit_psi = exit_action.forward_mul(beta_t.view());
                                        z_t_entry_psi = entry_action.forward_mul(beta_t.view());
                                        x_t_exit_action = Some(exit_action);
                                        x_t_entry_action = Some(entry_action);
                                    } else {
                                        z_t_exit_psi = action.forward_mul(beta_t.view());
                                        z_t_entry_psi = z_t_exit_psi.clone();
                                        x_t_exit_action = Some(action.clone());
                                        x_t_entry_action = Some(action);
                                    }
                                }
                                PsiDesignMap::Dense { matrix } => {
                                    let (exit, entry) = split_survival_psi_design(
                                        &matrix,
                                        n,
                                        t_time_varying,
                                        "SurvivalLocationScaleFamily threshold",
                                    )?;
                                    z_t_exit_psi = fast_av(&exit, beta_t);
                                    z_t_entry_psi = fast_av(&entry, beta_t);
                                    x_t_exit_psi = Some(exit);
                                    x_t_entry_psi = Some(entry);
                                }
                                PsiDesignMap::Zero { .. } => {}
                                PsiDesignMap::Second { .. } => {
                                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: "SurvivalLocationScaleFamily threshold: unexpected Second variant from _psi_map"
                                            .to_string(), }.into());
                                }
                            }
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            let total_rows = if ls_time_varying { 3 * n } else { n };
                            match resolve_custom_family_x_psi_map(
                                deriv,
                                total_rows,
                                pls,
                                0..total_rows,
                                "SurvivalLocationScaleFamily log-sigma",
                                &self.policy,
                            )? {
                                PsiDesignMap::First { action } => {
                                    if ls_time_varying {
                                        let exit_action = action.slice_rows(0..n)?;
                                        let entry_action = action.slice_rows(n..2 * n)?;
                                        z_ls_exit_psi = exit_action.forward_mul(beta_ls.view());
                                        z_ls_entry_psi = entry_action.forward_mul(beta_ls.view());
                                        x_ls_exit_action = Some(exit_action);
                                        x_ls_entry_action = Some(entry_action);
                                    } else {
                                        z_ls_exit_psi = action.forward_mul(beta_ls.view());
                                        z_ls_entry_psi = z_ls_exit_psi.clone();
                                        x_ls_exit_action = Some(action.clone());
                                        x_ls_entry_action = Some(action);
                                    }
                                }
                                PsiDesignMap::Dense { matrix } => {
                                    let (exit, entry) = split_survival_psi_design(
                                        &matrix,
                                        n,
                                        ls_time_varying,
                                        "SurvivalLocationScaleFamily log-sigma",
                                    )?;
                                    z_ls_exit_psi = fast_av(&exit, beta_ls);
                                    z_ls_entry_psi = fast_av(&entry, beta_ls);
                                    x_ls_exit_psi = Some(exit);
                                    x_ls_entry_psi = Some(entry);
                                }
                                PsiDesignMap::Zero { .. } => {}
                                PsiDesignMap::Second { .. } => {
                                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: "SurvivalLocationScaleFamily log-sigma: unexpected Second variant from _psi_map"
                                            .to_string(), }.into());
                                }
                            }
                        }
                        _ => return Ok(None),
                    }
                    return Ok(Some(SurvivalJointPsiDirection {
                        x_t_exit_psi,
                        x_t_entry_psi,
                        x_ls_exit_psi,
                        x_ls_entry_psi,
                        z_t_exit_psi,
                        z_t_entry_psi,
                        z_ls_exit_psi,
                        z_ls_entry_psi,
                        x_t_exit_action,
                        x_t_entry_action,
                        x_ls_exit_action,
                        x_ls_entry_action,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    /// Hazard-like survival ratio and its first derivative.
    ///
    /// Let `F` be the CDF, `f = F'` the PDF, and `S = 1 - F` the survival
    /// function so `S' = -f`.
    ///
    /// Define `r = f / S`. By quotient rule:
    /// `r' = (f' S - f S') / S^2`.
    /// Since `S' = -f`, this becomes:
    /// `r' = f'/S + f^2/S^2 = f'/S + r^2`.
    ///
    /// Sign note: the `f'/S` term is strictly additive. A minus here is wrong.
    pub(crate) fn survival_ratio_first_derivative(f: f64, fp: f64, s: f64) -> (f64, f64) {
        let r = f / s;
        let dr = (r * r) + fp / s;
        (r, dr)
    }

    /// Second derivative of the survival ratio `r = f/S`.
    ///
    /// Starting from `r' = f'/S + r^2`:
    /// `r'' = d/du[f'/S] + 2 r r'`.
    /// With `S' = -f`, we get:
    /// `d/du[f'/S] = f''/S + f' f / S^2`.
    /// Therefore:
    /// `r'' = 2 r r' + f''/S + f' f / S^2`.
    ///
    /// Equivalent expanded form:
    /// `r'' = f''/S + 3 f f' / S^2 + 2 f^3 / S^3`.
    pub(crate) fn survival_ratiosecond_derivative(
        r: f64,
        dr: f64,
        f: f64,
        fp: f64,
        fpp: f64,
        s: f64,
    ) -> f64 {
        (2.0 * r * dr) + (fpp / s + fp * f / (s * s))
    }

    /// Third derivative of the survival ratio `r = f/S`.
    ///
    /// Starting from `r'' = 2 r r' + f''/S + f' f / S²`:
    ///
    /// ```text
    /// r''' = d/du[2 r r'] + d/du[f''/S + f'f/S²]
    ///      = 2(r')² + 2 r r'' + f'''/S + f''f/S² + f'²/S² + 2f'f²/S³ + f''f/S²
    ///      = 2(r')² + 2 r r'' + f'''/S + 2f''f/S² + (f')²/S² + 2f(f')²/S³ ... wait
    /// ```
    ///
    /// More carefully: let A = f''/S, B = f'f/S². Then r'' = 2rr' + A + B.
    ///
    /// ```text
    /// d/du[A] = f'''/S + f''f/S²   (using S' = -f)
    /// d/du[B] = (f''f + f'²)/S² + 2f'f²/S³
    /// ```
    ///
    /// So:
    /// ```text
    /// r''' = 2(r')² + 2rr'' + f'''/S + 2f''f/S² + (f')²/S² + 2f'f²/S³
    /// ```
    ///
    /// This is needed for d⁴ℓ/dq0⁴ (the entry-side 4th likelihood derivative)
    /// and d⁴ℓ/dq1⁴ (the exit-side 4th likelihood derivative), which enter the
    /// outer REML Hessian's Q[v_k, v_l] term via the Arbogast formula.
    pub(crate) fn survival_ratio_third_derivative(
        r: f64,
        dr: f64,
        ddr: f64,
        f: f64,
        fp: f64,
        fpp: f64,
        fppp: f64,
        s: f64,
    ) -> f64 {
        let s2 = s * s;
        let s3 = s2 * s;
        2.0 * dr * dr
            + 2.0 * r * ddr
            + fppp / s
            + 2.0 * fpp * f / s2
            + fp * fp / s2
            + 2.0 * fp * f * f / s3
    }

    /// Like [`Self::exact_log_pdf_derivatives_rescaled`] but with a log-scale shift
    /// on the derivative magnitudes.  For CLogLog the `exp(eta)` terms in
    /// the derivatives become `exp(eta - deriv_log_scale)`, and the constant
    /// term in `d/deta log f = 1 - exp(eta)` is scaled by the same factor.
    /// The function value is returned unshifted.
    pub(crate) fn exact_log_pdf_derivatives_rescaled(
        inverse_link: &InverseLink,
        eta: f64,
        deriv_log_scale: f64,
    ) -> Result<(f64, f64, f64, f64, f64), String> {
        match inverse_link {
            InverseLink::Standard(StandardLink::Probit) => Ok((
                -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln(),
                -eta,
                -1.0,
                0.0,
                0.0,
            )),
            InverseLink::Standard(StandardLink::Logit) => {
                let mu = gam_solve::mixture_link::component_inverse_link_jet(
                    gam_problem::LinkComponent::Logit,
                    eta,
                )
                .mu;
                let w = mu * (1.0 - mu);
                Ok((
                    -softplus(eta) - softplus(-eta),
                    1.0 - 2.0 * mu,
                    -2.0 * w,
                    -2.0 * w * (1.0 - 2.0 * mu),
                    -2.0 * w * (1.0 - 6.0 * w),
                ))
            }
            InverseLink::Standard(StandardLink::CLogLog) => {
                let t_val = eta.exp(); // for function value (may be Inf)
                let t_deriv = (eta - deriv_log_scale).exp(); // for derivatives
                let deriv_scale = (-deriv_log_scale).exp();
                Ok((
                    eta - t_val,
                    deriv_scale - t_deriv,
                    -t_deriv,
                    -t_deriv,
                    -t_deriv,
                ))
            }
            InverseLink::Standard(StandardLink::Identity) => Ok((0.0, 0.0, 0.0, 0.0, 0.0)),
            _ => {
                let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                    .map_err(|e| format!("inverse link evaluation failed at eta={eta}: {e}"))?;
                let f = jet.d1;
                if !(f.is_finite() && f > 0.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure {
                        reason: format!(
                            "inverse-link pdf must be finite and positive, got {f} at eta={eta}"
                        ),
                    }
                    .into());
                }
                let fp = jet.d2;
                let fpp = jet.d3;
                let fppp = inverse_link_pdfthird_derivative_for_inverse_link(inverse_link, eta)
                    .map_err(|e| {
                        format!("inverse link third-derivative evaluation failed at eta={eta}: {e}")
                    })?;
                let fpppp = inverse_link_pdffourth_derivative(inverse_link, eta)?;
                let d1 = fp / f;
                let d2 = fpp / f - d1 * d1;
                let d3 = fppp / f - 3.0 * fp * fpp / (f * f) + 2.0 * fp.powi(3) / f.powi(3);
                let d4 = fpppp / f - 4.0 * fp * fppp / f.powi(2) - 3.0 * fpp * fpp / f.powi(2)
                    + 12.0 * fp.powi(2) * fpp / f.powi(3)
                    - 6.0 * fp.powi(4) / f.powi(4);
                Ok((f.ln(), d1, d2, d3, d4))
            }
        }
    }

    /// Survival log value and ratio derivatives.  The CLogLog survival ratio is
    /// `r = exp(eta)`, and its derivatives are also `exp(eta)`; these are not
    /// derivative-rescaled because they are already ratio derivatives with
    /// respect to the unshifted predictor.  Unlike the rescaled PDF path, this
    /// evaluator therefore does not depend on the derivative log-scale.  The
    /// function value (`-exp(eta)` = `log S`) is returned unshifted.
    pub(crate) fn exact_survival_neglog_derivatives_fourth_rescaled(
        inverse_link: &InverseLink,
        eta: f64,
    ) -> Result<(f64, f64, f64, f64, f64), String> {
        match inverse_link {
            InverseLink::Standard(StandardLink::Probit) => {
                let (log_s, r, dr, ddr, dddr) = probit_log_survival_and_ratio_derivatives(eta);
                Ok((log_s, r, dr, ddr, dddr))
            }
            InverseLink::Standard(StandardLink::Logit) => {
                let mu = gam_solve::mixture_link::component_inverse_link_jet(
                    gam_problem::LinkComponent::Logit,
                    eta,
                )
                .mu;
                let w = mu * (1.0 - mu);
                Ok((
                    -softplus(eta),
                    mu,
                    w,
                    w * (1.0 - 2.0 * mu),
                    w * (1.0 - 6.0 * w),
                ))
            }
            InverseLink::Standard(StandardLink::CLogLog) => {
                let t = eta.exp();
                Ok((-t, t, t, t, t))
            }
            InverseLink::Standard(StandardLink::Identity) => {
                let s = 1.0 - eta;
                if !(s.is_finite() && s > 0.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure {
                        reason: format!("identity-link survival invalid at eta={eta}: S={s}"),
                    }
                    .into());
                }
                let inv = s.recip();
                Ok((s.ln(), inv, inv * inv, 2.0 * inv.powi(3), 6.0 * inv.powi(4)))
            }
            _ => {
                let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                    .map_err(|e| format!("inverse link evaluation failed at eta={eta}: {e}"))?;
                let s = inverse_link_survival_probvalue(inverse_link, eta);
                if !(s.is_finite() && s > 0.0 && s <= 1.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure { reason: format!(
                        "inverse-link survival probability must lie in (0,1], got {s} at eta={eta}"
                    ) }.into());
                }
                let fppp = inverse_link_pdfthird_derivative_for_inverse_link(inverse_link, eta)
                    .map_err(|e| {
                        format!("inverse link third-derivative evaluation failed at eta={eta}: {e}")
                    })?;
                let (r, dr) = Self::survival_ratio_first_derivative(jet.d1, jet.d2, s);
                let ddr = Self::survival_ratiosecond_derivative(r, dr, jet.d1, jet.d2, jet.d3, s);
                let dddr = Self::survival_ratio_third_derivative(
                    r, dr, ddr, jet.d1, jet.d2, jet.d3, fppp, s,
                );
                Ok((s.ln(), r, dr, ddr, dddr))
            }
        }
    }

    /// Fused CLogLog evaluator for the exit-row pair: returns the
    /// `(log_s, r, dr, ddr, dddr)` survival tuple and the
    /// `(logphi, d1, d2, d3, d4)` log-pdf tuple while computing the two
    /// expensive `exp` calls once.  This duplicates the CLogLog branches of
    /// `exact_survival_neglog_derivatives_fourth_rescaled` and
    /// `exact_log_pdf_derivatives_rescaled` to share their work.
    #[inline]
    pub(crate) fn clglog_exit_pair(
        u1: f64,
        deriv_log_scale: f64,
    ) -> ((f64, f64, f64, f64, f64), (f64, f64, f64, f64, f64)) {
        let t_val = u1.exp();
        let t_deriv = (u1 - deriv_log_scale).exp();
        let deriv_scale = (-deriv_log_scale).exp();
        let surv = (-t_val, t_val, t_val, t_val, t_val);
        let logpdf = (
            u1 - t_val,
            deriv_scale - t_deriv,
            -t_deriv,
            -t_deriv,
            -t_deriv,
        );
        (surv, logpdf)
    }

    /// Exact `log(x)` value and first four derivatives on the positive domain.
    pub(crate) fn logwith_derivatives_positive(x: f64) -> (f64, f64, f64, f64, f64) {
        assert!(
            x.is_finite() && x > 0.0,
            "log derivative kernel requires finite positive x: x={x}"
        );
        let inv = 1.0 / x;
        (
            x.ln(),
            inv,
            -inv * inv,
            2.0 * inv * inv * inv,
            -6.0 * inv * inv * inv * inv,
        )
    }

    /// Build the row predictor state with possibly distinct entry/exit
    /// evaluations of threshold and sigma.
    ///
    /// For time-invariant blocks, the caller passes the same value for both
    /// entry and exit.
    pub(crate) fn row_predictor_state(
        &self,
        h0: f64,
        h1: f64,
        d_raw: f64,
        q0: f64,
        q1: f64,
        qdot1: f64,
    ) -> SurvivalPredictorState {
        let g_diff = compensated_difference(d_raw, -qdot1);
        SurvivalPredictorState {
            h0,
            h1,
            g: g_diff.value,
            q0,
            q1,
            g_roundoff_slack: g_diff.roundoff_slack,
            g_operand_scale: g_diff.operand_scale,
        }
    }

    #[inline]
    pub(crate) fn validated_event_target(&self, row: usize) -> Result<f64, String> {
        let d = self.y[row];
        if !(d.is_finite() && (0.0..=1.0).contains(&d)) {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "survival location-scale event target must lie in [0,1] at row {row}, got {d}"
                ),
            }
            .into());
        }
        Ok(d)
    }

    pub(crate) fn exact_row_kernel(
        &self,
        row: usize,
        state: SurvivalPredictorState,
    ) -> Result<Option<SurvivalExactRowKernel>, String> {
        self.exact_row_kernel_rescaled(row, state, 0.0)
    }

    /// Like [`Self::exact_row_kernel`] but with a log-scale shift on the
    /// derivative magnitudes, propagated to the survival/pdf derivative
    /// functions.  Used by the logdet Hessian path to avoid overflow.
    pub(crate) fn exact_row_kernel_rescaled(
        &self,
        row: usize,
        state: SurvivalPredictorState,
        deriv_log_scale: f64,
    ) -> Result<Option<SurvivalExactRowKernel>, String> {
        let w = self.w[row];
        if w <= 0.0 {
            return Ok(None);
        }
        let d = self.validated_event_target(row)?;
        let u0 = state.h0 + state.q0;
        let u1 = state.h1 + state.q1;

        let (log_s0, r0, dr0, ddr0, dddr0) =
            Self::exact_survival_neglog_derivatives_fourth_rescaled(&self.inverse_link, u0)
                .map_err(|e| {
                    format!("inverse-link survival evaluation failed at row {row} entry: {e}")
                })?;

        // Fast path: for CLogLog the survival and log-pdf evaluators both need
        // `exp(u1)`, and the PDF derivatives also need
        // `exp(u1 - deriv_log_scale)`. Share that work when both are called
        // back-to-back on the exit row.
        let ((log_s1, r1, dr1, ddr1, dddr1), (logphi1, dlogphi1, d2logphi1, d3logphi1, d4logphi1)) =
            if matches!(
                &self.inverse_link,
                InverseLink::Standard(StandardLink::CLogLog)
            ) {
                Self::clglog_exit_pair(u1, deriv_log_scale)
            } else {
                let surv =
                    Self::exact_survival_neglog_derivatives_fourth_rescaled(&self.inverse_link, u1)
                        .map_err(|e| {
                            format!(
                                "inverse-link survival evaluation failed at row {row} exit: {e}"
                            )
                        })?;

                let pdf = Self::exact_log_pdf_derivatives_rescaled(
                    &self.inverse_link,
                    u1,
                    deriv_log_scale,
                )
                .map_err(|e| {
                    format!("inverse-link log-pdf evaluation failed at row {row} exit: {e}")
                })?;
                (surv, pdf)
            };

        // Row degeneracy guard: when any hazard/pdf derivative is non-finite
        // (e.g. CLogLog with u > ~709 where exp(u) overflows), the row's
        // survival probability has underflowed to 0 and the derivatives
        // cannot be represented in f64.  Exclude the row — same principle
        // as the w <= 0 early-return above.
        if !(r0.is_finite()
            && dr0.is_finite()
            && ddr0.is_finite()
            && dddr0.is_finite()
            && r1.is_finite()
            && dr1.is_finite()
            && ddr1.is_finite()
            && dddr1.is_finite()
            && dlogphi1.is_finite()
            && d2logphi1.is_finite()
            && d3logphi1.is_finite()
            && d4logphi1.is_finite())
        {
            log::debug!(
                "skipping row {row}: survival derivatives non-finite \
                 (u0={u0:.2e}, u1={u1:.2e})"
            );
            return Ok(None);
        }

        let guard = self.time_derivative_lower_bound();
        let mut g = state.g;
        // Layer 4: NaN is a hard error (genuinely bad data or upstream logic
        // bug).  ±inf is clamped to finite extremes so downstream log(g) is
        // well-defined; the monotonicity guard will then floor g if needed.
        if g.is_nan() {
            return Err(SurvivalLocationScaleError::NumericalFailure { reason: format!(
                "survival location-scale time derivative is non-finite at row {row}: d_eta/dt={g}"
            ) }.into());
        }
        if g == f64::INFINITY {
            g = f64::MAX;
        } else if g == f64::NEG_INFINITY {
            g = f64::MIN;
        }
        // Adaptive roundoff slack for the monotonicity guard.
        //
        // `g` is now formed with a compensated subtraction, so the low-part
        // residual from that subtraction is the primary estimate of how much
        // rounding error the d_eta/dt reconstruction may have accumulated.
        // The older state-scale heuristic remains as a floor for moderate
        // inputs.
        let legacy_slack = MONOTONICITY_GUARD_SLACK_REL
            * (1.0
                + state
                    .h0
                    .abs()
                    .max(state.h1.abs())
                    .max(state.q0.abs())
                    .max(state.q1.abs()));
        let roundoff_slack = state.g_roundoff_slack.max(legacy_slack);
        if g < guard && g >= guard - roundoff_slack {
            g = guard;
        }
        // `d_raw` is structurally constrained, but the full event Jacobian is
        // `g = d_raw + qdot`. The threshold/log-sigma contribution can nudge an
        // otherwise valid monotone state below the numeric guard while still
        // remaining strictly positive. The row kernel only needs `log(g)` on the
        // positive domain, so clamp positive near-boundary values to the guard
        // and reserve hard failure for true non-monotone states.
        if g > 0.0 && g < guard {
            g = guard;
        }
        // Boundary cancellation floor. The constrained Newton solve bounds only
        // the structural `d_raw` channel; the additive `qdot = dq/dt` term from
        // the threshold/log-σ time-transform is unconstrained and, when it is of
        // comparable magnitude and opposite sign to `d_raw`, the difference
        // `g = d_raw + qdot` is a near-cancellation. At a feasible boundary
        // (`g → guard⁺`) the residual upstream error of that cancellation can
        // tip the reconstructed `g` a hair below zero (observed ~ -2e-7 with a
        // guard of 1e-6). Such a violation is strictly smaller than the modeling
        // guard itself and lives inside the cancellation resolution
        // `operand_scale`, so it cannot be distinguished from the feasible
        // boundary state the optimizer converged to — floor it to the guard,
        // exactly as the positive near-boundary branch above does. A genuinely
        // non-monotone fit produces `g` negative by far more than the guard,
        // which still hard-errors below.
        let cancellation_floor = guard + roundoff_slack;
        if g <= 0.0 && g >= -cancellation_floor {
            g = guard;
        }
        if g <= 0.0 {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "survival location-scale monotonicity violated at row {row}: \
                 d_eta/dt={g:.3e} <= 0 (lower_bound={guard:.3e}) \
                 (operand_scale={:.3e}, roundoff_slack={roundoff_slack:.3e})",
                    state.g_operand_scale
                ),
            }
            .into());
        }
        let (log_g, d_log_g, d2_log_g, d3_log_g, d4_log_g) = Self::logwith_derivatives_positive(g);

        Ok(Some(SurvivalExactRowKernel {
            w,
            d,
            log_s0,
            r0,
            dr0,
            ddr0,
            dddr0,
            log_s1,
            r1,
            dr1,
            ddr1,
            dddr1,
            logphi1,
            dlogphi1,
            d2logphi1,
            d3logphi1,
            d4logphi1,
            log_g,
            d_log_g,
            d2_log_g,
            d3_log_g,
            d4_log_g,
        }))
    }

    pub(crate) fn row_derivatives(
        &self,
        row: usize,
        state: SurvivalPredictorState,
    ) -> Result<Option<SurvivalRowDerivatives>, String> {
        self.row_derivatives_rescaled(row, state, 0.0)
    }

    pub(crate) fn row_derivatives_rescaled(
        &self,
        row: usize,
        state: SurvivalPredictorState,
        deriv_log_scale: f64,
    ) -> Result<Option<SurvivalRowDerivatives>, String> {
        let Some(kernel) = self.exact_row_kernel_rescaled(row, state, deriv_log_scale)? else {
            return Ok(None);
        };
        let tower = kernel.nll_index_tower();
        let d1_q0 = -tower.g[0];
        let d2_q0 = -tower.h[0][0];
        let d3_q0 = -tower.t3[0][0][0];
        let d1_q1 = -tower.g[1];
        let d2_q1 = -tower.h[1][1];
        let d3_q1 = -tower.t3[1][1][1];
        let d1_qdot1 = -tower.g[2];
        let d2_qdot1 = -tower.h[2][2];
        Ok(Some(SurvivalRowDerivatives {
            ll: kernel.log_likelihood(),
            d1_q0,
            d2_q0,
            d3_q0,
            d1_q1,
            d2_q1,
            d3_q1,
            d1_qdot1,
            d2_qdot1,
            grad_time_eta_h0: d1_q0,
            grad_time_eta_h1: d1_q1,
            grad_time_eta_d: d1_qdot1,
            h_time_h0: d2_q0,
            h_time_h1: d2_q1,
            h_time_d: d2_qdot1,
        }))
    }
}

/// Scalar chain-rule derivatives of
/// q(eta_t, eta_ls) = -eta_t * exp(-eta_ls).
///
/// Returns (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) — the full set of
/// partials up to third order needed by both the survival and GAMLSS engines.
#[inline]
pub(crate) fn q_chain_derivs_scalar(eta_t: f64, eta_ls: f64) -> (f64, f64, f64, f64, f64, f64) {
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    let q = -safe_product(eta_t, inv_sigma);
    (-inv_sigma, -q, inv_sigma, q, -inv_sigma, -q)
}
