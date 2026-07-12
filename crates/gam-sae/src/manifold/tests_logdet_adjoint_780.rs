//! Stationary-cache `∂log|H|/∂θ` adjoint regression tests (#1416),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet` are sourced from the sibling
//! `tests` module.

use super::construction::{active_softmax_gershgorin_majorizer_entry, softmax_majorizer_log_mean};
use super::derivative_oracle::{
    DerivativeTraceChannel, ExactTraceChannel, ExactTraceReport, MajorizerAnchorMode, PivotBranch,
    dual_spd_logdet, guarded_exact_trace_report,
};
use super::dual::{Dual, DualKinkBranch};
use super::tests::{fixed_state_logdet, gamma_fd_tiny_fixture};
use super::*;
use approx::assert_abs_diff_eq;

#[derive(Clone, Copy)]
struct TinyComplex {
    re: f64,
    im: f64,
}

impl TinyComplex {
    fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }

    fn exp(self) -> Self {
        let e = self.re.exp();
        Self {
            re: e * self.im.cos(),
            im: e * self.im.sin(),
        }
    }
}

fn real_softmax(logits: &[f64], tau: f64) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = logits
        .iter()
        .map(|&z| ((z - max_logit) / tau).exp())
        .collect();
    let sum: f64 = weights.iter().sum();
    for weight in weights.iter_mut() {
        *weight /= sum;
    }
    weights
}

fn complex_softmax_weight_product_derivative(
    logits: &[f64],
    tau: f64,
    atom_a: usize,
    atom_b: usize,
    atom_w: usize,
    block_inner: f64,
) -> f64 {
    let h = 1.0e-30;
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut denom = TinyComplex::real(0.0);
    let mut numer_a = TinyComplex::real(0.0);
    let mut numer_b = TinyComplex::real(0.0);
    for (atom, &logit) in logits.iter().enumerate() {
        let z = TinyComplex {
            re: (logit - max_logit) / tau,
            im: if atom == atom_w { h / tau } else { 0.0 },
        };
        let exp_z = z.exp();
        denom = denom.add(exp_z);
        if atom == atom_a {
            numer_a = exp_z;
        }
        if atom == atom_b {
            numer_b = exp_z;
        }
    }
    let a = numer_a.div(denom);
    let b = numer_b.div(denom);
    a.mul(b).mul(TinyComplex::real(block_inner)).im / h
}

fn dense_solve_for_logdet_2156(a: &Array2<f64>, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    assert_eq!(a.nrows(), n, "dense solve row count");
    assert_eq!(a.ncols(), n, "dense solve column count");
    let mut m = a.clone();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = m[[col, col]].abs();
        for row in col + 1..n {
            let candidate = m[[row, col]].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        assert!(
            pivot_abs > 1.0e-14,
            "dense solve singular pivot at col={col}, pivot_abs={pivot_abs:.8e}"
        );
        if pivot != col {
            for j in col..n {
                m.swap((col, j), (pivot, j));
            }
            rhs.swap(col, pivot);
        }
        let diag = m[[col, col]];
        for row in col + 1..n {
            let factor = m[[row, col]] / diag;
            m[[row, col]] = 0.0;
            for j in col + 1..n {
                m[[row, j]] -= factor * m[[col, j]];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0_f64; n];
    for offset in 0..n {
        let row = n - 1 - offset;
        let mut acc = rhs[row];
        for col in row + 1..n {
            acc -= m[[row, col]] * x[col];
        }
        x[row] = acc / m[[row, row]];
    }
    x
}

fn dense_cached_arrow_hessian_2156(cache: &ArrowFactorCache) -> Array2<f64> {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let mut h = Array2::<f64>::zeros((dim, dim));
    let mut e_t = Array1::<f64>::zeros(total_t);
    let mut e_beta = Array1::<f64>::zeros(cache.k);
    for col in 0..dim {
        if col < total_t {
            e_t[col] = 1.0;
        } else {
            e_beta[col - total_t] = 1.0;
        }
        let applied = apply_cached_arrow_hessian(cache, e_t.view(), e_beta.view())
            .expect("dense cached Hessian column");
        if col < total_t {
            e_t[col] = 0.0;
        } else {
            e_beta[col - total_t] = 0.0;
        }
        for row in 0..total_t {
            h[[row, col]] = applied.t[row];
        }
        for beta_row in 0..cache.k {
            h[[total_t + beta_row, col]] = applied.beta[beta_row];
        }
    }
    h
}

fn dense_cholesky_logdet_2156(h: &Array2<f64>) -> f64 {
    let n = h.nrows();
    assert_eq!(h.ncols(), n, "dense logdet matrix must be square");
    let mut lower = Array2::<f64>::zeros((n, n));
    for row in 0..n {
        for col in 0..=row {
            let mut sum = h[[row, col]];
            for inner in 0..col {
                sum -= lower[[row, inner]] * lower[[col, inner]];
            }
            if row == col {
                assert!(
                    sum.is_finite() && sum > 0.0,
                    "dense Cholesky non-positive pivot at row={row}: {sum:.12e}"
                );
                lower[[row, col]] = sum.sqrt();
            } else {
                lower[[row, col]] = sum / lower[[col, col]];
            }
        }
    }
    let mut acc = 0.0_f64;
    for idx in 0..n {
        acc += 2.0 * lower[[idx, idx]].ln();
    }
    acc
}

fn dual_half_logdet_trace_2156(cache: &ArrowFactorCache, h: &Array2<f64>, dh: &Array2<f64>) -> f64 {
    let report = dual_trace_report_2156(
        cache,
        h,
        vec![(DerivativeTraceChannel::Other("rho"), dh.clone())],
    );
    0.5 * report.total_derivative
}

fn relative_error_2156(exact: f64, production: f64) -> f64 {
    (exact - production).abs() / exact.abs().max(production.abs()).max(1.0)
}

fn assert_dual_trace_matches_analytic_2156(
    label: &str,
    coord: usize,
    exact_half: f64,
    analytic_half: f64,
) -> f64 {
    let rel = relative_error_2156(exact_half, analytic_half);
    assert!(
        rel < 1.0e-10,
        "{label} rho[{coord}] dual-vs-analytic logdet trace mismatch: \
         dual={exact_half:.16e}, analytic={analytic_half:.16e}, rel={rel:.3e}"
    );
    rel
}

fn row_deflation_pushforward_2156(
    cache: &ArrowFactorCache,
    row: usize,
    raw: &Array2<f64>,
) -> Array2<f64> {
    let q = raw.nrows();
    assert_eq!(raw.ncols(), q, "row derivative must be square");
    let Some(spec) = cache
        .deflation_row_spectra
        .get(row)
        .and_then(Option::as_ref)
    else {
        return raw.clone();
    };
    let u = &spec.evecs;
    assert_eq!(u.nrows(), q, "deflation eigenbasis row count");
    assert_eq!(u.ncols(), q, "deflation eigenbasis col count");
    let in_basis = u.t().dot(raw).dot(u);
    let mut pushed = Array2::<f64>::zeros((q, q));
    let eigen_scale = spec
        .raw_evals
        .iter()
        .chain(spec.cond_evals.iter())
        .copied()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    let gap_threshold = eigen_gap_threshold(eigen_scale, spec.raw_evals.len());
    for a in 0..q {
        for b in 0..q {
            let denom = spec.raw_evals[a] - spec.raw_evals[b];
            let factor = if denom.abs() > gap_threshold {
                (spec.cond_evals[a] - spec.cond_evals[b]) / denom
            } else if spec.cond_evals[a] == spec.raw_evals[a] {
                1.0
            } else {
                0.0
            };
            pushed[[a, b]] = factor * in_basis[[a, b]];
        }
    }
    u.dot(&pushed).dot(&u.t())
}

fn add_row_block_2156(
    out: &mut Array2<f64>,
    cache: &ArrowFactorCache,
    row: usize,
    block: &Array2<f64>,
) {
    let base = cache.row_offsets[row];
    for a in 0..block.nrows() {
        for b in 0..block.ncols() {
            out[[base + a, base + b]] += block[[a, b]];
        }
    }
}

fn smooth_rho_derivative_matrix_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    atom_idx: usize,
) -> Array2<f64> {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let mut dh = Array2::<f64>::zeros((dim, dim));
    let border = term
        .border_channels_for_cache(cache)
        .expect("border channels");
    let lambda = rho.lambda_smooth_vec().unwrap()[atom_idx];
    for left in &border {
        if left.atom != atom_idx {
            continue;
        }
        for right in &border {
            if right.atom != atom_idx {
                continue;
            }
            let s = &term.atoms[atom_idx].smooth_penalty;
            let sym_s =
                0.5 * (s[[left.basis_col, right.basis_col]] + s[[right.basis_col, left.basis_col]]);
            let output_dot = sae_dot(&left.output, &right.output);
            dh[[total_t + left.index, total_t + right.index]] += lambda * sym_s * output_dot;
        }
    }
    dh
}

fn softmax_sparse_rho_derivative_matrix_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> Array2<f64> {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let mut dh = Array2::<f64>::zeros((dim, dim));
    let AssignmentMode::Softmax {
        temperature,
        sparsity,
    } = term.assignment.mode
    else {
        panic!("softmax sparse derivative requires softmax mode");
    };
    let scale = rho.lambda_sparse().unwrap() * sparsity / (temperature * temperature);
    for row in 0..term.n_obs() {
        let assignments =
            crate::assignment::softmax_row(term.assignment.logits.row(row), temperature);
        let a = assignments.as_slice().expect("softmax row");
        let mean = softmax_majorizer_log_mean(a);
        let vars = term
            .row_vars_for_cache_row(row, cache)
            .expect("softmax row vars");
        let mut row_d = Array2::<f64>::zeros((cache.row_dims[row], cache.row_dims[row]));
        for (pos, var) in vars.iter().enumerate() {
            if let SaeLocalRowVar::Logit { atom } = *var {
                row_d[[pos, pos]] = active_softmax_gershgorin_majorizer_entry(a, atom, mean, scale);
            }
        }
        let pushed = row_deflation_pushforward_2156(cache, row, &row_d);
        add_row_block_2156(&mut dh, cache, row, &pushed);
    }
    dh
}

fn ordered_beta_bernoulli_sparse_rho_derivative_matrix_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> Array2<f64> {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let mut dh = Array2::<f64>::zeros((dim, dim));
    let k_atoms = term.k_atoms();
    let mut hdiag = assignment_prior_log_strength_hdiag(&term.assignment, rho)
        .expect("ordered Beta--Bernoulli hdiag");
    let channels = ordered_beta_bernoulli_psd_majorizer_third_channels(&term.assignment, rho)
        .expect("ordered Beta--Bernoulli channels")
        .expect(
            "ordered Beta--Bernoulli sparse derivative requires ordered Beta--Bernoulli channels",
        );
    // #2144/#1038: the production assembly PSD-majorizes the ordered Beta--Bernoulli curvature
    // UNCONDITIONALLY, so this mirror does too.
    for row in 0..term.n_obs() {
        for atom in 0..k_atoms {
            let slot = row * k_atoms + atom;
            hdiag[slot] = ordered_beta_bernoulli_majorized_hdiag_2156(
                &channels,
                row,
                k_atoms,
                atom,
                hdiag[slot],
            );
        }
    }
    for row in 0..term.n_obs() {
        let vars = term
            .row_vars_for_cache_row(row, cache)
            .expect("ordered Beta--Bernoulli row vars");
        let mut row_derivative = Array2::<f64>::zeros((cache.row_dims[row], cache.row_dims[row]));
        for (pos, var) in vars.iter().enumerate() {
            if let SaeLocalRowVar::Logit { atom } = *var {
                let slot = row * k_atoms + atom;
                row_derivative[[pos, pos]] = hdiag[slot];
            }
        }
        let pushed = row_deflation_pushforward_2156(cache, row, &row_derivative);
        add_row_block_2156(&mut dh, cache, row, &pushed);
    }
    dh
}

fn rho_logdet_derivative_matrix_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    coord: usize,
) -> Array2<f64> {
    if coord == 0 {
        match term.assignment.mode {
            AssignmentMode::Softmax { .. } => {
                softmax_sparse_rho_derivative_matrix_2156(term, rho, cache)
            }
            AssignmentMode::OrderedBetaBernoulli { .. } => {
                ordered_beta_bernoulli_sparse_rho_derivative_matrix_2156(term, rho, cache)
            }
            _ => {
                panic!("rho sparse derivative fixture must use softmax or ordered Beta--Bernoulli")
            }
        }
    } else {
        let atom = coord - 1;
        assert!(atom < term.k_atoms(), "smooth rho coordinate out of range");
        smooth_rho_derivative_matrix_2156(term, rho, cache, atom)
    }
}

fn ard_rho_derivative_matrix_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    atom: usize,
    axis: usize,
) -> Array2<f64> {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let mut dh = Array2::<f64>::zeros((dim, dim));
    let coord_offsets = term.assignment.coord_offsets();
    let periods = term.assignment.coords[atom].effective_axis_periods();
    let alpha = rho.log_ard[atom][axis].exp();
    for row in 0..term.n_obs() {
        let t = term.assignment.coords[atom].row(row)[axis];
        let hess = ArdAxisPrior::eval(alpha, t, periods[axis]).hess.max(0.0);
        let mut row_d = Array2::<f64>::zeros((cache.row_dims[row], cache.row_dims[row]));
        match term.last_row_layout.as_ref() {
            Some(layout) => {
                for (pos, &active_atom) in layout.active_atoms[row].iter().enumerate() {
                    if active_atom == atom {
                        let local = layout.coord_starts[row][pos] + axis;
                        row_d[[local, local]] = hess;
                    }
                }
            }
            None => {
                let local = coord_offsets[atom] + axis;
                row_d[[local, local]] = hess;
            }
        }
        let pushed = row_deflation_pushforward_2156(cache, row, &row_d);
        add_row_block_2156(&mut dh, cache, row, &pushed);
    }
    dh
}

fn dual_logdet_channel_2156(
    channel: DerivativeTraceChannel,
    h: &Array2<f64>,
    dh: &Array2<f64>,
    certificate: BranchCertificate,
) -> ExactTraceChannel {
    assert_eq!(h.raw_dim(), dh.raw_dim(), "dual channel shape");
    let n = h.nrows();
    let matrix: Vec<Vec<Dual>> = (0..n)
        .map(|row| {
            (0..n)
                .map(|col| Dual::with_derivative(h[[row, col]], dh[[row, col]]))
                .collect()
        })
        .collect();
    let logdet = dual_spd_logdet(&matrix).expect("dual SPD logdet channel");
    ExactTraceChannel {
        channel,
        value: logdet.re,
        derivative: logdet.eps,
        certificate,
    }
}

fn dual_trace_report_2156(
    cache: &ArrowFactorCache,
    h: &Array2<f64>,
    channels: Vec<(DerivativeTraceChannel, Array2<f64>)>,
) -> ExactTraceReport {
    let certificate = BranchCertificate::from_arrow_cache(cache, MajorizerAnchorMode::FrozenAnchor);
    let exact_channels: Vec<ExactTraceChannel> = channels
        .iter()
        .map(|(channel, dh)| dual_logdet_channel_2156(*channel, h, dh, certificate.clone()))
        .collect();
    guarded_exact_trace_report(certificate, exact_channels).expect("same-branch dual report")
}

fn exact_channel_derivative_2156(
    report: &ExactTraceReport,
    channel: DerivativeTraceChannel,
) -> f64 {
    report
        .channel_derivative(channel)
        .expect("missing exact channel derivative")
}

fn assert_close_2156(label: &str, exact: f64, production: f64, scale: f64) {
    let tol = 1.0e-8 * (1.0 + scale.abs().max(exact.abs()).max(production.abs()));
    assert!(
        (exact - production).abs() <= tol,
        "{label}: exact={exact:.12e}, production={production:.12e}, tol={tol:.3e}"
    );
}

fn dual_softmax_row_2156(logits: &[f64], tau: f64, seed_atom: usize) -> Vec<Dual> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let inv_tau = 1.0 / tau;
    let weights: Vec<Dual> = logits
        .iter()
        .enumerate()
        .map(|(atom, &logit)| {
            let value = ((logit - max_logit) * inv_tau).exp();
            let derivative = if atom == seed_atom {
                value * inv_tau
            } else {
                0.0
            };
            Dual::with_derivative(value, derivative)
        })
        .collect();
    let denom = weights
        .iter()
        .copied()
        .fold(Dual::constant(0.0), |acc, weight| acc + weight);
    weights.into_iter().map(|weight| weight / denom).collect()
}

fn softmax_logit_dual_channel_report_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    row: usize,
    local_w: usize,
) -> ExactTraceReport {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let base = cache.row_offsets[row];
    let vars = term
        .row_vars_for_cache_row(row, cache)
        .expect("softmax row vars");
    let SaeLocalRowVar::Logit { atom: seed_atom } = vars[local_w] else {
        panic!("softmax dual guard seed must be a logit slot");
    };
    let AssignmentMode::Softmax {
        temperature,
        sparsity,
    } = term.assignment.mode
    else {
        panic!("softmax dual guard requires softmax mode");
    };
    let logits: Vec<f64> = (0..term.k_atoms())
        .map(|atom| term.assignment.logits[[row, atom]])
        .collect();
    let dual_a = dual_softmax_row_2156(&logits, temperature, seed_atom);
    let mut assignments = Array1::<f64>::zeros(term.k_atoms());
    term.assignment
        .try_assignments_row_into(row, assignments.as_slice_mut().expect("assignment scratch"))
        .expect("softmax assignments");
    for atom in 0..term.k_atoms() {
        assert_close_2156(
            "dual softmax value",
            dual_a[atom].re,
            assignments[atom],
            1.0,
        );
    }

    let second_jets = term.atom_second_jets().expect("second jets");
    let border = term
        .border_channels_for_cache(cache)
        .expect("border channels");
    let jets = term
        .row_jets_for_logdet(row, vars, assignments.view(), &second_jets, &border)
        .expect("softmax row jets");
    let mut tt_data = Array2::<f64>::zeros((dim, dim));
    let mut tt_majorizer = Array2::<f64>::zeros((dim, dim));
    let mut t_beta = Array2::<f64>::zeros((dim, dim));
    let mut beta_beta = Array2::<f64>::zeros((dim, dim));

    for a in 0..jets.vars.len() {
        for b in 0..jets.vars.len() {
            let entry = sae_dot(jets.second(a, local_w), jets.first(b))
                + sae_dot(jets.first(a), jets.second(b, local_w));
            tt_data[[base + a, base + b]] = entry;
        }
    }

    let scale = rho.lambda_sparse().unwrap() * sparsity / (temperature * temperature);
    let majorizer_deriv = gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
        term.k_atoms(),
        temperature,
    )
    .row_psd_majorizer_logit_derivative(&logits, scale, seed_atom);
    for (pos, var) in jets.vars.iter().enumerate() {
        if let SaeLocalRowVar::Logit { atom } = *var {
            tt_majorizer[[base + pos, base + pos]] = majorizer_deriv[[atom, atom]];
        }
    }

    for a in 0..jets.vars.len() {
        for (beta_pos, channel) in border.iter().enumerate() {
            let entry = sae_dot(jets.second(a, local_w), jets.beta(beta_pos))
                + sae_dot(jets.first(a), jets.beta_deriv(local_w, beta_pos));
            let t_idx = base + a;
            let b_idx = total_t + channel.index;
            t_beta[[t_idx, b_idx]] = entry;
            t_beta[[b_idx, t_idx]] = entry;
        }
    }
    for (beta_i, channel_i) in border.iter().enumerate() {
        for (beta_j, channel_j) in border.iter().enumerate() {
            let entry = sae_dot(jets.beta_deriv(local_w, beta_i), jets.beta(beta_j))
                + sae_dot(jets.beta(beta_i), jets.beta_deriv(local_w, beta_j));
            beta_beta[[total_t + channel_i.index, total_t + channel_j.index]] = entry;
        }
    }

    let h = dense_cached_arrow_hessian_2156(cache);
    dual_trace_report_2156(
        cache,
        &h,
        vec![
            (DerivativeTraceChannel::Tt, tt_data),
            (DerivativeTraceChannel::Majorizer, tt_majorizer),
            (DerivativeTraceChannel::Border, t_beta),
            (DerivativeTraceChannel::Beta, beta_beta),
        ],
    )
}

fn install_low_rank_ordered_beta_bernoulli_metric_2156(term: &mut SaeManifoldTerm) {
    use gam_problem::{RowMetric, pack_probe_factors};
    use std::sync::Arc;

    let n = term.n_obs();
    let p = term.output_dim();
    let s = 2usize;
    let mut seed = 0x2156_2144_u64;
    let probes = Array3::<f64>::from_shape_fn((n, p, s), |(_, i, kk)| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let base = if kk == 0 && i == 0 {
            1.2
        } else if kk == 1 && i + 1 == p {
            1.0
        } else {
            0.0
        };
        base + 0.15 * (((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5)
    });
    let u = pack_probe_factors(probes.view()).expect("packed low-rank probes");
    term.set_row_metric(RowMetric::behavioral_fisher(Arc::new(u), p, s).expect("row metric"))
        .expect("install low-rank metric");
    assert!(
        term.row_metric()
            .is_some_and(|m| m.whitens_likelihood() && m.metric_rank() < p),
        "rank-{s} metric on p={p} must be a genuinely rank-deficient whitening metric"
    );
}

fn ordered_beta_bernoulli_majorized_hdiag_2156(
    channels: &OrderedBetaBernoulliHessianDiagThirdChannels,
    row: usize,
    k_atoms: usize,
    atom: usize,
    raw_hdiag: f64,
) -> f64 {
    let index = row * k_atoms + atom;
    if channels.diagonal_term[index] <= 0.0 {
        return 0.0;
    }
    let j = channels.z_jac[index];
    raw_hdiag - channels.mass_hessian_coefficient[atom] * j * j
}

fn dense_trace_hinv_dh_2156(h: &Array2<f64>, dh: &Array2<f64>) -> f64 {
    assert_eq!(h.raw_dim(), dh.raw_dim(), "trace shape");
    let dim = h.nrows();
    let mut trace = 0.0_f64;
    for col in 0..dim {
        let rhs: Vec<f64> = (0..dim).map(|row| dh[[row, col]]).collect();
        let solved = dense_solve_for_logdet_2156(h, &rhs);
        trace += solved[col];
    }
    trace
}

fn configure_decisive_softmax_logits_2156(term: &mut SaeManifoldTerm) {
    for r in 0..term.n_obs() {
        let center = 0.05 * (r as f64);
        let margin = 1.55 + 0.04 * (r as f64);
        if r % 2 == 0 {
            term.assignment.logits[[r, 0]] = center + margin;
            term.assignment.logits[[r, 1]] = center - margin;
        } else {
            term.assignment.logits[[r, 0]] = center - 0.85 * margin;
            term.assignment.logits[[r, 1]] = center + 0.85 * margin;
        }
    }
}

fn assert_branch_certificate_green_2156(label: &str, certificate: &BranchCertificate) {
    certificate
        .assert_derivative_reportable()
        .unwrap_or_else(|err| panic!("{label} derivative branch must be reportable: {err}"));
    assert!(
        certificate
            .kink_branches
            .iter()
            .all(|record| record.branch != DualKinkBranch::Tie),
        "{label} kink branch certificate must not contain a tie: {:?}",
        certificate.kink_branches
    );
    assert_eq!(
        certificate.min_row_pivot_branch,
        PivotBranch::Positive,
        "{label} row Cholesky branch must stay positive"
    );
    assert_eq!(
        certificate.min_pivot_branch,
        PivotBranch::Positive,
        "{label} global pivot branch must stay positive"
    );
    assert_eq!(
        certificate.max_pivot_branch,
        PivotBranch::Positive,
        "{label} max pivot branch must stay positive"
    );
    if certificate.beta_dim > 0 {
        assert_eq!(
            certificate.min_schur_pivot_branch,
            PivotBranch::Positive,
            "{label} Schur branch must stay positive"
        );
    }
}

fn assert_dual_rho_logdet_parity_2156(
    label: &str,
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> f64 {
    let certificate = BranchCertificate::from_arrow_cache(cache, MajorizerAnchorMode::FrozenAnchor);
    assert_branch_certificate_green_2156(label, &certificate);
    eprintln!("gam#2156 {label} rho branch certificate: {certificate:?}");
    let h = dense_cached_arrow_hessian_2156(cache);
    let solver = DeflatedArrowSolver::plain(cache);
    let mut max_rel = 0.0_f64;

    let sparse_dh = rho_logdet_derivative_matrix_2156(term, rho, cache, 0);
    let sparse_dual_half = dual_half_logdet_trace_2156(cache, &h, &sparse_dh);
    let sparse_analytic_half = term
        .assignment_log_strength_hessian_trace(rho, cache, &solver)
        .expect("production sparse rho trace");
    max_rel = max_rel.max(assert_dual_trace_matches_analytic_2156(
        label,
        0,
        sparse_dual_half,
        sparse_analytic_half,
    ));

    let lambda_smooth = rho.lambda_smooth_vec().unwrap();
    let smooth_analytic = term
        .decoder_smoothness_effective_dof_with_solver_per_atom(cache, &solver, &lambda_smooth)
        .expect("production smoothness rho trace");
    for atom in 0..rho.log_lambda_smooth.len() {
        let dh = rho_logdet_derivative_matrix_2156(term, rho, cache, atom + 1);
        let dual_half = dual_half_logdet_trace_2156(cache, &h, &dh);
        let analytic_half = 0.5 * smooth_analytic[atom];
        let rel =
            assert_dual_trace_matches_analytic_2156(label, atom + 1, dual_half, analytic_half);
        max_rel = max_rel.max(rel);
    }

    let ard_analytic = term
        .ard_log_precision_hessian_trace(rho, cache, &solver)
        .expect("production ARD rho trace");
    let mut flat = 1 + rho.log_lambda_smooth.len();
    for atom in 0..rho.log_ard.len() {
        for axis in 0..rho.log_ard[atom].len() {
            let dh = ard_rho_derivative_matrix_2156(term, rho, cache, atom, axis);
            let dual_half = dual_half_logdet_trace_2156(cache, &h, &dh);
            let analytic_half = ard_analytic[atom][axis];
            let rel =
                assert_dual_trace_matches_analytic_2156(label, flat, dual_half, analytic_half);
            max_rel = max_rel.max(rel);
            flat += 1;
        }
    }
    max_rel
}

fn perturb_theta_slot_2156(
    term: &mut SaeManifoldTerm,
    row: usize,
    var: SaeLocalRowVar,
    delta: f64,
) {
    match var {
        SaeLocalRowVar::Logit { atom } => {
            term.assignment.logits[[row, atom]] += delta;
        }
        SaeLocalRowVar::Coord { atom, axis } => {
            let mut flat = term.assignment.coords[atom].as_flat().clone();
            let idx = row * term.assignment.coords[atom].latent_dim() + axis;
            flat[idx] += delta;
            term.assignment.coords[atom].set_flat(flat.view());
        }
    }
}

fn assert_live_theta_logdet_fd_2156(
    label: &str,
    term: &SaeManifoldTerm,
    target: &Array2<f64>,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    probes: &[(usize, usize)],
) -> f64 {
    let certificate = BranchCertificate::from_arrow_cache(cache, MajorizerAnchorMode::FrozenAnchor);
    assert_branch_certificate_green_2156(label, &certificate);
    eprintln!("gam#2156 {label} live-theta branch certificate: {certificate:?}");
    let solver = DeflatedArrowSolver::plain(cache);
    let gamma = term
        .logdet_theta_adjoint(rho, cache, &solver)
        .expect("production theta adjoint");
    let h = 1.0e-5;
    let mut max_rel = 0.0_f64;
    for &(row, local_pos) in probes {
        let vars = term
            .row_vars_for_cache_row(row, cache)
            .expect("live theta FD row vars");
        let var = vars[local_pos];
        let mut plus = term.clone();
        let mut minus = term.clone();
        perturb_theta_slot_2156(&mut plus, row, var, h);
        perturb_theta_slot_2156(&mut minus, row, var, -h);
        let fd = (fixed_state_logdet(plus, target, rho) - fixed_state_logdet(minus, target, rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let rel = relative_error_2156(fd, analytic);
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "{label} live θ logdet FD mismatch row={row} local_pos={local_pos}: \
             fd={fd:.12e}, gamma={analytic:.12e}, abs_err={:.3e}, tol={tol:.3e}",
            (fd - analytic).abs()
        );
        max_rel = max_rel.max(rel);
    }
    max_rel
}

fn assert_dual_ard_logdet_parity_2156(
    label: &str,
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> f64 {
    let certificate = BranchCertificate::from_arrow_cache(cache, MajorizerAnchorMode::FrozenAnchor);
    assert_branch_certificate_green_2156(label, &certificate);
    eprintln!("gam#2156 {label} ARD branch certificate: {certificate:?}");
    let h = dense_cached_arrow_hessian_2156(cache);
    let solver = DeflatedArrowSolver::plain(cache);
    let ard_analytic = term
        .ard_log_precision_hessian_trace(rho, cache, &solver)
        .expect("production ARD rho trace");
    let mut max_rel = 0.0_f64;
    for atom in 0..rho.log_ard.len() {
        for axis in 0..rho.log_ard[atom].len() {
            let dh = ard_rho_derivative_matrix_2156(term, rho, cache, atom, axis);
            let dual_half = dual_half_logdet_trace_2156(cache, &h, &dh);
            let analytic_half = ard_analytic[atom][axis];
            let rel = relative_error_2156(dual_half, analytic_half);
            assert_dual_trace_matches_analytic_2156(label, atom + axis, dual_half, analytic_half);
            max_rel = max_rel.max(rel);
        }
    }
    max_rel
}

#[test]
pub(crate) fn end_to_end_dual_vs_analytic_logdet_parity_battery_2156_2144() {
    let (mut softmax_term, target, mut softmax_rho) = gamma_fd_tiny_fixture();
    softmax_rho.log_lambda_sparse = 0.5;
    softmax_rho.log_lambda_smooth = vec![-1.7, -1.2];
    softmax_term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &softmax_rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged softmax parity cache");
    configure_decisive_softmax_logits_2156(&mut softmax_term);
    let (softmax_value, softmax_loss, softmax_cache) = softmax_term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &softmax_rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("fixed-branch softmax parity cache");
    assert!(
        softmax_value.is_finite() && softmax_loss.total().is_finite(),
        "softmax parity fixture must produce a finite cache"
    );
    let softmax_theta_probes: Vec<(usize, usize)> = (0..softmax_cache.n_rows())
        .flat_map(|row| (0..softmax_cache.row_dims[row]).map(move |local| (row, local)))
        .collect();
    let softmax_theta_max_rel = assert_live_theta_logdet_fd_2156(
        "softmax",
        &softmax_term,
        &target,
        &softmax_rho,
        &softmax_cache,
        &softmax_theta_probes,
    );
    let softmax_max_rel =
        assert_dual_rho_logdet_parity_2156("softmax", &softmax_term, &softmax_rho, &softmax_cache);

    let (
        mut ordered_beta_bernoulli_term,
        ordered_beta_bernoulli_target,
        mut ordered_beta_bernoulli_rho,
    ) = gamma_fd_tiny_fixture();
    ordered_beta_bernoulli_term.assignment.mode =
        AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    install_low_rank_ordered_beta_bernoulli_metric_2156(&mut ordered_beta_bernoulli_term);
    ordered_beta_bernoulli_rho.log_lambda_sparse = 0.6;
    ordered_beta_bernoulli_rho.log_lambda_smooth = vec![-1.6, -1.1];
    let (ordered_beta_bernoulli_value, ordered_beta_bernoulli_loss, ordered_beta_bernoulli_cache) =
        ordered_beta_bernoulli_term
            .penalized_quasi_laplace_criterion_with_cache(
                ordered_beta_bernoulli_target.view(),
                &ordered_beta_bernoulli_rho,
                None,
                200,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("converged low-rank-metric ordered Beta--Bernoulli parity cache");
    assert!(
        ordered_beta_bernoulli_value.is_finite() && ordered_beta_bernoulli_loss.total().is_finite(),
        "ordered Beta--Bernoulli parity fixture must produce a finite cache"
    );
    let low_rank_certificate = BranchCertificate::from_arrow_cache(
        &ordered_beta_bernoulli_cache,
        MajorizerAnchorMode::FrozenAnchor,
    );
    assert!(
        ordered_beta_bernoulli_term
            .row_metric()
            .is_some_and(|m| m.whitens_likelihood()
                && m.metric_rank() < ordered_beta_bernoulli_term.output_dim()),
        "ordered Beta--Bernoulli parity fixture must exercise the low-rank metric branch; \
         certificate={low_rank_certificate:?}"
    );
    let ordered_beta_bernoulli_theta_probes: Vec<(usize, usize)> = (0
        ..ordered_beta_bernoulli_cache.n_rows())
        .flat_map(|row| {
            (0..ordered_beta_bernoulli_cache.row_dims[row]).map(move |local| (row, local))
        })
        .collect();
    let ordered_beta_bernoulli_theta_max_rel = assert_live_theta_logdet_fd_2156(
        "low_rank_metric_ordered_beta_bernoulli",
        &ordered_beta_bernoulli_term,
        &ordered_beta_bernoulli_target,
        &ordered_beta_bernoulli_rho,
        &ordered_beta_bernoulli_cache,
        &ordered_beta_bernoulli_theta_probes,
    );
    let ordered_beta_bernoulli_max_rel = assert_dual_rho_logdet_parity_2156(
        "low_rank_metric_ordered_beta_bernoulli",
        &ordered_beta_bernoulli_term,
        &ordered_beta_bernoulli_rho,
        &ordered_beta_bernoulli_cache,
    );

    let (mut deflated_term, deflated_target, mut deflated_rho) = gamma_fd_tiny_fixture();
    deflated_term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    deflated_rho.log_lambda_sparse = 0.5;
    let (deflated_value, deflated_loss, deflated_cache) = deflated_term
        .penalized_quasi_laplace_criterion_with_cache(
            deflated_target.view(),
            &deflated_rho,
            None,
            5,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged deflated parity cache");
    assert!(
        deflated_value.is_finite() && deflated_loss.total().is_finite(),
        "deflated parity fixture must produce a finite cache"
    );
    let deflated_certificate =
        BranchCertificate::from_arrow_cache(&deflated_cache, MajorizerAnchorMode::FrozenAnchor);
    assert!(
        deflated_certificate.deflated_rank > 0
            || deflated_certificate
                .deflated_per_row
                .iter()
                .any(|&count| count > 0)
            || deflated_certificate
                .spectral_deflated_rows
                .iter()
                .any(|&flag| flag),
        "deflated parity fixture must exercise deflated rows; certificate={deflated_certificate:?}"
    );
    let deflated_theta_probes: Vec<(usize, usize)> = (0..deflated_cache.n_rows())
        .flat_map(|row| (0..deflated_cache.row_dims[row]).map(move |local| (row, local)))
        .collect();
    let deflated_theta_max_rel = assert_live_theta_logdet_fd_2156(
        "deflated_rows_ordered_beta_bernoulli_theta",
        &deflated_term,
        &deflated_target,
        &deflated_rho,
        &deflated_cache,
        &deflated_theta_probes,
    );
    let deflated_ard_max_rel = assert_dual_ard_logdet_parity_2156(
        "deflated_rows_ard",
        &deflated_term,
        &deflated_rho,
        &deflated_cache,
    );

    eprintln!(
        "gam#2156/#2144 logdet parity max_rel: softmax_theta_live_fd={softmax_theta_max_rel:.3e}, softmax_rho={softmax_max_rel:.3e}, low_rank_metric_ordered_beta_bernoulli_theta_live_fd={ordered_beta_bernoulli_theta_max_rel:.3e}, low_rank_metric_ordered_beta_bernoulli_rho={ordered_beta_bernoulli_max_rel:.3e}, deflated_ordered_beta_bernoulli_theta_live_fd={deflated_theta_max_rel:.3e}, deflated_ard={deflated_ard_max_rel:.3e}"
    );
}

#[test]
pub(crate) fn branch_guarded_dual_oracle_pins_live_softmax_channels_2156() {
    let (mut softmax_term, target, mut softmax_rho) = gamma_fd_tiny_fixture();
    softmax_rho.log_lambda_sparse = 0.5;
    softmax_term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &softmax_rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged softmax cache");
    configure_decisive_softmax_logits_2156(&mut softmax_term);
    let (softmax_value, softmax_loss, softmax_cache) = softmax_term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &softmax_rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("fixed-state softmax cache");
    assert!(
        softmax_value.is_finite() && softmax_loss.total().is_finite(),
        "softmax guard fixture must produce a finite fixed-state cache"
    );
    let softmax_solver = DeflatedArrowSolver::plain(&softmax_cache);
    let softmax_gamma = softmax_term
        .logdet_theta_adjoint(&softmax_rho, &softmax_cache, &softmax_solver)
        .expect("softmax theta adjoint");
    let softmax_report =
        softmax_logit_dual_channel_report_2156(&softmax_term, &softmax_rho, &softmax_cache, 0, 0);
    eprintln!(
        "gam#2156 softmax branch certificate: {:?}",
        softmax_report.certificate
    );
    let softmax_tt = exact_channel_derivative_2156(&softmax_report, DerivativeTraceChannel::Tt);
    let softmax_majorizer =
        exact_channel_derivative_2156(&softmax_report, DerivativeTraceChannel::Majorizer);
    let softmax_t_beta =
        exact_channel_derivative_2156(&softmax_report, DerivativeTraceChannel::Border);
    let softmax_beta_beta =
        exact_channel_derivative_2156(&softmax_report, DerivativeTraceChannel::Beta);
    let softmax_exact_total = softmax_tt + softmax_majorizer + softmax_t_beta + softmax_beta_beta;
    let softmax_production = softmax_gamma.t[softmax_cache.row_offsets[0]];
    assert!(
        softmax_tt.abs() > 1.0e-10
            && softmax_majorizer.abs() > 1.0e-10
            && softmax_t_beta.abs() > 1.0e-10
            && softmax_beta_beta.abs() > 1.0e-10,
        "softmax guard must keep every channel live: tt={softmax_tt:.3e}, \
         majorizer={softmax_majorizer:.3e}, tβ={softmax_t_beta:.3e}, \
         ββ={softmax_beta_beta:.3e}"
    );
    assert_close_2156(
        "softmax live logdet_theta_adjoint total vs dual per-channel sum",
        softmax_exact_total,
        softmax_production,
        softmax_exact_total,
    );
}

#[test]
pub(crate) fn softmax_tt_weight_product_logit_adjoint_hits_both_factors_2156() {
    let logits = [0.31_f64, -0.27, 0.14, -0.08];
    let tau = 0.73_f64;
    let inv_tau = 1.0 / tau;
    let assignments = real_softmax(&logits, tau);
    let block_inner = 1.417_f64;

    for (atom_a, atom_b, atom_w) in [(0usize, 2usize, 1usize), (2usize, 2usize, 2usize)] {
        let h_ab = assignments[atom_a] * assignments[atom_b] * block_inner;
        let one_factor =
            h_ab * (if atom_w == atom_a { 1.0 } else { 0.0 } - assignments[atom_w]) * inv_tau;
        let fixed = h_ab
            * SaeManifoldTerm::softmax_data_weight_product_logit_factor(
                &assignments,
                atom_a,
                atom_b,
                atom_w,
                inv_tau,
            );
        let complex_step = complex_softmax_weight_product_derivative(
            &logits,
            tau,
            atom_a,
            atom_b,
            atom_w,
            block_inner,
        );
        let ratio = fixed / one_factor;
        assert!(
            (ratio - 2.0).abs() <= 1.0e-12,
            "one-factor softmax product derivative must be 2x low: got ratio {ratio:.12}"
        );
        assert!(
            (fixed - complex_step).abs() <= 1.0e-6 * (1.0 + complex_step.abs()),
            "fixed softmax product derivative must match complex-step: fixed={fixed:.12e}, complex={complex_step:.12e}"
        );
    }
}

#[test]
pub(crate) fn sae_logdet_theta_adjoint_logit0_dense_trace_localization_2156() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    rho.log_lambda_sparse = 0.5;
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        200,
        0.4,
        1.0e-6,
        1.0e-6,
    )
    .expect("converged cache");
    configure_decisive_softmax_logits_2156(&mut term);
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("off-kink fixed-state cache");

    let row = 0usize;
    let local_w = 0usize;
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let base = cache.row_offsets[row];
    let vars = term
        .row_vars_for_cache_row(row, &cache)
        .expect("row vars for localization");
    assert!(
        matches!(vars[local_w], SaeLocalRowVar::Logit { atom: 0 }),
        "gam#2156 localization probe must be row-0 logit-0"
    );
    let second_jets = term.atom_second_jets().expect("second jets");
    let border = term
        .border_channels_for_cache(&cache)
        .expect("border channels");
    let mut assignments = Array1::<f64>::zeros(term.k_atoms());
    term.assignment
        .try_assignments_row_into(row, assignments.as_slice_mut().expect("assignment scratch"))
        .expect("assignments");
    let jets = term
        .row_jets_for_logdet(row, vars, assignments.view(), &second_jets, &border)
        .expect("row jets");

    let mut dh = Array2::<f64>::zeros((dim, dim));
    let majorizer_deriv = match term.assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let scale = rho.lambda_sparse().unwrap() * sparsity / (temperature * temperature);
            let row_logits: Vec<f64> = (0..term.k_atoms())
                .map(|atom| term.assignment.logits[[row, atom]])
                .collect();
            gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                term.k_atoms(),
                temperature,
            )
            .row_psd_majorizer_logit_derivative(&row_logits, scale, 0)
        }
        _ => panic!("gam#2156 localization requires softmax mode"),
    };
    for a in 0..jets.vars.len() {
        for b in 0..jets.vars.len() {
            let mut entry = sae_dot(jets.second(a, local_w), jets.first(b))
                + sae_dot(jets.first(a), jets.second(b, local_w));
            if let (
                SaeLocalRowVar::Logit { atom: atom_a },
                SaeLocalRowVar::Logit { atom: atom_b },
            ) = (jets.vars[a], jets.vars[b])
            {
                if atom_a == atom_b {
                    entry += majorizer_deriv[[atom_a, atom_a]];
                }
            }
            let global_a = base + a;
            let global_b = base + b;
            dh[[global_a, global_b]] = entry;
        }
    }
    for a in 0..jets.vars.len() {
        for (beta_pos, channel) in border.iter().enumerate() {
            let entry = sae_dot(jets.second(a, local_w), jets.beta(beta_pos))
                + sae_dot(jets.first(a), jets.beta_deriv(local_w, beta_pos));
            let global_a = base + a;
            let global_beta = total_t + channel.index;
            dh[[global_a, global_beta]] = entry;
            dh[[global_beta, global_a]] = entry;
        }
    }
    for (beta_i, channel_i) in border.iter().enumerate() {
        for (beta_j, channel_j) in border.iter().enumerate() {
            let entry = sae_dot(jets.beta_deriv(local_w, beta_i), jets.beta(beta_j))
                + sae_dot(jets.beta(beta_i), jets.beta_deriv(local_w, beta_j));
            dh[[total_t + channel_i.index, total_t + channel_j.index]] = entry;
        }
    }

    let h_dense = dense_cached_arrow_hessian_2156(&cache);
    let dense_value = dense_cholesky_logdet_2156(&h_dense);
    let cache_value = cache
        .arrow_log_det()
        .expect("authoritative cache joint logdet");
    let evidence_value =
        arrow_log_det_from_cache(&cache).expect("evidence arrow logdet from cache");
    eprintln!(
        "gam#2156 value dense_H_apply={dense_value:.12e} cache_arrow={cache_value:.12e} evidence_arrow={evidence_value:.12e}"
    );
    let value_tol = 1.0e-9 * (1.0 + dense_value.abs().max(cache_value.abs()));
    assert!(
        (dense_value - cache_value).abs() <= value_tol,
        "gam#2156 value operator mismatch: dense H_apply logdet={dense_value:.12e}, cache.arrow_log_det={cache_value:.12e}, evidence={evidence_value:.12e}"
    );
    assert!(
        (evidence_value - cache_value).abs() <= value_tol,
        "gam#2156 evidence/cache logdet mismatch: evidence={evidence_value:.12e}, cache.arrow_log_det={cache_value:.12e}"
    );
    let trace = dense_trace_hinv_dh_2156(&h_dense, &dh);
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let analytic = gamma.t[base + local_w];
    let h = 1.0e-5;
    let at = |dl: f64| -> f64 {
        let mut t = term.clone();
        t.assignment.logits[[row, 0]] += dl;
        fixed_state_logdet(t, &target, &rho)
    };
    let fd_total = (at(h) - at(-h)) / (2.0 * h);
    eprintln!(
        "gam#2156 row=0 logit=0 dense_trace={trace:.12e} gamma={analytic:.12e} fd_total={fd_total:.12e} gamma_live_fd_abs_err={:.12e}",
        (analytic - fd_total).abs()
    );
    let trace_tol = 1.0e-5 * (1.0 + trace.abs().max(fd_total.abs()));
    assert!(
        (trace - fd_total).abs() <= trace_tol,
        "gam#2156 operator mismatch: dense trace from row jets does not match fixed-state FD: trace={trace:.12e}, fd_total={fd_total:.12e}"
    );
    let gamma_tol = 1.0e-5 * (1.0 + analytic.abs().max(fd_total.abs()));
    assert!(
        (analytic - fd_total).abs() <= gamma_tol,
        "gam#2156 live objective mismatch: gamma={analytic:.12e}, fd_total={fd_total:.12e}, dense trace={trace:.12e}"
    );
}

#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_on_tiny_fixture() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // The shared fixture default ships ρ at the −6.0 floor, where the undamped
    // joint Hessian has no interior PD minimum (the #1625 indefinite-basin
    // diagnosis): the inner solve never converges, so no stationary cache exists
    // at which the analytic adjoint can equal dense FD. Lift ρ_sparse into the PD
    // region AND give the inner Newton solve a budget large enough to reach a
    // tight optimum — at the converged cache the analytic `∂log|H|/∂θ` matches the
    // fixed-state central difference to ≈8 digits (verified across ρ ∈ [−1,3]).
    // This is a setup fix that makes the comparison point EXIST; no tolerance is
    // weakened.
    rho.log_lambda_sparse = 0.5;
    // Converge to a well-conditioned PD state, then replace only the softmax
    // logits by a deterministic, moderately decisive fixture. At the fitted
    // optimum an entropy-Hessian off-diagonal can sit exactly at the Gershgorin
    // `|H_kj|` sign-flip kink (acn116: fwd=-3.95, bwd=-19.16, central their
    // average), where no finite-difference stencil validates a subgradient. These
    // row-varying logit margins keep the softmax away from that kink without
    // saturating the row to a near-boundary PD block, so the fixed-state central
    // difference below differentiates a locally smooth majorizer branch.
    term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        200,
        0.4,
        1.0e-6,
        1.0e-6,
    )
    .expect("converged cache");
    for r in 0..term.n_obs() {
        let center = 0.05 * (r as f64);
        let margin = 1.55 + 0.04 * (r as f64);
        if r % 2 == 0 {
            term.assignment.logits[[r, 0]] = center + margin;
            term.assignment.logits[[r, 1]] = center - margin;
        } else {
            term.assignment.logits[[r, 0]] = center - 0.85 * margin;
            term.assignment.logits[[r, 1]] = center + 0.85 * margin;
        }
    }
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            0,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("off-kink fixed-state cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    let probes = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (3usize, 1usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
    ];
    for (row, local_pos, var) in probes {
        // Fixed-state central-difference `∂log|H|/∂θ` numerical oracle. NB: the
        // softmax entropy curvature written into `htt` is the Gershgorin
        // `|·|`-majorizer `D = diag(Σ_j|H_kj|)`, whose logit-derivative is
        // PIECEWISE (the `sign(H_kj)` flips where a `H_kj` crosses zero). A
        // *higher-order* stencil is therefore counterproductive; after the
        // decisive fixture above the narrow 2-point central difference is the
        // strongest smooth-branch oracle.
        let (logit_atom, coord_atom, coord_axis) = match var {
            SaeLocalRowVar::Logit { atom } => (Some(atom), None, 0usize),
            SaeLocalRowVar::Coord { atom, axis } => (None, Some(atom), axis),
        };
        let at = |dl: f64| -> f64 {
            let mut t = term.clone();
            if let Some(atom) = logit_atom {
                t.assignment.logits[[row, atom]] += dl;
            } else if let Some(atom) = coord_atom {
                let mut flat = t.assignment.coords[atom].as_flat().clone();
                let idx = row * t.assignment.coords[atom].latent_dim() + coord_axis;
                flat[idx] += dl;
                t.assignment.coords[atom].set_flat(flat.view());
            }
            fixed_state_logdet(t, &target, &rho)
        };
        let fd = (at(h) - at(-h)) / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 2.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ordered_beta_bernoulli() {
    // The integrated marginal's empirical-mass channel couples every row of
    // column `k`, so perturbing one logit shifts every retained row-local
    // assembled `htt` diagonal in that column. `fixed_state_logdet` rebuilds
    // H at the perturbed state, so a single-logit FD captures both the
    // row-local direct-z channel and the global `M_k` channel that
    // `logdet_theta_adjoint` accumulates column-wise. lambda_sparse is the
    // active prior weight (fixed alpha), so the channel is genuinely live.
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    // Same #1625 setup fix as the sibling `..._on_tiny_fixture`: the ordered Beta--Bernoulli prior
    // Hessian is genuinely indefinite in the low-`ρ_sparse` basin, so at the old
    // `ρ_sparse = −1.0` / 5-iter probe the assembled joint `H` was non-PD and
    // `log|H|` (and hence BOTH its FD and the analytic θ-adjoint contraction of
    // `H⁻¹`) is ill-conditioned — the −11 vs −13.6 mismatch was a near-singular
    // conditioning artifact, NOT a derivative error (the analytic matches dense
    // FD to tolerance once a PD stationary cache exists). Lift `ρ_sparse` into the
    // PD region and converge the inner solve so the comparison point EXISTS; no
    // tolerance is weakened.
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    // Probe both atoms across distinct rows so the shared-mass derivative is
    // exercised on both columns, and probe coordinate channels so the whole
    // theta adjoint remains on the same assembled curvature operator.
    //
    // Dense ordered Beta--Bernoulli layout (K = 2, `last_row_layout = None`): per row block, local
    // positions `0..K` are the logit slots (atom = local_pos) and `K..2K` are the
    // coordinate slots (atom = local_pos − K, axis 0), so local_pos 2 ↔ atom 0
    // coord and local_pos 3 ↔ atom 1 coord.
    let probes = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
        (7usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (1usize, 2usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
        (6usize, 3usize, SaeLocalRowVar::Coord { atom: 1, axis: 0 }),
    ];
    for (row, local_pos, var) in probes {
        let mut plus = term.clone();
        let mut minus = term.clone();
        match var {
            SaeLocalRowVar::Logit { atom } => {
                plus.assignment.logits[[row, atom]] += h;
                minus.assignment.logits[[row, atom]] -= h;
            }
            SaeLocalRowVar::Coord { atom, axis } => {
                let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                flat_p[idx] += h;
                flat_m[idx] -= h;
                plus.assignment.coords[atom].set_flat(flat_p.view());
                minus.assignment.coords[atom].set_flat(flat_m.view());
            }
        }
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "ordered Beta--Bernoulli Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

#[test]
pub(crate) fn exact_stationarity_a_minus_b_includes_ordered_beta_bernoulli_shared_mass_hvp() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged ordered Beta--Bernoulli exact-stationarity cache");

    let mut vector = SaeArrowVector {
        t: Array1::<f64>::zeros(cache.delta_t_len()),
        beta: Array1::<f64>::zeros(cache.k),
    };
    let mut flat_logit_direction = Array1::<f64>::zeros(term.n_obs() * term.k_atoms());
    let row_zero_vars = term
        .row_vars_for_cache_row(0, &cache)
        .expect("row-zero variable layout");
    let local = row_zero_vars
        .iter()
        .position(|var| matches!(var, SaeLocalRowVar::Logit { atom: 0 }))
        .expect("ordered assignment must expose atom-zero logit");
    vector.t[cache.row_offsets[0] + local] = 0.7;
    flat_logit_direction[0] = 0.7;

    let correction = term
        .apply_exact_hessian_minus_b(&rho, target.view(), &cache, &vector)
        .expect("base A-B apply");
    let mut doubled_rho = rho.clone();
    doubled_rho.log_lambda_sparse += 2.0_f64.ln();
    let doubled_correction = term
        .apply_exact_hessian_minus_b(&doubled_rho, target.view(), &cache, &vector)
        .expect("doubled-strength A-B apply");
    let expected =
        crate::assignment::ordered_beta_bernoulli_exact_hessian_minus_majorizer_hvp_weighted(
            &term.assignment,
            &rho,
            term.row_loss_weights.as_deref(),
            flat_logit_direction.view(),
        )
        .expect("ordered exact-Hessian helper");

    let mut saw_cross_row = false;
    for row in 0..term.n_obs() {
        let vars = term
            .row_vars_for_cache_row(row, &cache)
            .expect("row variable layout");
        for (local, var) in vars.iter().enumerate() {
            let index = cache.row_offsets[row] + local;
            let actual = doubled_correction.t[index] - correction.t[index];
            let wanted = match *var {
                SaeLocalRowVar::Logit { atom } => expected[row * term.k_atoms() + atom],
                SaeLocalRowVar::Coord { .. } => 0.0,
            };
            assert!(
                (actual - wanted).abs() <= 2.0e-9 * (1.0 + wanted.abs()),
                "row {row} local {local}: scaled A-B difference={actual}, expected={wanted}"
            );
            if row > 0 && matches!(var, SaeLocalRowVar::Logit { atom: 0 }) {
                saw_cross_row |= wanted.abs() > 1.0e-8;
            }
        }
    }
    for beta in 0..cache.k {
        assert_abs_diff_eq!(
            doubled_correction.beta[beta] - correction.beta[beta],
            0.0,
            epsilon = 2.0e-10
        );
    }
    assert!(
        saw_cross_row,
        "the exact integrated marginal must couple the one-row probe into other rows"
    );
}

/// The assembly PSD-majorizes the ordered Beta--Bernoulli curvature
/// unconditionally, so the
/// θ-adjoint must differentiate that SAME majorized operator. This is the
/// metric-first analogue of `..._ordered_beta_bernoulli`: install a rank-2 BehavioralFisher
/// metric (`s = 2 < p = 3`, a genuinely rank-deficient whitening) on the ordered Beta--Bernoulli tiny
/// fixture and check the analytic `Γ` matches the fixed-state dense FD of `log|H|`
/// — both flow through the majorized assembly (`fixed_state_logdet` rebuilds the
/// SAME majorized `H`). This guards the majorized θ-adjoint channels against the
/// majorized criterion log-det in the whitened+rank-deficient regime, where the
/// whitened data curvature cannot dominate the raw indefinite prior pieces.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ordered_beta_bernoulli_low_rank_metric_2144()
 {
    use gam_problem::{RowMetric, pack_probe_factors};
    use std::sync::Arc;
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    let n = term.n_obs();
    let p = term.output_dim();
    let s = 2usize;
    // Deterministic rank-2 output-Fisher sketch, directional (not a scalar × I) so
    // the metric genuinely whitens with a nontrivial null space.
    let mut seed = 0x2144_ABCD_u64;
    let probes = Array3::<f64>::from_shape_fn((n, p, s), |(_, i, kk)| {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let base = if kk == 0 && i == 0 {
            1.2
        } else if kk == 1 && i + 1 == p {
            1.0
        } else {
            0.0
        };
        base + 0.15 * (((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5)
    });
    let u = pack_probe_factors(probes.view()).unwrap();
    term.set_row_metric(RowMetric::behavioral_fisher(Arc::new(u), p, s).unwrap())
        .unwrap();
    assert!(
        term.row_metric()
            .is_some_and(|m| m.whitens_likelihood() && m.metric_rank() < p),
        "rank-{s} metric on p={p} must be a genuinely rank-deficient whitening metric"
    );
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged majorized cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    let probes_idx = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
        (1usize, 2usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
        (6usize, 3usize, SaeLocalRowVar::Coord { atom: 1, axis: 0 }),
    ];
    for (row, local_pos, var) in probes_idx {
        let mut plus = term.clone();
        let mut minus = term.clone();
        match var {
            SaeLocalRowVar::Logit { atom } => {
                plus.assignment.logits[[row, atom]] += h;
                minus.assignment.logits[[row, atom]] -= h;
            }
            SaeLocalRowVar::Coord { atom, axis } => {
                let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                flat_p[idx] += h;
                flat_m[idx] -= h;
                plus.assignment.coords[atom].set_flat(flat_p.view());
                minus.assignment.coords[atom].set_flat(flat_m.view());
            }
        }
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "majorized ordered Beta--Bernoulli Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// gam#2144 — the log-det row jets must be whitened whenever the metric
/// `whitens_likelihood()` at ANY rank, not only when rank-deficient. The
/// arrow-Schur assembly builds the likelihood Hessian from whitened Jacobians
/// (`Jᵀ U Uᵀ J`) under any whitening factor, so a FULL-RANK non-identity factor
/// (here `diag(1, 2, 1.5)`, `rank == p == 3`) rescales the output-space
/// derivatives just like a low-rank sketch does. The pre-fix code gated jet
/// whitening on `ordered_beta_bernoulli_low_rank_whiten()` (`whitens_likelihood && rank < p`), so
/// full-rank whitening left the row jets in RAW output space — differentiating
/// `JᵀJ` against an assembled `Jᵀ U Uᵀ J`. This pins the production
/// `logdet_theta_adjoint` against a fixed-state central difference of the
/// authoritative whitened joint `log|H|`; the unpatched (identity-on-the-jet)
/// path fails it.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_full_rank_whitening_2144() {
    use gam_problem::RowMetric;
    use std::sync::Arc;
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    let n = term.n_obs();
    let p = term.output_dim();
    // Full-rank (rank == p) DIAGONAL non-identity whitening factor U = diag(d).
    // M_n = U Uᵀ = diag(d²) is genuinely non-identity, so the whitened Jacobian
    // Jᵀ U Uᵀ J ≠ JᵀJ, yet the metric has NO null space — whitening engages with
    // no rank-deficiency in play.
    let d = [1.0_f64, 2.0, 1.5];
    assert_eq!(p, d.len(), "diagonal whitening factor width must equal p");
    let s = p;
    let mut u = Array2::<f64>::zeros((n, p * s));
    for row in 0..n {
        for i in 0..p {
            u[[row, i * s + i]] = d[i];
        }
    }
    term.set_row_metric(RowMetric::behavioral_fisher(Arc::new(u), p, s).unwrap())
        .unwrap();
    assert!(
        term.whiten_logdet_row_jets(),
        "full-rank whitening metric must whiten the log-det row jets"
    );
    assert!(
        term.row_metric().is_some_and(|m| m.metric_rank() == p),
        "rank-{s} == p={p} metric must be genuinely full-rank (this test discriminates \
         jet whitening from rank-deficiency handling)"
    );
    // #2144/#1038: the ordered Beta--Bernoulli PSD majorization is now UNCONDITIONAL (any rank, any
    // metric), so the joint Hessian here is the majorized operator too — the
    // historical #1416 non-PD landscape at `log_lambda_sparse = 0.5` no longer
    // exists. Keep the historical PD-island level `−0.8` for continuity (the
    // discriminating property of this test is unchanged either way:
    // `Jᵀ U Uᵀ J ≠ JᵀJ` separates whitened row jets from raw ones, which is
    // what the fixed-state FD comparison pins).
    rho.log_lambda_sparse = -0.8;
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged full-rank whitened cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    let probes_idx = [
        (0usize, 0usize, SaeLocalRowVar::Logit { atom: 0 }),
        (4usize, 1usize, SaeLocalRowVar::Logit { atom: 1 }),
        (1usize, 2usize, SaeLocalRowVar::Coord { atom: 0, axis: 0 }),
        (6usize, 3usize, SaeLocalRowVar::Coord { atom: 1, axis: 0 }),
    ];
    for (row, local_pos, var) in probes_idx {
        let mut plus = term.clone();
        let mut minus = term.clone();
        match var {
            SaeLocalRowVar::Logit { atom } => {
                plus.assignment.logits[[row, atom]] += h;
                minus.assignment.logits[[row, atom]] -= h;
            }
            SaeLocalRowVar::Coord { atom, axis } => {
                let mut flat_p = plus.assignment.coords[atom].as_flat().clone();
                let mut flat_m = minus.assignment.coords[atom].as_flat().clone();
                let idx = row * plus.assignment.coords[atom].latent_dim() + axis;
                flat_p[idx] += h;
                flat_m[idx] -= h;
                plus.assignment.coords[atom].set_flat(flat_p.view());
                minus.assignment.coords[atom].set_flat(flat_m.view());
            }
        }
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "full-rank whitened Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// The ordered Beta--Bernoulli fixed-alpha sparse-strength trace must
/// differentiate the same row-local PSD majorizer the factorization uses. For
/// fixed alpha the prior curvature scales with `lambda_sparse`, so the analytic
/// `assignment_log_strength_hessian_trace` returns `½ ∂log|H|/∂ρ_sparse`; this
/// pins it against a fixed-state central difference of the joint `log|H|`.
#[test]
pub(crate) fn ordered_beta_bernoulli_sparse_strength_trace_matches_dense_fd() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // Fixed-alpha ordered Beta--Bernoulli with an active sparse prior.
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    // Keep a moderate prior strength so the retained diagonal majorizer is live.
    rho.log_lambda_sparse = -0.8;
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let analytic = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("rho_sparse logdet trace");

    // Fixed-state central difference of log|H| w.r.t. ρ_sparse: vary λ_sparse,
    // hold (t, β) at the converged state (`fixed_state_logdet` re-assembles H
    // with inner_max_iter=0). The analytic trace is ½ ∂log|H|/∂ρ_sparse.
    let h = 1.0e-5;
    let mut rho_plus = rho.clone();
    let mut rho_minus = rho.clone();
    rho_plus.log_lambda_sparse += h;
    rho_minus.log_lambda_sparse -= h;
    let fd_half = 0.5
        * (fixed_state_logdet(term.clone(), &target, &rho_plus)
            - fixed_state_logdet(term.clone(), &target, &rho_minus))
        / (2.0 * h);
    let tol = 3.0e-3 * (1.0 + fd_half.abs().max(analytic.abs()));
    assert!(
        (fd_half - analytic).abs() <= tol,
        "ordered Beta--Bernoulli ρ_sparse logdet trace: fd(½∂log|H|/∂ρ)={fd_half:.8e}, \
         analytic={analytic:.8e}"
    );
}

/// Learnable-alpha ordered Beta--Bernoulli logit theta-adjoint.
/// `learnable_alpha = true`, a path the fixed-alpha `..._ordered_beta_bernoulli` sibling never
/// exercises. Under learnable α the resolved weight convention flips (`weight`
/// stays 1.0 and `log_lambda_sparse` drives `α` via `resolve_learnable_weight`
/// instead of scaling the prior), so a single logit perturbation holds alpha
/// fixed and moves only `M_k` and the local sigmoid gate.
///
/// The comparison point must EXIST and be STATIONARY: like the indefinite-basin
/// diagnosis driving the whole #1625 fix, the analytic
/// `Γ = tr(H⁻¹ ∂H/∂θ)` equals the fixed-state central difference of `log|H|`
/// only at a CONVERGED inner cache. A short inner budget (e.g. `iter = 5`) leaves
/// (t, β) non-stationary, and `fixed_state_logdet` (which re-solves with
/// `iter = 0`) then differences `log|H|` about a different state, manufacturing a
/// spurious O(several-%) mismatch that does NOT shrink with the FD step — the
/// tell that it is a state desync, not truncation. Converging the inner solve
/// (`iter = 200`, tol `1e-8`) makes Γ and the FD share one stationary state, and
/// the learnable-α logit adjoint then matches to ≈6 digits.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ordered_beta_bernoulli_learnable_alpha_1625()
 {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    // ρ₀ = 0.6 drives a PD learnable-α cache on this fixture (a sweep shows the
    // default 0.1 and rho0 <= -0.8 were poorly conditioned on this fixture).
    rho.log_lambda_sparse = 0.6;
    let (_value, _loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-8,
            1.0e-8,
        )
        .expect("converged learnable-α cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    // Probe both atoms across distinct rows so the shared-mass channel is
    // exercised on both columns under learnable alpha.
    let probes = [
        (0usize, 0usize, 0usize),
        (4usize, 1usize, 1usize),
        (7usize, 0usize, 0usize),
    ];
    for (row, local_pos, atom) in probes {
        let mut plus = term.clone();
        let mut minus = term.clone();
        plus.assignment.logits[[row, atom]] += h;
        minus.assignment.logits[[row, atom]] -= h;
        let fd = (fixed_state_logdet(plus, &target, &rho)
            - fixed_state_logdet(minus, &target, &rho))
            / (2.0 * h);
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let tol = 3.0e-3 * (1.0 + fd.abs().max(analytic.abs()));
        assert!(
            (fd - analytic).abs() <= tol,
            "learnable-α ordered Beta--Bernoulli Gamma row={row} local_pos={local_pos}: \
             fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// #2080 shared assertion: the matrix-free θ-adjoint
/// ([`SaeManifoldTerm::logdet_theta_adjoint_from_probes`]) reconstructed from the
/// FULL-BASIS probe bundle (`z_j = √k·e_j`, exact dense `S⁻¹` via
/// `cache.schur_inverse_apply`) must reproduce the dense selected-inverse
/// θ-adjoint ([`SaeManifoldTerm::logdet_theta_adjoint`]) on an UNDEFLATED cache —
/// where the plain-`S⁻¹` outer-product estimators are algebraically exact and the
/// Daleckii–Krein correction is identically zero. This isolates the from-probes
/// reconstruction (the dense adjoint is already FD-validated against `log|H|`).
fn assert_theta_adjoint_from_probes_matches_dense(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) {
    let deflated = cache.deflated_row_directions.iter().any(|d| !d.is_empty());
    assert!(
        !deflated,
        "the from-probes parity gate requires an UNDEFLATED cache (the plain-S⁻¹ \
         bundle cannot reconstruct the Daleckii–Krein correction); re-pick ρ so no \
         row deflates"
    );
    let solver = DeflatedArrowSolver::plain(cache);
    let dense = term
        .logdet_theta_adjoint(rho, cache, &solver)
        .expect("dense theta-adjoint");

    let k = cache.k;
    assert!(
        k > 0,
        "fixture must have a non-empty border to exercise S⁻¹ folds"
    );
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<ndarray::Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = ndarray::Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<ndarray::Array1<f64>> = probes
        .iter()
        .map(|v| {
            cache
                .schur_inverse_apply(v.view())
                .expect("schur_inverse_apply")
        })
        .collect();
    let mf = term
        .logdet_theta_adjoint_from_probes(rho, cache, &probes, &sinv)
        .expect("matrix-free theta-adjoint");

    assert_eq!(dense.t.len(), mf.t.len());
    assert_eq!(dense.beta.len(), mf.beta.len());
    let mut max_abs = 0.0_f64;
    for (i, (d, m)) in dense.t.iter().zip(mf.t.iter()).enumerate() {
        assert!(
            (d - m).abs() <= 1.0e-8 * (1.0 + d.abs()),
            "theta-adjoint gamma_t[{i}] mismatch: dense={d:.10e}, from_probes={m:.10e}"
        );
        max_abs = max_abs.max(d.abs());
    }
    for (i, (d, m)) in dense.beta.iter().zip(mf.beta.iter()).enumerate() {
        assert!(
            (d - m).abs() <= 1.0e-8 * (1.0 + d.abs()),
            "theta-adjoint gamma_beta[{i}] mismatch: dense={d:.10e}, from_probes={m:.10e}"
        );
        max_abs = max_abs.max(d.abs());
    }
    assert!(
        max_abs > 0.0 && max_abs.is_finite(),
        "the theta-adjoint must be non-trivial to make the parity check meaningful"
    );
}

/// #2080 θ-adjoint from-probes — SOFTMAX fixture. Exercises the softmax entropy
/// dense off-diagonal channel + the core t–t / t–β / β–β selected-inverse folds.
#[test]
fn sae_logdet_theta_adjoint_from_probes_matches_dense_softmax_2080() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    rho.log_lambda_sparse = 0.5;
    let (_v, _l, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged softmax cache");
    assert_theta_adjoint_from_probes_matches_dense(&term, &rho, &cache);
}

/// The ordered Beta--Bernoulli majorizer is row-local, so the full-basis probe
/// bundle must reproduce the dense theta adjoint exactly.
#[test]
fn sae_logdet_theta_adjoint_from_probes_matches_ordered_beta_bernoulli() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
    rho.log_lambda_sparse = 0.5;
    let (_v, _l, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged ordered Beta--Bernoulli cache");
    assert_theta_adjoint_from_probes_matches_dense(&term, &rho, &cache);
}

/// #2080 θ-adjoint from-probes — DEFLATION hard-refuse. On the known-deflating
/// PD-region learnable-ordered Beta--Bernoulli fixture (per-row gauge deflation surfaced into the cache),
/// the from-probes θ-adjoint must REFUSE (route to the dense channel) rather than
/// silently drop the Daleckii–Krein correction the plain-S⁻¹ bundle cannot rebuild.
#[test]
fn sae_logdet_theta_adjoint_from_probes_refuses_deflated_rows_2080() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
    rho.log_lambda_sparse = 0.5;
    let (_v, _l, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            5,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged learnable ordered Beta--Bernoulli cache");
    assert!(
        cache.deflated_row_directions.iter().any(|d| !d.is_empty()),
        "fixture must genuinely deflate to exercise the hard-refuse (re-pick ρ if not)"
    );
    let k = cache.k;
    let sqrt_k = (k as f64).sqrt();
    let probes: Vec<ndarray::Array1<f64>> = (0..k)
        .map(|j| {
            let mut v = ndarray::Array1::<f64>::zeros(k);
            v[j] = sqrt_k;
            v
        })
        .collect();
    let sinv: Vec<ndarray::Array1<f64>> = probes
        .iter()
        .map(|v| {
            cache
                .schur_inverse_apply(v.view())
                .expect("schur_inverse_apply")
        })
        .collect();
    let result = term.logdet_theta_adjoint_from_probes(&rho, &cache, &probes, &sinv);
    assert!(
        result.is_err(),
        "the from-probes theta-adjoint must refuse a deflated-row cache; got Ok"
    );
}
