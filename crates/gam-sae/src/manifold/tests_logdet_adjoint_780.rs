//! Stationary-cache `∂log|H|/∂θ` adjoint regression tests (#1416/#1417),
//! split verbatim out of `tests.rs` to keep that tracked file under the #780
//! 10k-line gate. Declared as a sibling `#[cfg(test)] mod` in `mod.rs`; shared
//! `gamma_fd_tiny_fixture` / `fixed_state_logdet` are sourced from the sibling
//! `tests` module.

use super::derivative_oracle::{
    BranchCertificate, DerivativeTraceChannel, ExactTraceChannel, ExactTraceReport,
    MajorizerAnchorMode, PivotBranch, dual_spd_logdet, guarded_exact_trace_report,
};
use super::tests::{fixed_state_logdet, gamma_fd_tiny_fixture};
use super::*;

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

fn dual_logdet_trace_2156(
    label: &'static str,
    cache: &ArrowFactorCache,
    h: &Array2<f64>,
    dh: &Array2<f64>,
) -> (f64, BranchCertificate) {
    let report = dual_trace_report_2156(
        cache,
        h,
        vec![(DerivativeTraceChannel::Other(label), dh.clone())],
    );
    (report.total_derivative, report.certificate)
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

fn assert_dual_parity_2156(label: &str, dual: f64, analytic: f64) {
    let rel = relative_error_2156(dual, analytic);
    assert!(
        rel < 1.0e-10,
        "{label} dual-vs-analytic logdet derivative mismatch: \
         dual={dual:.16e}, analytic={analytic:.16e}, rel={rel:.3e}"
    );
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

fn add_row_block_2156(out: &mut Array2<f64>, cache: &ArrowFactorCache, row: usize, block: &Array2<f64>) {
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
    let border = term.border_channels_for_cache(cache).expect("border channels");
    let lambda = rho.lambda_smooth_vec()[atom_idx];
    for left in &border {
        if left.atom != atom_idx {
            continue;
        }
        for right in &border {
            if right.atom != atom_idx {
                continue;
            }
            let s = &term.atoms[atom_idx].smooth_penalty;
            let sym_s = 0.5 * (s[[left.basis_col, right.basis_col]]
                + s[[right.basis_col, left.basis_col]]);
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
    let scale = rho.lambda_sparse() * sparsity / (temperature * temperature);
    for row in 0..term.n_obs() {
        let assignments = crate::assignment::softmax_row(term.assignment.logits.row(row), temperature);
        let a = assignments.as_slice().expect("softmax row");
        let mean = softmax_majorizer_log_mean(a);
        let vars = term
            .row_vars_for_cache_row(row, cache)
            .expect("softmax row vars");
        let mut row_d = Array2::<f64>::zeros((cache.row_dims[row], cache.row_dims[row]));
        for (pos, var) in vars.iter().enumerate() {
            if let SaeLocalRowVar::Logit { atom } = *var {
                row_d[[pos, pos]] = active_softmax_gershgorin_majorizer_entry(
                    a,
                    atom,
                    mean,
                    scale,
                );
            }
        }
        let pushed = row_deflation_pushforward_2156(cache, row, &row_d);
        add_row_block_2156(&mut dh, cache, row, &pushed);
    }
    dh
}

fn ibp_sparse_rho_derivative_matrix_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> Array2<f64> {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let mut dh = Array2::<f64>::zeros((dim, dim));
    let k_atoms = term.k_atoms();
    let mut hdiag = assignment_prior_log_strength_hdiag(&term.assignment, rho).expect("IBP hdiag");
    let mut channels = ibp_assignment_third_channels(&term.assignment, rho, false)
        .expect("IBP channels")
        .expect("IBP sparse derivative requires IBP channels");
    if term.ibp_low_rank_whiten() {
        for row in 0..term.n_obs() {
            for atom in 0..k_atoms {
                let slot = row * k_atoms + atom;
                hdiag[slot] = ibp_majorized_hdiag_2156(
                    &channels,
                    row,
                    k_atoms,
                    atom,
                    hdiag[slot],
                );
            }
        }
        for atom in 0..k_atoms {
            if channels.cross_row_d[atom] < 0.0 {
                channels.cross_row_d[atom] = 0.0;
            }
        }
    }

    let mut sites: Vec<Vec<(usize, usize)>> = vec![Vec::new(); k_atoms];
    for row in 0..term.n_obs() {
        let vars = term.row_vars_for_cache_row(row, cache).expect("IBP row vars");
        let mut row_no_self = Array2::<f64>::zeros((cache.row_dims[row], cache.row_dims[row]));
        let mut row_self = Array2::<f64>::zeros((cache.row_dims[row], cache.row_dims[row]));
        for (pos, var) in vars.iter().enumerate() {
            if let SaeLocalRowVar::Logit { atom } = *var {
                let slot = row * k_atoms + atom;
                let j = channels.z_jac[slot];
                let self_curv = channels.cross_row_d[atom] * j * j;
                row_no_self[[pos, pos]] = hdiag[slot] - self_curv;
                row_self[[pos, pos]] = self_curv;
                sites[atom].push((row, cache.row_offsets[row] + pos));
            }
        }
        let pushed = row_deflation_pushforward_2156(cache, row, &row_no_self);
        add_row_block_2156(&mut dh, cache, row, &pushed);
        add_row_block_2156(&mut dh, cache, row, &row_self);
    }

    for (atom, atom_sites) in sites.iter().enumerate() {
        let d_k = channels.cross_row_d[atom];
        if d_k == 0.0 {
            continue;
        }
        for &(row_i, idx_i) in atom_sites {
            let j_i = channels.z_jac[row_i * k_atoms + atom];
            for &(row_j, idx_j) in atom_sites {
                if row_i != row_j {
                    let j_j = channels.z_jac[row_j * k_atoms + atom];
                    dh[[idx_i, idx_j]] += d_k * j_i * j_j;
                }
            }
        }
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
            AssignmentMode::Softmax { .. } => softmax_sparse_rho_derivative_matrix_2156(term, rho, cache),
            AssignmentMode::IBPMap { .. } => ibp_sparse_rho_derivative_matrix_2156(term, rho, cache),
            _ => panic!("rho sparse derivative fixture must use softmax or IBP"),
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
    let alpha = SaeManifoldRho::stable_exp_strength(rho.log_ard[atom][axis]);
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

fn theta_derivative_matrix_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    row: usize,
    local_w: usize,
) -> Array2<f64> {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let base = cache.row_offsets[row];
    let vars = term
        .row_vars_for_cache_row(row, cache)
        .expect("theta derivative row vars");
    let seed = vars[local_w];
    let second_jets = term.atom_second_jets().expect("second jets");
    let border = term.border_channels_for_cache(cache).expect("border channels");
    let mut assignments = Array1::<f64>::zeros(term.k_atoms());
    term.assignment
        .try_assignments_row_for_rho_into(
            row,
            rho,
            assignments.as_slice_mut().expect("assignment scratch"),
        )
        .expect("theta derivative assignments");
    let mut jets = term
        .row_jets_for_logdet(rho, row, vars, assignments.view(), &second_jets, &border)
        .expect("theta derivative row jets");
    if term.ibp_low_rank_whiten() {
        whiten_row_jets_for_low_rank_metric_2156(term, row, &mut jets);
    }

    let softmax_majorizer = match (term.assignment.mode, seed) {
        (
            AssignmentMode::Softmax {
                temperature,
                sparsity,
            },
            SaeLocalRowVar::Logit { atom },
        ) => {
            let scale = rho.lambda_sparse() * sparsity / (temperature * temperature);
            let row_logits: Vec<f64> = (0..term.k_atoms())
                .map(|k| term.assignment.logits[[row, k]])
                .collect();
            Some((
                atom,
                1.0 / temperature,
                gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
                    term.k_atoms(),
                    temperature,
                )
                .row_psd_majorizer_logit_derivative(&row_logits, scale, atom),
            ))
        }
        _ => None,
    };
    let ibp_channels =
        ibp_assignment_third_channels(&term.assignment, rho, term.ibp_low_rank_whiten())
            .expect("IBP theta channels");

    let mut row_d = Array2::<f64>::zeros((cache.row_dims[row], cache.row_dims[row]));
    for a in 0..jets.vars.len() {
        for b in 0..jets.vars.len() {
            let mut entry = match (softmax_majorizer.as_ref(), jets.vars[a], jets.vars[b]) {
                (
                    Some((atom_w, inv_tau, _)),
                    SaeLocalRowVar::Coord { atom: atom_a, .. },
                    SaeLocalRowVar::Coord { atom: atom_b, .. },
                ) => {
                    let h_ab = sae_dot(&jets.first[a], &jets.first[b]);
                    h_ab
                        * SaeManifoldTerm::softmax_data_weight_product_logit_factor(
                            assignments.as_slice().expect("softmax assignments"),
                            atom_a,
                            atom_b,
                            *atom_w,
                            *inv_tau,
                        )
                }
                _ => {
                    sae_dot(&jets.second[a][local_w], &jets.first[b])
                        + sae_dot(&jets.first[a], &jets.second[b][local_w])
                }
            };
            if let (
                Some((_atom_w, _inv_tau, majorizer)),
                SaeLocalRowVar::Logit { atom: atom_a },
                SaeLocalRowVar::Logit { atom: atom_b },
            ) = (softmax_majorizer.as_ref(), jets.vars[a], jets.vars[b])
            {
                if atom_a == atom_b {
                    entry += majorizer[[atom_a, atom_a]];
                }
            }
            if a == b {
                entry += match jets.vars[a] {
                    SaeLocalRowVar::Logit { atom } => term
                        .assignment_prior_hdiag_derivative_entry(
                            rho,
                            row,
                            atom,
                            seed,
                            ibp_channels.as_ref(),
                        ),
                    SaeLocalRowVar::Coord { atom, axis } if a == local_w => {
                        term.ard_majorized_hessian_derivative(rho, row, atom, axis)
                    }
                    _ => 0.0,
                };
            }
            row_d[[a, b]] = entry;
        }
    }

    let row_d = row_deflation_pushforward_2156(cache, row, &row_d);
    let mut dh = Array2::<f64>::zeros((dim, dim));
    add_row_block_2156(&mut dh, cache, row, &row_d);
    for a in 0..jets.vars.len() {
        for (beta_pos, channel) in border.iter().enumerate() {
            let entry = sae_dot(&jets.second[a][local_w], &jets.beta[beta_pos])
                + sae_dot(&jets.first[a], &jets.beta_deriv[local_w][beta_pos]);
            let t_idx = base + a;
            let b_idx = total_t + channel.index;
            dh[[t_idx, b_idx]] = entry;
            dh[[b_idx, t_idx]] = entry;
        }
    }
    for (beta_i, channel_i) in border.iter().enumerate() {
        for (beta_j, channel_j) in border.iter().enumerate() {
            let entry = sae_dot(&jets.beta_deriv[local_w][beta_i], &jets.beta[beta_j])
                + sae_dot(&jets.beta[beta_i], &jets.beta_deriv[local_w][beta_j]);
            dh[[total_t + channel_i.index, total_t + channel_j.index]] = entry;
        }
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
    let certificate =
        BranchCertificate::from_arrow_cache(cache, MajorizerAnchorMode::FrozenAnchor);
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
        .try_assignments_row_for_rho_into(
            row,
            rho,
            assignments.as_slice_mut().expect("assignment scratch"),
        )
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
    let border = term.border_channels_for_cache(cache).expect("border channels");
    let jets = term
        .row_jets_for_logdet(rho, row, vars, assignments.view(), &second_jets, &border)
        .expect("softmax row jets");
    let mut tt_data = Array2::<f64>::zeros((dim, dim));
    let mut tt_majorizer = Array2::<f64>::zeros((dim, dim));
    let mut t_beta = Array2::<f64>::zeros((dim, dim));
    let mut beta_beta = Array2::<f64>::zeros((dim, dim));

    for a in 0..jets.vars.len() {
        for b in 0..jets.vars.len() {
            let entry = sae_dot(&jets.second[a][local_w], &jets.first[b])
                + sae_dot(&jets.first[a], &jets.second[b][local_w]);
            tt_data[[base + a, base + b]] = entry;
        }
    }

    let scale = rho.lambda_sparse() * sparsity / (temperature * temperature);
    let majorizer_deriv =
        gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty::new(
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
            let entry = sae_dot(&jets.second[a][local_w], &jets.beta[beta_pos])
                + sae_dot(&jets.first[a], &jets.beta_deriv[local_w][beta_pos]);
            let t_idx = base + a;
            let b_idx = total_t + channel.index;
            t_beta[[t_idx, b_idx]] = entry;
            t_beta[[b_idx, t_idx]] = entry;
        }
    }
    for (beta_i, channel_i) in border.iter().enumerate() {
        for (beta_j, channel_j) in border.iter().enumerate() {
            let entry = sae_dot(&jets.beta_deriv[local_w][beta_i], &jets.beta[beta_j])
                + sae_dot(&jets.beta[beta_i], &jets.beta_deriv[local_w][beta_j]);
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

fn install_low_rank_ibp_metric_2156(term: &mut SaeManifoldTerm) {
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
        term.ibp_low_rank_whiten(),
        "rank-{s} metric on p={p} must engage the low-rank IBP majorizer"
    );
}

fn whiten_vec_for_metric_2156(term: &SaeManifoldTerm, row: usize, values: &mut Vec<f64>) {
    let metric = term.row_metric().expect("row metric");
    let p = term.output_dim();
    assert_eq!(values.len(), p, "metric vector width");
    let rank = metric.metric_rank();
    let mut whitened = vec![0.0_f64; rank];
    for rank_col in 0..rank {
        let mut acc = 0.0_f64;
        for out_col in 0..p {
            acc += metric.factor_entry(row, out_col, rank_col) * values[out_col];
        }
        whitened[rank_col] = acc;
    }
    *values = whitened;
}

fn whiten_row_jets_for_low_rank_metric_2156(
    term: &SaeManifoldTerm,
    row: usize,
    jets: &mut SaeRowJets,
) {
    for first in jets.first.iter_mut() {
        whiten_vec_for_metric_2156(term, row, first);
    }
    for second_row in jets.second.iter_mut() {
        for second in second_row.iter_mut() {
            whiten_vec_for_metric_2156(term, row, second);
        }
    }
    for beta in jets.beta.iter_mut() {
        whiten_vec_for_metric_2156(term, row, beta);
    }
    for beta_deriv_row in jets.beta_deriv.iter_mut() {
        for beta_deriv in beta_deriv_row.iter_mut() {
            whiten_vec_for_metric_2156(term, row, beta_deriv);
        }
    }
    for beta_l_deriv_row in jets.beta_l_deriv.iter_mut() {
        for beta_l_deriv in beta_l_deriv_row.iter_mut() {
            whiten_vec_for_metric_2156(term, row, beta_l_deriv);
        }
    }
}

fn ibp_majorized_hdiag_2156(
    channels: &IbpHessianDiagThirdChannels,
    row: usize,
    k_atoms: usize,
    atom: usize,
    raw_hdiag: f64,
) -> f64 {
    let j = channels.z_jac[row * k_atoms + atom];
    let d = channels.cross_row_d[atom];
    let self_term = d * j * j;
    let diag_score_c = raw_hdiag - self_term;
    d.max(0.0) * j * j + diag_score_c.max(0.0)
}

fn ibp_logalpha_dual_channel_report_2156(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
) -> ExactTraceReport {
    let total_t = cache.delta_t_len();
    let dim = total_t + cache.k;
    let k_atoms = term.k_atoms();
    let alpha = term
        .assignment
        .resolved_ibp_alpha(rho)
        .expect("resolved learnable alpha");
    let inv_alpha1 = 1.0 / (alpha + 1.0);
    let second_jets = term.atom_second_jets().expect("second jets");
    let border = term.border_channels_for_cache(cache).expect("border channels");
    let mut assignments = Array1::<f64>::zeros(k_atoms);
    let mut tt_data = Array2::<f64>::zeros((dim, dim));
    let mut tt_majorizer = Array2::<f64>::zeros((dim, dim));
    let mut t_beta = Array2::<f64>::zeros((dim, dim));
    let mut beta_beta = Array2::<f64>::zeros((dim, dim));

    let kfac = |atom: usize, term: &SaeManifoldTerm| -> f64 {
        if term.assignment.ungated.get(atom).copied().unwrap_or(false) {
            0.0
        } else {
            (atom + 1) as f64
        }
    };

    for row in 0..term.n_obs() {
        let base = cache.row_offsets[row];
        let vars = term
            .row_vars_for_cache_row(row, cache)
            .expect("IBP row vars");
        term.assignment
            .try_assignments_row_for_rho_into(
                row,
                rho,
                assignments.as_slice_mut().expect("assignment scratch"),
            )
            .expect("IBP assignments");
        let mut jets = term
            .row_jets_for_logdet(rho, row, vars, assignments.view(), &second_jets, &border)
            .expect("IBP row jets");
        if term.ibp_low_rank_whiten() {
            whiten_row_jets_for_low_rank_metric_2156(term, row, &mut jets);
        }
        let var_atom: Vec<usize> = jets
            .vars
            .iter()
            .map(|v| match *v {
                SaeLocalRowVar::Logit { atom } => atom,
                SaeLocalRowVar::Coord { atom, .. } => atom,
            })
            .collect();

        for a in 0..jets.vars.len() {
            for b in 0..jets.vars.len() {
                let kw = kfac(var_atom[a], term) + kfac(var_atom[b], term);
                tt_data[[base + a, base + b]] =
                    inv_alpha1 * kw * sae_dot(&jets.first[a], &jets.first[b]);
            }
        }
        for a in 0..jets.vars.len() {
            for (beta_pos, channel) in border.iter().enumerate() {
                let kw = kfac(var_atom[a], term) + kfac(channel.atom, term);
                let entry = inv_alpha1 * kw * sae_dot(&jets.first[a], &jets.beta[beta_pos]);
                let t_idx = base + a;
                let b_idx = total_t + channel.index;
                t_beta[[t_idx, b_idx]] = entry;
                t_beta[[b_idx, t_idx]] = entry;
            }
        }
        for (beta_i, channel_i) in border.iter().enumerate() {
            for (beta_j, channel_j) in border.iter().enumerate() {
                let kw = kfac(channel_i.atom, term) + kfac(channel_j.atom, term);
                beta_beta[[total_t + channel_i.index, total_t + channel_j.index]] =
                    inv_alpha1 * kw * sae_dot(&jets.beta[beta_i], &jets.beta[beta_j]);
            }
        }
    }

    let mut hdiag =
        assignment_prior_log_strength_hdiag(&term.assignment, rho).expect("IBP hdiag");
    let mut channels = ibp_assignment_third_channels(&term.assignment, rho, false)
        .expect("IBP channels")
        .expect("IBP channels present");
    if term.ibp_low_rank_whiten() {
        for row in 0..term.n_obs() {
            for atom in 0..k_atoms {
                let slot = row * k_atoms + atom;
                hdiag[slot] =
                    ibp_majorized_hdiag_2156(&channels, row, k_atoms, atom, hdiag[slot]);
            }
        }
        for atom in 0..k_atoms {
            if channels.cross_row_d[atom] < 0.0 {
                channels.cross_row_d[atom] = 0.0;
                channels.cross_row_d_logalpha[atom] = 0.0;
            }
        }
    }

    let mut col_sites: Vec<Vec<(usize, usize)>> = vec![Vec::new(); k_atoms];
    for row in 0..term.n_obs() {
        let base = cache.row_offsets[row];
        let vars = term
            .row_vars_for_cache_row(row, cache)
            .expect("IBP majorizer row vars");
        for (pos, var) in vars.iter().enumerate() {
            if let SaeLocalRowVar::Logit { atom } = *var {
                let idx = base + pos;
                tt_majorizer[[idx, idx]] += hdiag[row * k_atoms + atom];
                col_sites[atom].push((row, idx));
            }
        }
    }
    for atom in 0..k_atoms {
        let d_k = channels.cross_row_d_logalpha[atom];
        if d_k == 0.0 {
            continue;
        }
        for &(row_i, idx_i) in &col_sites[atom] {
            let j_i = channels.z_jac[row_i * k_atoms + atom];
            for &(row_j, idx_j) in &col_sites[atom] {
                if row_i == row_j {
                    continue;
                }
                let j_j = channels.z_jac[row_j * k_atoms + atom];
                tt_majorizer[[idx_i, idx_j]] += d_k * j_i * j_j;
            }
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
    let certificate =
        BranchCertificate::from_arrow_cache(cache, MajorizerAnchorMode::FrozenAnchor);
    assert_branch_certificate_green_2156(label, &certificate);
    eprintln!("gam#2156 {label} rho branch certificate: {certificate:?}");
    let h = dense_cached_arrow_hessian_2156(cache);
    let solver = DeflatedArrowSolver::plain(cache);
    let mut max_rel = 0.0_f64;

    let sparse_dh = rho_logdet_derivative_matrix_2156(term, rho, cache, 0);
    let sparse_dual_half = dual_half_logdet_trace_2156(cache, &h, &sparse_dh);
    let sparse_analytic_half = term
        .assignment_log_strength_hessian_trace(rho, cache, &solver)
        .expect("production sparse rho trace")
        + term
            .learnable_ibp_data_logdet_alpha_trace(rho, cache, &solver)
            .expect("production learnable-alpha data trace");
    max_rel = max_rel.max(assert_dual_trace_matches_analytic_2156(
        label,
        0,
        sparse_dual_half,
        sparse_analytic_half,
    ));

    let lambda_smooth = rho.lambda_smooth_vec();
    let smooth_analytic = term
        .decoder_smoothness_effective_dof_with_solver_per_atom(
            cache,
            &solver,
            &lambda_smooth,
        )
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

fn assert_dual_theta_logdet_parity_2156(
    label: &str,
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
    cache: &ArrowFactorCache,
    probes: &[(usize, usize)],
) -> f64 {
    let certificate =
        BranchCertificate::from_arrow_cache(cache, MajorizerAnchorMode::FrozenAnchor);
    assert_branch_certificate_green_2156(label, &certificate);
    eprintln!("gam#2156 {label} theta branch certificate: {certificate:?}");
    let h = dense_cached_arrow_hessian_2156(cache);
    let solver = DeflatedArrowSolver::plain(cache);
    let gamma = term
        .logdet_theta_adjoint(rho, cache, &solver)
        .expect("production theta adjoint");
    let mut max_rel = 0.0_f64;
    for &(row, local_pos) in probes {
        let dh = theta_derivative_matrix_2156(term, rho, cache, row, local_pos);
        let (dual, theta_certificate) = dual_logdet_trace_2156("theta", cache, &h, &dh);
        eprintln!(
            "gam#2156 {label} theta row={row} local_pos={local_pos} branch certificate: {theta_certificate:?}"
        );
        assert_branch_certificate_green_2156(
            &format!("{label} theta row={row} local_pos={local_pos}"),
            &theta_certificate,
        );
        let analytic = gamma.t[cache.row_offsets[row] + local_pos];
        let rel = relative_error_2156(dual, analytic);
        assert_dual_parity_2156(
            &format!("{label} theta row={row} local_pos={local_pos}"),
            dual,
            analytic,
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
    let certificate =
        BranchCertificate::from_arrow_cache(cache, MajorizerAnchorMode::FrozenAnchor);
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
        .reml_criterion_with_cache(
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
        .reml_criterion_with_cache(
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
    let softmax_theta_max_rel = assert_dual_theta_logdet_parity_2156(
        "softmax",
        &softmax_term,
        &softmax_rho,
        &softmax_cache,
        &softmax_theta_probes,
    );
    let softmax_max_rel = assert_dual_rho_logdet_parity_2156(
        "softmax",
        &softmax_term,
        &softmax_rho,
        &softmax_cache,
    );

    let (mut ibp_term, ibp_target, mut ibp_rho) = gamma_fd_tiny_fixture();
    ibp_term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    install_low_rank_ibp_metric_2156(&mut ibp_term);
    ibp_rho.log_lambda_sparse = 0.6;
    ibp_rho.log_lambda_smooth = vec![-1.6, -1.1];
    let (ibp_value, ibp_loss, ibp_cache) = ibp_term
        .reml_criterion_with_cache(
            ibp_target.view(),
            &ibp_rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged low-rank-metric IBP parity cache");
    assert!(
        ibp_value.is_finite() && ibp_loss.total().is_finite(),
        "IBP parity fixture must produce a finite cache"
    );
    let low_rank_certificate =
        BranchCertificate::from_arrow_cache(&ibp_cache, MajorizerAnchorMode::FrozenAnchor);
    assert!(
        ibp_term.ibp_low_rank_whiten() && low_rank_certificate.cross_row_woodbury_rank > 0,
        "IBP parity fixture must exercise the low-rank metric and IBP Woodbury branch; \
         certificate={low_rank_certificate:?}"
    );
    let ibp_theta_probes: Vec<(usize, usize)> = (0..ibp_cache.n_rows())
        .flat_map(|row| (0..ibp_cache.row_dims[row]).map(move |local| (row, local)))
        .collect();
    let ibp_theta_max_rel = assert_dual_theta_logdet_parity_2156(
        "low_rank_metric_ibp",
        &ibp_term,
        &ibp_rho,
        &ibp_cache,
        &ibp_theta_probes,
    );
    let ibp_max_rel = assert_dual_rho_logdet_parity_2156(
        "low_rank_metric_ibp",
        &ibp_term,
        &ibp_rho,
        &ibp_cache,
    );

    let (mut deflated_term, deflated_target, mut deflated_rho) = gamma_fd_tiny_fixture();
    deflated_term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    deflated_rho.log_lambda_sparse = 0.5;
    let (deflated_value, deflated_loss, deflated_cache) = deflated_term
        .reml_criterion_with_cache(
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
    let deflated_theta_max_rel = assert_dual_theta_logdet_parity_2156(
        "deflated_rows_ibp_theta",
        &deflated_term,
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
        "gam#2156/#2144 dual logdet parity max_rel: softmax_theta={softmax_theta_max_rel:.3e}, softmax_rho={softmax_max_rel:.3e}, low_rank_metric_ibp_theta={ibp_theta_max_rel:.3e}, low_rank_metric_ibp_rho={ibp_max_rel:.3e}, deflated_ibp_theta={deflated_theta_max_rel:.3e}, deflated_ard={deflated_ard_max_rel:.3e}"
    );
}

#[test]
pub(crate) fn branch_guarded_dual_oracle_pins_live_softmax_and_ibp_channels_2156() {
    let (mut softmax_term, target, mut softmax_rho) = gamma_fd_tiny_fixture();
    softmax_rho.log_lambda_sparse = 0.5;
    softmax_term
        .reml_criterion_with_cache(
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
        .reml_criterion_with_cache(
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
    let softmax_report = softmax_logit_dual_channel_report_2156(
        &softmax_term,
        &softmax_rho,
        &softmax_cache,
        0,
        0,
    );
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
    let softmax_exact_total =
        softmax_tt + softmax_majorizer + softmax_t_beta + softmax_beta_beta;
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

    let (mut ibp_term, ibp_target, mut ibp_rho) = gamma_fd_tiny_fixture();
    ibp_term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    install_low_rank_ibp_metric_2156(&mut ibp_term);
    ibp_rho.log_lambda_sparse = 0.6;
    let (ibp_value, ibp_loss, ibp_cache) = ibp_term
        .reml_criterion_with_cache(
            ibp_target.view(),
            &ibp_rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged low-rank-metric IBP cache");
    assert!(
        ibp_value.is_finite() && ibp_loss.total().is_finite(),
        "IBP guard fixture must produce a finite low-rank cache"
    );
    let ibp_solver = DeflatedArrowSolver::plain(&ibp_cache);
    let ibp_report = ibp_logalpha_dual_channel_report_2156(&ibp_term, &ibp_rho, &ibp_cache);
    eprintln!(
        "gam#2156 low-rank IBP branch certificate: {:?}",
        ibp_report.certificate
    );
    let ibp_tt = exact_channel_derivative_2156(&ibp_report, DerivativeTraceChannel::Tt);
    let ibp_majorizer =
        exact_channel_derivative_2156(&ibp_report, DerivativeTraceChannel::Majorizer);
    let ibp_t_beta = exact_channel_derivative_2156(&ibp_report, DerivativeTraceChannel::Border);
    let ibp_beta_beta = exact_channel_derivative_2156(&ibp_report, DerivativeTraceChannel::Beta);
    let ibp_data_exact = 0.5 * (ibp_tt + ibp_t_beta + ibp_beta_beta);
    let ibp_majorizer_exact = 0.5 * ibp_majorizer;
    let ibp_data_production = ibp_term
        .learnable_ibp_data_logdet_alpha_trace(&ibp_rho, &ibp_cache, &ibp_solver)
        .expect("production low-rank IBP data alpha trace");
    let ibp_majorizer_production = ibp_term
        .assignment_log_strength_hessian_trace(&ibp_rho, &ibp_cache, &ibp_solver)
        .expect("production low-rank IBP majorizer alpha trace");
    assert!(
        ibp_tt.abs() > 1.0e-10
            && ibp_majorizer.abs() > 1.0e-10
            && ibp_t_beta.abs() > 1.0e-10
            && ibp_beta_beta.abs() > 1.0e-10,
        "IBP guard must keep every channel live: tt={ibp_tt:.3e}, \
         majorizer={ibp_majorizer:.3e}, tβ={ibp_t_beta:.3e}, ββ={ibp_beta_beta:.3e}"
    );
    assert_close_2156(
        "IBP learnable-alpha data trace vs dual tt/tβ/ββ channels",
        ibp_data_exact,
        ibp_data_production,
        ibp_data_exact,
    );
    assert_close_2156(
        "IBP assignment majorizer trace vs dual tt-majorizer channel",
        ibp_majorizer_exact,
        ibp_majorizer_production,
        ibp_majorizer_exact,
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
        let one_factor = h_ab
            * (if atom_w == atom_a { 1.0 } else { 0.0 } - assignments[atom_w])
            * inv_tau;
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
    term.reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    configure_decisive_softmax_logits_2156(&mut term);
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
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
        .try_assignments_row_for_rho_into(
            row,
            &rho,
            assignments.as_slice_mut().expect("assignment scratch"),
        )
        .expect("assignments");
    let jets = term
        .row_jets_for_logdet(&rho, row, vars, assignments.view(), &second_jets, &border)
        .expect("row jets");

    let mut dh = Array2::<f64>::zeros((dim, dim));
    let majorizer_deriv = match term.assignment.mode {
        AssignmentMode::Softmax {
            temperature,
            sparsity,
        } => {
            let scale = rho.lambda_sparse() * sparsity / (temperature * temperature);
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
            let mut entry = sae_dot(&jets.second[a][local_w], &jets.first[b])
                + sae_dot(&jets.first[a], &jets.second[b][local_w]);
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
            let entry = sae_dot(&jets.second[a][local_w], &jets.beta[beta_pos])
                + sae_dot(&jets.first[a], &jets.beta_deriv[local_w][beta_pos]);
            let global_a = base + a;
            let global_beta = total_t + channel.index;
            dh[[global_a, global_beta]] = entry;
            dh[[global_beta, global_a]] = entry;
        }
    }
    for (beta_i, channel_i) in border.iter().enumerate() {
        for (beta_j, channel_j) in border.iter().enumerate() {
            let entry = sae_dot(&jets.beta_deriv[local_w][beta_i], &jets.beta[beta_j])
                + sae_dot(&jets.beta[beta_i], &jets.beta_deriv[local_w][beta_j]);
            dh[[total_t + channel_i.index, total_t + channel_j.index]] = entry;
        }
    }

    let h_dense = dense_cached_arrow_hessian_2156(&cache);
    let dense_value = dense_cholesky_logdet_2156(&h_dense);
    let (cache_tt, cache_beta) = cache.arrow_log_det();
    let cache_value = cache_tt + cache_beta.expect("dense Schur logdet");
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
        "gam#2156 row=0 logit=0 dense_trace={trace:.12e} gamma={analytic:.12e} fd_total={fd_total:.12e}"
    );
    let trace_tol = 1.0e-5 * (1.0 + trace.abs().max(fd_total.abs()));
    assert!(
        (trace - fd_total).abs() <= trace_tol,
        "gam#2156 operator mismatch: dense trace from row jets does not match fixed-state FD: trace={trace:.12e}, fd_total={fd_total:.12e}"
    );
    let gamma_tol = 1.0e-5 * (1.0 + analytic.abs().max(trace.abs()));
    assert!(
        (analytic - trace).abs() <= gamma_tol,
        "gam#2156 trace assembly mismatch: gamma={analytic:.12e}, dense trace={trace:.12e}, fd_total={fd_total:.12e}"
    );
}

/// #1416 exact NUMERICAL ORACLE for the IBP cross-row log-det derivatives.
///
/// The issue pins a two-row, one-column interior example with a clean,
/// independently-derivable closed form: `α = 1.8`, `τ = 0.8`,
/// `ℓ = (0.2, −0.4)`, and a DATA curvature of exactly `1.2·I` on the two
/// (one-per-row) logit slots. The full joint Hessian is then
/// `H = 1.2·I + H_p`, where the IBP prior column Hessian is
/// `H_p = d·J Jᵀ + diag(s·c)` with `J_i = ∂z_i/∂ℓ_i`, `c_i = ∂²z_i/∂ℓ_i²`,
/// `s` the column score, and `d = ∂s/∂M` (`= w·s'` at unit weight). The
/// cross-row rank-one `d·J Jᵀ` couples the two rows, and its off-diagonal is
/// what the pre-#1416 diagonal-only contractions dropped.
///
/// Oracle values are the exact 2nd/3rd derivatives of the IMPLEMENTED IBP energy
/// (`IBPAssignmentPenalty::value`), verified three independent ways — a
/// from-scratch Python derivative, the production analytic contraction, and a
/// central FD of the cache-built `log|H|` — all agreeing to ≈8 digits:
///   * ρ-trace half-trace `½ tr(H⁻¹ H_p) = −0.1220750367`,
///   * logit adjoint `∂/∂ℓ_2 log|H| = −0.0229591145`.
/// (The pre-#1416 hand-derived constants `−0.1609707929` / `−0.0498935387` did
/// NOT match the implemented energy — the analytic adjoint was correct, those
/// oracle numbers were the mis-derivation; superseded here with FD cross-checks.)
///
/// To exercise the REAL derivative code paths (`assignment_log_strength_hessian_trace`
/// for the ρ-trace and `logdet_theta_adjoint` for the logit adjoint) on EXACTLY
/// this `H`, we drive the production arrow-Schur assembly directly: a 2-row,
/// K=1 IBP term carries the logits, and a hand-built [`ArrowSchurSystem`] with
/// one 1×1 logit slot per row, base diagonal `H₀ = 1.2 + (H_p)_ii`, and the
/// installed [`IbpCrossRowSource`] (the same source the live assembly emits) is
/// factored through `solve_arrow_newton_step_with_options`. The solver downdates
/// the rank-one self term and layers the exact Woodbury correction, so the
/// factored cache reconstructs `H = 1.2·I + H_p` to roundoff — the one operator
/// the value, log-det, ρ-trace, and θ-adjoint all differentiate. The diagonal-only
/// pre-fix contractions FAIL these tight (1e-7) assertions; the cross-row passes
/// pass them.
fn ibp_1416_oracle_term() -> (SaeManifoldTerm, SaeManifoldRho) {
    // A single trivial K=1 atom only supplies `assignment` (logits / mode) to the
    // derivative code; its decoder/coords are never read by the IBP logit-slot
    // contractions, and the cache layout below is hand-built, not assembled from
    // this atom. n = 2, p = 1.
    let n = 2usize;
    let p = 1usize;
    let m = 3usize;
    // A periodic-harmonic atom supplies the second-jet evaluator the θ-adjoint
    // needs, but its decoder is ZERO so the data Gauss-Newton block is identically
    // zero: the logit and coord slots are decoupled, and the data curvature on the
    // logit slots is injected by hand (1.2·I) in the cache builders below. The
    // coords are nonzero arbitrary phases (their jets are real, just multiplied by
    // the zero decoder).
    let coords = Array2::from_shape_vec((n, 1), vec![0.15_f64, 0.65_f64]).unwrap();
    let evaluator = std::sync::Arc::new(PeriodicHarmonicEvaluator::new(m).unwrap());
    // ZERO basis values AND zero decoder: the θ-adjoint reconstructs the data
    // Gauss-Newton jets from the atom's stored `basis_values`/decoder, so zeroing
    // BOTH makes every data jet (`jets.first`/`second`/`beta`) vanish. The data
    // block is then identically zero and `H` is block-diagonal across the logit,
    // coord, and β slots — leaving the IBP assignment-prior logit channels as the
    // sole live source for the logit-slot adjoint, on exactly the oracle `H`.
    let atom = SaeManifoldAtom::new(
        "ibp1416",
        SaeAtomBasisKind::Periodic,
        1,
        Array2::<f64>::zeros((n, m)),
        Array3::<f64>::zeros((n, m, 1)),
        Array2::<f64>::zeros((m, p)),
        Array2::<f64>::eye(m),
    )
    .unwrap()
    .with_basis_second_jet(evaluator);
    let logits = Array2::from_shape_vec((n, 1), vec![0.2_f64, -0.4_f64]).unwrap();
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        logits,
        vec![coords],
        vec![LatentManifold::Circle { period: 1.0 }],
        // alpha = 1.8, tau = 0.8, fixed alpha.
        AssignmentMode::ibp_map(0.8, 1.8, false),
    )
    .unwrap();
    let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    // log_lambda_sparse = 0 ⇒ λ_sparse = 1 ⇒ the IBP penalty weight w = 1 (the
    // oracle's unit weight). The single atom carries a one-element ARD vector.
    let rho = SaeManifoldRho::new(0.0, 0.0, vec![Array1::from_vec(vec![0.0])]);
    (term, rho)
}

/// Build the factored `ArrowFactorCache` for `H = 1.2·I + H_p` on the two
/// logit slots, using the production IBP source + Woodbury machinery. Each row
/// contributes ONE latent slot (its logit); the per-row base diagonal is the
/// FULL `H₀ = 1.2 + (H_p)_ii` (the solver downdates the rank-one self term and
/// re-adds the full `d·J Jᵀ` through the Woodbury carrier).
fn ibp_1416_oracle_cache(term: &SaeManifoldTerm, rho: &SaeManifoldRho) -> ArrowFactorCache {
    let n = term.n_obs();
    let channels = ibp_assignment_third_channels(&term.assignment, rho, false)
        .expect("channels")
        .expect("IBP mode must yield cross-row channels");
    // Full per-row IBP prior diagonal `(H_p)_ii = d·J_i² + s·c_i`, where the
    // diagonal `hessian_diag` already carries `d·J_i² + s·c_i` for IBP. Use the
    // penalty's assembled diagonal so the base matches the live assembly exactly.
    let hdiag = assignment_prior_log_strength_hdiag(&term.assignment, rho).expect("hdiag");
    let data_curv = 1.2_f64;
    let mut sys = ArrowSchurSystem::new(n, 1, 0);
    for row in 0..n {
        // `hdiag[row*K + 0]` is the assignment prior's full logit-slot curvature
        // `(H_p)_ii`; add the data curvature 1.2 to form the full `H₀` diagonal.
        sys.rows[row].htt[[0, 0]] = data_curv + hdiag[row];
    }
    // IBP source: rank R = 1, coefficient d_0 = w·s'_0 = cross_row_d[0]; the two
    // entries place `J_i = z_jac[i]` at row i's logit slot (global index i).
    let entries: Vec<(usize, usize, f64)> =
        (0..n).map(|i| (i, 0usize, channels.z_jac[i])).collect();
    let source = IbpCrossRowSource {
        r: 1,
        d: channels.cross_row_d.clone(),
        entries,
    };
    sys.set_ibp_cross_row_source(source);
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_dt, _db, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).expect("factor H");
    cache
}

/// Coord-aware variant of [`ibp_1416_oracle_cache`] for the θ-adjoint, which
/// (unlike the ρ-trace) walks the per-row jets and therefore needs the row's
/// coordinate slot present in the cache layout. Each row carries TWO latent
/// slots: the logit (local pos 0) and the atom's one coordinate (local pos 1).
/// The decoder is zero (see `ibp_1416_oracle_term`), so there is NO data
/// coupling between the logit and coord slots: the joint `H` is block-diagonal,
/// the logit 2×2 sub-block is exactly `1.2·I + H_p` (the issue's oracle `H`),
/// and the coord slots carry an independent PD curvature. Because `∂H/∂ℓ_w`
/// touches only the logit block and `H` is block-diagonal, the logit-adjoint
/// entry equals `tr((1.2·I+H_p)⁻¹ ∂(1.2·I+H_p)/∂ℓ_w)` — exactly the issue's
/// `∂log|H|/∂ℓ` — independent of the coord curvature value.
fn ibp_1416_oracle_cache_with_coord(
    term: &SaeManifoldTerm,
    rho: &SaeManifoldRho,
) -> ArrowFactorCache {
    let n = term.n_obs();
    let channels = ibp_assignment_third_channels(&term.assignment, rho, false)
        .expect("channels")
        .expect("IBP mode must yield cross-row channels");
    let hdiag = assignment_prior_log_strength_hdiag(&term.assignment, rho).expect("hdiag");
    let data_curv = 1.2_f64;
    // d = 2 latent slots per row ([logit, coord]); the decoder border carries
    // `border_dim = Σ_atoms m·p` channels so `border_channels_for_cache` (which
    // the θ-adjoint calls) matches `cache.k`. The decoder is zero, so the t↔β
    // coupling (`htbeta`) is zero and `H` stays block-diagonal across {logit,
    // coord} and β; `H_ββ = I` is an independent PD constant block whose log-det
    // is invariant under ℓ — it cancels in the logit derivative and the FD.
    let border_dim = term.factored_border_dim();
    let mut sys = ArrowSchurSystem::new(n, 2, border_dim);
    for c in 0..border_dim {
        sys.hbb[[c, c]] = 1.0;
    }
    for row in 0..n {
        sys.rows[row].htt[[0, 0]] = data_curv + hdiag[row]; // logit: full H₀ diagonal
        sys.rows[row].htt[[1, 1]] = 1.0; // coord: independent PD curvature
        // htbeta stays zero (decoder is zero ⇒ no t↔β data coupling).
    }
    // IBP source entries place `J_i` at row i's LOGIT slot, global index 2·i.
    let entries: Vec<(usize, usize, f64)> =
        (0..n).map(|i| (2 * i, 0usize, channels.z_jac[i])).collect();
    let source = IbpCrossRowSource {
        r: 1,
        d: channels.cross_row_d.clone(),
        entries,
    };
    sys.set_ibp_cross_row_source(source);
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_dt, _db, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).expect("factor H");
    cache
}

#[test]
pub(crate) fn ibp_rho_trace_matches_exact_numerical_oracle_1416() {
    let (term, rho) = ibp_1416_oracle_term();
    let cache = ibp_1416_oracle_cache(&term, &rho);
    let solver = DeflatedArrowSolver::plain(&cache);

    // The real ρ-trace contraction returns `½ tr(H⁻¹ ∂H_p/∂ρ) = ½ tr(H⁻¹ H_p)`
    // for fixed alpha (the whole IBP prior scales with λ_sparse = eᵖ).
    let analytic = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("rho-trace");

    // Exact half-trace `½ tr(H⁻¹ H_p)` for `H = 1.2·I + H_p` with `H_p` the TRUE
    // IBP-MAP energy Hessian (`H_p = s'·J Jᵀ + diag(s·c)`), INCLUDING the
    // cross-row off-diagonal `½ s' Σ_{i≠j}(H⁻¹)_{ij} J_i J_j`. Verified against a
    // from-scratch Python second-derivative of `IBPAssignmentPenalty::value` AND
    // the FD numerical oracle below. (The pre-#1416 hand-derived constant
    // `-0.1609707929` did NOT match the implemented energy; it is superseded.)
    const ORACLE: f64 = -0.1220750367;
    assert!(
        (analytic - ORACLE).abs() <= 1.0e-7,
        "IBP ρ-trace exact oracle: analytic={analytic:.10e}, oracle={ORACLE:.10e}"
    );

    // Independent numerical ground truth: for fixed-α IBP the whole prior scales
    // with `λ_sparse = e^ρ`, so `∂H/∂ρ = H_p` (the `1.2·I` data curvature is
    // ρ-independent) and `½ ∂log|H|/∂ρ = ½ tr(H⁻¹ H_p)`. Rebuild the SAME cache
    // at `ρ ± h` (the assembled `hdiag`/`cross_row_d` carry the `e^ρ` weight) and
    // central-difference `log|H|`; the analytic must equal half that FD.
    let fd_rho = |dr: f64| -> f64 {
        let mut r = rho.clone();
        r.log_lambda_sparse += dr;
        let c = ibp_1416_oracle_cache(&term, &r);
        let (tt, beta) = c.arrow_log_det();
        tt + beta.unwrap_or(0.0)
    };
    let h = 1.0e-6;
    let fd_half = 0.5 * (fd_rho(h) - fd_rho(-h)) / (2.0 * h);
    assert!(
        (fd_half - analytic).abs() <= 1.0e-5,
        "IBP ρ-trace vs ½ FD of log|H|: fd_half={fd_half:.8e}, analytic={analytic:.8e}"
    );
}

#[test]
pub(crate) fn ibp_logit_adjoint_matches_exact_numerical_oracle_1416() {
    let (term, rho) = ibp_1416_oracle_term();
    let cache = ibp_1416_oracle_cache_with_coord(&term, &rho);
    let solver = DeflatedArrowSolver::plain(&cache);

    // The real θ-adjoint returns Γ = tr(H⁻¹ ∂H/∂θ) = ∂log|H|/∂θ over the inner
    // variables. Row-1's logit slot is local position 0 of its block, global
    // t-index `row_offsets[1]`.
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("theta-adjoint");
    let analytic = gamma.t[cache.row_offsets[1]];

    // Exact value of `∂/∂ℓ_2 log|H|` for `H = 1.2·I + H_p` with `H_p` the TRUE
    // IBP-MAP energy Hessian (`H_p = s'·J Jᵀ + diag(s·c)`, `s' = ∂score/∂M`).
    // Verified three independent ways: the analytic θ-adjoint contraction below,
    // the central FD of the cache-built `log|H|`, and a from-scratch Python
    // second/third-derivative of `IBPAssignmentPenalty::value` — all agree to
    // ≈8 digits. (The pre-#1416 hand-derived constant `-0.0498935387` did NOT
    // match the implemented energy; it is superseded here.)
    const ORACLE: f64 = -0.0229591145;
    assert!(
        (analytic - ORACLE).abs() <= 1.0e-7,
        "IBP logit adjoint exact oracle ∂/∂ℓ_2 log|H|: analytic={analytic:.10e}, \
         oracle={ORACLE:.10e}"
    );

    // Cross-check the analytic adjoint against a central finite difference of the
    // joint log|H| w.r.t. ℓ_2, holding the rest of the state fixed. The cache is
    // rebuilt at each perturbed logit (its base + Woodbury both depend on ℓ_2),
    // so this FD differentiates the SAME `H = 1.2·I + H_p` the adjoint does — the
    // genuine numerical ground truth for the operator the θ-adjoint contracts.
    let fd_logdet = |dl: f64| -> f64 {
        let mut t = term.clone();
        t.assignment.logits[[1, 0]] += dl;
        let c = ibp_1416_oracle_cache_with_coord(&t, &rho);
        let (tt, beta) = c.arrow_log_det();
        tt + beta.unwrap_or(0.0)
    };
    let h = 1.0e-6;
    let fd = (fd_logdet(h) - fd_logdet(-h)) / (2.0 * h);
    assert!(
        (fd - analytic).abs() <= 1.0e-5,
        "IBP logit adjoint vs FD of log|H|: fd={fd:.8e}, analytic={analytic:.8e}"
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
    term.reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
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
        .reml_criterion_with_cache(target.view(), &rho, None, 0, 0.4, 1.0e-6, 1.0e-6)
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
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ibp_map() {
    // The #1006 empirical-π third channel: under IBP-MAP, pi_k(M_k) couples
    // every row of column k, so perturbing one logit shifts EVERY row's
    // assembled `htt` diagonal in that column. `fixed_state_logdet` rebuilds
    // H at the perturbed state, so a single-logit FD captures both the
    // row-local direct-z channel and the global cross-row M_k channel that
    // `logdet_theta_adjoint` accumulates column-wise. lambda_sparse is the
    // active prior weight (fixed alpha), so the channel is genuinely live.
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    // Same #1625 setup fix as the sibling `..._on_tiny_fixture`: the IBP prior
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
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    // Probe both atoms across distinct rows so the cross-row coupling (different
    // rows sharing a column) is exercised on both columns, AND probe the COORD
    // channel — the #1641 defect made BOTH the logit and the coord channel of the
    // IBP θ-adjoint disagree with dense FD (logit ~4× over tol; coord ~10× off),
    // because the cross-row Woodbury pass double-counted the rank-one self term and
    // carried the ρ-trace ½ instead of the full trace. The coord slots do not pass
    // through the Woodbury pass, but they contract the SAME assembled `htt`
    // (whose IBP diagonal carries the cross-row self curvature), so they guard the
    // one-operator consistency of the whole θ-adjoint, not just the logit lane.
    //
    // Dense IBP layout (K = 2, `last_row_layout = None`): per row block, local
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
            "IBP Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// gam#2144 consistency: under a RANK-DEFICIENT whitening metric the assembly
/// PSD-majorizes the IBP curvature (`ibp_psd_majorized_hdiag` + clamped Woodbury
/// `d`), so the θ-adjoint must differentiate that SAME majorized operator. This is
/// the metric-first analogue of `..._ibp_map`: install a rank-2 BehavioralFisher
/// metric (`s = 2 < p = 3`, so `ibp_low_rank_whiten()` engages) on the IBP tiny
/// fixture and check the analytic `Γ` matches the fixed-state dense FD of `log|H|`
/// — both flow through the majorized assembly (`fixed_state_logdet` rebuilds the
/// SAME majorized `H`). The current FD adjoint tests use NO metric, so this is the
/// only guard that the majorized θ-adjoint channels agree with the majorized
/// evidence log-det; without the gam#2144 contraction gating it disagrees.
#[test]
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ibp_low_rank_metric_2144() {
    use gam_problem::{RowMetric, pack_probe_factors};
    use std::sync::Arc;
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
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
        term.ibp_low_rank_whiten(),
        "rank-{s} metric on p={p} must engage the gam#2144 low-rank IBP majorizer"
    );
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
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
            "majorized IBP Gamma row={row} local_pos={local_pos}: fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// #1416 — the IBP fixed-alpha `ρ_sparse`-trace `½ tr(H⁻¹ ∂H_p/∂ρ_sparse)` must
/// include the FULL cross-row off-diagonal of the rank-one Woodbury source, not
/// just the diagonal. Under IBP-MAP the per-column empirical-mass `M_k` couples
/// every row of column `k` through `H_p = d·J Jᵀ + diag(s, c)`, and for fixed
/// alpha the entire IBP prior scales with `λ_sparse = eᵖ`, so
/// `∂H_p/∂ρ_sparse = H_p`. The analytic
/// `assignment_log_strength_hessian_trace` returns `½ ∂log|H|/∂ρ_sparse`; this
/// pins it against a fixed-state central difference of the joint `log|H|`. A
/// diagonal-only contraction (the pre-#1416 bug) would miss the
/// `½ d Σ_{i≠j}(H⁻¹)_{ij} J_i J_j` cross-row term and fail this FD.
#[test]
pub(crate) fn ibp_rho_sparse_logdet_trace_matches_dense_fd_1416() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // Fixed-alpha IBP-MAP with an active sparse prior so the cross-row Woodbury
    // source is genuinely live.
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    // Fixed-alpha IBP-MAP is PD only on a JAGGED ρ_sparse landscape on this n=10
    // periodic-bilinear fixture: most values (including the previously-pinned 1.0,
    // which a `log_lambda_sparse`-sweep shows is non-PD — the converge call panics
    // with "Schur complement Cholesky failed: non-PD pivot") leave the undamped
    // joint Hessian indefinite at setup. The contiguous island ρ_sparse ∈
    // [−1.0, −0.4] is solidly PD, and −0.8 sits in its interior: it converges to
    // the SAME PD cache for every inner budget (iter ∈ {5…40}), the cross-row
    // Woodbury source is genuinely live there (max|d_k| ≈ 0.21), and the analytic
    // ρ_sparse trace matches the fixed-state central difference of log|H| to ≈10
    // digits. Setup fix only — no tolerance weakened.
    rho.log_lambda_sparse = -0.8;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
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
        "IBP ρ_sparse logdet trace: fd(½∂log|H|/∂ρ)={fd_half:.8e}, \
         analytic={analytic:.8e}"
    );
}

/// #1417 — for LEARNABLE IBP alpha the joint Laplace `log|H|` depends on alpha
/// not only through the prior Hessian but EXPLICITLY through the data
/// Gauss-Newton blocks: `a_ik = σ(ℓ/τ)·π_k(α)`, so `H_ββ`, `H_tβ`, `H_tt` all
/// carry `α`. The complete `½ ∂log|H|/∂logα` is therefore the prior-Hessian
/// trace (`assignment_log_strength_hessian_trace`) PLUS the data trace
/// (`learnable_ibp_data_logdet_alpha_trace`, #1417). The learnable-alpha control
/// is `α(ρ₀) = α_base·e^{ρ₀}` (`resolve_learnable_weight`), so `∂logα/∂ρ₀ = 1`
/// and a fixed-state central difference of `log|H|` w.r.t. ρ₀ must equal twice
/// the SUM of both analytic traces. Omitting the data trace (the pre-#1417 bug)
/// would fail this FD.
#[test]
pub(crate) fn learnable_ibp_alpha_logdet_trace_matches_dense_fd_1417() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    // Learnable-alpha IBP-MAP: ρ₀ (log_lambda_sparse) now drives alpha. The
    // default ρ₀ = 0.1 sits in the indefinite basin and panics at setup (the same
    // basin the passing `..._pd_region_deflation` sibling documents); ρ₀ = 0.5 is
    // PD — exactly the value and inner budget that sibling pins for this same
    // learnable-α fixture, and at it the prior+data trace matches the fixed-state
    // central difference of log|H| to ≈9 digits. Setup fix only — no tolerance
    // weakened.
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    rho.log_lambda_sparse = 0.5;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
        .expect("converged cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    // The full ½ ∂log|H|/∂logα = prior trace + data trace, exactly as
    // `analytic_outer_rho_gradient_components` folds into `logdet_trace[0]`.
    let prior_trace = term
        .assignment_log_strength_hessian_trace(&rho, &cache, &solver)
        .expect("prior-Hessian alpha trace");
    let data_trace = term
        .learnable_ibp_data_logdet_alpha_trace(&rho, &cache, &solver)
        .expect("data-Hessian alpha trace");
    let analytic = prior_trace + data_trace;

    // Fixed-state central difference of log|H| w.r.t. ρ₀ (= log α offset).
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
        "learnable-α logdet trace: fd(½∂log|H|/∂logα)={fd_half:.8e}, \
         analytic(prior+data)={analytic:.8e} (prior={prior_trace:.6e}, \
         data={data_trace:.6e})"
    );
    // The data trace must be a genuine, nonzero contribution (the #1417 term the
    // diagonal-only prior trace omitted) — otherwise the test would pass even if
    // `learnable_ibp_data_logdet_alpha_trace` returned 0.
    assert!(
        data_trace.abs() > 1.0e-9,
        "the #1417 data-Hessian alpha trace must be a live nonzero term; got \
         {data_trace:.3e}"
    );
}

/// #1625 (scope expansion) — the LEARNABLE-α IBP-MAP logit θ-adjoint. This is
/// the cross-row Woodbury logit channel of `Γ = tr(H⁻¹ ∂H/∂ℓ)` under
/// `learnable_alpha = true`, a path the fixed-alpha `..._ibp_map` sibling never
/// exercises. Under learnable α the resolved weight convention flips (`weight`
/// stays 1.0 and `log_lambda_sparse` drives `α` via `resolve_learnable_weight`
/// instead of scaling the prior), so the per-column Woodbury coefficient
/// `d_k = w·s'_k` and its mass-derivative `dd_k = w·s''_k` take DIFFERENT numeric
/// values than the fixed-alpha path — yet a single logit perturbation holds α
/// fixed (it only moves `M_k` and the local `z`), so the same off-diagonal
/// cross-row contraction must hold.
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
pub(crate) fn sae_logdet_theta_adjoint_matches_dense_fd_ibp_map_learnable_alpha_1625() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
    // ρ₀ = 0.6 drives a PD learnable-α cache on this fixture (a sweep shows the
    // default 0.1 and ρ₀ ≤ −0.8 are non-PD for learnable α); the cross-row
    // Woodbury source is genuinely live there.
    rho.log_lambda_sparse = 0.6;
    let (_value, _loss, cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-8, 1.0e-8)
        .expect("converged learnable-α cache");
    let solver = DeflatedArrowSolver::plain(&cache);
    let gamma = term
        .logdet_theta_adjoint(&rho, &cache, &solver)
        .expect("Gamma");
    let h = 1.0e-5;
    // Probe both atoms across distinct rows so the cross-row coupling (different
    // rows sharing a column) is exercised on both columns under learnable α.
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
            "learnable-α IBP Gamma row={row} local_pos={local_pos}: \
             fd={fd:.8e}, analytic={analytic:.8e}"
        );
    }
}

/// #1416 (compact-layout completion) — the IBP cross-row ρ-trace
/// (`assignment_log_strength_hessian_trace`) must add the
/// `½ d_k Σ_{i≠j}(H⁻¹)_{ij} J_i J_j` off-diagonal term under the COMPACT
/// (#1420 top-`k`) row layout, not only the dense layout. The cross-row Woodbury
/// source is installed for both layouts and the θ-adjoint already differentiates
/// both, but the ρ-trace cross-row pass (and the deflation self-curvature
/// downdate) were gated `if last_row_layout.is_none()` — so whenever the budget /
/// `top_k` engaged the compact layout the ρ-gradient of `log|H|` silently dropped
/// the cross-row term.
///
/// A FULL-SUPPORT compact layout (every row active for both atoms) is
/// geometrically IDENTICAL to dense — same logit slots, same assembled `H` — so
/// its `½ ∂log|H|/∂ρ_sparse` must equal both the dense analytic trace and the
/// dense fixed-state central difference. Before the fix the compact trace skipped
/// the cross-row pass and diverged from both; the sibling
/// `ibp_rho_sparse_logdet_trace_matches_dense_fd_1416` confirms the dropped
/// off-diagonal term is genuinely nonzero at this ρ (max|d_k| ≈ 0.21), so this
/// equality is non-vacuous.
#[test]
pub(crate) fn ibp_rho_sparse_logdet_trace_compact_layout_matches_dense_1416() {
    let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
    term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
    // Same solidly-PD island the dense sibling pins (ρ_sparse ∈ [−1.0, −0.4]).
    rho.log_lambda_sparse = -0.8;

    // Converge the inner fit. The tiny fixture's dense Gram is far under the host
    // budget, so production keeps the dense layout (`last_row_layout = None`);
    // this also mutates `term` to the converged (t, β, logit) state.
    let (_value, _loss, dense_cache) = term
        .reml_criterion_with_cache(target.view(), &rho, None, 200, 0.4, 1.0e-6, 1.0e-6)
        .expect("dense converged cache");
    let dense_solver = DeflatedArrowSolver::plain(&dense_cache);
    let analytic_dense = term
        .assignment_log_strength_hessian_trace(&rho, &dense_cache, &dense_solver)
        .expect("dense rho_sparse trace");

    // Re-assemble the SAME converged state under a forced full-support compact
    // layout, factor it, and recompute the ρ-trace. `assemble_arrow_schur_inner`
    // sets `last_row_layout = Some(layout)`, so the trace takes the compact path.
    let n = target.nrows();
    let coord_dims = vec![1usize, 1usize];
    let coord_offsets = term.assignment.coord_offsets();
    let full_active: Vec<Vec<usize>> = (0..n).map(|_| vec![0usize, 1usize]).collect();
    let layout = SaeRowLayout::from_active_atoms(full_active, coord_dims, coord_offsets);
    let probe = SAE_DENSE_BETA_PENALTY_PROBE_MAX_DIM;
    let sys = term
        .assemble_arrow_schur_inner(target.view(), &rho, None, 1.0, probe, Some(Some(layout)))
        .expect("full-support compact assembly");
    let options = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let (_dt, _db, compact_cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options).expect("compact factor");
    let compact_solver = DeflatedArrowSolver::plain(&compact_cache);
    let analytic_compact = term
        .assignment_log_strength_hessian_trace(&rho, &compact_cache, &compact_solver)
        .expect("compact rho_sparse trace");

    // Full-support compact must reproduce the dense trace to roundoff.
    let struct_tol = 1.0e-7 * (1.0 + analytic_dense.abs());
    assert!(
        (analytic_dense - analytic_compact).abs() <= struct_tol,
        "compact-layout IBP ρ_sparse logdet trace must equal the dense trace on \
         full support: dense={analytic_dense:.10e}, compact={analytic_compact:.10e}"
    );

    // And the compact trace must independently match the dense fixed-state central
    // difference of log|H| (the full ½ ∂log|H|/∂ρ_sparse including the cross-row
    // off-diagonal) — FD-validating the compact path itself, not just the
    // dense/compact equality.
    let h = 1.0e-5;
    let mut rho_plus = rho.clone();
    let mut rho_minus = rho.clone();
    rho_plus.log_lambda_sparse += h;
    rho_minus.log_lambda_sparse -= h;
    let fd_half = 0.5
        * (fixed_state_logdet(term.clone(), &target, &rho_plus)
            - fixed_state_logdet(term.clone(), &target, &rho_minus))
        / (2.0 * h);
    let fd_tol = 3.0e-3 * (1.0 + fd_half.abs().max(analytic_compact.abs()));
    assert!(
        (fd_half - analytic_compact).abs() <= fd_tol,
        "compact-layout IBP ρ_sparse logdet trace vs dense FD: \
         fd(½∂log|H|/∂ρ)={fd_half:.8e}, compact analytic={analytic_compact:.8e}"
    );
}
