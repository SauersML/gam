use ndarray::{ArrayView1, ArrayView2, ArrayView3, ArrayViewD, IxDyn};

/// Pivot magnitude at or below which a small matrix is declared singular during
/// Gauss-Jordan inversion.
const SMALL_MATRIX_SINGULAR_TOL: f64 = 1.0e-12;

/// Floor on the SO(3) rotation angle (axis-angle magnitude) used to normalise
/// the rotation axis, avoiding a 0/0 at the identity rotation.
const SO3_ANGLE_FLOOR: f64 = 1.0e-12;

/// Tikhonov ridge added to the diagonal of `WᵀW` before inverting it for the
/// orthogonal projector `P_perp`, keeping the small solve well-conditioned for
/// near-rank-deficient atom bases.
const PROJECTION_WTW_RIDGE: f64 = 1.0e-6;

/// Floor inside the ARD log-bandwidth penalty `½·ln(floor + bw²)`, bounding the
/// log below as `bw → 0` so the penalty stays finite.
const ARD_BANDWIDTH_FLOOR: f64 = 1.0e-3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EquivariantGroup {
    SO2,
    SO3,
    R1,
    Trivial,
}

impl EquivariantGroup {
    fn name(self) -> &'static str {
        match self {
            EquivariantGroup::SO2 => "SO2",
            EquivariantGroup::SO3 => "SO3",
            EquivariantGroup::R1 => "R1",
            EquivariantGroup::Trivial => "TRIVIAL",
        }
    }

    fn rep_dim(self) -> usize {
        match self {
            EquivariantGroup::SO2 => 2,
            EquivariantGroup::SO3 => 3,
            EquivariantGroup::R1 | EquivariantGroup::Trivial => 1,
        }
    }

    fn parse(group: &str) -> Result<Self, String> {
        match group {
            "SO2" => Ok(EquivariantGroup::SO2),
            "SO3" => Ok(EquivariantGroup::SO3),
            "R1" => Ok(EquivariantGroup::R1),
            "Trivial" | "TRIVIAL" => Ok(EquivariantGroup::Trivial),
            other => Err(format!(
                "group must be 'SO2', 'SO3', 'R1', or 'Trivial'; got {other:?}"
            )),
        }
    }
}

fn dynamic_value(
    values: &ArrayViewD<'_, f64>,
    index: &[usize],
    label: &str,
) -> Result<f64, String> {
    values
        .get(IxDyn(index))
        .copied()
        .ok_or_else(|| format!("{label}: index {index:?} out of bounds"))
}

fn invert_small_matrix(matrix: &[Vec<f64>], context: &str) -> Result<Vec<Vec<f64>>, String> {
    let n = matrix.len();
    if n == 0 {
        return Err(format!("{context}: matrix must not be empty"));
    }
    let mut aug = vec![vec![0.0_f64; 2 * n]; n];
    for i in 0..n {
        if matrix[i].len() != n {
            return Err(format!("{context}: matrix must be square"));
        }
        for j in 0..n {
            let value = matrix[i][j];
            if !value.is_finite() {
                return Err(format!("{context}: matrix entry [{i},{j}] is not finite"));
            }
            aug[i][j] = value;
        }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut pivot_abs = aug[col][col].abs();
        for row in (col + 1)..n {
            let candidate = aug[row][col].abs();
            if candidate > pivot_abs {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if pivot_abs <= SMALL_MATRIX_SINGULAR_TOL {
            return Err(format!("{context}: matrix is singular at pivot {col}"));
        }
        if pivot != col {
            aug.swap(pivot, col);
        }
        let scale = aug[col][col];
        for item in &mut aug[col] {
            *item /= scale;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    let mut inverse = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = aug[i][n + j];
        }
    }
    Ok(inverse)
}

fn square_matmul(left: &[Vec<f64>], right: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            let left_ik = left[i][k];
            for j in 0..n {
                out[i][j] += left_ik * right[k][j];
            }
        }
    }
    out
}

fn equivariant_rotation(
    group: EquivariantGroup,
    g: ArrayViewD<'_, f64>,
    batch: usize,
    atom: usize,
) -> Result<Vec<Vec<f64>>, String> {
    match group {
        EquivariantGroup::SO2 => {
            if g.ndim() != 2 {
                return Err("SO2 group coordinates must have shape (B, A)".to_string());
            }
            let theta = dynamic_value(&g, &[batch, atom], "SO2 group coordinates")?;
            let (s, c) = theta.sin_cos();
            Ok(vec![vec![c, -s], vec![s, c]])
        }
        EquivariantGroup::SO3 => {
            if g.ndim() != 3 || g.shape()[2] != 3 {
                return Err("SO3 group coordinates must have shape (B, A, 3)".to_string());
            }
            let ox = dynamic_value(&g, &[batch, atom, 0], "SO3 group coordinates")?;
            let oy = dynamic_value(&g, &[batch, atom, 1], "SO3 group coordinates")?;
            let oz = dynamic_value(&g, &[batch, atom, 2], "SO3 group coordinates")?;
            let angle = (ox * ox + oy * oy + oz * oz).sqrt().max(SO3_ANGLE_FLOOR);
            let ax = ox / angle;
            let ay = oy / angle;
            let az = oz / angle;
            let k = vec![vec![0.0, -az, ay], vec![az, 0.0, -ax], vec![-ay, ax, 0.0]];
            let kk = square_matmul(&k, &k, 3);
            let s = angle.sin();
            let one_minus_c = 1.0 - angle.cos();
            let mut out = vec![vec![0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j] =
                        if i == j { 1.0 } else { 0.0 } + s * k[i][j] + one_minus_c * kk[i][j];
                }
            }
            Ok(out)
        }
        EquivariantGroup::R1 | EquivariantGroup::Trivial => {
            if g.ndim() != 2 {
                return Err(format!(
                    "{} group coordinates must have shape (B, A)",
                    group.name()
                ));
            }
            Ok(vec![vec![1.0]])
        }
    }
}

/// Gauge-companion loss for equivariant terms.
///
/// Penalizes the fitted angular field `theta` against the auxiliary targets
/// `aux_values` (each component in `[0, 1)`). The first companion wraps onto the
/// circle as `1 - cos(theta_0 - 2*pi*aux_0)`; any further companions
/// (`k = 1, 2`, gated by `d_aux` and the available columns) are squared
/// cosine-deviation terms `(cos(theta_k) - (2*aux_k - 1))^2`. Each companion's
/// per-row mean is averaged and scaled by `weight`. Returns `0.0` for an empty
/// batch. This is the single source of truth shared by the CLI/core and the
/// `equivariant_gauge_companion_loss` FFI shim.
pub fn gauge_companion_loss(
    aux_values: ArrayView2<'_, f64>,
    theta: ArrayView2<'_, f64>,
    d_aux: usize,
    weight: f64,
) -> Result<f64, String> {
    if !weight.is_finite() {
        return Err("gauge companion weight must be finite".to_string());
    }
    if aux_values.ncols() < 1 {
        return Err("aux_values must have at least one column".to_string());
    }
    if theta.nrows() != aux_values.nrows() {
        return Err("aux_values and theta must agree in row count".to_string());
    }
    let n = aux_values.nrows();
    if n == 0 {
        return Ok(0.0);
    }
    let n_f = n as f64;
    let mut terms: Vec<f64> = Vec::new();
    let two_pi = std::f64::consts::TAU;
    let mut term0 = 0.0_f64;
    for row in 0..n {
        let h_rad = aux_values[[row, 0]] * two_pi;
        term0 += 1.0 - (theta[[row, 0]] - h_rad).cos();
    }
    terms.push(term0 / n_f);
    if d_aux >= 2 && theta.ncols() >= 2 && aux_values.ncols() >= 2 {
        let mut term1 = 0.0_f64;
        for row in 0..n {
            let diff = theta[[row, 1]].cos() - (2.0 * aux_values[[row, 1]] - 1.0);
            term1 += diff * diff;
        }
        terms.push(term1 / n_f);
    }
    if d_aux >= 3 && theta.ncols() >= 3 && aux_values.ncols() >= 3 {
        let mut term2 = 0.0_f64;
        for row in 0..n {
            let diff = theta[[row, 2]].cos() - (2.0 * aux_values[[row, 2]] - 1.0);
            term2 += diff * diff;
        }
        terms.push(term2 / n_f);
    }
    let total: f64 = terms.iter().sum();
    Ok(weight * total / (terms.len() as f64))
}

/// Computes a weighted projection residual for each atom and batch:
/// 0.5 * z[b, a] * ||P_perp W_a rho(g_ba)||_F^2, where
/// P_perp = I - W_a (W_a^T W_a + eps I)^-1 W_a^T.
///
/// This is not the commutator W rho(g) - rho(g) W. It measures the
/// Frobenius-norm energy in the rotated basis columns that falls outside the
/// column space of W_a.
pub fn equivariant_penalty_value(
    group: &str,
    w: ArrayView3<'_, f64>,
    g: ArrayViewD<'_, f64>,
    z: ArrayView2<'_, f64>,
    weight: f64,
    ard_weight: f64,
    log_bandwidth: Option<ArrayView1<'_, f64>>,
) -> Result<f64, String> {
    if !(weight.is_finite() && weight > 0.0) {
        return Err(format!("weight must be finite and > 0, got {weight}"));
    }
    if !(ard_weight.is_finite() && ard_weight >= 0.0) {
        return Err(format!(
            "ard_weight must be finite and >= 0, got {ard_weight}"
        ));
    }
    let group = EquivariantGroup::parse(group)?;
    let expected_r = group.rep_dim();
    let (n_atoms, ambient_dim, rep_dim) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    if rep_dim != expected_r {
        return Err(format!(
            "{} requires W.shape[2] == {expected_r}; got {rep_dim}",
            group.name()
        ));
    }
    if z.ncols() != n_atoms {
        return Err(format!("z has {} atoms but W has {n_atoms}", z.ncols()));
    }
    let batches = z.nrows();
    if g.ndim() < 2 || g.shape()[0] != batches || g.shape()[1] != n_atoms {
        return Err(format!(
            "g leading dimensions must match z shape ({batches}, {n_atoms})"
        ));
    }
    if let Some(log_bw) = log_bandwidth.as_ref() {
        if log_bw.len() != n_atoms {
            return Err(format!(
                "log_bandwidth length {} must equal atom count {n_atoms}",
                log_bw.len()
            ));
        }
    }

    let mut projection_total = 0.0_f64;
    for atom in 0..n_atoms {
        let mut wtw = vec![vec![0.0_f64; rep_dim]; rep_dim];
        for r1 in 0..rep_dim {
            for r2 in 0..rep_dim {
                let mut acc = 0.0_f64;
                for d in 0..ambient_dim {
                    acc += w[[atom, d, r1]] * w[[atom, d, r2]];
                }
                if r1 == r2 {
                    acc += PROJECTION_WTW_RIDGE;
                }
                wtw[r1][r2] = acc;
            }
        }
        let inv = invert_small_matrix(&wtw, "equivariant_penalty_value WtW")?;
        for batch in 0..batches {
            let rotation = equivariant_rotation(group, g.view(), batch, atom)?;
            let mut w_rot = vec![vec![0.0_f64; rep_dim]; ambient_dim];
            for d in 0..ambient_dim {
                for s_col in 0..rep_dim {
                    let mut acc = 0.0_f64;
                    for r_col in 0..rep_dim {
                        acc += w[[atom, d, r_col]] * rotation[r_col][s_col];
                    }
                    w_rot[d][s_col] = acc;
                }
            }
            let mut cross = vec![vec![0.0_f64; rep_dim]; rep_dim];
            for r_col in 0..rep_dim {
                for s_col in 0..rep_dim {
                    let mut acc = 0.0_f64;
                    for d in 0..ambient_dim {
                        acc += w[[atom, d, r_col]] * w_rot[d][s_col];
                    }
                    cross[r_col][s_col] = acc;
                }
            }
            let solve = square_matmul(&inv, &cross, rep_dim);
            let mut sq = 0.0_f64;
            for d in 0..ambient_dim {
                for s_col in 0..rep_dim {
                    let mut projection = 0.0_f64;
                    for r_col in 0..rep_dim {
                        projection += w[[atom, d, r_col]] * solve[r_col][s_col];
                    }
                    let residual = w_rot[d][s_col] - projection;
                    sq += residual * residual;
                }
            }
            projection_total += 0.5 * z[[batch, atom]] * sq;
        }
    }

    let mut value = weight * projection_total / ((batches * n_atoms) as f64);
    if let Some(log_bw) = log_bandwidth {
        if ard_weight > 0.0 {
            let mut bw_value = 0.0_f64;
            for bandwidth in log_bw.iter().copied() {
                if !bandwidth.is_finite() {
                    return Err("log_bandwidth entries must be finite".to_string());
                }
                bw_value += 0.5 * (ARD_BANDWIDTH_FLOOR + bandwidth * bandwidth).ln();
            }
            value += ard_weight * bw_value;
        }
    }
    Ok(value)
}
