//! Exact coefficient-directional derivatives of the joint Hessian along every coefficient axis
//! for a time wiggle on every frame the ζ composition serves (gam#2893): the second `D²H[u, e_a]`,
//! the third `D³H[u, v, e_a]`, and the design-ψ mixed third derivatives the explicit Jeffreys
//! curvature reads.
//!
//! A row's negative log-likelihood is `ℓ(G(ζ))`. The row coordinate ζ is affine in β. Its
//! z block holds the entry index `h₀`, the exit index `h₁`, the raw derivative index `d`
//! and the wiggle coefficients γ; after it come the linear primaries, which are the slope
//! and the identity-mapped flex coordinates. `G` maps ζ to the primaries. Only three of them
//! are nonlinear: `q₀ = h₀ + B(h₀)·γ`, `q₁ = h₁ + B(h₁)·γ` and `q̇₁ = (1 + B'(h₁)·γ)·d`,
//! and each is at most linear in γ and in `d`. The row Hessian is `Ãᵀ ∇²_ζ(ℓ∘G) Ã`, so
//! `D²H[u, w] = Ãᵀ ∇⁴_ζ(ℓ∘G)[Ãu, Ãw] Ã` and `D³H[u, v, w] = Ãᵀ ∇⁵_ζ(ℓ∘G)[Ãu, Ãv, Ãw] Ã`.
//! A design ψ moves `Ã` itself, through the ζ image `Ã_ψ` of its design-derivative row.
//!
//! Each derivative of the composition is Faà di Bruno over the set partitions of the two free
//! axes and the directions: 5 for the third, 15 for the fourth, 52 for the fifth. When a
//! partition puts both free axes in one block, the ℓ contraction of the other blocks weights a
//! curvature of `G` over both axes. When it separates them, an ℓ derivative sits between two
//! one-axis derivatives of `G`. Every derivative of `G` has a closed form in the wiggle basis
//! derivatives. Every ℓ contraction is linear in the ζ-axis direction. So each row contracts
//! once per primary axis, assembles once per ζ axis, and pulls back once per coefficient axis.

use super::information_third::static_row_fifth;
use super::information_third_dynamic::{contract_fifth_all_primary_axes, dynamic_row_fifth};
use super::*;

/// z-block positions of the entry index, the exit index and the raw derivative index; the
/// wiggle coefficients follow.
const ZETA_H0: usize = 0;
const ZETA_H1: usize = 1;
const ZETA_DR: usize = 2;
const ZETA_GAMMA: usize = 3;
/// The wiggle basis derivatives `B⁽⁰⁾..B⁽⁶⁾`: the fifth derivative of `q̇₁` reaches `m₆`.
const WIGGLE_ORDERS: usize = 7;
/// Bits of the directions `u`, `v` and the ζ-axis direction `z` in a subset mask.
const U: u8 = 0b001;
const V: u8 = 0b010;
const Z: u8 = 0b100;

/// One timepoint's basis derivative rows `B⁽ᵏ⁾(h)` and `m_k = B⁽ᵏ⁾(h)·γ`.
type SideBasis = (Vec<Array1<f64>>, [f64; WIGGLE_ORDERS]);

/// One timepoint's image of a ζ direction: its index component `h` and `B⁽ᵏ⁾(h)·γ` of its
/// wiggle component at every order `k`.
#[derive(Clone, Copy)]
struct SideImage {
    h: f64,
    b: [f64; WIGGLE_ORDERS],
}

/// The z-block image of a ζ direction on both timepoints.
#[derive(Clone, Copy)]
struct ZetaDirection {
    entry: SideImage,
    exit: SideImage,
    dr: f64,
}

impl ZetaDirection {
    const ZERO: Self = Self {
        entry: SideImage {
            h: 0.0,
            b: [0.0; WIGGLE_ORDERS],
        },
        exit: SideImage {
            h: 0.0,
            b: [0.0; WIGGLE_ORDERS],
        },
        dr: 0.0,
    };
}

/// The members of a subset mask over the three directions.
fn members(set: u8) -> impl Iterator<Item = usize> {
    (0..3usize).filter(move |&j| set & (1 << j) != 0)
}

/// `Π_{j∈S} h_j` on one timepoint.
fn side_product(dirs: [&SideImage; 3], set: u8) -> f64 {
    members(set).map(|j| dirs[j].h).product()
}

/// `D^{|S|} m_k [S]` on one timepoint,
/// `m_{k+|S|}·Π_{j∈S} h_j + Σ_{j∈S} (B^{(k+|S|−1)}·γ_j)·Π_{i∈S∖j} h_i`.
/// `m_k` is linear in γ and `h` is affine in β, so no other term survives.
fn side_moved_m(m: &[f64; WIGGLE_ORDERS], dirs: [&SideImage; 3], k: usize, set: u8) -> f64 {
    let size = set.count_ones() as usize;
    let mut value = m[k + size] * side_product(dirs, set);
    for j in members(set) {
        value += dirs[j].b[k + size - 1] * side_product(dirs, set & !(1 << j));
    }
    value
}

/// A row's time-wiggle geometry on both timepoints: the basis derivative rows `B⁽ᵏ⁾(h)` and
/// `m_k = B⁽ᵏ⁾(h)·γ`, with `m₁` carrying the identity of `q = h + B(h)·γ`.
struct WiggleRowGeometry {
    dr: f64,
    entry_basis: Vec<Array1<f64>>,
    exit_basis: Vec<Array1<f64>>,
    entry_m: [f64; WIGGLE_ORDERS],
    exit_m: [f64; WIGGLE_ORDERS],
}

impl WiggleRowGeometry {
    fn gamma_width(&self) -> usize {
        self.entry_basis[0].len()
    }

    /// The image of a z-block direction.
    fn direction(&self, z: ArrayView1<'_, f64>) -> ZetaDirection {
        let gamma = z.slice(s![ZETA_GAMMA..]);
        ZetaDirection {
            entry: SideImage {
                h: z[ZETA_H0],
                b: std::array::from_fn(|k| self.entry_basis[k].dot(&gamma)),
            },
            exit: SideImage {
                h: z[ZETA_H1],
                b: std::array::from_fn(|k| self.exit_basis[k].dot(&gamma)),
            },
            dr: z[ZETA_DR],
        }
    }

    /// `Dⁿ(q₀, q₁, q̇₁)[S]` for a nonempty subset `S`.
    fn q_derivative(&self, dirs: [&ZetaDirection; 3], set: u8) -> [f64; 3] {
        let entry = dirs.map(|direction| &direction.entry);
        let exit = dirs.map(|direction| &direction.exit);
        let mut qd1 = side_moved_m(&self.exit_m, exit, 1, set) * self.dr;
        for j in members(set) {
            qd1 += dirs[j].dr * side_moved_m(&self.exit_m, exit, 1, set & !(1 << j));
        }
        [
            side_moved_m(&self.entry_m, entry, 0, set),
            side_moved_m(&self.exit_m, exit, 0, set),
            qd1,
        ]
    }

    /// `D^{1+|S|}(q₀, q₁, q̇₁)[·, S]` over the z-block axes, one row per nonlinear primary.
    fn q_rows(&self, dirs: [&ZetaDirection; 3], set: u8) -> Array2<f64> {
        let size = set.count_ones() as usize;
        let entry = dirs.map(|direction| &direction.entry);
        let exit = dirs.map(|direction| &direction.exit);
        let entry_product = side_product(entry, set);
        let exit_product = side_product(exit, set);
        let mut rows = Array2::<f64>::zeros((3, ZETA_GAMMA + self.gamma_width()));
        rows[[0, ZETA_H0]] = side_moved_m(&self.entry_m, entry, 1, set);
        rows[[1, ZETA_H1]] = side_moved_m(&self.exit_m, exit, 1, set);
        let mut qd1_h1 = side_moved_m(&self.exit_m, exit, 2, set) * self.dr;
        for j in members(set) {
            qd1_h1 += dirs[j].dr * side_moved_m(&self.exit_m, exit, 2, set & !(1 << j));
        }
        rows[[2, ZETA_H1]] = qd1_h1;
        rows[[2, ZETA_DR]] = side_moved_m(&self.exit_m, exit, 1, set);
        for l in 0..self.gamma_width() {
            rows[[0, ZETA_GAMMA + l]] = self.entry_basis[size][l] * entry_product;
            rows[[1, ZETA_GAMMA + l]] = self.exit_basis[size][l] * exit_product;
            let mut qd1_gamma = self.exit_basis[size + 1][l] * exit_product * self.dr;
            for j in members(set) {
                qd1_gamma +=
                    dirs[j].dr * self.exit_basis[size][l] * side_product(exit, set & !(1 << j));
            }
            rows[[2, ZETA_GAMMA + l]] = qd1_gamma;
        }
        rows
    }

    /// `D^{2+|S|}(q₀, q₁, q̇₁)[·, ·, S]` over the z-block axes, one matrix per nonlinear
    /// primary.
    fn q_matrices(&self, dirs: [&ZetaDirection; 3], set: u8) -> [Array2<f64>; 3] {
        let size = set.count_ones() as usize;
        let entry = dirs.map(|direction| &direction.entry);
        let exit = dirs.map(|direction| &direction.exit);
        let entry_product = side_product(entry, set);
        let exit_product = side_product(exit, set);
        let width = ZETA_GAMMA + self.gamma_width();
        let mut q0 = Array2::<f64>::zeros((width, width));
        let mut q1 = Array2::<f64>::zeros((width, width));
        let mut qd1 = Array2::<f64>::zeros((width, width));
        q0[[ZETA_H0, ZETA_H0]] = side_moved_m(&self.entry_m, entry, 2, set);
        q1[[ZETA_H1, ZETA_H1]] = side_moved_m(&self.exit_m, exit, 2, set);
        let mut qd1_hh = side_moved_m(&self.exit_m, exit, 3, set) * self.dr;
        for j in members(set) {
            qd1_hh += dirs[j].dr * side_moved_m(&self.exit_m, exit, 3, set & !(1 << j));
        }
        qd1[[ZETA_H1, ZETA_H1]] = qd1_hh;
        let qd1_hd = side_moved_m(&self.exit_m, exit, 2, set);
        qd1[[ZETA_H1, ZETA_DR]] = qd1_hd;
        qd1[[ZETA_DR, ZETA_H1]] = qd1_hd;
        for l in 0..self.gamma_width() {
            let g = ZETA_GAMMA + l;
            let entry_cross = self.entry_basis[size + 1][l] * entry_product;
            q0[[ZETA_H0, g]] = entry_cross;
            q0[[g, ZETA_H0]] = entry_cross;
            let exit_cross = self.exit_basis[size + 1][l] * exit_product;
            q1[[ZETA_H1, g]] = exit_cross;
            q1[[g, ZETA_H1]] = exit_cross;
            qd1[[ZETA_DR, g]] = exit_cross;
            qd1[[g, ZETA_DR]] = exit_cross;
            let mut qd1_hg = self.exit_basis[size + 2][l] * exit_product * self.dr;
            for j in members(set) {
                qd1_hg += dirs[j].dr
                    * self.exit_basis[size + 1][l]
                    * side_product(exit, set & !(1 << j));
            }
            qd1[[ZETA_H1, g]] = qd1_hg;
            qd1[[g, ZETA_H1]] = qd1_hg;
        }
        [q0, q1, qd1]
    }
}

/// The sparse ζ image of one coefficient axis. A base time column reaches the entry, exit
/// and derivative indices, which is the most entries any axis has.
#[derive(Clone, Copy, Default)]
struct ZetaImage {
    len: usize,
    entries: [(usize, f64); 3],
}

impl ZetaImage {
    fn push(&mut self, zeta: usize, weight: f64) {
        if weight != 0.0 {
            self.entries[self.len] = (zeta, weight);
            self.len += 1;
        }
    }

    fn entries(&self) -> &[(usize, f64)] {
        &self.entries[..self.len]
    }
}

/// Where the primaries sit in ζ: the nonlinear primaries are the rows of `Ĵ` over the z
/// block, and each linear primary is one ζ coordinate after it.
struct ZetaLayout {
    /// Primary indices of `q₀`, `q₁` and `q̇₁`.
    q: [usize; 3],
    /// `(primary index, ζ index)` of every linear primary.
    linear: Vec<(usize, usize)>,
    z_width: usize,
    width: usize,
}

impl ZetaLayout {
    fn nonlinear_block(&self, m: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn((3, 3), |(i, j)| m[[self.q[i], self.q[j]]])
    }

    /// `Ĵ·ζ`, where `Ĵ = ∂G/∂ζ` has the nonlinear rows `jq` and the identity on the linear
    /// primaries.
    fn primary_image(
        &self,
        jq: &Array2<f64>,
        zeta: &Array1<f64>,
        primary_total: usize,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(primary_total);
        let z = zeta.slice(s![..self.z_width]);
        for (row, &k) in self.q.iter().enumerate() {
            out[k] = jq.row(row).dot(&z);
        }
        for &(k, index) in &self.linear {
            out[k] = zeta[index];
        }
        out
    }

    /// A nonlinear-primary vector on its primary indices.
    fn on_primaries(&self, values: [f64; 3], primary_total: usize) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(primary_total);
        for (row, &k) in self.q.iter().enumerate() {
            out[k] = values[row];
        }
        out
    }

    /// `Φ += Σ_q c_q·K_q` over the z block.
    fn add_curvature(
        &self,
        phi: &mut Array2<f64>,
        coefficients: &Array1<f64>,
        matrices: &[Array2<f64>; 3],
    ) {
        let mut block = phi.slice_mut(s![..self.z_width, ..self.z_width]);
        for (row, &k) in self.q.iter().enumerate() {
            if coefficients[k] != 0.0 {
                block.scaled_add(coefficients[k], &matrices[row]);
            }
        }
    }

    /// `Φ += Ĵᵀ M Ĵ`.
    fn add_full_sandwich(&self, phi: &mut Array2<f64>, jq: &Array2<f64>, m: &Array2<f64>) {
        let zw = self.z_width;
        let zz = jq.t().dot(&self.nonlinear_block(m).dot(jq));
        {
            let mut block = phi.slice_mut(s![..zw, ..zw]);
            block += &zz;
        }
        for &(k, zeta) in &self.linear {
            let mut column = Array1::<f64>::zeros(zw);
            for (row, &qk) in self.q.iter().enumerate() {
                column.scaled_add(m[[qk, k]], &jq.row(row));
            }
            {
                let mut lower = phi.slice_mut(s![..zw, zeta]);
                lower += &column;
            }
            {
                let mut upper = phi.slice_mut(s![zeta, ..zw]);
                upper += &column;
            }
            for &(l, zeta_l) in &self.linear {
                phi[[zeta, zeta_l]] += m[[k, l]];
            }
        }
    }

    /// `Φ += Xᵀ M Ĵ + (Xᵀ M Ĵ)ᵀ` for nonlinear rows `X` over the z block.
    fn add_symmetric_half_sandwich(
        &self,
        phi: &mut Array2<f64>,
        x: &Array2<f64>,
        m: &Array2<f64>,
        jq: &Array2<f64>,
    ) {
        let zw = self.z_width;
        let zz = x.t().dot(&self.nonlinear_block(m).dot(jq));
        {
            let mut block = phi.slice_mut(s![..zw, ..zw]);
            block += &zz;
            block += &zz.t();
        }
        for &(k, zeta) in &self.linear {
            let mut column = Array1::<f64>::zeros(zw);
            for (row, &qk) in self.q.iter().enumerate() {
                column.scaled_add(m[[qk, k]], &x.row(row));
            }
            {
                let mut lower = phi.slice_mut(s![..zw, zeta]);
                lower += &column;
            }
            let mut upper = phi.slice_mut(s![zeta, ..zw]);
            upper += &column;
        }
    }

    /// `Φ += Xᵀ M Y + (Xᵀ M Y)ᵀ` for nonlinear rows `X` and `Y` over the z block.
    fn add_symmetric_row_sandwich(
        &self,
        phi: &mut Array2<f64>,
        x: &Array2<f64>,
        m: &Array2<f64>,
        y: &Array2<f64>,
    ) {
        let zz = x.t().dot(&self.nonlinear_block(m).dot(y));
        let mut block = phi.slice_mut(s![..self.z_width, ..self.z_width]);
        block += &zz;
        block += &zz.t();
    }
}

/// The direction-independent ζ frame of a family: its block ranges, where the primaries sit
/// in ζ, and the ζ coordinate of every linear coefficient.
struct ZetaFrame<'a> {
    knots: &'a Array1<f64>,
    degree: usize,
    slices: BlockSlices,
    primary: FlexPrimarySlices,
    layout: ZetaLayout,
    time_tail: std::ops::Range<usize>,
    /// The ζ index of every slope channel, in the order of `SlopeLayout::primary_channels`: one
    /// for a time-constant slope, three for a follow-up-varying one.
    slope_channels: Vec<usize>,
    /// `(coefficient index, ζ index)` of every identity-mapped flex coefficient.
    identity_images: Vec<(usize, usize)>,
    /// The ζ index of the absorbed influence offset, whose coefficients load through `Z̃_infl`.
    influence_zeta: Option<usize>,
}

/// One row's time-wiggle geometry and the ζ image `Ã` of every coefficient axis.
struct ZetaRow {
    geometry: WiggleRowGeometry,
    images: Vec<ZetaImage>,
}

/// One row's ζ calculus: its geometry and images, the q-map Jacobian `Ĵ` and curvature `D²q`
/// over the z block, and the ℓ derivatives every sweep contracts.
struct ZetaRowCalculus<'a> {
    zeta_row: ZetaRow,
    jq: Array2<f64>,
    k0: [Array2<f64>; 3],
    gradient: Array1<f64>,
    hessian: Array2<f64>,
    /// `ℓ³[e_k]` along every primary axis `k`.
    third: Vec<Array2<f64>>,
    /// The row program every higher ℓ contraction reads.
    program: ZetaRowProgram<'a>,
}

/// `Ã·direction` for the column images `images`.
fn zeta_image_of(images: &[ZetaImage], direction: &Array1<f64>, width: usize) -> Array1<f64> {
    let mut zeta = Array1::<f64>::zeros(width);
    for (c, image) in images.iter().enumerate() {
        if direction[c] != 0.0 {
            for &(index, weight) in image.entries() {
                zeta[index] += weight * direction[c];
            }
        }
    }
    zeta
}

/// `Σ_k w_k·axes[k]` over the primary axes.
fn combine_axes(axes: &[Array2<f64>], weights: &Array1<f64>, primary_total: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((primary_total, primary_total));
    for (axis, &weight) in axes.iter().zip(weights.iter()) {
        if weight != 0.0 {
            out.scaled_add(weight, axis);
        }
    }
    out
}

/// `out += scale·Σ_{(ζ, w)∈image} w·Φ_ζ` over the per-ζ-axis derivatives `phi_axes`.
fn add_combined_axes(phi_axes: &[Array2<f64>], image: &ZetaImage, scale: f64, out: &mut Array2<f64>) {
    for &(zeta, weight) in image.entries() {
        out.scaled_add(scale * weight, &phi_axes[zeta]);
    }
}

/// `acc += Lᵀ Φ R` for one row, where column `c` of `L` is `left[c]` and column `c` of `R`
/// is `right[c]`; `scratch` holds `Φ R`.
fn add_zeta_sandwich(
    phi: &Array2<f64>,
    left: &[ZetaImage],
    right: &[ZetaImage],
    scratch: &mut Array2<f64>,
    acc: &mut Array2<f64>,
) {
    scratch.fill(0.0);
    for (c, image) in right.iter().enumerate() {
        let mut column = scratch.column_mut(c);
        for &(zeta, weight) in image.entries() {
            column.scaled_add(weight, &phi.column(zeta));
        }
    }
    for (c, image) in left.iter().enumerate() {
        let mut row = acc.row_mut(c);
        for &(zeta, weight) in image.entries() {
            row.scaled_add(weight, &scratch.row(zeta));
        }
    }
}

/// `acc[c] += Ãᵀ (Σ_ζ Ã_{ζc}·Φ_ζ) Ã` for every coefficient axis `c` of one row, from the
/// per-ζ-axis derivatives `phi_axes`; `scratch` and `phi_axis` are scratch.
fn pull_back_axes(
    phi_axes: &[Array2<f64>],
    images: &[ZetaImage],
    scratch: &mut Array2<f64>,
    phi_axis: &mut Array2<f64>,
    acc: &mut [Array2<f64>],
) {
    for (c, image) in images.iter().enumerate() {
        if image.entries().is_empty() {
            continue;
        }
        phi_axis.fill(0.0);
        add_combined_axes(phi_axes, image, 1.0, phi_axis);
        add_zeta_sandwich(phi_axis, images, images, scratch, &mut acc[c]);
    }
}

/// The ζ image `Ã_ψ` of every coefficient axis under a design ψ that moves block `block_idx`
/// through the design-derivative row `x_psi` of row `row`: a marginal row moves both indices, and
/// a slope row moves every slope channel. A follow-up-varying slope lifts a covariate row onto its
/// three channels through the layout's time margin.
fn psi_zeta_images(
    family: &SurvivalMarginalSlopeFamily,
    frame: &ZetaFrame<'_>,
    row: usize,
    block_idx: usize,
    x_psi: &Array1<f64>,
) -> Result<Vec<ZetaImage>, String> {
    let slices = &frame.slices;
    let mut images = vec![ZetaImage::default(); slices.total];
    let (range, channel_rows): (std::ops::Range<usize>, Vec<(usize, Array1<f64>)>) =
        match block_idx {
            1 => (
                slices.marginal.clone(),
                vec![(ZETA_H0, x_psi.clone()), (ZETA_H1, x_psi.clone())],
            ),
            2 => {
                let channel_rows = match (
                    family.slope_layout.time_margin(),
                    frame.slope_channels.as_slice(),
                ) {
                    (Some(margin), channels) => channels
                        .iter()
                        .copied()
                        .zip(margin.lift_row(row, x_psi))
                        .collect(),
                    (None, &[zeta]) => vec![(zeta, x_psi.clone())],
                    (None, channels) => {
                        return Err(format!(
                            "time-wiggle ζ composition: a slope with {} channels records no time \
                             margin to lift a covariate design ψ onto them",
                            channels.len()
                        ));
                    }
                };
                (slices.slope.clone(), channel_rows)
            }
            _ => {
                return Err(format!(
                    "time-wiggle ζ composition: a design ψ on block {block_idx} has no ζ image"
                ));
            }
        };
    for (zeta, design_row) in &channel_rows {
        if design_row.len() != range.len() {
            return Err(format!(
                "time-wiggle ζ composition: a design ψ row has {} entries for a block of width {}",
                design_row.len(),
                range.len()
            ));
        }
        for (local, &value) in design_row.iter().enumerate() {
            images[range.start + local].push(*zeta, value);
        }
    }
    Ok(images)
}

/// The direction-independent parts of one row's ζ composition that every order reads: where the
/// primaries sit in ζ, the wiggle geometry, `Ĵ` and `D²q` over the z block, and the ℓ gradient and
/// Hessian.
struct ZetaParts<'a> {
    layout: &'a ZetaLayout,
    geometry: &'a WiggleRowGeometry,
    jq: &'a Array2<f64>,
    k0: &'a [Array2<f64>; 3],
    gradient: &'a Array1<f64>,
    hessian: &'a Array2<f64>,
}

impl ZetaParts<'_> {
    /// `∇_ζ(ℓ∘G) = Ĵᵀ∇ℓ`.
    fn order_one(&self) -> Array1<f64> {
        let layout = self.layout;
        let mut out = Array1::<f64>::zeros(layout.width);
        for (row, &k) in layout.q.iter().enumerate() {
            let mut z_block = out.slice_mut(s![..layout.z_width]);
            z_block.scaled_add(self.gradient[k], &self.jq.row(row));
        }
        for &(k, zeta) in &layout.linear {
            out[zeta] = self.gradient[k];
        }
        out
    }

    /// `∇²_ζ(ℓ∘G) = Ĵᵀ∇²ℓĴ + Σ_q ∂_qℓ·D²q`.
    fn order_two(&self) -> Array2<f64> {
        let mut phi = Array2::<f64>::zeros((self.layout.width, self.layout.width));
        self.layout.add_full_sandwich(&mut phi, self.jq, self.hessian);
        self.layout.add_curvature(&mut phi, self.gradient, self.k0);
        phi
    }

    /// `∇³_ζ(ℓ∘G)[x]` for a ζ direction `x`, reading the ℓ contraction `third` along primary
    /// directions: Faà di Bruno over the 5 set partitions of the two free axes and `x`.
    fn order_three(
        &self,
        x: &Array1<f64>,
        third: &dyn Fn(&Array1<f64>) -> Result<Array2<f64>, String>,
    ) -> Result<Array2<f64>, String> {
        let (layout, geometry) = (self.layout, self.geometry);
        let p_primary = self.gradient.len();
        let dir_x = geometry.direction(x.slice(s![..layout.z_width]));
        let dirs = [&ZetaDirection::ZERO, &ZetaDirection::ZERO, &dir_x];
        let jx = layout.primary_image(self.jq, x, p_primary);
        let jx1 = geometry.q_rows(dirs, Z);
        let mut phi = Array2::<f64>::zeros((layout.width, layout.width));

        // Both free axes in one block.
        layout.add_curvature(&mut phi, &self.hessian.dot(&jx), self.k0);
        layout.add_curvature(&mut phi, self.gradient, &geometry.q_matrices(dirs, Z));

        // Free axes in separate blocks.
        layout.add_full_sandwich(&mut phi, self.jq, &third(&jx)?);
        layout.add_symmetric_half_sandwich(&mut phi, &jx1, self.hessian, self.jq);
        Ok(phi)
    }

    /// `∇⁴_ζ(ℓ∘G)[u, v]` for ζ directions `u` and `v`, reading the ℓ contractions `third` and
    /// `fourth` along primary directions: Faà di Bruno over the 15 set partitions of the two free
    /// axes, `u` and `v`. `timewiggle_order_four_axes` evaluates the same composition along every
    /// ζ axis at once.
    fn order_four(
        &self,
        u: &Array1<f64>,
        v: &Array1<f64>,
        third: &dyn Fn(&Array1<f64>) -> Result<Array2<f64>, String>,
        fourth: &dyn Fn(&Array1<f64>, &Array1<f64>) -> Result<Array2<f64>, String>,
    ) -> Result<Array2<f64>, String> {
        let (layout, geometry) = (self.layout, self.geometry);
        let (jq, hessian) = (self.jq, self.hessian);
        let p_primary = self.gradient.len();
        let dir_u = geometry.direction(u.slice(s![..layout.z_width]));
        let dir_v = geometry.direction(v.slice(s![..layout.z_width]));
        let dirs = [&dir_u, &ZetaDirection::ZERO, &dir_v];
        let ju = layout.primary_image(jq, u, p_primary);
        let jv = layout.primary_image(jq, v, p_primary);
        let g2uv = layout.on_primaries(geometry.q_derivative(dirs, U | Z), p_primary);
        let ju1 = geometry.q_rows(dirs, U);
        let jv1 = geometry.q_rows(dirs, Z);
        let juv2 = geometry.q_rows(dirs, U | Z);
        let t_u = third(&ju)?;
        let t_v = third(&jv)?;
        let mut phi = Array2::<f64>::zeros((layout.width, layout.width));

        // Both free axes in one block: the ℓ contraction of the remaining blocks weights the
        // curvature of G over both axes.
        layout.add_curvature(&mut phi, &(t_u.dot(&jv) + hessian.dot(&g2uv)), self.k0);
        layout.add_curvature(&mut phi, &hessian.dot(&jv), &geometry.q_matrices(dirs, U));
        layout.add_curvature(&mut phi, &hessian.dot(&ju), &geometry.q_matrices(dirs, Z));
        layout.add_curvature(&mut phi, self.gradient, &geometry.q_matrices(dirs, U | Z));

        // Free axes in separate blocks: an ℓ derivative between one-axis derivatives of G.
        layout.add_full_sandwich(&mut phi, jq, &(fourth(&ju, &jv)? + third(&g2uv)?));
        layout.add_symmetric_half_sandwich(&mut phi, &ju1, &t_v, jq);
        layout.add_symmetric_half_sandwich(&mut phi, &jv1, &t_u, jq);
        layout.add_symmetric_row_sandwich(&mut phi, &ju1, hessian, &jv1);
        layout.add_symmetric_half_sandwich(&mut phi, &juv2, hessian, jq);
        Ok(phi)
    }
}

/// `out += scale·Lᵀ values` for one row, where column `c` of `L` is `images[c]`.
fn add_pulled_back(images: &[ZetaImage], values: &Array1<f64>, scale: f64, out: &mut Array1<f64>) {
    for (c, image) in images.iter().enumerate() {
        for &(zeta, weight) in image.entries() {
            out[c] += scale * weight * values[zeta];
        }
    }
}

impl ZetaRowCalculus<'_> {
    /// The direction-independent parts every order of the composition reads.
    fn parts<'b>(&'b self, layout: &'b ZetaLayout) -> ZetaParts<'b> {
        ZetaParts {
            layout,
            geometry: &self.zeta_row.geometry,
            jq: &self.jq,
            k0: &self.k0,
            gradient: &self.gradient,
            hessian: &self.hessian,
        }
    }
}

/// A row's ℓ contractions along primary directions from the family's own row program: the FLEX
/// program through its direction-independent base, or the rigid program at its resolved primary
/// point, each built once per row.
enum ZetaRowProgram<'a> {
    Flex(FlexThirdRowBase),
    Rigid(RigidRowPoint<'a>),
}

/// A rigid row's program point, resolved once per row: its scalar inputs and its primaries in the
/// family's slope frame, and, where the composition reaches order five, its fifth likelihood
/// derivative tensor. Resolving the primaries rebuilds the row's time-wiggle I-spline geometry,
/// and the ζ composition contracts ℓ once per primary axis and direction, so re-resolving them
/// per contraction rebuilt that basis per row, per axis (gam#3304). Every contraction here runs
/// only the jet at this point, and the direction-free fifth tensor is contracted per `(u, v)`.
struct RigidRowPoint<'a> {
    inputs: RigidRowInputs<'a>,
    primaries: Vec<f64>,
    fifth: Option<RigidRowFifth>,
}

/// The fifth likelihood derivatives `ℓ_{abcde}` of a rigid row, in its slope frame.
enum RigidRowFifth {
    Static(Box<[[[[[f64; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES]; STATIC_SLOPE_PRIMARIES]>),
    Dynamic(Box<[[[[[f64; DYNAMIC_SLOPE_PRIMARIES]; DYNAMIC_SLOPE_PRIMARIES]; DYNAMIC_SLOPE_PRIMARIES]; DYNAMIC_SLOPE_PRIMARIES]; DYNAMIC_SLOPE_PRIMARIES]>),
}

/// A stack `P × P` primary tensor as an `Array2`.
fn primary_square_array<const P: usize>(tensor: [[f64; P]; P]) -> Array2<f64> {
    Array2::from_shape_fn((P, P), |(a, b)| tensor[a][b])
}

/// `primaries` in the frame of width `P` it was resolved in.
fn rigid_frame_point<const P: usize>(primaries: &[f64]) -> Result<[f64; P], String> {
    <[f64; P]>::try_from(primaries).map_err(|_| {
        format!(
            "rigid row point holds {} primaries, but its slope frame has {P}",
            primaries.len()
        )
    })
}

/// One row's ζ value calculus for the design-ψ terms: its geometry and images, `Ĵ` and `D²q` over
/// the z block, and the ℓ gradient, Hessian and contractions of the family's own row program.
struct ZetaPsiRow<'a> {
    zeta_row: ZetaRow,
    jq: Array2<f64>,
    k0: [Array2<f64>; 3],
    gradient: Array1<f64>,
    hessian: Array2<f64>,
    program: ZetaRowProgram<'a>,
}

impl ZetaPsiRow<'_> {
    /// The direction-independent parts every order of the composition reads.
    fn parts<'b>(&'b self, layout: &'b ZetaLayout) -> ZetaParts<'b> {
        ZetaParts {
            layout,
            geometry: &self.zeta_row.geometry,
            jq: &self.jq,
            k0: &self.k0,
            gradient: &self.gradient,
            hessian: &self.hessian,
        }
    }
}

/// `∇³_ζ(ℓ∘G)[e_ζ]` for every ζ axis.
fn order_three_axes(
    frame: &ZetaFrame<'_>,
    calc: &ZetaRowCalculus,
) -> Result<Vec<Array2<f64>>, String> {
    let layout = &frame.layout;
    let p_primary = frame.primary.total;
    let parts = calc.parts(layout);
    let third = |direction: &Array1<f64>| -> Result<Array2<f64>, String> {
        Ok(combine_axes(&calc.third, direction, p_primary))
    };
    (0..layout.width)
        .map(|zeta_axis| {
            let mut unit = Array1::<f64>::zeros(layout.width);
            unit[zeta_axis] = 1.0;
            parts.order_three(&unit, &third)
        })
        .collect()
}

/// The flat coefficient vector of `block_states`, in block order.
fn flat_beta(block_states: &[ParameterBlockState]) -> Result<Array1<f64>, String> {
    let views: Vec<ArrayView1<'_, f64>> = block_states.iter().map(|state| state.beta.view()).collect();
    ndarray::concatenate(Axis(0), &views).map_err(|error| error.to_string())
}

impl SurvivalMarginalSlopeFamily {
    /// The ζ frame of this family. A family without a time-wiggle basis has no z block, and the
    /// FLEX row program carries one slope primary, so a follow-up-varying slope beside it has no ζ
    /// frame; both are refused.
    fn timewiggle_zeta_frame(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<ZetaFrame<'_>, String> {
        let flex = self.flex_active();
        if flex && self.slope_is_follow_up_varying() {
            return Err(
                "time-wiggle ζ composition: the FLEX row program carries one slope primary, so a \
                 follow-up-varying slope has no ζ frame"
                    .to_string(),
            );
        }
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Err(
                "time-wiggle ζ composition: the family has no time-wiggle basis".to_string(),
            );
        };
        let slices = block_slices(self, block_states);
        let mut primary = flex_primary_slices(self);
        if !flex {
            // The rigid row program's own frame: four primaries, or six beside a follow-up-varying
            // slope.
            primary.total = self.core_primary_dimension();
        }
        let time_tail = self.time_wiggle_range();
        let z_width = ZETA_GAMMA + time_tail.len();
        let q = [primary.q0, primary.q1, primary.qd1];
        let linear: Vec<(usize, usize)> = (0..primary.total)
            .filter(|k| !q.contains(k))
            .enumerate()
            .map(|(position, k)| (k, z_width + position))
            .collect();
        let mut zeta_of_primary = vec![None; primary.total];
        for &(k, zeta) in &linear {
            zeta_of_primary[k] = Some(zeta);
        }
        let slope_channels = self
            .slope_layout
            .primary_channels()
            .as_slice()
            .iter()
            .map(|&(slope_primary, design)| {
                zeta_of_primary[slope_primary].ok_or_else(|| {
                    format!(
                        "time-wiggle ζ composition: the slope primary {slope_primary} of a \
                         {}-column channel has no ζ coordinate",
                        design.ncols()
                    )
                })
            })
            .collect::<Result<Vec<usize>, String>>()?;
        let mut identity_images = Vec::new();
        for (primary_range, joint_range) in flex_identity_block_pairs(&primary, &slices) {
            if primary_range.len() != joint_range.len() {
                return Err(format!(
                    "time-wiggle ζ composition: flex primaries {primary_range:?} and \
                     coefficients {joint_range:?} differ in width"
                ));
            }
            for local in 0..primary_range.len() {
                let zeta = zeta_of_primary[primary_range.start + local].ok_or_else(|| {
                    "time-wiggle ζ composition: a flex primary has no ζ coordinate".to_string()
                })?;
                identity_images.push((joint_range.start + local, zeta));
            }
        }
        let influence_zeta = primary
            .infl
            .map(|k| {
                zeta_of_primary[k].ok_or_else(|| {
                    "time-wiggle ζ composition: the influence primary has no ζ coordinate".to_string()
                })
            })
            .transpose()?;
        let layout = ZetaLayout {
            q,
            width: z_width + linear.len(),
            z_width,
            linear,
        };
        Ok(ZetaFrame {
            knots,
            degree,
            slices,
            primary,
            layout,
            time_tail,
            slope_channels,
            identity_images,
            influence_zeta,
        })
    }

    /// Row `row`'s design rows, indices and wiggle basis derivatives, and the ζ image of every
    /// coefficient axis.
    fn timewiggle_zeta_row(
        &self,
        frame: &ZetaFrame<'_>,
        block_states: &[ParameterBlockState],
        row: usize,
    ) -> Result<ZetaRow, String> {
        let slices = &frame.slices;
        let p_base = frame.time_tail.start;
        let gamma_width = frame.time_tail.len();
        let beta_time = &block_states[0].beta;
        let beta_base = beta_time.slice(s![..p_base]);
        let gamma = beta_time.slice(s![frame.time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
        let derivative_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
        let marginal_chunk = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
        let xe = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let xx = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let xd = derivative_chunk.row(0).slice(s![..p_base]).to_owned();
        let mr = marginal_chunk.row(0);
        let bm = block_states[1].eta[row];
        let h0 = xe.dot(&beta_base) + self.offset_entry[row] + bm;
        let h1 = xx.dot(&beta_base) + self.offset_exit[row] + bm;
        let dr = xd.dot(&beta_base) + self.derivative_offset_exit[row];
        let side = |h: f64| -> Result<SideBasis, String> {
            let seed = Array1::from_vec(vec![h]);
            let mut basis = Vec::with_capacity(WIGGLE_ORDERS);
            for order in 0..WIGGLE_ORDERS {
                let rows = monotone_wiggle_basis_with_derivative_order(
                    seed.view(),
                    frame.knots,
                    frame.degree,
                    order,
                )?;
                if rows.ncols() != gamma_width {
                    return Err(format!(
                        "time-wiggle ζ composition: basis derivative {order} has {} columns for \
                         {gamma_width} wiggle coefficients",
                        rows.ncols()
                    ));
                }
                basis.push(rows.row(0).to_owned());
            }
            let mut m: [f64; WIGGLE_ORDERS] = std::array::from_fn(|k| basis[k].dot(&gamma));
            m[1] += 1.0;
            Ok((basis, m))
        };
        let (entry_basis, entry_m) = side(h0)?;
        let (exit_basis, exit_m) = side(h1)?;

        let mut images = vec![ZetaImage::default(); slices.total];
        for a in 0..p_base {
            let image = &mut images[slices.time.start + a];
            image.push(ZETA_H0, xe[a]);
            image.push(ZETA_H1, xx[a]);
            image.push(ZETA_DR, xd[a]);
        }
        for l in 0..gamma_width {
            images[slices.time.start + p_base + l].push(ZETA_GAMMA + l, 1.0);
        }
        for j in 0..slices.marginal.len() {
            let image = &mut images[slices.marginal.start + j];
            image.push(ZETA_H0, mr[j]);
            image.push(ZETA_H1, mr[j]);
        }
        let slope_designs = self.slope_layout.primary_channels();
        for (channel, &zeta) in frame.slope_channels.iter().enumerate() {
            let chunk = slope_designs.as_slice()[channel]
                .1
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("slope channel design try_row_chunk: {e}"))?;
            for (b, &value) in chunk.row(0).iter().enumerate() {
                images[slices.slope.start + b].push(zeta, value);
            }
        }
        for &(joint, zeta) in &frame.identity_images {
            images[joint].push(zeta, 1.0);
        }
        if let (Some(zeta), Some(range), Some(z_tilde)) = (
            frame.influence_zeta,
            slices.influence.as_ref(),
            self.influence_absorber.as_ref(),
        ) {
            for (local, &value) in z_tilde.row(row).iter().enumerate() {
                images[range.start + local].push(zeta, value);
            }
        }
        Ok(ZetaRow {
            geometry: WiggleRowGeometry {
                dr,
                entry_basis,
                exit_basis,
                entry_m,
                exit_m,
            },
            images,
        })
    }

    /// Row `row`'s ζ calculus from the family's own row program: the FLEX base, built through the
    /// order-five moments where `fifth_order` holds and the order-four moments otherwise, or the
    /// rigid row program with its closed-form likelihood derivatives.
    fn timewiggle_zeta_row_calculus(
        &self,
        frame: &ZetaFrame<'_>,
        block_states: &[ParameterBlockState],
        row: usize,
        fifth_order: bool,
    ) -> Result<ZetaRowCalculus<'_>, String> {
        let zeta_row = self.timewiggle_zeta_row(frame, block_states, row)?;
        let zero = [&ZetaDirection::ZERO; 3];
        let jq = zeta_row.geometry.q_rows(zero, 0);
        let k0 = zeta_row.geometry.q_matrices(zero, 0);
        let (gradient, hessian, program) = if self.flex_active() {
            let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
            let (_, gradient, hessian) = self.compute_row_flex_primary_gradient_hessian_exact(
                row,
                block_states,
                &q_geom,
                &frame.primary,
            )?;
            let base = if fifth_order {
                self.build_row_flex_fifth_base_with_states(row, block_states, &frame.primary)?
            } else {
                self.build_row_flex_third_base_with_states(row, block_states, &frame.primary)?
            };
            (gradient, hessian, ZetaRowProgram::Flex(base))
        } else {
            let point = self.rigid_row_point(block_states, row, fifth_order)?;
            let (gradient, hessian) = self.rigid_row_point_gradient_hessian(&point)?;
            (gradient, hessian, ZetaRowProgram::Rigid(point))
        };
        let p_primary = frame.primary.total;
        let mut third = Vec::with_capacity(p_primary);
        for k in 0..p_primary {
            let mut axis = Array1::<f64>::zeros(p_primary);
            axis[k] = 1.0;
            third.push(self.zeta_row_third(&program, &axis)?);
        }
        Ok(ZetaRowCalculus {
            zeta_row,
            jq,
            k0,
            gradient,
            hessian,
            third,
            program,
        })
    }

    /// `∇⁴_ζ(ℓ∘G)[u, e_ζ]` for every ζ axis, for a ζ direction `u`: Faà di Bruno over the 15 set
    /// partitions of the two free axes, `u` and the ζ axis.
    fn timewiggle_order_four_axes(
        &self,
        frame: &ZetaFrame<'_>,
        calc: &ZetaRowCalculus,
        u_zeta: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let layout = &frame.layout;
        let p_primary = frame.primary.total;
        let geometry = &calc.zeta_row.geometry;
        let (jq, hessian, gradient) = (&calc.jq, &calc.hessian, &calc.gradient);

        // ── Derivatives of G that do not read the ζ axis ──
        let dir_u = geometry.direction(u_zeta.slice(s![..layout.z_width]));
        let fixed = [&dir_u, &ZetaDirection::ZERO, &ZetaDirection::ZERO];
        let ju = layout.primary_image(jq, u_zeta, p_primary);
        let ju1 = geometry.q_rows(fixed, U);
        let ku = geometry.q_matrices(fixed, U);

        // ── ℓ contractions along u, once per primary axis ──
        let mut fourth_u = Vec::with_capacity(p_primary);
        for k in 0..p_primary {
            let mut axis = Array1::<f64>::zeros(p_primary);
            axis[k] = 1.0;
            fourth_u.push(self.zeta_row_fourth(&calc.program, &ju, &axis)?);
        }
        let t_u = combine_axes(&calc.third, &ju, p_primary);
        let c_z = hessian.dot(&ju);

        // ── ∇⁴_ζ(ℓ∘G)[u, e_ζ], once per ζ axis ──
        let mut phi_axes = Vec::with_capacity(layout.width);
        for zeta_axis in 0..layout.width {
            let mut unit = Array1::<f64>::zeros(layout.width);
            unit[zeta_axis] = 1.0;
            let dir_z = geometry.direction(unit.slice(s![..layout.z_width]));
            let dirs = [&dir_u, &ZetaDirection::ZERO, &dir_z];
            let jz = layout.primary_image(jq, &unit, p_primary);
            let g2uz = layout.on_primaries(geometry.q_derivative(dirs, U | Z), p_primary);
            let jz1 = geometry.q_rows(dirs, Z);
            let juz2 = geometry.q_rows(dirs, U | Z);
            let t_z = combine_axes(&calc.third, &jz, p_primary);
            let mut phi = Array2::<f64>::zeros((layout.width, layout.width));

            // Both free axes in one block: the ℓ contraction of the remaining blocks weights
            // the curvature of G over both axes.
            let c_empty = t_u.dot(&jz) + hessian.dot(&g2uz);
            let c_u = hessian.dot(&jz);
            layout.add_curvature(&mut phi, &c_empty, &calc.k0);
            layout.add_curvature(&mut phi, &c_u, &ku);
            layout.add_curvature(&mut phi, &c_z, &geometry.q_matrices(dirs, Z));
            layout.add_curvature(&mut phi, gradient, &geometry.q_matrices(dirs, U | Z));

            // Free axes in separate blocks: an ℓ derivative between one-axis derivatives of G.
            let m_empty =
                combine_axes(&fourth_u, &jz, p_primary) + combine_axes(&calc.third, &g2uz, p_primary);
            layout.add_full_sandwich(&mut phi, jq, &m_empty);
            layout.add_symmetric_half_sandwich(&mut phi, &ju1, &t_z, jq);
            layout.add_symmetric_half_sandwich(&mut phi, &jz1, &t_u, jq);
            layout.add_symmetric_row_sandwich(&mut phi, &ju1, hessian, &jz1);
            layout.add_symmetric_half_sandwich(&mut phi, &juz2, hessian, jq);
            phi_axes.push(phi);
        }
        Ok(phi_axes)
    }

    /// `∇⁵_ζ(ℓ∘G)[u, v, e_ζ]` for every ζ axis, for ζ directions `u` and `v`: Faà di Bruno over
    /// the 52 set partitions of the two free axes, `u`, `v` and the ζ axis. `calc` must contract
    /// the fifth-moment base.
    fn timewiggle_order_five_axes(
        &self,
        frame: &ZetaFrame<'_>,
        calc: &ZetaRowCalculus,
        u_zeta: &Array1<f64>,
        v_zeta: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let layout = &frame.layout;
        let p_primary = frame.primary.total;
        let geometry = &calc.zeta_row.geometry;
        let (jq, hessian, gradient, third) = (&calc.jq, &calc.hessian, &calc.gradient, &calc.third);

        // ── Derivatives of G that do not read the ζ axis ──
        let dir_u = geometry.direction(u_zeta.slice(s![..layout.z_width]));
        let dir_v = geometry.direction(v_zeta.slice(s![..layout.z_width]));
        let fixed = [&dir_u, &dir_v, &ZetaDirection::ZERO];
        let ju = layout.primary_image(jq, u_zeta, p_primary);
        let jv = layout.primary_image(jq, v_zeta, p_primary);
        let g2uv = layout.on_primaries(geometry.q_derivative(fixed, U | V), p_primary);
        let ju1 = geometry.q_rows(fixed, U);
        let jv1 = geometry.q_rows(fixed, V);
        let juv2 = geometry.q_rows(fixed, U | V);
        let ku = geometry.q_matrices(fixed, U);
        let kv = geometry.q_matrices(fixed, V);
        let kuv = geometry.q_matrices(fixed, U | V);

        // ── ℓ contractions along u and v, once per primary axis ──
        let mut fourth_u = Vec::with_capacity(p_primary);
        let mut fourth_v = Vec::with_capacity(p_primary);
        let mut fourth_uv = Vec::with_capacity(p_primary);
        for k in 0..p_primary {
            let mut axis = Array1::<f64>::zeros(p_primary);
            axis[k] = 1.0;
            fourth_u.push(self.zeta_row_fourth(&calc.program, &ju, &axis)?);
            fourth_v.push(self.zeta_row_fourth(&calc.program, &jv, &axis)?);
            fourth_uv.push(self.zeta_row_fourth(&calc.program, &g2uv, &axis)?);
        }
        let fifth =
            self.zeta_row_fifth_all_primary_axes(&calc.program, &ju, &jv)?;
        let t_u = combine_axes(third, &ju, p_primary);
        let t_v = combine_axes(third, &jv, p_primary);
        let q_uv = combine_axes(&fourth_u, &jv, p_primary);
        let m_z = &q_uv + &combine_axes(third, &g2uv, p_primary);
        let c_z = t_u.dot(&jv) + hessian.dot(&g2uv);
        let c_uz = hessian.dot(&jv);
        let c_vz = hessian.dot(&ju);

        // ── ∇⁵_ζ(ℓ∘G)[u, v, e_ζ], once per ζ axis ──
        let mut phi_axes = Vec::with_capacity(layout.width);
        for zeta_axis in 0..layout.width {
            let mut unit = Array1::<f64>::zeros(layout.width);
            unit[zeta_axis] = 1.0;
            let dir_z = geometry.direction(unit.slice(s![..layout.z_width]));
            let dirs = [&dir_u, &dir_v, &dir_z];
            let jz = layout.primary_image(jq, &unit, p_primary);
            let g2uz = layout.on_primaries(geometry.q_derivative(dirs, U | Z), p_primary);
            let g2vz = layout.on_primaries(geometry.q_derivative(dirs, V | Z), p_primary);
            let g3uvz = layout.on_primaries(geometry.q_derivative(dirs, U | V | Z), p_primary);
            let jz1 = geometry.q_rows(dirs, Z);
            let juz2 = geometry.q_rows(dirs, U | Z);
            let jvz2 = geometry.q_rows(dirs, V | Z);
            let juvz3 = geometry.q_rows(dirs, U | V | Z);
            let t_z = combine_axes(third, &jz, p_primary);
            let mut phi = Array2::<f64>::zeros((layout.width, layout.width));

            // Both free axes in one block: the ℓ contraction of the remaining blocks weights
            // the curvature of G over both axes.
            let c_empty = q_uv.dot(&jz)
                + t_z.dot(&g2uv)
                + t_v.dot(&g2uz)
                + t_u.dot(&g2vz)
                + hessian.dot(&g3uvz);
            let c_u = t_v.dot(&jz) + hessian.dot(&g2vz);
            let c_v = t_u.dot(&jz) + hessian.dot(&g2uz);
            let c_uv = hessian.dot(&jz);
            layout.add_curvature(&mut phi, &c_empty, &calc.k0);
            layout.add_curvature(&mut phi, &c_u, &ku);
            layout.add_curvature(&mut phi, &c_v, &kv);
            layout.add_curvature(&mut phi, &c_z, &geometry.q_matrices(dirs, Z));
            layout.add_curvature(&mut phi, &c_uv, &kuv);
            layout.add_curvature(&mut phi, &c_uz, &geometry.q_matrices(dirs, U | Z));
            layout.add_curvature(&mut phi, &c_vz, &geometry.q_matrices(dirs, V | Z));
            layout.add_curvature(&mut phi, gradient, &geometry.q_matrices(dirs, U | V | Z));

            // Free axes in separate blocks: an ℓ derivative between one-axis derivatives of G.
            let m_empty = combine_axes(&fifth, &jz, p_primary)
                + combine_axes(&fourth_uv, &jz, p_primary)
                + combine_axes(&fourth_v, &g2uz, p_primary)
                + combine_axes(&fourth_u, &g2vz, p_primary)
                + combine_axes(third, &g3uvz, p_primary);
            let m_u = combine_axes(&fourth_v, &jz, p_primary) + combine_axes(third, &g2vz, p_primary);
            let m_v = combine_axes(&fourth_u, &jz, p_primary) + combine_axes(third, &g2uz, p_primary);
            layout.add_full_sandwich(&mut phi, jq, &m_empty);
            layout.add_symmetric_half_sandwich(&mut phi, &ju1, &m_u, jq);
            layout.add_symmetric_half_sandwich(&mut phi, &jv1, &m_v, jq);
            layout.add_symmetric_half_sandwich(&mut phi, &jz1, &m_z, jq);
            layout.add_symmetric_row_sandwich(&mut phi, &ju1, &t_z, &jv1);
            layout.add_symmetric_row_sandwich(&mut phi, &ju1, &t_v, &jz1);
            layout.add_symmetric_row_sandwich(&mut phi, &jv1, &t_u, &jz1);
            layout.add_symmetric_half_sandwich(&mut phi, &juv2, &t_z, jq);
            layout.add_symmetric_half_sandwich(&mut phi, &juz2, &t_v, jq);
            layout.add_symmetric_half_sandwich(&mut phi, &jvz2, &t_u, jq);
            layout.add_symmetric_row_sandwich(&mut phi, &juv2, hessian, &jz1);
            layout.add_symmetric_row_sandwich(&mut phi, &juz2, hessian, &jv1);
            layout.add_symmetric_row_sandwich(&mut phi, &jvz2, hessian, &ju1);
            layout.add_symmetric_half_sandwich(&mut phi, &juvz3, hessian, jq);
            phi_axes.push(phi);
        }
        Ok(phi_axes)
    }

    /// Second directional derivative `D²H[u, e_a]` of the joint Hessian along every coefficient
    /// axis, for a time wiggle on every frame the ζ composition serves (gam#2893). One row pass
    /// serves every axis, where the single-direction evaluator rebuilds each row's flex base
    /// once per axis. The module documentation derives the ζ composition this evaluates.
    pub(crate) fn exact_newton_joint_hessian_second_directional_derivative_timewiggle_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((width, width));
                for row in range {
                    let calc = self.timewiggle_zeta_row_calculus(&frame, block_states, row, false)?;
                    let images = &calc.zeta_row.images;
                    let u_zeta = zeta_image_of(images, d_u, width);
                    let phi_axes = self.timewiggle_order_four_axes(&frame, &calc, &u_zeta)?;
                    pull_back_axes(&phi_axes, images, &mut scratch, &mut phi_axis, &mut acc);
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// Third directional derivative `D³H[u, v, e_a]` of the joint Hessian along every
    /// coefficient axis, for a time wiggle on every frame the ζ composition serves (gam#2893).
    /// The module documentation derives the ζ composition this evaluates.
    pub(crate) fn exact_newton_joint_hessian_third_directional_derivative_timewiggle_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((width, width));
                for row in range {
                    let calc = self.timewiggle_zeta_row_calculus(&frame, block_states, row, true)?;
                    let images = &calc.zeta_row.images;
                    let u_zeta = zeta_image_of(images, d_u, width);
                    let v_zeta = zeta_image_of(images, d_v, width);
                    let phi_axes = self.timewiggle_order_five_axes(&frame, &calc, &u_zeta, &v_zeta)?;
                    pull_back_axes(&phi_axes, images, &mut scratch, &mut phi_axis, &mut acc);
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// `{D_β_a D_β ∂_ψ H[v]}` along every coefficient axis `a` for a design ψ, under the row
    /// measure `row_weights` (gam#2893). With `Ã_ψ` the ζ image of the design motion and
    /// `w = Ã_ψ β`, a row contributes `Ãᵀ(∇⁵[w, Ãv, Ãe_a] + ∇⁴[Ã_ψv, Ãe_a] + ∇⁴[Ãv, Ã_ψe_a])Ã`
    /// and `Ã_ψᵀ ∇⁴[Ãv, Ãe_a] Ã` with its transpose. Returns `None` where the family has no ψ
    /// block for the axis.
    pub(crate) fn timewiggle_design_psi_third_information_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta: &Array1<f64>,
        row_weights: &[f64],
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        let Some((block_idx, local_idx, p_psi, label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        if d_beta.len() != p_total || row_weights.len() != self.n {
            return Err(format!(
                "time-wiggle design ψ third information derivative requires a direction of length \
                 {p_total} and {} row weights",
                self.n
            ));
        }
        let beta = flat_beta(block_states)?;
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let psi_map = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_idx][local_idx],
            self.n,
            p_psi,
            0..self.n,
            label,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((width, width));
                for row in range {
                    let weight = row_weights[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let calc = self.timewiggle_zeta_row_calculus(&frame, block_states, row, true)?;
                    let images = &calc.zeta_row.images;
                    let x_psi = psi_map.row_vector(row).map_err(|error| {
                        format!("time-wiggle design ψ third information row: {error}")
                    })?;
                    let psi_images = psi_zeta_images(self, &frame, row, block_idx, &x_psi)?;
                    let w = zeta_image_of(&psi_images, &beta, width);
                    let v_zeta = zeta_image_of(images, d_beta, width);
                    let psi_v_zeta = zeta_image_of(&psi_images, d_beta, width);
                    let fourth_v = self.timewiggle_order_four_axes(&frame, &calc, &v_zeta)?;
                    let fifth_wv = self.timewiggle_order_five_axes(&frame, &calc, &w, &v_zeta)?;
                    let fourth_psi_v = self.timewiggle_order_four_axes(&frame, &calc, &psi_v_zeta)?;
                    for c in 0..p_total {
                        let (image, psi_image) = (&images[c], &psi_images[c]);
                        if image.entries().is_empty() && psi_image.entries().is_empty() {
                            continue;
                        }
                        phi_axis.fill(0.0);
                        add_combined_axes(&fifth_wv, image, weight, &mut phi_axis);
                        add_combined_axes(&fourth_psi_v, image, weight, &mut phi_axis);
                        add_combined_axes(&fourth_v, psi_image, weight, &mut phi_axis);
                        add_zeta_sandwich(&phi_axis, images, images, &mut scratch, &mut acc[c]);
                        if !image.entries().is_empty() {
                            phi_axis.fill(0.0);
                            add_combined_axes(&fourth_v, image, weight, &mut phi_axis);
                            add_zeta_sandwich(&phi_axis, &psi_images, images, &mut scratch, &mut acc[c]);
                            add_zeta_sandwich(&phi_axis, images, &psi_images, &mut scratch, &mut acc[c]);
                        }
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(Some(result))
    }

    /// `{D_β_a ∂_ψ H}` along every coefficient axis `a` for a design ψ, under the row measure
    /// `row_weights` (gam#2893). With `Ã_ψ` the ζ image of the design motion and `w = Ã_ψ β`, a
    /// row contributes `Ãᵀ(∇⁴[w, Ãe_a] + ∇³[Ã_ψe_a])Ã` and `Ã_ψᵀ ∇³[Ãe_a] Ã` with its
    /// transpose, so one row pass serves every axis. Returns `None` where the family has no ψ
    /// block for the axis.
    pub(crate) fn timewiggle_design_psi_hessian_all_beta_axes(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        row_weights: &[f64],
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        let Some((block_idx, local_idx, p_psi, label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        if row_weights.len() != self.n {
            return Err(format!(
                "time-wiggle design ψ Hessian sweep has {} row weights for {} rows",
                row_weights.len(),
                self.n
            ));
        }
        let beta = flat_beta(block_states)?;
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let psi_map = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_idx][local_idx],
            self.n,
            p_psi,
            0..self.n,
            label,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((width, width));
                for row in range {
                    let weight = row_weights[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let calc = self.timewiggle_zeta_row_calculus(&frame, block_states, row, false)?;
                    let images = &calc.zeta_row.images;
                    let x_psi = psi_map
                        .row_vector(row)
                        .map_err(|error| format!("time-wiggle design ψ Hessian sweep row: {error}"))?;
                    let psi_images = psi_zeta_images(self, &frame, row, block_idx, &x_psi)?;
                    let w = zeta_image_of(&psi_images, &beta, width);
                    let third_axes = order_three_axes(&frame, &calc)?;
                    let fourth_w = self.timewiggle_order_four_axes(&frame, &calc, &w)?;
                    for c in 0..p_total {
                        let (image, psi_image) = (&images[c], &psi_images[c]);
                        if image.entries().is_empty() && psi_image.entries().is_empty() {
                            continue;
                        }
                        phi_axis.fill(0.0);
                        add_combined_axes(&fourth_w, image, weight, &mut phi_axis);
                        add_combined_axes(&third_axes, psi_image, weight, &mut phi_axis);
                        add_zeta_sandwich(&phi_axis, images, images, &mut scratch, &mut acc[c]);
                        if !image.entries().is_empty() {
                            phi_axis.fill(0.0);
                            add_combined_axes(&third_axes, image, weight, &mut phi_axis);
                            add_zeta_sandwich(&phi_axis, &psi_images, images, &mut scratch, &mut acc[c]);
                            add_zeta_sandwich(&phi_axis, images, &psi_images, &mut scratch, &mut acc[c]);
                        }
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(Some(result))
    }

    /// The z-block motion `w_θ` of row `row` along the baseline-chart coordinate `axis`: the
    /// chart moves the entry index, the exit index and the raw derivative index through their
    /// offsets, and no other ζ coordinate.
    fn baseline_zeta_motion(
        geometry: &crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry,
        row: usize,
        axis: usize,
        width: usize,
    ) -> Result<Array1<f64>, String> {
        Ok(Self::baseline_zeta_image(
            Self::rigid_baseline_primary_first::<4>(geometry, row, axis)?,
            width,
        ))
    }

    /// The z-block second motion `w_θθ'` of row `row` along a pair of baseline-chart
    /// coordinates, on the same three ζ coordinates as [`Self::baseline_zeta_motion`].
    fn baseline_zeta_second_motion(
        geometry: &crate::survival::construction::SurvivalMarginalSlopeOffsetGeometry,
        row: usize,
        axis: usize,
        other_axis: usize,
        width: usize,
    ) -> Result<Array1<f64>, String> {
        Ok(Self::baseline_zeta_image(
            Self::rigid_baseline_primary_second::<4>(geometry, row, axis, other_axis)?,
            width,
        ))
    }

    /// Place a chart motion of the entry index, the exit index and the raw derivative index on
    /// its ζ coordinates.
    fn baseline_zeta_image(
        [entry, exit, derivative_exit, _]: [f64; 4],
        width: usize,
    ) -> Array1<f64> {
        let mut w = Array1::<f64>::zeros(width);
        w[ZETA_H0] = entry;
        w[ZETA_H1] = exit;
        w[ZETA_DR] = derivative_exit;
        w
    }

    /// `∂_θ ℓ̄`, `∂_θ ∇_β ℓ̄` and `∂_θ H` for the baseline-chart coordinate `axis` under the row
    /// measure of `options` (gam#3061). The chart leaves `Ã` fixed and moves ζ by `w_θ`, so a
    /// row contributes `∇·w_θ`, `Ãᵀ∇²w_θ` and `Ãᵀ∇³[w_θ]Ã`, the design-ψ terms without the
    /// motion of `Ã`.
    pub(crate) fn timewiggle_baseline_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<ExactNewtonJointPsiTerms, String> {
        let geometry = self.rigid_baseline_geometry()?;
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let flex = self.effective_flex_active(block_states)?;
        let row_weights = self.rigid_third_row_weights(options);
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        let zeros = || {
            (
                0.0,
                Array1::<f64>::zeros(p_total),
                Array2::<f64>::zeros((p_total, p_total)),
            )
        };
        let (objective_psi, score_psi, hessian_psi) =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                self.n,
                |range| -> Result<(f64, Array1<f64>, Array2<f64>), String> {
                    let (mut objective, mut score, mut hessian) = zeros();
                    let mut scratch = Array2::<f64>::zeros((width, p_total));
                    for row in range {
                        let weight = row_weights[row];
                        if weight == 0.0 {
                            continue;
                        }
                        let w = Self::baseline_zeta_motion(geometry, row, axis, width)?;
                        let psi_row = self.timewiggle_zeta_psi_row(&frame, block_states, row, flex)?;
                        let parts = psi_row.parts(&frame.layout);
                        let images = &psi_row.zeta_row.images;
                        let third = |direction: &Array1<f64>| -> Result<Array2<f64>, String> {
                            self.zeta_row_third(&psi_row.program, direction)
                        };
                        objective += weight * parts.order_one().dot(&w);
                        add_pulled_back(images, &parts.order_two().dot(&w), weight, &mut score);
                        let third_w = parts.order_three(&w, &third)? * weight;
                        add_zeta_sandwich(&third_w, images, images, &mut scratch, &mut hessian);
                    }
                    Ok((objective, score, hessian))
                },
                |left, right| -> Result<_, String> {
                    Ok((left.0 + right.0, left.1 + right.1, left.2 + right.2))
                },
            )?
            .unwrap_or_else(zeros);
        Ok(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        })
    }

    /// `D_β ∂_θ H[v]` for the baseline-chart coordinate `axis` along the coefficient direction
    /// `d_beta` under the row measure of `options` (gam#3304). With `w_θ` the chart's z-block
    /// motion and `Ã` fixed, a row contributes `Ãᵀ∇⁴[w_θ, Ãv]Ã`, the single-direction case of
    /// [`Self::timewiggle_baseline_psi_hessian_all_beta_axes`].
    pub(crate) fn timewiggle_baseline_psi_hessian_drift(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        d_beta: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Array2<f64>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let flex = self.effective_flex_active(block_states)?;
        let row_weights = self.rigid_third_row_weights(options);
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        if d_beta.len() != p_total {
            return Err(format!(
                "time-wiggle baseline Hessian drift needs a direction of length {p_total}, got {}",
                d_beta.len()
            ));
        }
        let zeros = || Array2::<f64>::zeros((p_total, p_total));
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                for row in range {
                    let weight = row_weights[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let w = Self::baseline_zeta_motion(geometry, row, axis, width)?;
                    let psi_row = self.timewiggle_zeta_psi_row(&frame, block_states, row, flex)?;
                    let parts = psi_row.parts(&frame.layout);
                    let images = &psi_row.zeta_row.images;
                    let v_zeta = zeta_image_of(images, d_beta, width);
                    let third = |direction: &Array1<f64>| -> Result<Array2<f64>, String> {
                        self.zeta_row_third(&psi_row.program, direction)
                    };
                    let fourth =
                        |left: &Array1<f64>, right: &Array1<f64>| -> Result<Array2<f64>, String> {
                            self.zeta_row_fourth(&psi_row.program, left, right)
                        };
                    let inner = parts.order_four(&w, &v_zeta, &third, &fourth)? * weight;
                    add_zeta_sandwich(&inner, images, images, &mut scratch, &mut acc);
                }
                Ok(acc)
            },
            |left, right| -> Result<_, String> { Ok(left + right) },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// `∂²_θθ' ℓ̄`, `∂²_θθ' ∇_β ℓ̄` and `∂²_θθ' H` for a pair of baseline-chart coordinates under
    /// the row measure of `options` (gam#3304). With `w_θ`, `w_θ'` the chart's z-block motions,
    /// `w_θθ'` its second motion and `Ã` fixed, a row contributes `∇·w_θθ' + w_θᵀ∇²w_θ'`,
    /// `Ãᵀ(∇³[w_θ]w_θ' + ∇²w_θθ')` and `Ãᵀ(∇⁴[w_θ, w_θ'] + ∇³[w_θθ'])Ã`.
    pub(crate) fn timewiggle_baseline_psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        other_axis: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
        let geometry = self.rigid_baseline_geometry()?;
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let flex = self.effective_flex_active(block_states)?;
        let row_weights = self.rigid_third_row_weights(options);
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        let zeros = || {
            (
                0.0,
                Array1::<f64>::zeros(p_total),
                Array2::<f64>::zeros((p_total, p_total)),
            )
        };
        let (objective_psi_psi, score_psi_psi, hessian_psi_psi) =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                self.n,
                |range| -> Result<(f64, Array1<f64>, Array2<f64>), String> {
                    let (mut objective, mut score, mut hessian) = zeros();
                    let mut scratch = Array2::<f64>::zeros((width, p_total));
                    for row in range {
                        let weight = row_weights[row];
                        if weight == 0.0 {
                            continue;
                        }
                        let w_i = Self::baseline_zeta_motion(geometry, row, axis, width)?;
                        let w_j = Self::baseline_zeta_motion(geometry, row, other_axis, width)?;
                        let w_ij =
                            Self::baseline_zeta_second_motion(geometry, row, axis, other_axis, width)?;
                        let psi_row = self.timewiggle_zeta_psi_row(&frame, block_states, row, flex)?;
                        let parts = psi_row.parts(&frame.layout);
                        let images = &psi_row.zeta_row.images;
                        let third = |direction: &Array1<f64>| -> Result<Array2<f64>, String> {
                            self.zeta_row_third(&psi_row.program, direction)
                        };
                        let fourth = |left: &Array1<f64>,
                                      right: &Array1<f64>|
                         -> Result<Array2<f64>, String> {
                            self.zeta_row_fourth(&psi_row.program, left, right)
                        };
                        let second = parts.order_two();
                        objective += weight * (parts.order_one().dot(&w_ij) + w_i.dot(&second.dot(&w_j)));
                        let score_zeta = parts.order_three(&w_i, &third)?.dot(&w_j) + second.dot(&w_ij);
                        add_pulled_back(images, &score_zeta, weight, &mut score);
                        let inner = (parts.order_four(&w_i, &w_j, &third, &fourth)?
                            + parts.order_three(&w_ij, &third)?)
                            * weight;
                        add_zeta_sandwich(&inner, images, images, &mut scratch, &mut hessian);
                    }
                    Ok((objective, score, hessian))
                },
                |left, right| -> Result<_, String> {
                    Ok((left.0 + right.0, left.1 + right.1, left.2 + right.2))
                },
            )?
            .unwrap_or_else(zeros);
        Ok(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    /// `∂²_θψ ℓ̄`, `∂²_θψ ∇_β ℓ̄` and `∂²_θψ H` for the baseline-chart coordinate `baseline_axis`
    /// and the design ψ `design_psi_index` under the row measure of `options` (gam#3304). The
    /// chart moves ζ by `w_θ` and leaves `Ã` fixed; the design ψ moves `Ã` by `Ã_ψ` and ζ by
    /// `w_ψ = Ã_ψβ`. A row contributes `w_θᵀ∇²w_ψ`, `Ã_ψᵀ∇²w_θ + Ãᵀ∇³[w_θ]w_ψ` and
    /// `Ãᵀ∇⁴[w_θ, w_ψ]Ã + Ã_ψᵀ∇³[w_θ]Ã + Ãᵀ∇³[w_θ]Ã_ψ`. Returns `None` where the family has no
    /// ψ block for the design axis.
    pub(crate) fn timewiggle_baseline_design_psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        baseline_axis: usize,
        design_psi_index: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((block_idx, psi_map)) =
            self.timewiggle_design_psi_map(derivative_blocks, design_psi_index)?
        else {
            return Ok(None);
        };
        let geometry = self.rigid_baseline_geometry()?;
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let flex = self.effective_flex_active(block_states)?;
        let row_weights = self.rigid_third_row_weights(options);
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        let beta = flat_beta(block_states)?;
        let zeros = || {
            (
                0.0,
                Array1::<f64>::zeros(p_total),
                Array2::<f64>::zeros((p_total, p_total)),
            )
        };
        let (objective_psi_psi, score_psi_psi, hessian_psi_psi) =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                self.n,
                |range| -> Result<(f64, Array1<f64>, Array2<f64>), String> {
                    let (mut objective, mut score, mut hessian) = zeros();
                    let mut scratch = Array2::<f64>::zeros((width, p_total));
                    for row in range {
                        let weight = row_weights[row];
                        if weight == 0.0 {
                            continue;
                        }
                        let w_theta = Self::baseline_zeta_motion(geometry, row, baseline_axis, width)?;
                        let psi_row = self.timewiggle_zeta_psi_row(&frame, block_states, row, flex)?;
                        let parts = psi_row.parts(&frame.layout);
                        let images = &psi_row.zeta_row.images;
                        let x_psi = psi_map.row_vector(row).map_err(|error| {
                            format!("time-wiggle baseline-by-design ψ pair row: {error}")
                        })?;
                        let psi_images = psi_zeta_images(self, &frame, row, block_idx, &x_psi)?;
                        let w_psi = zeta_image_of(&psi_images, &beta, width);
                        let third = |direction: &Array1<f64>| -> Result<Array2<f64>, String> {
                            self.zeta_row_third(&psi_row.program, direction)
                        };
                        let fourth = |left: &Array1<f64>,
                                      right: &Array1<f64>|
                         -> Result<Array2<f64>, String> {
                            self.zeta_row_fourth(&psi_row.program, left, right)
                        };
                        let second_w_theta = parts.order_two().dot(&w_theta);
                        let third_theta = parts.order_three(&w_theta, &third)?;
                        objective += weight * second_w_theta.dot(&w_psi);
                        add_pulled_back(&psi_images, &second_w_theta, weight, &mut score);
                        add_pulled_back(images, &third_theta.dot(&w_psi), weight, &mut score);
                        let fourth_theta_psi =
                            parts.order_four(&w_theta, &w_psi, &third, &fourth)? * weight;
                        add_zeta_sandwich(&fourth_theta_psi, images, images, &mut scratch, &mut hessian);
                        let third_theta = third_theta * weight;
                        add_zeta_sandwich(&third_theta, &psi_images, images, &mut scratch, &mut hessian);
                        add_zeta_sandwich(&third_theta, images, &psi_images, &mut scratch, &mut hessian);
                    }
                    Ok((objective, score, hessian))
                },
                |left, right| -> Result<_, String> {
                    Ok((left.0 + right.0, left.1 + right.1, left.2 + right.2))
                },
            )?
            .unwrap_or_else(zeros);
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        }))
    }

    /// `{D_β_a ∂_θ H}` along every coefficient axis `a` for the baseline-chart coordinate
    /// `axis`, under the row measure `row_weights` (gam#3061). The chart moves the index offsets
    /// and leaves `Ã` fixed, so with `w_θ` the z-block motion of the entry index, the exit index
    /// and the raw derivative index, a row contributes `Ãᵀ ∇⁴[w_θ, Ãe_a] Ã`, and one row pass
    /// serves every axis, where the single-direction evaluator reruns each row's program once
    /// per axis.
    pub(crate) fn timewiggle_baseline_psi_hessian_all_beta_axes(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        row_weights: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        if row_weights.len() != self.n {
            return Err(format!(
                "time-wiggle baseline Hessian sweep has {} row weights for {} rows",
                row_weights.len(),
                self.n
            ));
        }
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((width, width));
                for row in range {
                    let weight = row_weights[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let w = Self::baseline_zeta_motion(geometry, row, axis, width)?;
                    let calc = self.timewiggle_zeta_row_calculus(&frame, block_states, row, false)?;
                    let images = &calc.zeta_row.images;
                    let fourth_w = self.timewiggle_order_four_axes(&frame, &calc, &w)?;
                    for (c, image) in images.iter().enumerate() {
                        if image.entries().is_empty() {
                            continue;
                        }
                        phi_axis.fill(0.0);
                        add_combined_axes(&fourth_w, image, weight, &mut phi_axis);
                        add_zeta_sandwich(&phi_axis, images, images, &mut scratch, &mut acc[c]);
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// `{D_β_a D_β ∂_θ H[v]}` along every coefficient axis `a` for the baseline-chart coordinate
    /// `axis`, under the row measure `row_weights` (gam#3061). With `w_θ` the chart's z-block
    /// motion and `Ã` fixed, a row contributes `Ãᵀ ∇⁵[w_θ, Ãv, Ãe_a] Ã`.
    pub(crate) fn timewiggle_baseline_psi_third_information_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        d_beta: &Array1<f64>,
        row_weights: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        if d_beta.len() != p_total || row_weights.len() != self.n {
            return Err(format!(
                "time-wiggle baseline third information derivative requires a direction of length \
                 {p_total} and {} row weights",
                self.n
            ));
        }
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((width, width));
                for row in range {
                    let weight = row_weights[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let w = Self::baseline_zeta_motion(geometry, row, axis, width)?;
                    let calc = self.timewiggle_zeta_row_calculus(&frame, block_states, row, true)?;
                    let images = &calc.zeta_row.images;
                    let v_zeta = zeta_image_of(images, d_beta, width);
                    let fifth_wv =
                        self.timewiggle_order_five_axes(&frame, &calc, &w, &v_zeta)?;
                    for (c, image) in images.iter().enumerate() {
                        if image.entries().is_empty() {
                            continue;
                        }
                        phi_axis.fill(0.0);
                        add_combined_axes(&fifth_wv, image, weight, &mut phi_axis);
                        add_zeta_sandwich(&phi_axis, images, images, &mut scratch, &mut acc[c]);
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// `{D_β_a ∂²_θθ' H}` along every coefficient axis `a` for a pair of baseline-chart
    /// coordinates, under the row measure `row_weights` (gam#3061). With `w_θ`, `w_θ'` the
    /// chart's z-block motions, `w_θθ'` its second motion and `Ã` fixed, a row contributes
    /// `Ãᵀ(∇⁵[w_θ, w_θ', Ãe_a] + ∇⁴[w_θθ', Ãe_a])Ã`.
    pub(crate) fn timewiggle_baseline_psi_pair_third_information_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        axis: usize,
        other_axis: usize,
        row_weights: &[f64],
    ) -> Result<Vec<Array2<f64>>, String> {
        let geometry = self.rigid_baseline_geometry()?;
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        if row_weights.len() != self.n {
            return Err(format!(
                "time-wiggle baseline-pair third information derivative has {} row weights for \
                 {} rows",
                row_weights.len(),
                self.n
            ));
        }
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((width, width));
                for row in range {
                    let weight = row_weights[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let w_i = Self::baseline_zeta_motion(geometry, row, axis, width)?;
                    let w_j = Self::baseline_zeta_motion(geometry, row, other_axis, width)?;
                    let w_ij =
                        Self::baseline_zeta_second_motion(geometry, row, axis, other_axis, width)?;
                    let calc = self.timewiggle_zeta_row_calculus(&frame, block_states, row, true)?;
                    let images = &calc.zeta_row.images;
                    let fifth_ij =
                        self.timewiggle_order_five_axes(&frame, &calc, &w_i, &w_j)?;
                    let fourth_ij = self.timewiggle_order_four_axes(&frame, &calc, &w_ij)?;
                    for (c, image) in images.iter().enumerate() {
                        if image.entries().is_empty() {
                            continue;
                        }
                        phi_axis.fill(0.0);
                        add_combined_axes(&fifth_ij, image, weight, &mut phi_axis);
                        add_combined_axes(&fourth_ij, image, weight, &mut phi_axis);
                        add_zeta_sandwich(&phi_axis, images, images, &mut scratch, &mut acc[c]);
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(result)
    }

    /// `{D_β_a ∂²_ψiψj H}` along every coefficient axis `a` for a pair of design ψ, under the
    /// row measure `row_weights` (gam#2893). With the ζ images `Ã_i`, `Ã_j` and `Ã_ij` of the
    /// design motions, `w_• = Ã_• β` and `z_• = Ã_• e_a`, a row contributes
    /// `Ãᵀ(∇⁵[w_i, w_j, z] + ∇⁴[w_ij, z] + ∇⁴[w_j, z_i] + ∇⁴[w_i, z_j] + ∇³[z_ij])Ã`,
    /// `Ã_iᵀ(∇⁴[w_j, z] + ∇³[z_j])Ã` and `Ã_jᵀ(∇⁴[w_i, z] + ∇³[z_i])Ã` with their transposes,
    /// and `Ã_ijᵀ ∇³[z] Ã + Ã_iᵀ ∇³[z] Ã_j` with theirs. `Ã_ij` is zero across blocks. Returns
    /// `None` where the family has no ψ block for either axis.
    pub(crate) fn timewiggle_design_psi_pair_third_information_all_axes(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        row_weights: &[f64],
    ) -> Result<Option<Vec<Array2<f64>>>, String> {
        let Some((block_i, local_i, p_psi_i, label_i)) =
            self.psi_block_info(derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some((block_j, local_j, p_psi_j, label_j)) =
            self.psi_block_info(derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        let n = self.n;
        if row_weights.len() != n {
            return Err(format!(
                "time-wiggle design ψ-pair third information derivative has {} row weights for \
                 {n} rows",
                row_weights.len()
            ));
        }
        let beta = flat_beta(block_states)?;
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let map_i = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_i][local_i],
            n,
            p_psi_i,
            0..n,
            label_i,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let map_j = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_j][local_j],
            n,
            p_psi_j,
            0..n,
            label_j,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let map_ij = if block_i == block_j {
            Some(
                crate::custom_family::resolve_custom_family_x_psi_psi_map(
                    &derivative_blocks[block_i][local_i],
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    n,
                    p_psi_i,
                    0..n,
                    label_i,
                    &policy,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        let row_error = |error| format!("time-wiggle design ψ-pair third information row: {error}");
        let zeros = || vec![Array2::<f64>::zeros((p_total, p_total)); p_total];
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            n,
            |range| -> Result<Vec<Array2<f64>>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                let mut phi_axis = Array2::<f64>::zeros((width, width));
                for row in range {
                    let weight = row_weights[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let calc = self.timewiggle_zeta_row_calculus(&frame, block_states, row, true)?;
                    let images = &calc.zeta_row.images;
                    let images_i =
                        psi_zeta_images(self, &frame, row, block_i, &map_i.row_vector(row).map_err(row_error)?)?;
                    let images_j =
                        psi_zeta_images(self, &frame, row, block_j, &map_j.row_vector(row).map_err(row_error)?)?;
                    let images_ij = map_ij
                        .as_ref()
                        .map(|map| -> Result<Vec<ZetaImage>, String> {
                            psi_zeta_images(self, &frame, row, block_i, &map.row_vector(row).map_err(row_error)?)
                        })
                        .transpose()?;
                    let w_i = zeta_image_of(&images_i, &beta, width);
                    let w_j = zeta_image_of(&images_j, &beta, width);
                    let third_axes = order_three_axes(&frame, &calc)?;
                    let fourth_i = self.timewiggle_order_four_axes(&frame, &calc, &w_i)?;
                    let fourth_j = self.timewiggle_order_four_axes(&frame, &calc, &w_j)?;
                    let fifth_ij = self.timewiggle_order_five_axes(&frame, &calc, &w_i, &w_j)?;
                    let fourth_ij = images_ij
                        .as_ref()
                        .map(|images_ij| {
                            self.timewiggle_order_four_axes(
                                &frame,
                                &calc,
                                &zeta_image_of(images_ij, &beta, width),
                            )
                        })
                        .transpose()?;
                    for c in 0..p_total {
                        let (image, image_i, image_j) = (&images[c], &images_i[c], &images_j[c]);
                        let image_ij = images_ij.as_ref().map(|images_ij| &images_ij[c]);
                        if image.entries().is_empty()
                            && image_i.entries().is_empty()
                            && image_j.entries().is_empty()
                            && image_ij.is_none_or(|image_ij| image_ij.entries().is_empty())
                        {
                            continue;
                        }
                        // `Ã_ijᵀ ∇³[z] Ã + Ã_iᵀ ∇³[z] Ã_j`, with their transposes.
                        if !image.entries().is_empty() {
                            phi_axis.fill(0.0);
                            add_combined_axes(&third_axes, image, weight, &mut phi_axis);
                            if let Some(images_ij) = images_ij.as_ref() {
                                add_zeta_sandwich(&phi_axis, images_ij, images, &mut scratch, &mut acc[c]);
                                add_zeta_sandwich(&phi_axis, images, images_ij, &mut scratch, &mut acc[c]);
                            }
                            add_zeta_sandwich(&phi_axis, &images_i, &images_j, &mut scratch, &mut acc[c]);
                            add_zeta_sandwich(&phi_axis, &images_j, &images_i, &mut scratch, &mut acc[c]);
                        }
                        // `Ã_iᵀ(∇⁴[w_j, z] + ∇³[z_j])Ã`, with its transpose.
                        phi_axis.fill(0.0);
                        add_combined_axes(&fourth_j, image, weight, &mut phi_axis);
                        add_combined_axes(&third_axes, image_j, weight, &mut phi_axis);
                        add_zeta_sandwich(&phi_axis, &images_i, images, &mut scratch, &mut acc[c]);
                        add_zeta_sandwich(&phi_axis, images, &images_i, &mut scratch, &mut acc[c]);
                        // `Ã_jᵀ(∇⁴[w_i, z] + ∇³[z_i])Ã`, with its transpose.
                        phi_axis.fill(0.0);
                        add_combined_axes(&fourth_i, image, weight, &mut phi_axis);
                        add_combined_axes(&third_axes, image_i, weight, &mut phi_axis);
                        add_zeta_sandwich(&phi_axis, &images_j, images, &mut scratch, &mut acc[c]);
                        add_zeta_sandwich(&phi_axis, images, &images_j, &mut scratch, &mut acc[c]);
                        // `Ãᵀ(∇⁵[w_i, w_j, z] + ∇⁴[w_ij, z] + ∇⁴[w_j, z_i] + ∇⁴[w_i, z_j] + ∇³[z_ij])Ã`.
                        phi_axis.fill(0.0);
                        add_combined_axes(&fifth_ij, image, weight, &mut phi_axis);
                        add_combined_axes(&fourth_j, image_i, weight, &mut phi_axis);
                        add_combined_axes(&fourth_i, image_j, weight, &mut phi_axis);
                        if let (Some(fourth_ij), Some(image_ij)) = (fourth_ij.as_ref(), image_ij) {
                            add_combined_axes(fourth_ij, image, weight, &mut phi_axis);
                            add_combined_axes(&third_axes, image_ij, weight, &mut phi_axis);
                        }
                        add_zeta_sandwich(&phi_axis, images, images, &mut scratch, &mut acc[c]);
                    }
                }
                Ok(acc)
            },
            |mut a, b| -> Result<_, String> {
                for (ai, bi) in a.iter_mut().zip(b.into_iter()) {
                    *ai += &bi;
                }
                Ok(a)
            },
        )?
        .unwrap_or_else(zeros);
        Ok(Some(result))
    }
}

impl SurvivalMarginalSlopeFamily {
    /// Whether the ζ composition serves this family (gam#2893): a time wiggle with a single score
    /// slope. On such a frame it serves the design-ψ terms, their drift and pair terms, the joint
    /// `D²H`/`D³H` sweeps, the Jeffreys third information derivative and the design-ψ mixed third
    /// derivatives. The FLEX row program serves a time-constant slope. The rigid program serves a
    /// time-constant or follow-up-varying slope on its own four or six primaries.
    pub(crate) fn timewiggle_zeta_available(&self) -> bool {
        self.flex_timewiggle_active()
            && !self.per_z_slope_active()
            && !(self.flex_active() && self.slope_is_follow_up_varying())
    }

    /// Whether the ζ composition serves the order-five sweeps: the Jeffreys third information
    /// derivative and the ψ-mixed third information derivatives. The FLEX base carries a declared
    /// latent law through its anchored timepoints (gam#2948); the rigid row reads the Gaussian
    /// lowering's closed-form fifth tensor, which a declared law does not have (gam#3304).
    pub(crate) fn timewiggle_zeta_fifth_available(&self) -> bool {
        self.timewiggle_zeta_available() && (self.flex_active() || !self.anchored_law_active())
    }

    /// Row `row`'s ζ value calculus for the design-ψ terms, from the FLEX row program when `flex`
    /// holds and from the rigid one otherwise.
    fn timewiggle_zeta_psi_row(
        &self,
        frame: &ZetaFrame<'_>,
        block_states: &[ParameterBlockState],
        row: usize,
        flex: bool,
    ) -> Result<ZetaPsiRow<'_>, String> {
        let zeta_row = self.timewiggle_zeta_row(frame, block_states, row)?;
        let zero = [&ZetaDirection::ZERO; 3];
        let jq = zeta_row.geometry.q_rows(zero, 0);
        let k0 = zeta_row.geometry.q_matrices(zero, 0);
        let (gradient, hessian, program) = if flex {
            let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
            let (_, gradient, hessian) = self.compute_row_flex_primary_gradient_hessian_exact(
                row,
                block_states,
                &q_geom,
                &frame.primary,
            )?;
            let base =
                self.build_row_flex_third_base_with_states(row, block_states, &frame.primary)?;
            (gradient, hessian, ZetaRowProgram::Flex(base))
        } else {
            let point = self.rigid_row_point(block_states, row, false)?;
            let (gradient, hessian) = self.rigid_row_point_gradient_hessian(&point)?;
            (gradient, hessian, ZetaRowProgram::Rigid(point))
        };
        Ok(ZetaPsiRow {
            zeta_row,
            jq,
            k0,
            gradient,
            hessian,
            program,
        })
    }

    /// Row `row`'s rigid program point, with its fifth likelihood derivatives where `fifth_order`
    /// holds.
    fn rigid_row_point(
        &self,
        block_states: &[ParameterBlockState],
        row: usize,
        fifth_order: bool,
    ) -> Result<RigidRowPoint<'_>, String> {
        let inputs = rigid_row_inputs(self, block_states, row, "survival marginal-slope ζ rigid row")?;
        let primaries = in_slope_frame!(self, P, Frame, {
            rigid_row_kernel_primaries::<P, Frame>(self, block_states, row)?.to_vec()
        });
        let fifth = if !fifth_order {
            None
        } else if self.anchored_law_active() {
            // The anchor is an implicit function of the declared law; its derivatives exist only
            // through the jet lift, which stops at order four (gam#2923).
            return Err(format!(
                "survival marginal-slope ζ rigid row {row}: the fifth likelihood derivatives are \
                 the Gaussian lowering's closed form, which a declared latent law does not have"
            ));
        } else if self.slope_is_follow_up_varying() {
            let point = rigid_frame_point::<DYNAMIC_SLOPE_PRIMARIES>(&primaries)?;
            Some(RigidRowFifth::Dynamic(Box::new(dynamic_row_fifth(&point, &inputs)?)))
        } else {
            let point = rigid_frame_point::<STATIC_SLOPE_PRIMARIES>(&primaries)?;
            Some(RigidRowFifth::Static(Box::new(static_row_fifth(&point, &inputs)?)))
        };
        Ok(RigidRowPoint {
            inputs,
            primaries,
            fifth,
        })
    }

    /// The rigid row program's primary gradient and Hessian at `point`.
    fn rigid_row_point_gradient_hessian(
        &self,
        point: &RigidRowPoint<'_>,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        in_slope_frame!(self, P, Frame, {
            let primaries = rigid_frame_point::<P>(&point.primaries)?;
            let (_, gradient, hessian) =
                Self::rigid_gradient_hessian_at::<P, Frame>(&primaries, &point.inputs)?;
            Ok((gradient, hessian))
        })
    }

    /// `ℓ³[dir]` of the row `program` describes.
    fn zeta_row_third(
        &self,
        program: &ZetaRowProgram<'_>,
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        match program {
            ZetaRowProgram::Flex(base) => self.row_flex_third_contract_from_base(base, dir),
            ZetaRowProgram::Rigid(point) => in_slope_frame!(self, P, Frame, {
                let primaries = rigid_frame_point::<P>(&point.primaries)?;
                Ok(primary_square_array(Self::rigid_third_contracted_at::<P, Frame>(
                    &primaries,
                    &point.inputs,
                    dir.view(),
                )?))
            }),
        }
    }

    /// `Σ_{cde} ℓ_{abcde} u_c v_d (e_k)_e` for every primary axis `k` of the row `program`
    /// describes.
    fn zeta_row_fifth_all_primary_axes(
        &self,
        program: &ZetaRowProgram<'_>,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        match program {
            ZetaRowProgram::Flex(base) => {
                self.row_flex_fifth_contract_all_primary_axes_from_base(base, u, v)
            }
            ZetaRowProgram::Rigid(point) => match &point.fifth {
                Some(RigidRowFifth::Static(fifth)) => contract_fifth_all_primary_axes(fifth, u, v),
                Some(RigidRowFifth::Dynamic(fifth)) => contract_fifth_all_primary_axes(fifth, u, v),
                None => Err(format!(
                    "survival marginal-slope ζ rigid row {}: a fifth contraction reads a row \
                     point resolved without its fifth likelihood derivatives",
                    point.inputs.row
                )),
            },
        }
    }

    /// `ℓ⁴[u, v]` of the row `program` describes.
    fn zeta_row_fourth(
        &self,
        program: &ZetaRowProgram<'_>,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        match program {
            ZetaRowProgram::Flex(base) => self.row_flex_fourth_contract_from_base(base, u, v),
            ZetaRowProgram::Rigid(point) => in_slope_frame!(self, P, Frame, {
                let primaries = rigid_frame_point::<P>(&point.primaries)?;
                Ok(primary_square_array(Self::rigid_fourth_contracted_at::<P, Frame>(
                    &primaries,
                    &point.inputs,
                    u.view(),
                    v.view(),
                )?))
            }),
        }
    }

    /// The block of design ψ `psi_index` and its design-derivative row map, or `None` where the
    /// family has no ψ block for the axis.
    fn timewiggle_design_psi_map(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<(usize, crate::custom_family::PsiDesignMap)>, String> {
        let Some((block_idx, local_idx, p_psi, label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let map = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_idx][local_idx],
            self.n,
            p_psi,
            0..self.n,
            label,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        Ok(Some((block_idx, map)))
    }

    /// `∂_ψ ℓ̄`, `∂_ψ ∇_β ℓ̄` and `∂_ψ H` for a design ψ under the row measure of `options`
    /// (gam#2893). A design ψ moves `Ã` by its ζ image `Ã_ψ`, so ζ moves by `w = Ã_ψβ` and a row
    /// contributes `∇·w`, `Ã_ψᵀ∇ + Ãᵀ∇²w` and `Ãᵀ∇³[w]Ã + Ã_ψᵀ∇²Ã + Ãᵀ∇²Ã_ψ`. The time-wiggle map
    /// moves through `G`, so its Jacobian needs no hand lift. Returns `None` where the family has
    /// no ψ block for the axis.
    pub(crate) fn timewiggle_design_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let Some((block_idx, psi_map)) =
            self.timewiggle_design_psi_map(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let flex = self.effective_flex_active(block_states)?;
        let row_weights = self.rigid_third_row_weights(options);
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        let beta = flat_beta(block_states)?;
        let zeros = || {
            (
                0.0,
                Array1::<f64>::zeros(p_total),
                Array2::<f64>::zeros((p_total, p_total)),
            )
        };
        let (objective_psi, score_psi, hessian_psi) =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                self.n,
                |range| -> Result<(f64, Array1<f64>, Array2<f64>), String> {
                    let (mut objective, mut score, mut hessian) = zeros();
                    let mut scratch = Array2::<f64>::zeros((width, p_total));
                    for row in range {
                        let weight = row_weights[row];
                        if weight == 0.0 {
                            continue;
                        }
                        let psi_row = self.timewiggle_zeta_psi_row(&frame, block_states, row, flex)?;
                        let parts = psi_row.parts(&frame.layout);
                        let images = &psi_row.zeta_row.images;
                        let x_psi = psi_map
                            .row_vector(row)
                            .map_err(|error| format!("time-wiggle design ψ terms row: {error}"))?;
                        let psi_images = psi_zeta_images(self, &frame, row, block_idx, &x_psi)?;
                        let w = zeta_image_of(&psi_images, &beta, width);
                        let third = |direction: &Array1<f64>| -> Result<Array2<f64>, String> {
                            self.zeta_row_third(&psi_row.program, direction)
                        };
                        let gradient = parts.order_one();
                        let second = parts.order_two();
                        objective += weight * gradient.dot(&w);
                        add_pulled_back(&psi_images, &gradient, weight, &mut score);
                        add_pulled_back(images, &second.dot(&w), weight, &mut score);
                        let third_w = parts.order_three(&w, &third)? * weight;
                        add_zeta_sandwich(&third_w, images, images, &mut scratch, &mut hessian);
                        let second = second * weight;
                        add_zeta_sandwich(&second, &psi_images, images, &mut scratch, &mut hessian);
                        add_zeta_sandwich(&second, images, &psi_images, &mut scratch, &mut hessian);
                    }
                    Ok((objective, score, hessian))
                },
                |left, right| -> Result<_, String> {
                    Ok((left.0 + right.0, left.1 + right.1, left.2 + right.2))
                },
            )?
            .unwrap_or_else(zeros);
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    /// `D_β ∂_ψ H[v]` for a design ψ along the coefficient direction `d_beta` under the row measure
    /// of `options` (gam#2893). With `w = Ã_ψβ`, a row contributes
    /// `Ãᵀ(∇⁴[w, Ãv] + ∇³[Ã_ψv])Ã + Ã_ψᵀ∇³[Ãv]Ã + Ãᵀ∇³[Ãv]Ã_ψ`, the single-direction case of
    /// `timewiggle_design_psi_hessian_all_beta_axes`. Returns `None` where the family has no ψ
    /// block for the axis.
    pub(crate) fn timewiggle_design_psi_hessian_drift(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let Some((block_idx, psi_map)) =
            self.timewiggle_design_psi_map(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let flex = self.effective_flex_active(block_states)?;
        let row_weights = self.rigid_third_row_weights(options);
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        if d_beta.len() != p_total {
            return Err(format!(
                "time-wiggle design ψ Hessian drift needs a direction of length {p_total}, got {}",
                d_beta.len()
            ));
        }
        let beta = flat_beta(block_states)?;
        let zeros = || Array2::<f64>::zeros((p_total, p_total));
        let result = gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
            self.n,
            |range| -> Result<Array2<f64>, String> {
                let mut acc = zeros();
                let mut scratch = Array2::<f64>::zeros((width, p_total));
                for row in range {
                    let weight = row_weights[row];
                    if weight == 0.0 {
                        continue;
                    }
                    let psi_row = self.timewiggle_zeta_psi_row(&frame, block_states, row, flex)?;
                    let parts = psi_row.parts(&frame.layout);
                    let images = &psi_row.zeta_row.images;
                    let x_psi = psi_map
                        .row_vector(row)
                        .map_err(|error| format!("time-wiggle design ψ Hessian drift row: {error}"))?;
                    let psi_images = psi_zeta_images(self, &frame, row, block_idx, &x_psi)?;
                    let w = zeta_image_of(&psi_images, &beta, width);
                    let v_zeta = zeta_image_of(images, d_beta, width);
                    let psi_v_zeta = zeta_image_of(&psi_images, d_beta, width);
                    let third = |direction: &Array1<f64>| -> Result<Array2<f64>, String> {
                        self.zeta_row_third(&psi_row.program, direction)
                    };
                    let fourth =
                        |left: &Array1<f64>, right: &Array1<f64>| -> Result<Array2<f64>, String> {
                            self.zeta_row_fourth(&psi_row.program, left, right)
                        };
                    let inner = parts.order_four(&w, &v_zeta, &third, &fourth)?
                        + parts.order_three(&psi_v_zeta, &third)?;
                    add_zeta_sandwich(&(inner * weight), images, images, &mut scratch, &mut acc);
                    let third_v = parts.order_three(&v_zeta, &third)? * weight;
                    add_zeta_sandwich(&third_v, &psi_images, images, &mut scratch, &mut acc);
                    add_zeta_sandwich(&third_v, images, &psi_images, &mut scratch, &mut acc);
                }
                Ok(acc)
            },
            |left, right| -> Result<_, String> { Ok(left + right) },
        )?
        .unwrap_or_else(zeros);
        Ok(Some(result))
    }

    /// `∂²_ψiψj ℓ̄`, `∂²_ψiψj ∇_β ℓ̄` and `∂²_ψiψj H` for a pair of design ψ under the row measure
    /// of `options` (gam#2893). With the ζ images `Ã_i`, `Ã_j` and `Ã_ij` of the design motions and
    /// `w_• = Ã_•β`, a row contributes `∇·w_ij + w_iᵀ∇²w_j`,
    /// `Ã_ijᵀ∇ + Ã_iᵀ∇²w_j + Ã_jᵀ∇²w_i + Ãᵀ(∇³[w_i]w_j + ∇²w_ij)` and
    /// `Ãᵀ(∇⁴[w_i, w_j] + ∇³[w_ij])Ã`, plus `Ã_iᵀ∇³[w_j]Ã`, `Ã_jᵀ∇³[w_i]Ã`, `Ã_ijᵀ∇²Ã` and
    /// `Ã_iᵀ∇²Ã_j`, each with its transpose. `Ã_ij` is zero across blocks. Returns `None` where the
    /// family has no ψ block for either axis.
    pub(crate) fn timewiggle_design_psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some((block_i, local_i, p_psi_i, label_i)) =
            self.psi_block_info(derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some((block_j, local_j, p_psi_j, label_j)) =
            self.psi_block_info(derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let frame = self.timewiggle_zeta_frame(block_states)?;
        let flex = self.effective_flex_active(block_states)?;
        let row_weights = self.rigid_third_row_weights(options);
        let width = frame.layout.width;
        let p_total = frame.slices.total;
        let n = self.n;
        let beta = flat_beta(block_states)?;
        let policy = gam_runtime::resource::ResourcePolicy::default_library();
        let map_i = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_i][local_i],
            n,
            p_psi_i,
            0..n,
            label_i,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let map_j = crate::custom_family::resolve_custom_family_x_psi_map(
            &derivative_blocks[block_j][local_j],
            n,
            p_psi_j,
            0..n,
            label_j,
            &policy,
        )
        .map_err(|error| error.to_string())?;
        let map_ij = if block_i == block_j {
            Some(
                crate::custom_family::resolve_custom_family_x_psi_psi_map(
                    &derivative_blocks[block_i][local_i],
                    &derivative_blocks[block_j][local_j],
                    local_j,
                    n,
                    p_psi_i,
                    0..n,
                    label_i,
                    &policy,
                )
                .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        let zeros = || {
            (
                0.0,
                Array1::<f64>::zeros(p_total),
                Array2::<f64>::zeros((p_total, p_total)),
            )
        };
        let (objective_psi_psi, score_psi_psi, hessian_psi_psi) =
            gam_linalg::pairwise_reduce::par_deterministic_try_block_fold(
                n,
                |range| -> Result<(f64, Array1<f64>, Array2<f64>), String> {
                    let (mut objective, mut score, mut hessian) = zeros();
                    let mut scratch = Array2::<f64>::zeros((width, p_total));
                    for row in range {
                        let weight = row_weights[row];
                        if weight == 0.0 {
                            continue;
                        }
                        let psi_row = self.timewiggle_zeta_psi_row(&frame, block_states, row, flex)?;
                        let parts = psi_row.parts(&frame.layout);
                        let images = &psi_row.zeta_row.images;
                        let x_i = map_i
                            .row_vector(row)
                            .map_err(|error| format!("time-wiggle design ψ pair terms row: {error}"))?;
                        let x_j = map_j
                            .row_vector(row)
                            .map_err(|error| format!("time-wiggle design ψ pair terms row: {error}"))?;
                        let images_i = psi_zeta_images(self, &frame, row, block_i, &x_i)?;
                        let images_j = psi_zeta_images(self, &frame, row, block_j, &x_j)?;
                        let images_ij = match map_ij.as_ref() {
                            Some(map) => {
                                let x_ij = map.row_vector(row).map_err(|error| {
                                    format!("time-wiggle design ψ pair terms row: {error}")
                                })?;
                                Some(psi_zeta_images(self, &frame, row, block_i, &x_ij)?)
                            }
                            None => None,
                        };
                        let w_i = zeta_image_of(&images_i, &beta, width);
                        let w_j = zeta_image_of(&images_j, &beta, width);
                        let third = |direction: &Array1<f64>| -> Result<Array2<f64>, String> {
                            self.zeta_row_third(&psi_row.program, direction)
                        };
                        let fourth = |left: &Array1<f64>,
                                      right: &Array1<f64>|
                         -> Result<Array2<f64>, String> {
                            self.zeta_row_fourth(&psi_row.program, left, right)
                        };
                        let gradient = parts.order_one();
                        let second = parts.order_two();
                        let third_i = parts.order_three(&w_i, &third)?;
                        let third_j = parts.order_three(&w_j, &third)?;
                        let second_w_i = second.dot(&w_i);
                        let second_w_j = second.dot(&w_j);
                        objective += weight * w_i.dot(&second_w_j);
                        add_pulled_back(&images_i, &second_w_j, weight, &mut score);
                        add_pulled_back(&images_j, &second_w_i, weight, &mut score);
                        add_pulled_back(images, &third_i.dot(&w_j), weight, &mut score);
                        let mut inner = parts.order_four(&w_i, &w_j, &third, &fourth)?;
                        if let Some(images_ij) = images_ij.as_ref() {
                            let w_ij = zeta_image_of(images_ij, &beta, width);
                            objective += weight * gradient.dot(&w_ij);
                            add_pulled_back(images_ij, &gradient, weight, &mut score);
                            add_pulled_back(images, &second.dot(&w_ij), weight, &mut score);
                            inner += &parts.order_three(&w_ij, &third)?;
                            let second_ij = &second * weight;
                            add_zeta_sandwich(&second_ij, images_ij, images, &mut scratch, &mut hessian);
                            add_zeta_sandwich(&second_ij, images, images_ij, &mut scratch, &mut hessian);
                        }
                        add_zeta_sandwich(&(inner * weight), images, images, &mut scratch, &mut hessian);
                        let third_j = third_j * weight;
                        add_zeta_sandwich(&third_j, &images_i, images, &mut scratch, &mut hessian);
                        add_zeta_sandwich(&third_j, images, &images_i, &mut scratch, &mut hessian);
                        let third_i = third_i * weight;
                        add_zeta_sandwich(&third_i, &images_j, images, &mut scratch, &mut hessian);
                        add_zeta_sandwich(&third_i, images, &images_j, &mut scratch, &mut hessian);
                        let second = second * weight;
                        add_zeta_sandwich(&second, &images_i, &images_j, &mut scratch, &mut hessian);
                        add_zeta_sandwich(&second, &images_j, &images_i, &mut scratch, &mut hessian);
                    }
                    Ok((objective, score, hessian))
                },
                |left, right| -> Result<_, String> {
                    Ok((left.0 + right.0, left.1 + right.1, left.2 + right.2))
                },
            )?
            .unwrap_or_else(zeros);
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        }))
    }
}
