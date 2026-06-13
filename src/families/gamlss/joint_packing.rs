//! Pure dense block-assembly helpers that pack per-parameter score vectors and
//! symmetric Hessian blocks into the joint (location, scale[, wiggle]) layout.
//! Each is a dependency-free `Array{1,2}` slice-assign kernel relocated verbatim
//! from `gamlss.rs`; the parent re-imports them so every call site is unchanged.

use super::mirror_upper_to_lower;
use ndarray::{Array1, Array2, s};

pub(crate) fn gaussian_pack_joint_score(
    scoremu: &Array1<f64>,
    score_ls: &Array1<f64>,
) -> Array1<f64> {
    let pmu = scoremu.len();
    let p_ls = score_ls.len();
    let mut out = Array1::<f64>::zeros(pmu + p_ls);
    out.slice_mut(s![0..pmu]).assign(scoremu);
    out.slice_mut(s![pmu..pmu + p_ls]).assign(score_ls);
    out
}

pub(crate) fn gaussian_pack_joint_symmetrichessian(
    hmumu: &Array2<f64>,
    hmu_ls: &Array2<f64>,
    h_ls_ls: &Array2<f64>,
) -> Array2<f64> {
    let pmu = hmumu.nrows();
    let p_ls = h_ls_ls.nrows();
    let total = pmu + p_ls;
    let mut out = Array2::<f64>::zeros((total, total));
    out.slice_mut(s![0..pmu, 0..pmu]).assign(hmumu);
    out.slice_mut(s![0..pmu, pmu..total]).assign(hmu_ls);
    out.slice_mut(s![pmu..total, pmu..total]).assign(h_ls_ls);
    mirror_upper_to_lower(&mut out);
    out
}

pub(crate) fn gaussian_pack_wiggle_joint_score(
    score_mu: &Array1<f64>,
    score_ls: &Array1<f64>,
    score_w: &Array1<f64>,
) -> Array1<f64> {
    let pmu = score_mu.len();
    let p_ls = score_ls.len();
    let pw = score_w.len();
    let total = pmu + p_ls + pw;
    let mut out = Array1::<f64>::zeros(total);
    out.slice_mut(s![0..pmu]).assign(score_mu);
    out.slice_mut(s![pmu..pmu + p_ls]).assign(score_ls);
    out.slice_mut(s![pmu + p_ls..total]).assign(score_w);
    out
}

pub(crate) fn gaussian_pack_wiggle_joint_symmetrichessian(
    h_mm: &Array2<f64>,
    h_ml: &Array2<f64>,
    h_mw: &Array2<f64>,
    h_ll: &Array2<f64>,
    h_lw: &Array2<f64>,
    h_ww: &Array2<f64>,
) -> Array2<f64> {
    let pmu = h_mm.nrows();
    let p_ls = h_ll.nrows();
    let pw = h_ww.nrows();
    let total = pmu + p_ls + pw;
    let mut out = Array2::<f64>::zeros((total, total));
    out.slice_mut(s![0..pmu, 0..pmu]).assign(h_mm);
    out.slice_mut(s![0..pmu, pmu..pmu + p_ls]).assign(h_ml);
    out.slice_mut(s![0..pmu, pmu + p_ls..total]).assign(h_mw);
    out.slice_mut(s![pmu..pmu + p_ls, pmu..pmu + p_ls])
        .assign(h_ll);
    out.slice_mut(s![pmu..pmu + p_ls, pmu + p_ls..total])
        .assign(h_lw);
    out.slice_mut(s![pmu + p_ls..total, pmu + p_ls..total])
        .assign(h_ww);
    mirror_upper_to_lower(&mut out);
    out
}

pub(crate) fn binomial_pack_mean_wiggle_joint_score(
    score_eta: &Array1<f64>,
    score_w: &Array1<f64>,
) -> Array1<f64> {
    let p_eta = score_eta.len();
    let pw = score_w.len();
    let mut out = Array1::<f64>::zeros(p_eta + pw);
    out.slice_mut(s![0..p_eta]).assign(score_eta);
    out.slice_mut(s![p_eta..p_eta + pw]).assign(score_w);
    out
}

pub(crate) fn binomial_pack_mean_wiggle_joint_symmetrichessian(
    h_eta_eta: &Array2<f64>,
    h_eta_w: &Array2<f64>,
    h_ww: &Array2<f64>,
) -> Array2<f64> {
    let p_eta = h_eta_eta.nrows();
    let pw = h_ww.nrows();
    let total = p_eta + pw;
    let mut out = Array2::<f64>::zeros((total, total));
    out.slice_mut(s![0..p_eta, 0..p_eta]).assign(h_eta_eta);
    out.slice_mut(s![0..p_eta, p_eta..total]).assign(h_eta_w);
    out.slice_mut(s![p_eta..total, p_eta..total]).assign(h_ww);
    mirror_upper_to_lower(&mut out);
    out
}
