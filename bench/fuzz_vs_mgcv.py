#!/usr/bin/env python3
"""
Adversarial fuzzer: find datasets where mgcv massively outperforms Rust GAM.

Generates extremely diverse regression datasets across multiple model types
(GAM, GAMLSS, Duchon, multidimensional) and compares Rust vs mgcv, surfacing
the worst failures with full diagnostic detail.

Model types tested:
  - gaussian GAM (ps, tps, duchon)
  - binomial GAM (ps, tps, duchon)
  - gaussian GAMLSS location-scale (ps, duchon)
  - mgcv gaulss comparison

Usage:
    python bench/fuzz_vs_mgcv.py                        # 100 trials
    python bench/fuzz_vs_mgcv.py --n-trials 500         # more
    python bench/fuzz_vs_mgcv.py --resume                # continue
    python bench/fuzz_vs_mgcv.py --model-type gamlss     # only gamlss
    python bench/fuzz_vs_mgcv.py --family binomial       # only binomial
"""

from __future__ import annotations

# Force unbuffered stdout so per-trial output appears in real time
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import inspect
import json
import math
import os
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Import metrics from the main benchmark suite
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_suite import (
    auc_score,
    brier_score,
    log_loss_score,
    nagelkerke_r2_score,
    rmse_score,
    r2_score,
    mae_score,
    gaussian_log_loss_score,
    zscore_train_test,
    make_folds,
    Fold,
)

ROOT = Path(__file__).resolve().parent.parent
RUST_BINARY = ROOT / "target" / "release" / "gam"
RESULTS_FILE = Path(__file__).resolve().parent / "fuzz_results.jsonl"
DEFAULT_R_TIMEOUT = 180
DEFAULT_RUST_TIMEOUT = 180

# ═══════════════════════════════════════════════════════════════════════════
# SMOOTH FUNCTION LIBRARY — 35+ functions
# ═══════════════════════════════════════════════════════════════════════════

def _f_sine(x, rng):
    return np.sin(rng.uniform(0.5, 8) * x + rng.uniform(0, 2*np.pi))

def _f_cosine_beat(x, rng):
    f1, f2 = rng.uniform(2, 6), rng.uniform(5, 10)
    return np.cos(f1 * x) * np.cos(f2 * x)

def _f_poly(x, rng):
    return np.polyval(rng.randn(rng.randint(2, 7) + 1), x)

def _f_step(x, rng):
    cuts = np.sort(rng.uniform(np.percentile(x, 5), np.percentile(x, 95), rng.randint(1, 8)))
    levels = rng.randn(len(cuts) + 1) * 2
    out = np.full_like(x, levels[0])
    for i, c in enumerate(cuts):
        out = np.where(x > c, levels[i + 1], out)
    return out

def _f_spike(x, rng):
    c = rng.uniform(np.percentile(x, 15), np.percentile(x, 85))
    w = rng.uniform(0.01, 0.2) * (np.ptp(x) + 1e-8)
    return rng.uniform(2, 10) * np.exp(-0.5 * ((x - c) / w)**2)

def _f_multi_spike(x, rng):
    return sum(_f_spike(x, rng) for _ in range(rng.randint(2, 5)))

def _f_plateau(x, rng):
    lo = rng.uniform(np.percentile(x, 5), np.percentile(x, 40))
    hi = rng.uniform(np.percentile(x, 60), np.percentile(x, 95))
    s = rng.uniform(5, 40) / (np.ptp(x) + 1e-8)
    return rng.uniform(1, 5) * (1/(1+np.exp(-s*(x-lo))) - 1/(1+np.exp(-s*(x-hi))))

def _f_wiggly(x, rng):
    return sum(rng.uniform(.2, 2)*np.sin(rng.uniform(1, 20)*x + rng.uniform(0, 2*np.pi))
               for _ in range(rng.randint(4, 12)))

def _f_linear(x, rng):
    return rng.uniform(-3, 3) * x

def _f_quadratic(x, rng):
    return rng.randn() * x**2 + rng.randn() * x + rng.randn()

def _f_cubic(x, rng):
    c = rng.randn(4); return c[0]*x**3 + c[1]*x**2 + c[2]*x + c[3]

def _f_sqrt_abs(x, rng):
    return rng.uniform(1, 5) * np.sign(x) * np.sqrt(np.abs(x))

def _f_log_abs(x, rng):
    return rng.uniform(0.5, 3) * np.log1p(np.abs(x))

def _f_sawtooth(x, rng):
    f = rng.uniform(1, 6)
    return 2 * (f*x/(2*np.pi) - np.floor(0.5 + f*x/(2*np.pi)))

def _f_chirp(x, rng):
    f0, f1 = rng.uniform(0.5, 2), rng.uniform(4, 15)
    t = (x - x.min()) / (np.ptp(x) + 1e-8)
    return np.sin(2*np.pi*(f0*t + (f1-f0)*t**2/2))

def _f_runge(x, rng):
    c = rng.uniform(-0.5, 0.5) * np.ptp(x)
    s = rng.uniform(0.05, 0.3) * np.ptp(x)
    return 1 / (1 + ((x - c) / s)**2)

def _f_abs_sin(x, rng):
    return np.abs(np.sin(rng.uniform(1, 6) * x))

def _f_piecewise_linear(x, rng):
    breaks = np.sort(rng.uniform(x.min(), x.max(), rng.randint(2, 7)))
    slopes = rng.randn(len(breaks) + 1) * 2
    out = np.zeros_like(x); prev = x.min(); val = 0.0
    for i, b in enumerate(breaks):
        m = (x >= prev) & (x < b); out[m] = val + slopes[i]*(x[m]-prev)
        val += slopes[i]*(b-prev); prev = b
    out[x >= prev] = val + slopes[-1]*(x[x >= prev]-prev)
    return out

def _f_exp_decay(x, rng):
    return rng.uniform(1, 5) * np.exp(-rng.uniform(0.5, 5) * (x - x.min()) / (np.ptp(x)+1e-8))

def _f_logistic(x, rng):
    c = rng.uniform(np.percentile(x, 20), np.percentile(x, 80))
    s = rng.uniform(0.05, 0.5) * np.ptp(x)
    return rng.uniform(1, 5) / (1 + np.exp(-(x - c) / s))

def _f_interaction_proxy(x, rng):
    mid = np.median(x)
    return np.sin(rng.uniform(2, 6)*x)*(x < mid) + rng.uniform(.3, 2)*x*(x >= mid)

def _f_modulated_sine(x, rng):
    return np.sin(rng.uniform(3, 10)*x) * np.cos(rng.uniform(0.5, 2)*x)

def _f_cauchy_bump(x, rng):
    c = rng.uniform(np.percentile(x, 20), np.percentile(x, 80))
    w = rng.uniform(0.02, 0.15) * np.ptp(x)
    return rng.uniform(1, 5) / (1 + ((x - c)/w)**2)

def _f_triangle_wave(x, rng):
    f = rng.uniform(1, 6)
    return 2*np.abs(2*(f*x/(2*np.pi) - np.floor(f*x/(2*np.pi) + 0.5))) - 1

def _f_double_well(x, rng):
    return rng.uniform(0.5, 3) * (x**4 - 2*x**2)

def _f_near_flat_edge(x, rng):
    edge = rng.choice([x.min(), x.max()])
    w = 0.05 * np.ptp(x)
    return rng.uniform(3, 10) * np.exp(-((x - edge)/w)**2)

def _f_heterogeneous(x, rng):
    q = np.percentile(x, [25, 50, 75])
    return (np.sin(5*x)*(x<q[0]) + 2.0*(x>=q[0])*(x<q[1])
            - 3*(x-q[2])*(x>=q[1])*(x<q[2]) + np.cos(8*x)*(x>=q[2]))

def _f_linear_wiggle(x, rng):
    return rng.uniform(-3, 3)*x + rng.uniform(0.05, 0.3)*np.sin(rng.uniform(5, 15)*x)

def _f_sinc(x, rng):
    s = rng.uniform(2, 8)
    u = s * x
    safe_u = np.where(np.abs(u) < 1e-8, 1.0, u)
    return np.where(np.abs(u) < 1e-8, 1.0, np.sin(safe_u) / safe_u) * rng.uniform(1, 5)

def _f_wavelet(x, rng):
    s = rng.uniform(0.1, 0.5) * np.ptp(x)
    c = rng.uniform(np.percentile(x, 20), np.percentile(x, 80))
    t = (x - c) / s
    return (1 - t**2) * np.exp(-0.5 * t**2) * rng.uniform(2, 8)

def _f_fractal_sum(x, rng):
    """Self-similar multi-scale oscillation."""
    out = np.zeros_like(x)
    for k in range(1, rng.randint(4, 8)):
        out += np.sin(2**k * x + rng.uniform(0, 2*np.pi)) / 2**k
    return out * rng.uniform(2, 5)

def _f_smooth_then_jump(x, rng):
    jump_at = rng.uniform(np.percentile(x, 30), np.percentile(x, 70))
    jump_size = rng.uniform(2, 8) * rng.choice([-1, 1])
    return np.sin(2*x) + jump_size * (x > jump_at).astype(float)

def _f_polynomial_ratio(x, rng):
    return (rng.randn()*x**2 + rng.randn()*x) / (1 + x**2)

SMOOTH_FN = {
    "sine": _f_sine, "cosine_beat": _f_cosine_beat, "poly": _f_poly,
    "step": _f_step, "spike": _f_spike, "multi_spike": _f_multi_spike,
    "plateau": _f_plateau, "wiggly": _f_wiggly, "linear": _f_linear,
    "quadratic": _f_quadratic, "cubic": _f_cubic, "sqrt_abs": _f_sqrt_abs,
    "log_abs": _f_log_abs, "sawtooth": _f_sawtooth, "chirp": _f_chirp,
    "runge": _f_runge, "abs_sin": _f_abs_sin, "piecewise_linear": _f_piecewise_linear,
    "exp_decay": _f_exp_decay, "logistic": _f_logistic,
    "interaction_proxy": _f_interaction_proxy, "modulated_sine": _f_modulated_sine,
    "cauchy_bump": _f_cauchy_bump, "triangle_wave": _f_triangle_wave,
    "double_well": _f_double_well, "near_flat_edge": _f_near_flat_edge,
    "heterogeneous": _f_heterogeneous, "linear_wiggle": _f_linear_wiggle,
    "sinc": _f_sinc, "wavelet": _f_wavelet, "fractal_sum": _f_fractal_sum,
    "smooth_then_jump": _f_smooth_then_jump, "polynomial_ratio": _f_polynomial_ratio,
}

# ═══════════════════════════════════════════════════════════════════════════
# 2D / MULTIDIMENSIONAL SMOOTH FUNCTIONS (for Duchon / TPS)
# ═══════════════════════════════════════════════════════════════════════════

def _f2d_saddle(x1, x2, rng):
    return rng.uniform(1, 4) * (x1**2 - x2**2)

def _f2d_dome(x1, x2, rng):
    return rng.uniform(2, 6) * np.exp(-0.5*(x1**2 + x2**2))

def _f2d_ridge(x1, x2, rng):
    angle = rng.uniform(0, np.pi)
    u = np.cos(angle)*x1 + np.sin(angle)*x2
    return _f_wiggly(u, rng)

def _f2d_spiral(x1, x2, rng):
    r = np.sqrt(x1**2 + x2**2)
    theta = np.arctan2(x2, x1)
    return np.sin(rng.uniform(1, 4)*r + theta) * rng.uniform(1, 4)

def _f2d_checkerboard(x1, x2, rng):
    freq = rng.uniform(1, 4)
    return np.sign(np.sin(freq*np.pi*x1) * np.sin(freq*np.pi*x2))

def _f2d_manifold_1d(x1, x2, rng):
    """Signal lives on a 1D manifold in 2D space."""
    u = rng.uniform(-1, 1)*x1 + rng.uniform(-1, 1)*x2
    return _f_wiggly(u, rng)

def _f2d_radial_wave(x1, x2, rng):
    r = np.sqrt(x1**2 + x2**2 + 0.01)
    return np.sin(rng.uniform(2, 8)*r) / r * rng.uniform(1, 5)

def _f2d_product(x1, x2, rng):
    """Pure interaction: f(x1) * g(x2)."""
    f1_name = rng.choice(list(SMOOTH_FN.keys()))
    f2_name = rng.choice(list(SMOOTH_FN.keys()))
    return SMOOTH_FN[f1_name](x1, rng) * SMOOTH_FN[f2_name](x2, rng)

def _f2d_additive_plus_interaction(x1, x2, rng):
    main1 = SMOOTH_FN[rng.choice(list(SMOOTH_FN.keys()))](x1, rng)
    main2 = SMOOTH_FN[rng.choice(list(SMOOTH_FN.keys()))](x2, rng)
    interaction = rng.uniform(0.2, 1.5) * x1 * x2
    return main1 + main2 + interaction

def _f2d_cliff(x1, x2, rng):
    """Sharp boundary in 2D."""
    angle = rng.uniform(0, np.pi)
    u = np.cos(angle)*x1 + np.sin(angle)*x2
    return rng.uniform(3, 8) / (1 + np.exp(-rng.uniform(5, 30)*u))

def _f2d_volcano(x1, x2, rng):
    r = np.sqrt(x1**2 + x2**2)
    peak_r = rng.uniform(0.3, 1.5)
    return rng.uniform(2, 6) * np.exp(-((r - peak_r)/0.3)**2)

SMOOTH_FN_2D = {
    "saddle": _f2d_saddle, "dome": _f2d_dome, "ridge": _f2d_ridge,
    "spiral": _f2d_spiral, "checkerboard": _f2d_checkerboard,
    "manifold_1d": _f2d_manifold_1d, "radial_wave": _f2d_radial_wave,
    "product": _f2d_product, "additive_interaction": _f2d_additive_plus_interaction,
    "cliff": _f2d_cliff, "volcano": _f2d_volcano,
}

# ═══════════════════════════════════════════════════════════════════════════
# NOISE GENERATORS — 15 types
# ═══════════════════════════════════════════════════════════════════════════

def _n_gaussian(n, sd, rng, **kw):     return rng.randn(n) * sd
def _n_t(n, sd, rng, **kw):
    df = rng.uniform(2.5, 8); return rng.standard_t(df, n) * sd / np.sqrt(df/(df-2))
def _n_laplace(n, sd, rng, **kw):      return rng.laplace(0, sd/np.sqrt(2), n)
def _n_cauchy(n, sd, rng, **kw):       return rng.standard_cauchy(n) * sd * 0.3
def _n_skew(n, sd, rng, **kw):
    a = rng.uniform(2, 8)*rng.choice([-1, 1]); z = rng.randn(n); u = rng.randn(n)
    raw = np.where(u < a*z, z, -z); return raw * sd/(np.std(raw)+1e-8)
def _n_mixture(n, sd, rng, **kw):
    mix = rng.uniform(.1, .4); s2 = rng.uniform(3, 10)
    noise = rng.randn(n)*sd; noise[rng.random(n) < mix] *= s2; return noise
def _n_uniform(n, sd, rng, **kw):      return rng.uniform(-sd*np.sqrt(3), sd*np.sqrt(3), n)
def _n_hetero(n, sd, rng, x=None, **kw):
    if x is None: x = np.linspace(0, 1, n)
    t = (x - x.min())/(np.ptp(x)+1e-8); return rng.randn(n)*sd*(0.2 + 2*t)
def _n_periodic_het(n, sd, rng, x=None, **kw):
    if x is None: x = np.linspace(0, 1, n)
    t = (x - x.min())/(np.ptp(x)+1e-8)
    return rng.randn(n)*sd*(0.5 + np.abs(np.sin(rng.uniform(2, 6)*np.pi*t)))
def _n_lognormal(n, sd, rng, **kw):
    raw = rng.lognormal(0, 1, n); raw -= raw.mean(); return raw*sd/(np.std(raw)+1e-8)
def _n_sparse_outlier(n, sd, rng, **kw):
    noise = rng.randn(n)*sd
    k = max(1, int(n*rng.uniform(.01, .05)))
    noise[rng.choice(n, k, replace=False)] = rng.randn(k)*sd*rng.uniform(10, 50)
    return noise
def _n_quantized(n, sd, rng, **kw):
    lev = rng.uniform(5, 20); return np.round(rng.randn(n)*sd*lev)/lev
def _n_ar1(n, sd, rng, **kw):
    """Autocorrelated noise."""
    phi = rng.uniform(0.3, 0.95); e = rng.randn(n)*sd*np.sqrt(1-phi**2)
    out = np.zeros(n); out[0] = e[0]
    for i in range(1, n): out[i] = phi*out[i-1] + e[i]
    return out
def _n_bimodal(n, sd, rng, **kw):
    mask = rng.random(n) < 0.5
    return np.where(mask, rng.randn(n)*sd - sd*1.5, rng.randn(n)*sd + sd*1.5)
def _n_contaminated(n, sd, rng, **kw):
    """Normal with 5% contamination from a different mean."""
    noise = rng.randn(n)*sd
    k = max(1, int(n*0.05))
    noise[rng.choice(n, k, replace=False)] += rng.choice([-1, 1])*sd*rng.uniform(5, 15)
    return noise

NOISE_FN = {
    "gaussian": _n_gaussian, "t": _n_t, "laplace": _n_laplace,
    "cauchy": _n_cauchy, "skew": _n_skew, "mixture": _n_mixture,
    "uniform": _n_uniform, "heteroscedastic": _n_hetero,
    "periodic_het": _n_periodic_het, "lognormal": _n_lognormal,
    "sparse_outlier": _n_sparse_outlier, "quantized": _n_quantized,
    "ar1": _n_ar1, "bimodal": _n_bimodal, "contaminated": _n_contaminated,
}

# ═══════════════════════════════════════════════════════════════════════════
# X DISTRIBUTIONS — 12 types
# ═══════════════════════════════════════════════════════════════════════════

def _x_uniform(n, k, rng):         return rng.uniform(-1, 1, (n, k))
def _x_normal(n, k, rng):          return rng.randn(n, k)
def _x_skewed(n, k, rng):          r = rng.exponential(1, (n, k)); return r - r.mean(0)
def _x_heavy(n, k, rng):           return rng.standard_t(3, (n, k))
def _x_clustered(n, k, rng):
    nc = rng.randint(2, 6); c = rng.randn(nc, k)*3
    return c[rng.randint(0, nc, n)] + rng.randn(n, k)*0.3
def _x_bimodal(n, k, rng):
    m = rng.random((n, k)) < 0.5; return np.where(m, rng.randn(n, k)-2, rng.randn(n, k)+2)
def _x_uniform_wide(n, k, rng):    return rng.uniform(-10, 10, (n, k))
def _x_sparse(n, k, rng):
    r = rng.randn(n, k); r[rng.random((n, k)) < 0.8] *= 0.1; return r
def _x_grid(n, k, rng):
    lev = rng.randint(5, 15); return np.round(rng.uniform(-1, 1, (n, k))*lev)/lev
def _x_correlated(n, k, rng):
    z = rng.randn(n, k); L = rng.randn(k, k)*0.3; np.fill_diagonal(L, 1.0); return z @ L
def _x_low_dim_manifold(n, k, rng):
    """Features live on a low-dimensional manifold (intrinsic dim < k)."""
    intrinsic = max(1, k // 2)
    z = rng.randn(n, intrinsic)
    A = rng.randn(intrinsic, k)
    return z @ A + rng.randn(n, k) * 0.05
def _x_mixture_of_lines(n, k, rng):
    """Points clustered along random lines in feature space."""
    n_lines = rng.randint(2, 5)
    out = np.zeros((n, k))
    for i in range(n):
        line_id = rng.randint(0, n_lines)
        direction = rng.randn(k)
        direction /= np.linalg.norm(direction) + 1e-8
        origin = rng.randn(k) * 2
        t = rng.randn() * 2
        out[i] = origin + t * direction + rng.randn(k) * 0.1
    return out

XDIST_FN = {
    "uniform": _x_uniform, "normal": _x_normal, "skewed": _x_skewed,
    "heavy_tailed": _x_heavy, "clustered": _x_clustered, "bimodal": _x_bimodal,
    "uniform_wide": _x_uniform_wide, "sparse": _x_sparse, "grid": _x_grid,
    "correlated": _x_correlated, "low_dim_manifold": _x_low_dim_manifold,
    "mixture_of_lines": _x_mixture_of_lines,
}

# ═══════════════════════════════════════════════════════════════════════════
# SIGMA (VARIANCE) FUNCTIONS — for GAMLSS
# ═══════════════════════════════════════════════════════════════════════════

def _sigma_constant(x, rng):
    return np.ones_like(x[:, 0]) * rng.uniform(0.3, 3.0)

def _sigma_linear(x, rng):
    return np.exp(rng.uniform(-1, 1) * x[:, 0])

def _sigma_smooth(x, rng):
    fn = SMOOTH_FN[rng.choice(list(SMOOTH_FN.keys()))]
    raw = fn(x[:, 0], rng)
    return np.exp(raw / (np.std(raw) + 1e-8) * 0.5)

def _sigma_bimodal(x, rng):
    mid = np.median(x[:, 0])
    return np.where(x[:, 0] < mid, rng.uniform(0.3, 1.0), rng.uniform(1.5, 5.0))

def _sigma_periodic(x, rng):
    return np.exp(0.5 * np.sin(rng.uniform(1, 5) * x[:, 0]))

SIGMA_FN = {
    "constant": _sigma_constant, "linear": _sigma_linear,
    "smooth": _sigma_smooth, "bimodal": _sigma_bimodal,
    "periodic": _sigma_periodic,
}

# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL STRUCTURE GENERATORS — how smooth components combine
# ═══════════════════════════════════════════════════════════════════════════

def _build_additive_signal(X, smooth_kinds, rng):
    """Standard additive: f1(x1) + f2(x2) + ..."""
    eta = np.zeros(X.shape[0])
    for j, kind in enumerate(smooth_kinds):
        contrib = SMOOTH_FN[kind](X[:, j % X.shape[1]], rng)
        s = np.std(contrib)
        if s > 1e-8: contrib /= s
        eta += contrib
    return eta

def _build_additive_with_interactions(X, smooth_kinds, rng):
    """Additive + pairwise interactions."""
    eta = _build_additive_signal(X, smooth_kinds, rng)
    k = X.shape[1]
    if k >= 2:
        n_inter = min(rng.randint(1, 4), k * (k - 1) // 2)
        for _ in range(n_inter):
            i, j = rng.choice(k, 2, replace=False)
            strength = rng.uniform(0.2, 2.0)
            eta += strength * X[:, i] * X[:, j]
    return eta

def _build_2d_surface(X, smooth_kinds, rng):
    """Use 2D smooth functions on pairs of features."""
    eta = np.zeros(X.shape[0])
    k = X.shape[1]
    fn2d_names = list(SMOOTH_FN_2D.keys())
    # Add 2D surfaces on consecutive pairs
    for j in range(0, k - 1, 2):
        fn2d = SMOOTH_FN_2D[rng.choice(fn2d_names)]
        contrib = fn2d(X[:, j], X[:, j + 1], rng)
        s = np.std(contrib)
        if s > 1e-8: contrib /= s
        eta += contrib
    # If odd number, add a 1D smooth for the last feature
    if k % 2 == 1:
        fn1d = SMOOTH_FN[rng.choice(list(SMOOTH_FN.keys()))]
        contrib = fn1d(X[:, -1], rng)
        s = np.std(contrib)
        if s > 1e-8: contrib /= s
        eta += contrib
    return eta

def _build_low_dim_manifold(X, smooth_kinds, rng):
    """Signal depends on a low-dim projection of features."""
    k = X.shape[1]
    dim = max(1, min(k // 2, 3))
    proj = rng.randn(k, dim)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
    Z = X @ proj
    eta = np.zeros(X.shape[0])
    for d in range(dim):
        fn = SMOOTH_FN[rng.choice(list(SMOOTH_FN.keys()))]
        contrib = fn(Z[:, d], rng)
        s = np.std(contrib)
        if s > 1e-8: contrib /= s
        eta += contrib
    return eta

def _build_stacked(X, smooth_kinds, rng):
    """Composition: g(f1(x1) + f2(x2))."""
    inner = _build_additive_signal(X, smooth_kinds, rng)
    outer_fn = SMOOTH_FN[rng.choice(list(SMOOTH_FN.keys()))]
    inner_norm = inner / (np.std(inner) + 1e-8)
    return outer_fn(inner_norm, rng)

def _build_regime(X, smooth_kinds, rng):
    """Different functions in different regions of feature space."""
    k = X.shape[1]
    split_dim = rng.randint(0, k)
    threshold = np.median(X[:, split_dim])
    mask = X[:, split_dim] < threshold

    smooth_names = list(SMOOTH_FN.keys())
    eta = np.zeros(X.shape[0])
    for j in range(min(k, len(smooth_kinds))):
        fn_a = SMOOTH_FN[rng.choice(smooth_names)]
        fn_b = SMOOTH_FN[rng.choice(smooth_names)]
        ca = fn_a(X[:, j], rng); cb = fn_b(X[:, j], rng)
        for arr in [ca, cb]:
            s = np.std(arr)
            if s > 1e-8: arr /= s
        eta += np.where(mask, ca, cb)
    return eta

SIGNAL_BUILDERS = {
    "additive": _build_additive_signal,
    "additive_interaction": _build_additive_with_interactions,
    "surface_2d": _build_2d_surface,
    "low_dim_manifold": _build_low_dim_manifold,
    "stacked": _build_stacked,
    "regime": _build_regime,
}

# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FuzzScenario:
    seed: int
    family: str               # gaussian, binomial
    model_type: str            # gam, gamlss
    n_obs: int
    n_smooths: int
    knots: int
    double_penalty: bool
    noise_sd: float
    noise_kind: str
    smooth_kinds: list
    x_distribution: str
    basis_type: str            # ps, tps, duchon
    collinear_strength: float
    signal_structure: str      # additive, additive_interaction, surface_2d, ...
    sigma_kind: str            # for gamlss: constant, linear, smooth, ...
    duchon_order: int          # 0 or 1
    duchon_power: int          # 1 or 2
    n_duchon_dims: int         # how many features go into the duchon term (2 or 3)

    def tag(self) -> str:
        return f"s{self.seed}_{self.family}_{self.model_type}_{self.basis_type}"


def estimate_scenario_cost(sc: FuzzScenario) -> float:
    cost = float(max(sc.n_obs, 1) * max(sc.n_smooths, 1) * max(sc.knots, 1))
    if sc.family == "binomial":
        cost *= 2.0
    if sc.model_type == "gamlss":
        cost *= 1.8
    cost *= {
        "ps": 1.6,
        "tps": 3.2,
        "duchon": 1.2,
    }[sc.basis_type]
    if sc.double_penalty:
        cost *= 1.15
    if sc.signal_structure in {"surface_2d", "stacked", "regime"}:
        cost *= 1.2
    if sc.noise_kind in {"heteroscedastic", "periodic_het", "lognormal", "sparse_outlier"}:
        cost *= 1.15
    return cost


def generate_scenario(seed: int, family_filter=None, model_type_filter=None) -> FuzzScenario:
    rng = np.random.RandomState(seed)
    choice = lambda lst: lst[rng.randint(0, len(lst))]

    family = family_filter or choice(["gaussian"]*3 + ["binomial"]*2)
    model_type = model_type_filter or choice(["gam"]*4 + ["gamlss"]*2)
    if family == "binomial":
        model_type = "gam"  # no binomial gamlss for now

    n_obs = choice([50, 100, 200, 500, 1000, 2000, 5000])
    n_smooths = choice([1, 1, 2, 2, 3, 3, 5, 7, 10])
    knots = choice([4, 5, 7, 8, 10, 12, 15, 20])
    double_penalty = rng.random() < 0.5
    noise_sd = choice([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    noise_kind = choice(list(NOISE_FN.keys()))
    smooth_kinds = [choice(list(SMOOTH_FN.keys())) for _ in range(n_smooths)]
    x_distribution = choice(list(XDIST_FN.keys()))
    basis_type = choice(["ps"]*3 + ["tps"]*2 + ["duchon"]*3)
    collinear_strength = choice([0]*4 + [0.3, 0.5, 0.7, 0.9])
    signal_structure = choice(list(SIGNAL_BUILDERS.keys()))
    sigma_kind = choice(list(SIGMA_FN.keys())) if model_type == "gamlss" else "constant"
    duchon_order = choice([0, 0, 1])
    duchon_power = choice([1, 2, 2])
    n_duchon_dims = choice([2, 2, 3]) if n_smooths >= 2 else 2

    # Constraints
    max_knots = max(3, n_obs // max(n_smooths + 1, 2) - 2)
    knots = min(knots, max_knots)
    if family == "binomial":
        knots = min(knots, max(3, n_obs // 5))
    if basis_type == "duchon" and n_smooths < 2:
        n_smooths = 2  # duchon needs at least 2 dims
        smooth_kinds = [choice(list(SMOOTH_FN.keys())) for _ in range(n_smooths)]
    if basis_type == "duchon":
        n_smooths = max(n_smooths, 3)
        while len(smooth_kinds) < n_smooths:
            smooth_kinds.append(choice(list(SMOOTH_FN.keys())))
        n_duchon_dims = min(max(n_duchon_dims, 2), n_smooths)
        duchon_order = 1
        max_knots = max(3, n_obs // max(n_smooths + 1, 2) - 2)
        knots = min(knots, max_knots)
        if duchon_order >= 1 and knots < n_duchon_dims + 1:
            n_duchon_dims = max(2, min(n_duchon_dims, knots - 1))
        duchon_power = _admissible_duchon_power(n_duchon_dims, duchon_order, 1)

    return FuzzScenario(
        seed=seed, family=family, model_type=model_type, n_obs=n_obs,
        n_smooths=n_smooths, knots=knots, double_penalty=double_penalty,
        noise_sd=noise_sd, noise_kind=noise_kind, smooth_kinds=smooth_kinds,
        x_distribution=x_distribution, basis_type=basis_type,
        collinear_strength=collinear_strength, signal_structure=signal_structure,
        sigma_kind=sigma_kind, duchon_order=duchon_order, duchon_power=duchon_power,
        n_duchon_dims=n_duchon_dims,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_data(sc: FuzzScenario):
    """Returns (train_df, test_df, feature_cols)."""
    rng = np.random.RandomState(sc.seed)
    n, k = sc.n_obs, sc.n_smooths
    cols = [f"x{i}" for i in range(k)]

    X = XDIST_FN[sc.x_distribution](n, k, rng)
    if sc.collinear_strength > 0 and k >= 2:
        mix = sc.collinear_strength
        for j in range(1, k):
            X[:, j] = mix * X[:, 0] + (1 - mix) * X[:, j]

    builder = SIGNAL_BUILDERS.get(sc.signal_structure, _build_additive_signal)
    eta = builder(X, sc.smooth_kinds, rng)

    if sc.family == "gaussian":
        signal_sd = max(np.std(eta), 1e-8)
        noise_fn = NOISE_FN[sc.noise_kind]
        sig = inspect.signature(noise_fn)
        kw = {"x": X[:, 0]} if "x" in sig.parameters else {}
        noise = noise_fn(n, sc.noise_sd * signal_sd, rng, **kw)

        if sc.model_type == "gamlss":
            sigma_vals = SIGMA_FN[sc.sigma_kind](X, rng)
            noise = noise * sigma_vals / (np.std(noise * sigma_vals) + 1e-8) * sc.noise_sd * signal_sd

        y = eta + noise
    elif sc.family == "binomial":
        eta = eta - np.mean(eta)
        eta_sd = max(np.std(eta), 1e-8)
        eta = eta * 2.0 / eta_sd
        y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)

    df = pd.DataFrame(X, columns=cols)
    df["y"] = y
    folds = make_folds(y, n_splits=1, seed=sc.seed, stratified=(sc.family == "binomial"))
    fold = folds[0]
    train_df = df.iloc[fold.train_idx].reset_index(drop=True)
    test_df = df.iloc[fold.test_idx].reset_index(drop=True)
    train_df, test_df = zscore_train_test(train_df, test_df, cols)
    return train_df, test_df, cols


# ═══════════════════════════════════════════════════════════════════════════
# FORMULA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _rhs_from_terms(terms):
    return " + ".join(terms) if terms else "1"


def _formula_from_terms(response, terms):
    return f"{response} ~ {_rhs_from_terms(terms)}"


def _admissible_duchon_power(dim: int, order: int, requested_power: int) -> int:
    """Smallest power satisfying value, gradient, and Hessian collocation."""
    power = max(0, int(requested_power))
    while 2 * (int(order) + power) <= int(dim) + 2:
        power += 1
    return power


def _duchon_dims_for_centers(cols, sc, centers: int) -> int:
    dims = min(max(int(sc.n_duchon_dims), 2), len(cols))
    if int(sc.duchon_order) >= 1 and centers < dims + 1:
        dims = max(2, min(dims, centers - 1))
    return dims


def _rust_mean_terms(cols, sc):
    dp = "true" if sc.double_penalty else "false"
    if sc.basis_type == "duchon":
        dims = _duchon_dims_for_centers(cols, sc, sc.knots)
        power = _admissible_duchon_power(dims, sc.duchon_order, sc.duchon_power)
        d_cols = cols[:dims]
        duchon_term = f"duchon({', '.join(d_cols)}, centers={sc.knots}, order={sc.duchon_order}, power={power}, double_penalty={dp})"
        extra = [f"s({c}, type=ps, knots={sc.knots}, double_penalty={dp})" for c in cols[dims:]]
        return [duchon_term] + extra
    elif sc.basis_type == "tps":
        return [f"s({c}, type=tps, centers={sc.knots}, double_penalty={dp})" for c in cols]
    else:
        return [f"s({c}, type=ps, knots={sc.knots}, double_penalty={dp})" for c in cols]


def rust_mean_formula(cols, sc):
    return _formula_from_terms("y", _rust_mean_terms(cols, sc))


def rust_noise_terms(cols, sc):
    """Noise terms for GAMLSS; --predict-noise expects only the RHS."""
    dp = "true" if sc.double_penalty else "false"
    if sc.basis_type == "duchon" and len(cols) >= 2:
        centers = max(3, sc.knots // 2)
        dims = _duchon_dims_for_centers(cols, sc, centers)
        power = _admissible_duchon_power(dims, sc.duchon_order, sc.duchon_power)
        d_cols = cols[:dims]
        return f"duchon({', '.join(d_cols)}, centers={centers}, order={sc.duchon_order}, power={power}, double_penalty={dp})"
    return f"s({cols[0]}, type=ps, knots={max(3, sc.knots // 2)}, double_penalty={dp})"


def build_rust_fit_cmd(sc, train_csv, model_json, cols):
    fit_cmd = [str(RUST_BINARY), "fit", "--out", model_json]
    if sc.model_type == "gamlss":
        fit_cmd += ["--predict-noise", rust_noise_terms(cols, sc)]
    fit_cmd += [train_csv, rust_mean_formula(cols, sc)]
    return fit_cmd


def mgcv_formula(cols, sc):
    if sc.basis_type == "duchon":
        dims = _duchon_dims_for_centers(cols, sc, sc.knots)
        d_cols = cols[:dims]
        m_vals = f"c({_admissible_duchon_power(dims, sc.duchon_order, sc.duchon_power)},{sc.duchon_order})"
        k_val = sc.knots
        duchon_term = f"s({','.join(d_cols)}, bs='ds', m={m_vals}, k=min({k_val}, nrow(train_df)-1))"
        extra = [f"s({c}, bs='ps', k=min({sc.knots + 4}, nrow(train_df)-1))" for c in cols[dims:]]
        return "y ~ " + " + ".join([duchon_term] + extra)
    elif sc.basis_type == "tps":
        terms = [f"s({c}, bs='tp', k=min({sc.knots}, nrow(train_df)-1))" for c in cols]
        return "y ~ " + " + ".join(terms)
    else:
        terms = [f"s({c}, bs='ps', k=min({sc.knots + 4}, nrow(train_df)-1))" for c in cols]
        return "y ~ " + " + ".join(terms)


def mgcv_sigma_formula(cols, sc):
    if sc.basis_type == "duchon" and len(cols) >= 2:
        k_val = max(3, sc.knots // 2)
        dims = _duchon_dims_for_centers(cols, sc, k_val)
        d_cols = cols[:dims]
        m_vals = f"c({_admissible_duchon_power(dims, sc.duchon_order, sc.duchon_power)},{sc.duchon_order})"
        return f"~ s({','.join(d_cols)}, bs='ds', m={m_vals}, k=min({k_val}, nrow(train_df)-1))"
    k_val = max(3, sc.knots // 2) + (4 if sc.basis_type == "ps" else 0)
    bs = "ps" if sc.basis_type == "ps" else "tp"
    return f"~ s({cols[0]}, bs='{bs}', k=min({k_val}, nrow(train_df)-1))"


# ═══════════════════════════════════════════════════════════════════════════
# RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def run_rust(sc, train_df, test_df, cols, tmpdir, rust_timeout):
    train_csv = os.path.join(tmpdir, "train.csv")
    test_csv = os.path.join(tmpdir, "test.csv")
    model_json = os.path.join(tmpdir, "model.json")
    pred_csv = os.path.join(tmpdir, "pred.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    y_test = test_df["y"].to_numpy(float)
    y_train = train_df["y"].to_numpy(float)

    t0 = time.time()
    try:
        fit_cmd = build_rust_fit_cmd(sc, train_csv, model_json, cols)

        r = subprocess.run(fit_cmd, capture_output=True, text=True, timeout=rust_timeout)
        if r.returncode != 0:
            return {"error": f"fit rc={r.returncode}", "stderr": r.stderr,
                    "stdout": r.stdout, "cmd": " ".join(fit_cmd), "time": time.time()-t0}

        r = subprocess.run(
            [str(RUST_BINARY), "predict", model_json, test_csv, "--out", pred_csv],
            capture_output=True, text=True, timeout=rust_timeout,
        )
        if r.returncode != 0:
            return {"error": f"predict rc={r.returncode}", "stderr": r.stderr,
                    "stdout": r.stdout, "time": time.time()-t0}

        pred_df = pd.read_csv(pred_csv)
        preds = pred_df["mean"].to_numpy(float)
        if len(preds) != len(y_test):
            return {"error": f"pred count {len(preds)} vs {len(y_test)}", "time": time.time()-t0}

        metrics = _compute_metrics(sc.family, y_test, y_train, preds)
        metrics["time"] = time.time() - t0

        if sc.family == "gaussian":
            try:
                with open(model_json) as f:
                    model = json.load(f)
                sigma = model.get("fit_result", {}).get("standard_deviation")
                if sigma and sigma > 0:
                    metrics["logloss"] = gaussian_log_loss_score(y_test, preds, sigma)
            except Exception:
                pass
        return metrics

    except subprocess.TimeoutExpired as e:
        # subprocess.run(capture_output=True) buffers stderr/stdout; on timeout
        # the TimeoutExpired exception exposes whatever was captured before the
        # kill. Surface it so diagnostic logs emitted before the 60s wall make
        # it into fuzz_results.jsonl (otherwise every timeout is opaque and we
        # can't tell which iteration / function was stuck).
        return {
            "error": "timeout",
            "time": rust_timeout,
            "stderr": (e.stderr or "") if hasattr(e, "stderr") else "",
            "stdout": (e.stdout or "") if hasattr(e, "stdout") else "",
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc(), "time": time.time()-t0}


def run_mgcv(sc, train_df, test_df, cols, tmpdir, r_timeout):
    train_csv = os.path.join(tmpdir, "mgcv_train.csv")
    test_csv = os.path.join(tmpdir, "mgcv_test.csv")
    out_json = os.path.join(tmpdir, "mgcv_out.json")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    mu_formula = mgcv_formula(cols, sc)
    select_str = "TRUE" if sc.double_penalty else "FALSE"
    fam_str = "binomial(link='logit')" if sc.family == "binomial" else "gaussian(link='identity')"

    if sc.model_type == "gamlss" and sc.family == "gaussian":
        sigma_f = mgcv_sigma_formula(cols, sc)
        fit_line = f"fit <- gam(list({mu_formula}, {sigma_f}), family=gaulss(), data=train_df, method='REML', select={select_str})"
        pred_line = """
pred_raw <- predict(fit, newdata=test_df, type='response')
if (is.list(pred_raw)) {
    p <- as.numeric(pred_raw[[1]])
    inv_sigma <- as.numeric(pred_raw[[2]])
} else if (is.matrix(pred_raw)) {
    p <- as.numeric(pred_raw[,1])
    inv_sigma <- as.numeric(pred_raw[,2])
} else {
    p <- as.numeric(pred_raw)
    inv_sigma <- rep(1.0, length(p))
}
sigma_hat <- 1.0 / pmax(inv_sigma, 1e-12)
"""
        gaussian_metrics = """
rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst > 0) 1.0 - sum((y_test - p)^2) / sst else 0.0
logloss <- mean(0.5 * log(2 * pi * sigma_hat^2) + ((y_test - p)^2) / (2 * sigma_hat^2))
result <- list(status='ok', r2=r2, rmse=rmse, mae=mae, logloss=logloss)
"""
    else:
        fit_line = f"fit <- gam({mu_formula}, family={fam_str}, data=train_df, method='REML', select={select_str})"
        pred_line = "p <- as.numeric(predict(fit, newdata=test_df, type='response'))"
        if sc.family == "binomial":
            gaussian_metrics = """
p_safe <- pmin(pmax(p, 1e-12), 1 - 1e-12)
n_pos <- sum(y_test > 0.5); n_neg <- sum(y_test <= 0.5)
if (n_pos > 0 && n_neg > 0) {
  ord <- order(p); yy <- y_test[ord]; ranks <- seq_along(yy)
  auc <- (sum(ranks[yy > 0.5]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
} else { auc <- 0.5 }
brier <- mean((y_test - p)^2)
logloss <- mean(-(y_test*log(p_safe) + (1-y_test)*log(1-p_safe)))
p_mean <- mean(y_train)
nagelkerke_r2 <- NULL
if (is.finite(p_mean) && p_mean > 0 && p_mean < 1) {
  ll_null <- sum(y_test*log(p_mean)+(1-y_test)*log(1-p_mean))
  ll_model <- sum(y_test*log(p_safe)+(1-y_test)*log(1-p_safe))
  n_obs <- length(y_test)
  r2_cs <- 1-exp((2/n_obs)*(ll_null-ll_model))
  max_r2_cs <- 1-exp((2/n_obs)*ll_null)
  nagelkerke_r2 <- if (max_r2_cs > 0) r2_cs/max_r2_cs else NULL
}
result <- list(status='ok', auc=auc, brier=brier, logloss=logloss, nagelkerke_r2=nagelkerke_r2)
"""
        else:
            gaussian_metrics = """
sigma_hat <- NA_real_
fit_scale <- tryCatch(as.numeric(summary(fit)$scale), error=function(e) NA_real_)
if (is.finite(fit_scale) && fit_scale > 0) { sigma_hat <- sqrt(fit_scale)
} else { p_train <- as.numeric(predict(fit, newdata=train_df, type='response')); sigma_hat <- sqrt(mean((y_train - p_train)^2)) }
sigma_hat <- max(as.numeric(sigma_hat), 1e-12)
rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst > 0) 1.0 - sum((y_test - p)^2) / sst else 0.0
logloss <- mean(0.5 * log(2 * pi * sigma_hat^2) + ((y_test - p)^2) / (2 * sigma_hat^2))
result <- list(status='ok', r2=r2, rmse=rmse, mae=mae, logloss=logloss)
"""

    r_script = f"""
suppressMessages({{ library(mgcv); library(jsonlite) }})
train_df <- read.csv("{train_csv}")
test_df <- read.csv("{test_csv}")
y_test <- test_df$y
y_train <- train_df$y
tryCatch({{
    {fit_line}
    {pred_line}
    {gaussian_metrics}
    writeLines(toJSON(result, auto_unbox=TRUE, null="null"), "{out_json}")
}}, error=function(e) {{
    writeLines(toJSON(list(status="error", message=conditionMessage(e)), auto_unbox=TRUE), "{out_json}")
}})
"""
    script_path = os.path.join(tmpdir, "mgcv.R")
    with open(script_path, "w") as f:
        f.write(r_script)

    t0 = time.time()
    try:
        r = subprocess.run(["Rscript", script_path], capture_output=True, text=True, timeout=r_timeout)
        elapsed = time.time() - t0

        if not os.path.exists(out_json):
            return {"error": f"no R output rc={r.returncode}", "stderr": r.stderr,
                    "stdout": r.stdout, "time": elapsed}

        with open(out_json) as f:
            out = json.load(f)

        if out.get("status") != "ok":
            return {"error": out.get("message", "R error"), "stderr": r.stderr, "time": elapsed}

        result = {"time": elapsed}
        for key in ["r2", "rmse", "mae", "logloss", "auc", "brier", "nagelkerke_r2"]:
            if key in out and out[key] is not None:
                result[key] = float(out[key])
        return result

    except subprocess.TimeoutExpired as e:
        return {
            "error": "timeout",
            "time": r_timeout,
            "stderr": (e.stderr or "") if hasattr(e, "stderr") else "",
            "stdout": (e.stdout or "") if hasattr(e, "stdout") else "",
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc(), "time": time.time()-t0}


def _compute_metrics(family, y_test, y_train, preds):
    m = {}
    if family == "gaussian":
        m["r2"] = r2_score(y_test, preds)
        m["rmse"] = rmse_score(y_test, preds)
        m["mae"] = mae_score(y_test, preds)
    elif family == "binomial":
        m["auc"] = auc_score(y_test, preds)
        m["brier"] = brier_score(y_test, preds)
        m["logloss"] = log_loss_score(y_test, preds)
        nr2 = nagelkerke_r2_score(y_test, preds, null_mean=float(np.mean(y_train)))
        if nr2 is not None:
            m["nagelkerke_r2"] = nr2
    return m


# ═══════════════════════════════════════════════════════════════════════════
# RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FuzzResult:
    scenario: dict
    rust: dict
    mgcv: dict
    primary_gap: Optional[float] = None
    primary_metric: Optional[str] = None

    def compute_gap(self):
        fam = self.scenario["family"]
        if fam == "gaussian":
            self.primary_metric = "r2"
        else:
            self.primary_metric = "auc"
        rv = self.rust.get(self.primary_metric)
        mv = self.mgcv.get(self.primary_metric)
        if rv is not None and mv is not None:
            self.primary_gap = mv - rv
        elif mv is not None and rv is None:
            self.primary_gap = mv + 1.0  # rust failed


ABS_GAP_WARN_THRESHOLD = 0.05
ABS_GAP_FAIL_THRESHOLD = 0.1
GAUSSIAN_R2_WARN_FLOOR = 0.01
GAUSSIAN_R2_FAIL_FLOOR = 0.02
GAUSSIAN_RMSE_WARN_RATIO = 1.5
GAUSSIAN_RMSE_FAIL_RATIO = 2.0


def gaussian_rmse_ratio(result: FuzzResult) -> Optional[float]:
    if result.scenario["family"] != "gaussian":
        return None

    rust_rmse = result.rust.get("rmse")
    mgcv_rmse = result.mgcv.get("rmse")
    if rust_rmse is None or mgcv_rmse is None:
        return None

    rust_rmse = float(rust_rmse)
    mgcv_rmse = float(mgcv_rmse)
    if not math.isfinite(rust_rmse) or not math.isfinite(mgcv_rmse):
        return None
    if rust_rmse < 0 or mgcv_rmse < 0:
        return None
    if mgcv_rmse <= 1e-12:
        return math.inf if rust_rmse > 1e-12 else 1.0
    return rust_rmse / mgcv_rmse


def classify_primary_divergence(result: FuzzResult) -> tuple[str, str]:
    gap = result.primary_gap
    if gap is None or gap <= 0:
        return "", ""

    metric = result.primary_metric or "metric"
    if gap > ABS_GAP_FAIL_THRESHOLD:
        return "fail", f"{metric}_gap={gap:+.4f}"

    if result.scenario["family"] == "gaussian":
        rmse_ratio = gaussian_rmse_ratio(result)
        if rmse_ratio is not None:
            if gap > GAUSSIAN_R2_FAIL_FLOOR and rmse_ratio > GAUSSIAN_RMSE_FAIL_RATIO:
                return "fail", f"r2_gap={gap:+.4f}, rmse_ratio={rmse_ratio:.2f}x"
            if gap > GAUSSIAN_R2_WARN_FLOOR and rmse_ratio > GAUSSIAN_RMSE_WARN_RATIO:
                return "warn", f"r2_gap={gap:+.4f}, rmse_ratio={rmse_ratio:.2f}x"

    if gap > ABS_GAP_WARN_THRESHOLD:
        return "warn", f"{metric}_gap={gap:+.4f}"
    return "", ""


def run_trial(sc, rust_timeout, r_timeout):
    train_df, test_df, cols = generate_data(sc)
    with tempfile.TemporaryDirectory(prefix="gam_fuzz_") as tmpdir:
        rust_out = run_rust(sc, train_df, test_df, cols, tmpdir, rust_timeout)
        mgcv_out = run_mgcv(sc, train_df, test_df, cols, tmpdir, r_timeout)
    result = FuzzResult(scenario=asdict(sc), rust=rust_out, mgcv=mgcv_out)
    result.compute_gap()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def print_leaderboard(results, top_n=25):
    groups = {}
    for r in results:
        key = (r.scenario["family"], r.scenario["model_type"], r.scenario["basis_type"])
        groups.setdefault(key, []).append(r)

    # Also print overall
    groups[("ALL", "ALL", "ALL")] = results

    for (fam, mt, basis), subset in sorted(groups.items()):
        valid = [r for r in subset if r.primary_gap is not None]
        if not valid:
            continue
        valid.sort(key=lambda r: -(r.primary_gap or 0))

        metric = valid[0].primary_metric or "?"
        label = f"{fam}/{mt}/{basis}" if fam != "ALL" else "ALL TRIALS"

        print(f"\n{'=' * 120}")
        print(f"  {label} — {metric} gap (mgcv - rust)  |  {len(valid)} valid / {len(subset)} total")
        print("=" * 120)
        print(f"{'#':>3}  {'gap':>8}  {'rust':>9}  {'mgcv':>9}  {'n':>5}  {'k':>2}  {'kn':>3}  {'dp':>3}  "
              f"{'noise':>6}  {'noise_t':>12}  {'basis':>6}  {'signal':>15}  {'x_dist':>12}  {'smooths'}")
        print("-" * 120)

        for i, r in enumerate(valid[:top_n]):
            s = r.scenario
            gap_s = f"{r.primary_gap:+.4f}" if r.primary_gap is not None else "  N/A "
            rv = r.rust.get(r.primary_metric or "r2")
            mv = r.mgcv.get(r.primary_metric or "r2")
            rs = f"{rv:.4f}" if rv is not None else "  FAIL"
            ms = f"{mv:.4f}" if mv is not None else "  FAIL"
            kinds = ",".join(k[:3] for k in s["smooth_kinds"][:4])
            if len(s["smooth_kinds"]) > 4: kinds += f"+{len(s['smooth_kinds'])-4}"
            print(f"{i+1:3d}  {gap_s}  {rs:>9}  {ms:>9}  {s['n_obs']:5d}  {s['n_smooths']:2d}  "
                  f"{s['knots']:3d}  {'Y' if s['double_penalty'] else 'N':>3}  "
                  f"{s['noise_sd']:6.2f}  {s['noise_kind']:>12}  {s['basis_type']:>6}  "
                  f"{s['signal_structure']:>15}  {s['x_distribution']:>12}  {kinds}")

        gaps = [r.primary_gap for r in valid]
        mgcv_w = sum(1 for g in gaps if g > 0.01)
        rust_w = sum(1 for g in gaps if g < -0.01)
        ties = len(gaps) - mgcv_w - rust_w
        print(f"\n  mgcv wins: {mgcv_w} | rust wins: {rust_w} | ties: {ties} | median gap: {np.median(gaps):+.4f}")

    # Failure details
    rust_fails = [r for r in results if r.rust.get("error")]
    mgcv_fails = [r for r in results if r.mgcv.get("error")]
    if rust_fails or mgcv_fails:
        print(f"\n{'=' * 120}")
        print(f"  FAILURES — rust: {len(rust_fails)} | mgcv: {len(mgcv_fails)}")
        print("=" * 120)
        for label, fails in [("RUST", rust_fails), ("MGCV", mgcv_fails)]:
            for r in fails[:10]:
                s = r.scenario
                err = r.rust.get("error", "") if label == "RUST" else r.mgcv.get("error", "")
                stderr = r.rust.get("stderr", "") if label == "RUST" else r.mgcv.get("stderr", "")
                cmd = r.rust.get("cmd", "") if label == "RUST" else ""
                print(f"  [{label}] seed={s['seed']} {s['family']}/{s['model_type']}/{s['basis_type']} "
                      f"n={s['n_obs']} k={s['n_smooths']} kn={s['knots']}")
                print(f"    error: {err[:200]}")
                if stderr:
                    print(f"    stderr: {stderr[:200]}")
                if cmd:
                    print(f"    cmd: {cmd[:200]}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Adversarial fuzzer: Rust GAM vs mgcv")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--family", type=str, default=None, choices=["gaussian", "binomial"])
    parser.add_argument("--model-type", type=str, default=None, choices=["gam", "gamlss"])
    parser.add_argument("--basis", type=str, default=None, choices=["ps", "tps", "duchon"])
    parser.add_argument("--rust-timeout", type=int, default=DEFAULT_RUST_TIMEOUT)
    parser.add_argument("--r-timeout", type=int, default=DEFAULT_R_TIMEOUT)
    parser.add_argument("--max-total-seconds", type=int, default=None)
    args = parser.parse_args()

    if not RUST_BINARY.exists():
        print(f"Rust binary not found: {RUST_BINARY}\nBuild first: cargo build --release")
        sys.exit(1)
    r_check = subprocess.run(["Rscript", "-e", "library(mgcv); cat('ok')"],
                              capture_output=True, text=True, timeout=30)
    if "ok" not in (r_check.stdout or ""):
        print("R + mgcv not available"); sys.exit(1)

    existing_seeds = set()
    results = []
    if not args.resume and RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    if args.resume and RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                fr = FuzzResult(scenario=obj["scenario"], rust=obj["rust"], mgcv=obj["mgcv"])
                fr.compute_gap()
                results.append(fr)
                existing_seeds.add(obj["scenario"]["seed"])
        print(f"Loaded {len(results)} existing results")

    seeds = [args.seed_start + i for i in range(args.n_trials)]
    seeds = [s for s in seeds if s not in existing_seeds]

    if not seeds:
        print_leaderboard(results, top_n=args.top); return

    print(f"Running {len(seeds)} trials")
    print(f"  Smooth functions:  {len(SMOOTH_FN)} 1D + {len(SMOOTH_FN_2D)} 2D")
    print(f"  Noise types:       {len(NOISE_FN)}")
    print(f"  X distributions:   {len(XDIST_FN)}")
    print(f"  Signal structures: {len(SIGNAL_BUILDERS)}")
    print(f"  Sigma functions:   {len(SIGMA_FN)}")
    print(f"  Filters: family={args.family or 'all'} model={args.model_type or 'all'} basis={args.basis or 'all'}")
    print(f"  Rust timeout:      {args.rust_timeout}s")
    print(f"  R timeout:         {args.r_timeout}s")
    print(f"  Time budget:       {args.max_total_seconds}s" if args.max_total_seconds else "  Time budget:       none")
    print(f"  Results: {RESULTS_FILE}\n")

    # Pre-generate all scenarios so we can sort by estimated cost
    # (smallest/cheapest first) while keeping the randomized generation
    scenarios = []
    for seed in seeds:
        sc = generate_scenario(seed, family_filter=args.family, model_type_filter=args.model_type)
        if args.basis:
            sc.basis_type = args.basis
            if args.basis == "duchon" and sc.n_smooths < 2:
                sc.n_smooths = 2
                sc.smooth_kinds = [sc.smooth_kinds[0], sc.smooth_kinds[0]]
        scenarios.append(sc)
    scenarios.sort(key=estimate_scenario_cost)

    started_at = time.time()
    with open(RESULTS_FILE, "a") as out_f:
        for i, sc in enumerate(scenarios):
            if args.max_total_seconds is not None:
                elapsed = time.time() - started_at
                if elapsed >= args.max_total_seconds:
                    print(
                        f"\nReached wall-clock budget after {i} completed trial(s) "
                        f"({elapsed:.1f}s >= {args.max_total_seconds}s). Stopping early.",
                        flush=True,
                    )
                    break
            seed = sc.seed
            result = run_trial(sc, args.rust_timeout, args.r_timeout)

            results.append(result)
            out_f.write(json.dumps({"scenario": result.scenario, "rust": result.rust,
                                     "mgcv": result.mgcv, "primary_gap": result.primary_gap,
                                     "primary_metric": result.primary_metric}, default=str) + "\n")
            out_f.flush()

            m = result.primary_metric or "?"
            gap_s = f"{result.primary_gap:+.4f}" if result.primary_gap is not None else " N/A "
            rv = result.rust.get(m); mv = result.mgcv.get(m)
            rs = f"{rv:.4f}" if rv is not None else " FAIL"
            ms = f"{mv:.4f}" if mv is not None else " FAIL"
            err_r = " [R:ERR]" if result.rust.get("error") else ""
            err_m = " [M:ERR]" if result.mgcv.get("error") else ""
            divergence_level, _ = classify_primary_divergence(result)
            flag = " !!!" if divergence_level == "fail" else (" !!" if divergence_level == "warn" else "")
            t_rust = result.rust.get("time", 0) or 0
            t_mgcv = result.mgcv.get("time", 0) or 0
            time_s = f"  rust={t_rust:.1f}s mgcv={t_mgcv:.1f}s" if max(t_rust, t_mgcv) > 0.5 else ""
            print(
                f"  [{i+1:3d}/{len(scenarios)}] seed={seed:4d} {sc.family[:4]}/{sc.model_type[:5]}/{sc.basis_type[:5]:5s} "
                f"{m}:rust={rs} {m}:mgcv={ms} gap={gap_s} "
                f"n={sc.n_obs:4d} k={sc.n_smooths:2d} kn={sc.knots:2d} "
                f"sig={sc.signal_structure[:8]:8s} noise={sc.noise_kind[:7]:7s}"
                f"{err_r}{err_m}{flag}{time_s}",
                flush=True
            )

            # Print full error details immediately on failure
            for label, out in [("RUST", result.rust), ("MGCV", result.mgcv)]:
                if out.get("error"):
                    print(f"    ┌── {label} FAILURE ──", flush=True)
                    print(f"    │ error: {out['error']}", flush=True)
                    if out.get("stderr"):
                        print(f"    │ stderr: {out['stderr']}", flush=True)
                    if out.get("stdout"):
                        print(f"    │ stdout: {out['stdout']}", flush=True)
                    if out.get("cmd"):
                        print(f"    │ cmd: {out['cmd']}", flush=True)
                    if out.get("traceback"):
                        print(f"    │ traceback: {out['traceback']}", flush=True)
                    print(f"    └──────────────────────", flush=True)

    print_leaderboard(results, top_n=args.top)

    # ─── CI fail policy ────────────────────────────────────────────────
    #
    # Until this block was added, `main()` returned normally after
    # print_leaderboard regardless of per-trial outcomes, so the fuzz CI
    # workflow was green even when every trial hit "[R:ERR]" or reported a
    # large mgcv-vs-rust disagreement. `!!!` / `[R:ERR]` were purely human-
    # readable log decorations with no effect on exit code.
    #
    # Two regression classes are now blocking:
    #
    # 1. **rust-only failures** — any trial where `rust.error` is set while
    #    `mgcv.error` is None. This is the harness's unambiguous "rust
    #    regressed on a scenario mgcv handles" signal. Transient timeouts
    #    are counted; if a timeout is genuinely uninformative it is a
    #    separate bug to fix upstream, not a failure mode to tolerate.
    #
    # 2. **large correctness gap** — any trial where
    #    `classify_primary_divergence(r)[0] == "fail"`. Absolute primary-
    #    metric gaps still block for both families, but gaussian trials get
    #    one extra scale-aware check: when mgcv already has very low RMSE on
    #    the shared test fold, even a modest `r2` gap can hide a several-fold
    #    inflation in rust RMSE. Example: `r2=0.99` vs `r2=0.94` is only a
    #    0.05 absolute gap, yet rust leaves 6x more unexplained variance.
    #    The gaussian guard therefore requires both a nontrivial `r2` gap
    #    and a materially worse RMSE ratio before escalating to `!!` / `!!!`.
    #
    # The per-trial prints/jsonl payload stay the same; only the `!!` / `!!!`
    # marker logic and final exit policy now share the same classifier.
    rust_only_failures = [
        r for r in results
        if r.rust.get("error") and not r.mgcv.get("error")
    ]
    large_gap_regressions = [
        (r, reason) for r in results
        for level, reason in [classify_primary_divergence(r)]
        if level == "fail"
    ]
    if rust_only_failures or large_gap_regressions:
        print(f"\n{'=' * 120}")
        print("  CI FAIL: fuzz harness detected blocking regressions")
        print("=" * 120)
        if rust_only_failures:
            print(
                f"  rust-only failures (rust.error set, mgcv.error unset): "
                f"{len(rust_only_failures)}"
            )
            for r in rust_only_failures[:10]:
                s = r.scenario
                err = str(r.rust.get("error", ""))[:200]
                print(
                    f"    seed={s['seed']} {s['family']}/{s['model_type']}/"
                    f"{s['basis_type']} n={s['n_obs']} k={s['n_smooths']} "
                    f"kn={s['knots']} :: {err}"
                )
        if large_gap_regressions:
            print(
                "  large benchmark regressions "
                "(absolute primary-metric gap, or gaussian RMSE inflation on the same fold): "
                f"{len(large_gap_regressions)}"
            )
            large_gap_regressions.sort(
                key=lambda item: (
                    item[0].primary_gap or 0,
                    gaussian_rmse_ratio(item[0]) or 1.0,
                ),
                reverse=True,
            )
            for r, reason in large_gap_regressions[:10]:
                s = r.scenario
                rv = r.rust.get(r.primary_metric or "r2")
                mv = r.mgcv.get(r.primary_metric or "r2")
                rs = f"{rv:.4f}" if rv is not None else "FAIL"
                ms = f"{mv:.4f}" if mv is not None else "FAIL"
                print(
                    f"    seed={s['seed']} {s['family']}/{s['model_type']}/"
                    f"{s['basis_type']} n={s['n_obs']} k={s['n_smooths']} "
                    f"kn={s['knots']} :: {r.primary_metric}: "
                    f"rust={rs} mgcv={ms} gap={r.primary_gap:+.4f} ({reason})"
                )
        sys.exit(1)


if __name__ == "__main__":
    main()
