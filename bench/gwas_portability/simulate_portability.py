#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_OUT = Path("gwas_portability_runs/manual")
N_PCS = 5
TRAIN_DEME = "deme_0"
HORIZON = 10.0


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@dataclass(frozen=True)
class DemeSpec:
    name: str
    distance: int
    grid_x: int
    grid_y: int


def build_serial1d_demography(n_demes: int, split_spacing: int) -> tuple[msprime.Demography, list[DemeSpec]]:
    if n_demes < 3:
        raise ValueError("serial1d requires at least 3 demes")
    dem = msprime.Demography()
    dem.add_population("deme_0", initial_size=12_000)
    for i in range(1, n_demes):
        dem.add_population(f"deme_{i}", initial_size=10_000)
        dem.add_population_split(
            time=float(split_spacing * i),
            derived=[f"deme_{i}"],
            ancestral=f"deme_{i - 1}",
        )
    dem.sort_events()
    specs = [DemeSpec(f"deme_{i}", i, i, 0) for i in range(n_demes)]
    return dem, specs


def build_grid2d_demography(side: int, split_gens: int, migration_rate: float) -> tuple[msprime.Demography, list[DemeSpec]]:
    if side < 2:
        raise ValueError("grid2d side must be at least 2")
    dem = msprime.Demography()
    specs: list[DemeSpec] = []
    for y in range(side):
        for x in range(side):
            idx = y * side + x
            name = f"deme_{idx}"
            dem.add_population(name, initial_size=10_000)
            specs.append(DemeSpec(name, abs(x) + abs(y), x, y))
    dem.add_population("ancestral", initial_size=12_000)
    dem.add_population_split(time=float(split_gens), derived=[s.name for s in specs], ancestral="ancestral")
    for a in specs:
        for b in specs:
            if abs(a.grid_x - b.grid_x) + abs(a.grid_y - b.grid_y) == 1:
                dem.set_migration_rate(a.name, b.name, float(migration_rate))
    dem.sort_events()
    return dem, specs


def diploid_pairs(ts: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sample_nodes = ts.samples()
    node_to_col = {int(n): i for i, n in enumerate(sample_nodes)}
    a: list[int] = []
    b: list[int] = []
    pop_names: list[str] = []
    for ind_id in range(ts.num_individuals):
        ind = ts.individual(ind_id)
        nodes = list(ind.nodes)
        if len(nodes) != 2:
            continue
        n0, n1 = int(nodes[0]), int(nodes[1])
        if n0 not in node_to_col or n1 not in node_to_col:
            continue
        a.append(node_to_col[n0])
        b.append(node_to_col[n1])
        pop_id = int(ts.node(n0).population)
        pop_names.append(str(ts.population(pop_id).metadata.get("name", f"pop_{pop_id}")))
    if not a:
        raise RuntimeError("msprime tree sequence did not contain diploid individuals")
    return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), pop_names


def sample_sites_by_maf(
    ts: Any,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    *,
    pca_sites: int,
    causal_sites: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng_pca = np.random.default_rng(seed + 11)
    rng_causal = np.random.default_rng(seed + 17)
    pca: list[int] = []
    causal: list[int] = []
    pca_seen = 0
    causal_seen = 0
    denom = 2.0 * len(a_idx)
    for var in ts.variants():
        if len(var.alleles) != 2:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        dosage = g[a_idx] + g[b_idx]
        af = float(dosage.sum()) / denom
        maf = min(af, 1.0 - af)
        sid = int(var.site.id)
        if maf >= 0.05:
            pca_seen += 1
            if len(pca) < pca_sites:
                pca.append(sid)
            else:
                j = int(rng_pca.integers(0, pca_seen))
                if j < pca_sites:
                    pca[j] = sid
        if maf >= 0.01:
            causal_seen += 1
            if len(causal) < causal_sites:
                causal.append(sid)
            else:
                j = int(rng_causal.integers(0, causal_seen))
                if j < causal_sites:
                    causal[j] = sid
    if len(pca) < min(50, pca_sites) or len(causal) < min(50, causal_sites):
        raise RuntimeError(f"Not enough usable sites: pca={len(pca)} causal={len(causal)}")
    return pca, causal


def dosage_matrix_for_sites(ts: Any, a_idx: np.ndarray, b_idx: np.ndarray, site_ids: set[int]) -> np.ndarray:
    cols: list[np.ndarray] = []
    for var in ts.variants():
        if int(var.site.id) not in site_ids or len(var.alleles) != 2:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        cols.append((g[a_idx] + g[b_idx]).astype(np.float64))
    if not cols:
        raise RuntimeError("No dosage columns extracted")
    return np.column_stack(cols)


def compute_pcs_and_liability(ts: Any, a_idx: np.ndarray, b_idx: np.ndarray, pca_ids: list[int], causal_ids: list[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    pca_g = dosage_matrix_for_sites(ts, a_idx, b_idx, set(pca_ids))
    pca_g = StandardScaler().fit_transform(pca_g)
    pcs = PCA(n_components=N_PCS, random_state=seed).fit_transform(pca_g)

    causal_g = dosage_matrix_for_sites(ts, a_idx, b_idx, set(causal_ids))
    causal_g = StandardScaler().fit_transform(causal_g)
    rng = np.random.default_rng(seed + 101)
    effects = rng.normal(0.0, 1.0, causal_g.shape[1])
    effects *= rng.choice([0.0, 1.0], size=effects.size, p=[0.65, 0.35])
    if not np.any(effects):
        effects[0] = 1.0
    g_true = causal_g @ effects
    g_true = (g_true - float(np.mean(g_true))) / float(np.std(g_true, ddof=0))
    return pcs, g_true


def split_groups(df: pd.DataFrame, n_train: int, n_train_test: int, n_other: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 303)
    out = df.copy()
    out["split"] = "unused"
    train_idx = out.index[out["deme"] == TRAIN_DEME].to_numpy()
    rng.shuffle(train_idx)
    if train_idx.size < n_train + n_train_test:
        raise RuntimeError(f"{TRAIN_DEME} has {train_idx.size} rows; need {n_train + n_train_test}")
    out.loc[train_idx[:n_train], "split"] = "train"
    out.loc[train_idx[n_train:n_train + n_train_test], "split"] = "train_heldout"
    for deme in sorted(set(out["deme"]) - {TRAIN_DEME}):
        idx = out.index[out["deme"] == deme].to_numpy()
        rng.shuffle(idx)
        if idx.size < n_other:
            raise RuntimeError(f"{deme} has {idx.size} rows; need {n_other}")
        out.loc[idx[:n_other], "split"] = "other_test"
    return out[out["split"] != "unused"].reset_index(drop=True)


def add_phenotypes(df: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + (701 if mode == "deme_varying" else 601))
    out = df.copy()
    demes = sorted(out["deme"].unique())
    if mode == "constant":
        offsets = {d: 0.0 for d in demes}
    elif mode == "deme_varying":
        offsets = {d: 0.18 * float(out.loc[out["deme"] == d, "distance"].iloc[0]) for d in demes}
    else:
        raise ValueError(f"unknown phenotype mode: {mode}")
    baseline = out["deme"].map(offsets).to_numpy(dtype=float)
    liability = 0.85 * out["G_true"].to_numpy(dtype=float) + baseline
    intercept = -1.55 - float(np.mean(liability))
    p = expit(intercept + liability)
    out["binary_y"] = rng.binomial(1, p).astype(np.int8)
    out["binary_prob_true"] = p

    log_h = math.log(0.035) + 0.55 * out["G_true"].to_numpy(dtype=float) + baseline
    hazard = np.exp(log_h)
    event_time = rng.exponential(1.0 / hazard)
    censor_time = rng.uniform(4.0, 14.0, size=len(out))
    exit_time = np.minimum(event_time, censor_time)
    out["entry"] = 0.0
    out["exit"] = np.maximum(exit_time, 1.0e-3)
    out["event"] = (event_time <= censor_time).astype(np.int8)
    out["risk_horizon_true"] = 1.0 - np.exp(-hazard * HORIZON)
    out["phenotype_mode"] = mode
    return out


def run_cmd(cmd: list[str], desc: str) -> None:
    log(f"{desc}: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{desc} failed rc={res.returncode}\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}")


def plink_bin() -> str:
    p = shutil.which("plink2")
    if p is None:
        raise RuntimeError("plink2 is required on PATH")
    return p


def write_vcf_and_bed(ts: Any, out_prefix: Path, iids: list[str], threads: int, memory_mb: int) -> Path:
    vcf = out_prefix.with_suffix(".vcf")
    with vcf.open("w", encoding="utf-8") as fh:
        ts.write_vcf(fh, individual_names=iids, position_transform=lambda x: np.asarray(x) + 1)
    cmd = [
        plink_bin(),
        "--vcf",
        str(vcf),
        "--make-bed",
        "--allow-extra-chr",
        "--threads",
        str(threads),
        "--memory",
        str(memory_mb),
        "--out",
        str(out_prefix),
    ]
    run_cmd(cmd, "PLINK2 VCF to BED")
    return out_prefix


def fam_iid_to_fid(prefix: Path) -> dict[str, str]:
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, dtype=str)
    fam.columns = ["FID", "IID", "PAT", "MAT", "SEX", "PHENO"]
    return dict(zip(fam["IID"], fam["FID"]))


def write_keep(df: pd.DataFrame, iid_to_fid: dict[str, str], path: Path) -> None:
    pd.DataFrame({"FID": [iid_to_fid[i] for i in df["IID"]], "IID": df["IID"]}).to_csv(path, sep="\t", index=False, header=False)


def make_split_bed(all_prefix: Path, df: pd.DataFrame, split: str, iid_to_fid: dict[str, str], work: Path, threads: int, memory_mb: int) -> Path:
    keep = work / f"{split}.keep"
    prefix = work / split
    write_keep(df[df["split"] == split], iid_to_fid, keep)
    run_cmd([
        plink_bin(), "--bfile", str(all_prefix), "--keep", str(keep), "--make-bed",
        "--allow-extra-chr", "--threads", str(threads), "--memory", str(memory_mb), "--out", str(prefix)
    ], f"PLINK2 make {split} BED")
    return prefix


def write_pheno_and_covar(train_df: pd.DataFrame, iid_to_fid: dict[str, str], work: Path) -> tuple[Path, Path]:
    pheno = work / "train_binary.phen"
    pd.DataFrame({"FID": [iid_to_fid[i] for i in train_df["IID"]], "IID": train_df["IID"], "y": train_df["binary_y"].astype(int)}).to_csv(pheno, sep=" ", index=False, header=False)
    covar = work / "train.covar"
    cov = {"FID": [iid_to_fid[i] for i in train_df["IID"]], "IID": train_df["IID"]}
    for k in range(1, N_PCS + 1):
        cov[f"pc{k}"] = train_df[f"pc{k}"].to_numpy(dtype=float)
    pd.DataFrame(cov).to_csv(covar, sep=" ", index=False, header=False)
    return pheno, covar


def read_score(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    iid_col = "IID" if "IID" in df.columns else "#IID"
    out = df[[iid_col, "SCORE1_AVG"]].copy()
    out.columns = ["IID", "PGS"]
    out["IID"] = out["IID"].astype(str)
    out["PGS"] = pd.to_numeric(out["PGS"], errors="raise").astype(float)
    return out


def run_pt(all_prefix: Path, train_prefix: Path, train_df: pd.DataFrame, iid_to_fid: dict[str, str], work: Path, threads: int, memory_mb: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pheno, covar = write_pheno_and_covar(train_df, iid_to_fid, work)
    common = ["--allow-extra-chr", "--threads", str(threads), "--memory", str(memory_mb)]
    prune = work / "pt.prune"
    run_cmd([plink_bin(), "--bfile", str(train_prefix), "--indep-pairwise", "250", "50", "0.2", *common, "--out", str(prune)], "P+T LD pruning")
    gwas = work / "pt.gwas"
    run_cmd([
        plink_bin(), "--bfile", str(train_prefix), "--pheno", str(pheno), "--pheno-col-nums", "3",
        "--1", "--covar", str(covar), "--glm", "hide-covar", "--allow-no-sex", *common, "--out", str(gwas)
    ], "P+T training-deme GWAS")
    glm_files = sorted(work.glob("pt.gwas*.glm.*"))
    glm_path = next((p for p in glm_files if ".glm.logistic" in p.name), None)
    if glm_path is None:
        raise RuntimeError(f"No PLINK logistic GWAS output found in {work}; files={[p.name for p in glm_files]}")
    g = pd.read_csv(glm_path, sep=r"\s+")
    if "TEST" in g.columns:
        add = g[g["TEST"].astype(str) == "ADD"].copy()
        if len(add):
            g = add
    eff_col = "BETA" if "BETA" in g.columns else ("LOG(OR)" if "LOG(OR)" in g.columns else None)
    if eff_col is None:
        raise RuntimeError(f"GWAS output lacks BETA/LOG(OR): {list(g.columns)}")
    weights = pd.DataFrame({
        "ID": g["ID"].astype(str),
        "A1": g["A1"].astype(str),
        "EFF": pd.to_numeric(g[eff_col], errors="coerce"),
        "P": pd.to_numeric(g["P"], errors="coerce"),
    }).dropna()
    prune_in = prune.with_suffix(".prune.in")
    keep = set(prune_in.read_text(encoding="utf-8").splitlines())
    weights = weights[weights["ID"].isin(keep)].sort_values("P")
    if weights.empty:
        raise RuntimeError("P+T had no LD-pruned GWAS rows")
    thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    score_all = work / "pt.all.score"
    qfile = work / "pt.qvals.tsv"
    ranges = work / "pt.ranges.tsv"
    weights[["ID", "A1", "EFF"]].to_csv(score_all, sep="\t", index=False)
    weights[["ID", "P"]].to_csv(qfile, sep="\t", index=False)
    with ranges.open("w", encoding="utf-8") as fh:
        fh.write("RANGE\tLOW\tHIGH\n")
        for t in thresholds:
            fh.write(f"T{str(t).replace('.', 'p')}\t0\t{t}\n")
    qout = work / "pt.qtrain"
    run_cmd([
        plink_bin(), "--bfile", str(train_prefix), "--score", str(score_all), "1", "2", "3", "header",
        "--q-score-range", str(ranges), str(qfile), "1", "2", "header", "--allow-no-sex", *common, "--out", str(qout)
    ], "P+T score training thresholds")
    rows = []
    y = train_df.set_index("IID")["binary_y"].astype(int)
    for t in thresholds:
        label = f"T{str(t).replace('.', 'p')}"
        s = read_score(Path(f"{qout}.{label}.sscore"))
        merged = s.set_index("IID").join(y, how="inner")
        x = merged["PGS"].to_numpy(dtype=float).reshape(-1, 1)
        yy = merged["binary_y"].to_numpy(dtype=int)
        if np.unique(yy).size < 2 or float(np.std(x)) <= 1e-12:
            auc = -np.inf
            ll = np.inf
        else:
            lr = LogisticRegression(max_iter=1000).fit(x, yy)
            p = lr.predict_proba(x)[:, 1]
            auc = float(roc_auc_score(yy, p))
            ll = float(log_loss(yy, p, labels=[0, 1]))
        rows.append({"p_threshold": t, "n_snps": int(np.count_nonzero(weights["P"].to_numpy() <= t)), "train_auc": auc, "train_log_loss": ll})
    metrics = pd.DataFrame(rows)
    best_row = metrics.sort_values(["train_auc", "n_snps"], ascending=[False, False]).iloc[0]
    best_t = float(best_row["p_threshold"])
    best = weights[weights["P"] <= best_t]
    if best.empty:
        raise RuntimeError("P+T selected an empty threshold")
    best_score = work / "pt.best.score"
    best[["ID", "A1", "EFF"]].to_csv(best_score, sep="\t", index=False, header=False)
    sout = work / "pt.all_samples"
    run_cmd([plink_bin(), "--bfile", str(all_prefix), "--score", str(best_score), "1", "2", "3", "--allow-no-sex", *common, "--out", str(sout)], "P+T score all samples")
    metrics["selected"] = metrics["p_threshold"] == best_t
    return read_score(Path(f"{sout}.sscore")), metrics


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    ok = np.isfinite(t) & np.isfinite(r)
    t, e, r = t[ok], e[ok], r[ok]
    conc = ties = comp = 0
    n = len(t)
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j]:
                continue
            if t[i] < t[j] and e[i] == 1:
                comp += 1
                if r[i] > r[j]:
                    conc += 1
                elif r[i] == r[j]:
                    ties += 1
            elif t[j] < t[i] and e[j] == 1:
                comp += 1
                if r[j] > r[i]:
                    conc += 1
                elif r[i] == r[j]:
                    ties += 1
    return float((conc + 0.5 * ties) / comp) if comp else float("nan")


class CoxPH:
    def __init__(self) -> None:
        self.beta: np.ndarray | None = None
        self.event_times: np.ndarray | None = None
        self.base_cumhaz: np.ndarray | None = None

    def fit(self, x: np.ndarray, time: np.ndarray, event: np.ndarray) -> "CoxPH":
        x = np.asarray(x, dtype=float)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=int)
        order = np.argsort(time)
        x, time, event = x[order], time[order], event[order]

        def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
            eta = x @ beta
            exp_eta = np.exp(np.clip(eta, -40, 40))
            suffix_exp = np.cumsum(exp_eta[::-1])[::-1]
            suffix_xexp = np.cumsum((x * exp_eta[:, None])[::-1], axis=0)[::-1]
            idx = np.where(event == 1)[0]
            pll = float(np.sum(eta[idx] - np.log(suffix_exp[idx])))
            grad = np.sum(x[idx] - suffix_xexp[idx] / suffix_exp[idx, None], axis=0)
            pen = 1e-4 * float(beta @ beta)
            return -pll + pen, -grad + 2e-4 * beta

        res = minimize(lambda b: objective(b)[0], np.zeros(x.shape[1]), jac=lambda b: objective(b)[1], method="BFGS")
        if not res.success:
            raise RuntimeError(f"CoxPH fit failed: {res.message}")
        self.beta = np.asarray(res.x, dtype=float)
        eta = x @ self.beta
        exp_eta = np.exp(np.clip(eta, -40, 40))
        event_times = np.sort(np.unique(time[event == 1]))
        ch = []
        total = 0.0
        for tt in event_times:
            d = float(np.sum((time == tt) & (event == 1)))
            risk_sum = float(np.sum(exp_eta[time >= tt]))
            total += d / max(risk_sum, 1e-12)
            ch.append(total)
        self.event_times = event_times
        self.base_cumhaz = np.asarray(ch, dtype=float)
        return self

    def predict_risk(self, x: np.ndarray, horizon: float) -> np.ndarray:
        if self.beta is None or self.event_times is None or self.base_cumhaz is None:
            raise RuntimeError("CoxPH is not fitted")
        idx = np.searchsorted(self.event_times, float(horizon), side="right") - 1
        h0 = 0.0 if idx < 0 else float(self.base_cumhaz[idx])
        eta = np.asarray(x, dtype=float) @ self.beta
        return 1.0 - np.exp(-h0 * np.exp(np.clip(eta, -40, 40)))


def gamfit_issue(out_dir: Path, label: str, exc: BaseException) -> Path:
    p = out_dir / "gamfit_issues.md"
    with p.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {label}\n\n")
        fh.write(f"- exception: `{type(exc).__name__}`\n")
        fh.write(f"- message: `{str(exc)}`\n\n")
        fh.write("```text\n")
        fh.write("".join(traceback.format_exception(exc)))
        fh.write("\n```\n")
    return p


def fit_predict_methods(df: pd.DataFrame, out_dir: Path, gam_centers: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"].copy()
    test = df[df["split"].isin(["train_heldout", "other_test"])].copy()
    mu = float(train["PGS"].mean())
    sd = float(train["PGS"].std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise RuntimeError("Training PGS is degenerate")
    for frame in (train, test):
        frame["pgs_z"] = (frame["PGS"].to_numpy(dtype=float) - mu) / sd
    pc_cols = [f"pc{k}" for k in range(1, N_PCS + 1)]
    method_rows: list[pd.DataFrame] = []

    y = train["binary_y"].to_numpy(dtype=int)
    for method, cols in [("z_norm", ["pgs_z"]), ("linear_pc_main", ["pgs_z", *pc_cols])]:
        lr = LogisticRegression(max_iter=1000).fit(train[cols].to_numpy(dtype=float), y)
        p = lr.predict_proba(test[cols].to_numpy(dtype=float))[:, 1]
        part = test[["IID", "deme", "distance", "split", "phenotype_mode", "binary_y", "entry", "exit", "event"]].copy()
        part["outcome"] = "binary"
        part["method"] = method
        part["pred_risk"] = p
        method_rows.append(part)

    for method, cols in [("z_norm", ["pgs_z"]), ("linear_pc_main", ["pgs_z", *pc_cols])]:
        cox = CoxPH().fit(train[cols].to_numpy(dtype=float), train["exit"].to_numpy(dtype=float), train["event"].to_numpy(dtype=int))
        risk = cox.predict_risk(test[cols].to_numpy(dtype=float), HORIZON)
        part = test[["IID", "deme", "distance", "split", "phenotype_mode", "binary_y", "entry", "exit", "event"]].copy()
        part["outcome"] = "survival"
        part["method"] = method
        part["pred_risk"] = risk
        method_rows.append(part)

    try:
        import gamfit

        smooth = f"duchon(pc1, pc2, pc3, centers={int(gam_centers)}, order=1)"
        gm = gamfit.fit(
            train,
            f"binary_y ~ {smooth}",
            family="bernoulli-marginal-slope",
            z_column="pgs_z",
            logslope_formula=smooth,
        )
        pred = gm.predict(test, return_type="dict")
        part = test[["IID", "deme", "distance", "split", "phenotype_mode", "binary_y", "entry", "exit", "event"]].copy()
        part["outcome"] = "binary"
        part["method"] = "gamfit_marginal_slope"
        part["pred_risk"] = np.asarray(pred["mean"], dtype=float)
        method_rows.append(part)
    except BaseException as exc:
        issue = gamfit_issue(out_dir, f"bernoulli marginal-slope phenotype_mode={train['phenotype_mode'].iloc[0]}", exc)
        log(f"Recorded gamfit issue at {issue}")

    try:
        import gamfit

        smooth = f"duchon(pc1, pc2, pc3, centers={int(gam_centers)}, order=1)"
        sm = gamfit.fit(
            train,
            f"Surv(entry, exit, event) ~ {smooth}",
            survival_likelihood="marginal-slope",
            z_column="pgs_z",
            logslope_formula=smooth,
        )
        sp = sm.predict(test)
        risk = 1.0 - np.asarray(sp.survival_at([HORIZON]), dtype=float).reshape(len(test), -1)[:, 0]
        part = test[["IID", "deme", "distance", "split", "phenotype_mode", "binary_y", "entry", "exit", "event"]].copy()
        part["outcome"] = "survival"
        part["method"] = "gamfit_marginal_slope"
        part["pred_risk"] = risk
        method_rows.append(part)
    except BaseException as exc:
        issue = gamfit_issue(out_dir, f"survival marginal-slope phenotype_mode={train['phenotype_mode'].iloc[0]}", exc)
        log(f"Recorded gamfit issue at {issue}")

    pred_df = pd.concat(method_rows, ignore_index=True)
    metrics = pooled_metrics(pred_df)
    cal = calibration(pred_df)
    return metrics, cal, pred_df


def pooled_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "train_heldout": pred["split"] == "train_heldout",
        "other_pooled": pred["split"] == "other_test",
        "all_pooled": pred["split"].isin(["train_heldout", "other_test"]),
    }
    for (mode, outcome, method), g in pred.groupby(["phenotype_mode", "outcome", "method"]):
        for scope, mask_all in scopes.items():
            sub = g[mask_all.loc[g.index]].copy()
            if sub.empty:
                continue
            if outcome == "binary":
                y = sub["binary_y"].to_numpy(dtype=int)
                p = np.clip(sub["pred_risk"].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
                rows.append({
                    "phenotype_mode": mode, "outcome": outcome, "method": method, "scope": scope,
                    "n": int(len(sub)),
                    "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
                    "brier": float(brier_score_loss(y, p)),
                    "log_loss": float(log_loss(y, p, labels=[0, 1])),
                    "c_index": np.nan,
                })
            else:
                risk = np.clip(sub["pred_risk"].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
                rows.append({
                    "phenotype_mode": mode, "outcome": outcome, "method": method, "scope": scope,
                    "n": int(len(sub)),
                    "auc": np.nan,
                    "brier": np.nan,
                    "log_loss": np.nan,
                    "c_index": harrell_c_index(sub["exit"].to_numpy(dtype=float), sub["event"].to_numpy(dtype=int), risk),
                })
    return pd.DataFrame(rows)


def calibration(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in pred.groupby(["phenotype_mode", "outcome", "method", "deme", "distance", "split"]):
        mode, outcome, method, deme, distance, split = keys
        p = np.clip(sub["pred_risk"].to_numpy(dtype=float), 0.0, 1.0)
        if outcome == "binary":
            obs = sub["binary_y"].to_numpy(dtype=float)
            rows.append({
                "phenotype_mode": mode, "outcome": outcome, "method": method, "deme": deme,
                "distance": int(distance), "split": split, "n": int(len(sub)),
                "mean_predicted_risk": float(np.mean(p)),
                "observed_risk": float(np.mean(obs)),
                "calibration_error": float(np.mean(p) - np.mean(obs)),
            })
        else:
            usable = ~((sub["event"].to_numpy(dtype=int) == 0) & (sub["exit"].to_numpy(dtype=float) < HORIZON))
            if not np.any(usable):
                continue
            obs = ((sub["event"].to_numpy(dtype=int) == 1) & (sub["exit"].to_numpy(dtype=float) <= HORIZON))[usable].astype(float)
            rows.append({
                "phenotype_mode": mode, "outcome": outcome, "method": method, "deme": deme,
                "distance": int(distance), "split": split, "n": int(np.sum(usable)),
                "mean_predicted_risk": float(np.mean(p[usable])),
                "observed_risk": float(np.mean(obs)),
                "calibration_error": float(np.mean(p[usable]) - np.mean(obs)),
            })
    return pd.DataFrame(rows)


def plot_outputs(metrics: pd.DataFrame, cal: pd.DataFrame, score_diag: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({"figure.dpi": 160, "axes.spines.top": False, "axes.spines.right": False})
    if not metrics.empty:
        for outcome in sorted(metrics["outcome"].unique()):
            sub = metrics[(metrics["outcome"] == outcome) & (metrics["scope"].isin(["train_heldout", "other_pooled"]))]
            if sub.empty:
                continue
            metric_col = "auc" if outcome == "binary" else "c_index"
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = [f"{m}\n{s}\n{p}" for m, s, p in zip(sub["method"], sub["scope"], sub["phenotype_mode"])]
            ax.bar(np.arange(len(sub)), sub[metric_col].to_numpy(dtype=float), color="#4C78A8")
            ax.set_xticks(np.arange(len(sub)), labels, rotation=45, ha="right")
            ax.set_ylabel(metric_col)
            ax.set_title(f"Pooled {outcome} accuracy")
            fig.tight_layout()
            fig.savefig(out_dir / f"pooled_{outcome}_accuracy.png")
            plt.close(fig)
    if not cal.empty:
        for outcome in sorted(cal["outcome"].unique()):
            fig, ax = plt.subplots(figsize=(9, 5))
            sub = cal[(cal["outcome"] == outcome) & (cal["split"] == "other_test")]
            for (mode, method), g in sub.groupby(["phenotype_mode", "method"]):
                by = g.groupby("distance", as_index=False).agg(calibration_error=("calibration_error", "mean"))
                ax.plot(by["distance"], by["calibration_error"], marker="o", label=f"{mode}/{method}")
            ax.axhline(0.0, color="#333333", linewidth=0.8)
            ax.set_xlabel("Genetic distance from training deme")
            ax.set_ylabel("Mean predicted risk - observed risk")
            ax.set_title(f"{outcome} calibration by distance")
            ax.legend(frameon=False, fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"{outcome}_calibration_by_distance.png")
            plt.close(fig)
    if not score_diag.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for demog, g in score_diag.groupby("demography"):
            ax.plot(g["distance"], g["pgs_gtrue_corr"], marker="o", label=demog)
        ax.set_xlabel("Genetic distance from training deme")
        ax.set_ylabel("corr(P+T PGS, true genetic liability)")
        ax.set_title("PGS portability diagnostic by distance")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "pgs_true_liability_correlation_by_distance.png")
        plt.close(fig)


def run_demography(args: argparse.Namespace, demography_name: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if demography_name == "serial1d":
        dem, specs = build_serial1d_demography(args.serial_demes, args.serial_split_spacing)
    elif demography_name == "grid2d":
        dem, specs = build_grid2d_demography(args.grid_side, args.grid_split_gens, args.grid_migration_rate)
    else:
        raise ValueError(demography_name)

    samples = {s.name: (args.n_train + args.n_train_test if s.name == TRAIN_DEME else args.n_other) for s in specs}
    log(f"{demography_name}: sim_ancestry samples={samples}")
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=dem,
        sequence_length=float(args.sequence_length),
        recombination_rate=float(args.recombination_rate),
        ploidy=2,
        random_seed=int(args.seed + (1000 if demography_name == "grid2d" else 0)),
    )
    ts = msprime.sim_mutations(ts, rate=float(args.mutation_rate), random_seed=int(args.seed + 1 + (1000 if demography_name == "grid2d" else 0)), discrete_genome=True)
    log(f"{demography_name}: tree sequence sites={ts.num_sites} individuals={ts.num_individuals}")
    a_idx, b_idx, pop_names = diploid_pairs(ts)
    pca_ids, causal_ids = sample_sites_by_maf(ts, a_idx, b_idx, pca_sites=args.pca_sites, causal_sites=args.causal_sites, seed=args.seed)
    pcs, g_true = compute_pcs_and_liability(ts, a_idx, b_idx, pca_ids, causal_ids, args.seed)
    spec_map = {s.name: s for s in specs}
    base = pd.DataFrame({
        "IID": [f"ind{i + 1}" for i in range(len(pop_names))],
        "deme": pop_names,
        "distance": [spec_map[p].distance for p in pop_names],
        "grid_x": [spec_map[p].grid_x for p in pop_names],
        "grid_y": [spec_map[p].grid_y for p in pop_names],
        "G_true": g_true,
    })
    for k in range(N_PCS):
        base[f"pc{k + 1}"] = pcs[:, k]
    base = split_groups(base, args.n_train, args.n_train_test, args.n_other, args.seed)
    demog_dir = out_dir / demography_name
    work = demog_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    all_prefix = write_vcf_and_bed(ts, work / "all_samples", [f"ind{i + 1}" for i in range(len(pop_names))], args.threads, args.plink_memory_mb)
    iid_to_fid = fam_iid_to_fid(all_prefix)
    train_prefix = make_split_bed(all_prefix, base, "train", iid_to_fid, work, args.threads, args.plink_memory_mb)
    pgs, pt_metrics = run_pt(all_prefix, train_prefix, base[base["split"] == "train"].copy(), iid_to_fid, work, args.threads, args.plink_memory_mb)
    base = base.merge(pgs, on="IID", how="left", validate="one_to_one")
    if base["PGS"].isna().any():
        raise RuntimeError("Missing P+T scores after merge")
    pt_metrics.insert(0, "demography", demography_name)
    pt_metrics.to_csv(demog_dir / "pt_thresholds.tsv", sep="\t", index=False)

    score_rows = []
    for (deme, distance), sub in base.groupby(["deme", "distance"]):
        score_rows.append(
            {
                "deme": deme,
                "distance": int(distance),
                "pgs_gtrue_corr": float(np.corrcoef(sub["PGS"], sub["G_true"])[0, 1]),
                "n": int(len(sub)),
            }
        )
    score_diag = pd.DataFrame(score_rows)
    score_diag.insert(0, "demography", demography_name)

    metrics_parts = []
    cal_parts = []
    pred_parts = []
    for mode in ["constant", "deme_varying"]:
        phen = add_phenotypes(base, mode, args.seed)
        phen.insert(0, "demography", demography_name)
        phen.to_csv(demog_dir / f"simulated_data_{mode}.tsv", sep="\t", index=False)
        m, c, p = fit_predict_methods(phen, demog_dir, args.gam_centers)
        m.insert(0, "demography", demography_name)
        c.insert(0, "demography", demography_name)
        p.insert(0, "demography", demography_name)
        metrics_parts.append(m)
        cal_parts.append(c)
        pred_parts.append(p)
    pd.concat(pred_parts, ignore_index=True).to_csv(demog_dir / "predictions.tsv", sep="\t", index=False)
    return pd.concat(metrics_parts, ignore_index=True), pd.concat(cal_parts, ignore_index=True), score_diag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--demographies", nargs="+", choices=["serial1d", "grid2d"], default=["serial1d", "grid2d"])
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-train-test", type=int, default=3000)
    parser.add_argument("--n-other", type=int, default=2000)
    parser.add_argument("--serial-demes", type=int, default=6)
    parser.add_argument("--serial-split-spacing", type=int, default=650)
    parser.add_argument("--grid-side", type=int, default=3)
    parser.add_argument("--grid-split-gens", type=int, default=2200)
    parser.add_argument("--grid-migration-rate", type=float, default=2e-5)
    parser.add_argument("--sequence-length", type=int, default=15_000_000)
    parser.add_argument("--recombination-rate", type=float, default=1.1e-8)
    parser.add_argument("--mutation-rate", type=float, default=1.25e-8)
    parser.add_argument("--pca-sites", type=int, default=2500)
    parser.add_argument("--causal-sites", type=int, default=800)
    parser.add_argument("--gam-centers", type=int, default=40)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--plink-memory-mb", type=int, default=64000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    all_metrics = []
    all_cal = []
    all_score = []
    for demog in args.demographies:
        metrics, cal, score = run_demography(args, demog, args.out)
        all_metrics.append(metrics)
        all_cal.append(cal)
        all_score.append(score)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    cal_df = pd.concat(all_cal, ignore_index=True)
    score_df = pd.concat(all_score, ignore_index=True)
    metrics_df.to_csv(args.out / "metrics_global_pooled.tsv", sep="\t", index=False)
    cal_df.to_csv(args.out / "calibration_by_deme_distance.tsv", sep="\t", index=False)
    score_df.to_csv(args.out / "pgs_portability_by_distance.tsv", sep="\t", index=False)
    plot_outputs(metrics_df, cal_df, score_df, args.out)
    log(f"Wrote global pooled metrics: {args.out / 'metrics_global_pooled.tsv'}")
    log(f"Wrote stratified calibration: {args.out / 'calibration_by_deme_distance.tsv'}")
    log(f"Wrote plots under: {args.out}")


if __name__ == "__main__":
    main()
