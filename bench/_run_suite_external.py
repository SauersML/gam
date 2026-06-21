from __future__ import annotations

import json
import math
import re
import tempfile
import typing
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def configure(context: dict[str, typing.Any]) -> None:
    globals().update(context)


def _is_gamlss_benchmark_scenario(scenario_name: str) -> bool:
    cfg = _scenario_fit_mapping(scenario_name)
    if cfg is None:
        return False
    return not _requires_joint_spatial_term(cfg)


def _gamlss_mu_formula_for_scenario(scenario_name: str, ds: typing.Any) -> typing.Any:
    cfg = _scenario_fit_mapping(scenario_name)
    if cfg is None:
        return None
    if ds["family"] not in ("binomial", "gaussian"):
        return None
    if _requires_joint_spatial_term(cfg):
        return None

    terms = [str(c) for c in cfg.get("linear_cols", [])]
    knot_count = max(4, int(cfg.get("knots", 8)))
    knot_expr = f"max(1, min({knot_count}, nrow(train_df)-1))"
    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        for col in smooth_cols:
            # Use the scenario knot count as an actual spline-basis size control
            # rather than misusing pb(..., df=...) as if it were equivalent to
            # mgcv/Rust basis rank.
            terms.append(f"pb({col}, inter={knot_expr})")
    elif cfg.get("smooth_col"):
        col = cfg["smooth_col"]
        terms.append(f"pb({col}, inter={knot_expr})")

    if not terms:
        return None
    return f"{ds['target']} ~ " + " + ".join(terms)


def _scenario_knot_count(scenario_name: str | None, default: int = 8) -> int:
    if not scenario_name:
        return int(default)
    cfg = _scenario_fit_mapping(scenario_name)
    if cfg is None:
        return int(default)
    return max(4, int(cfg.get("knots", default)))


def _feature_should_be_smooth(ds: dict[str, typing.Any], col: str) -> bool:
    vals = pd.to_numeric(pd.Series([row.get(col) for row in ds["rows"]]), errors="coerce").dropna()
    if vals.empty:
        return False
    unique_vals = np.unique(vals.to_numpy(dtype=float))
    return unique_vals.size > 2


def _sigma_feature_terms(ds: dict[str, typing.Any], *, scenario_name: str | None, backend: str) -> list[str]:
    """Build the per-feature term list for the GAMLSS sigma block.

    Contract (joint-PC, applied uniformly across backends):

    * If the dataset's features include 2+ PC columns, they are emitted as
      ONE joint multi-D smooth via `_emit_joint_pc_term`. Backends without a
      clean multi-D smoother (gamlss, gamboostlss) raise — the harness must
      filter those lanes out rather than fall back to per-axis 1D fits.

    * Non-PC features keep their existing per-feature smooth/linear treatment.

    * Single-PC scenarios (rare; only `pc1`) fall through to the per-feature
      branch since there's nothing to "join".

    The DoF budget computation for mgcv/brms/bamlss counts the joint PC term
    as ONE smooth regardless of dimension; this matches the actual GAM model
    structure rather than the surface formula syntax.
    """
    knot_count = _scenario_knot_count(scenario_name)
    knot_expr = f"max(1, min({knot_count}, nrow(train_df)-1))"
    features = [str(c) for c in ds.get("features", [])]
    pc_features, other_features = _split_pc_columns(features)
    smoothed_pc = [c for c in pc_features if _feature_should_be_smooth(ds, c)]
    smoothed_other = [c for c in other_features if _feature_should_be_smooth(ds, c)]

    cfg = _effective_scenario_fit_mapping(scenario_name) if scenario_name else None
    pc_basis = _joint_pc_basis((cfg or {}).get("smooth_basis", "ps"))
    use_joint_pc = len(smoothed_pc) >= 2

    # Total smooth-block count for GAULSS-style total-DoF caps. The joint PC
    # term counts as 1 regardless of dimension because it shares one λ in
    # the mu+sigma optimization; per-axis on `other_features` each count as 1.
    n_sigma_smooth = (1 if use_joint_pc else len(smoothed_pc)) + len(smoothed_other)
    gaulss_total = 2 * max(1, n_sigma_smooth)
    mgcv_k = (
        f"min({knot_count + 4}, "
        f"max(4L, as.integer(floor(nrow(train_df) * 0.8 / {gaulss_total}))))"
    )
    brms_k = (
        f"min({knot_count}, "
        f"max(4L, as.integer(floor(nrow(train_df) * 0.8 / {gaulss_total}))))"
    )

    terms: list[str] = []

    if use_joint_pc:
        terms.append(
            _emit_joint_pc_term(
                backend,
                smoothed_pc,
                knot_count=knot_count,
                pc_basis=pc_basis,
                double_penalty=True,
            )
        )

    # Per-feature smooth/linear treatment for non-PC features (and for the
    # rare 1-PC case where there's nothing to join).
    leftover = (smoothed_pc if not use_joint_pc else []) + other_features
    for col in leftover:
        if _feature_should_be_smooth(ds, col):
            if backend == "rust":
                terms.append(f"s({col}, type=ps, knots={knot_count})")
            elif backend == "r_gamlss":
                terms.append(f"pb({col}, inter={knot_expr})")
            elif backend == "mgcv":
                terms.append(f"s({col}, bs='ps', k={mgcv_k})")
            elif backend == "gamboostlss":
                terms.append(f"bbs({col}, knots={knot_expr})")
            elif backend == "bamlss":
                terms.append(f"s({col}, bs='ps', k={mgcv_k})")
            elif backend == "brms":
                terms.append(f"s({col}, k={brms_k})")
            else:
                raise RuntimeError(f"unsupported sigma backend '{backend}'")
        else:
            if backend == "rust":
                terms.append(f"linear({col})")
            elif backend == "gamboostlss":
                terms.append(f"bols({col}, intercept=FALSE)")
            else:
                terms.append(col)
    return terms


def _sigma_feature_formula(ds: dict[str, typing.Any], *, scenario_name: str | None, backend: str) -> str:
    terms = _sigma_feature_terms(ds, scenario_name=scenario_name, backend=backend)
    if backend == "r_gamlss" and not terms:
        raise RuntimeError(
            f"r_gamlss requires a non-constant sigma model for scenario "
            f"'{scenario_name or ds.get('name', 'unknown')}', but no sigma terms were generated"
        )
    return "~ " + (" + ".join(terms) if terms else "1")


def run_external_r_gamlss_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    scenario_name = scenario["name"]
    if not _is_gamlss_benchmark_scenario(scenario_name):
        return None

    if ds is None:
        ds = dataset_for_scenario(scenario)
    family = ds["family"]
    if family not in ("gaussian", "binomial"):
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    mu_formula = _gamlss_mu_formula_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None
    sigma_formula = _sigma_feature_formula(ds, scenario_name=scenario_name, backend="r_gamlss")

    with _workspace_tempdir(prefix="gam_bench_r_gamlss_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_gamlss_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario_name,
            "family": family,
            "mu_formula": mu_formula,
            "sigma_formula": sigma_formula,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(gamlss)
  library(gamlss.dist)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
family_name <- as.character(payload$family)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.formula(as.character(payload$sigma_formula))
train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_train <- as.numeric(df[[target_name]])[train_idx]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

t0 <- proc.time()[["elapsed"]]
fit_formula <- mu_formula
fit_family <- if (family_name == "binomial") BI() else NO()
fit <- tryCatch(
  gamlss(
    as.formula(fit_formula),
    sigma.formula = sigma_formula,
    family = fit_family,
    data = train_df,
    control = gamlss.control(n.cyc=200, trace=FALSE)
  ),
  error = function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(
    status="failed",
    error=paste0("r_gamlss fit failed: ", conditionMessage(fit))
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, what="mu", type="response")),
  error = function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(
    status="failed",
    error=paste0("r_gamlss predict failed: ", conditionMessage(p))
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

sigma_hat <- tryCatch(
  pmax(as.numeric(predict(fit, newdata=test_df, what="sigma", type="response")), 1e-12),
  error = function(e) e
)
if (inherits(sigma_hat, "error")) {
  out <- list(status="failed", error=paste0("r_gamlss sigma predict failed: ", conditionMessage(sigma_hat)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(sigma_hat) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_gamlss sigma length mismatch (got ", length(sigma_hat), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(sigma_hat) | sigma_hat <= 0)) {
  out <- list(status="failed", error="r_gamlss sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (family_name == "binomial") {
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    pred=as.numeric(p),
    sigma=as.numeric(sigma_hat),
    model_spec=paste0("gamlss(BI; sigma.formula=", deparse(sigma_formula), "): ", fit_formula)
  )
} else {
  rmse <- sqrt(mean((y_test - p)^2))
  mae <- mean(abs(y_test - p))
  sst <- sum((y_test - mean(y_test))^2)
  r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
  logloss <- mean(0.5 * log(2 * pi * sigma_hat^2) + ((y_test - p)^2) / (2 * sigma_hat^2))
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    pred=as.numeric(p),
    auc=NULL,
    brier=NULL,
    logloss=logloss,
    nagelkerke_r2=NULL,
    rmse=rmse,
    mae=mae,
    r2=r2,
    model_spec=paste0("gamlss(NO; sigma.formula=", deparse(sigma_formula), "): ", fit_formula)
  )
}
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        plot_payload = _init_plot_payload(ds)
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_gamlss",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_gamlss",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_gamlss fold failed")),
                }
            pred = np.asarray(fold_row.get("pred", []), dtype=float).reshape(-1)
            if pred.shape[0] != len(fold.test_idx):
                return {
                    "contender": "r_gamlss",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (
                        "r_gamlss fold output missing/invalid prediction vector "
                        f"(got {pred.shape[0]}, expected {len(fold.test_idx)})"
                    ),
                }
            test_df = pd.DataFrame(ds["rows"]).iloc[fold.test_idx].copy()
            _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
            fit_sec = float(fold_row.get("fit_sec", 0.0))
            predict_sec = float(fold_row.get("predict_sec", 0.0))
            model_spec = str(fold_row.get("model_spec", "r_gamlss"))
            if family == "binomial":
                sigma_hat = np.asarray(fold_row.get("sigma", []), dtype=float).reshape(-1)
                if sigma_hat.shape[0] != len(fold.test_idx):
                    return {
                        "contender": "r_gamlss",
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "r_gamlss fold output missing/invalid sigma vector "
                            f"(got {sigma_hat.shape[0]}, expected {len(fold.test_idx)})"
                        ),
                    }
                y_test = test_df[ds["target"]].to_numpy(dtype=float)
                train_df = pd.DataFrame(ds["rows"]).iloc[fold.train_idx].copy()
                y_train = train_df[ds["target"]].to_numpy(dtype=float)
                cv_rows.append(
                    {
                        "fit_sec": fit_sec,
                        "predict_sec": predict_sec,
                        "auc": auc_score(y_test, pred),
                        "brier": brier_score(y_test, pred),
                        "logloss": log_loss_score(y_test, pred),
                        "nagelkerke_r2": nagelkerke_r2_score(
                            y_test, pred, null_mean=float(np.mean(y_train))
                        ),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": model_spec,
                    }
                )
            else:
                fold_row.pop("pred", None)
                fold_row["n_test"] = int(len(fold.test_idx))
                cv_rows.append(fold_row)

    return _finalize_cv_result(
        contender="r_gamlss",
        scenario_name=scenario_name,
        family=family,
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_mgcv_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if folds is None:
        folds = folds_for_dataset(ds)
    contender_name = "r_survival_coxph" if ds["family"] == "survival" else "r_mgcv"
    mgcv_formula = None
    rust_cfg = _scenario_fit_mapping(scenario["name"])
    use_select = bool((rust_cfg or {}).get("double_penalty", True))
    if ds["family"] != "survival":
        mgcv_formula = _mgcv_formula_for_scenario(scenario["name"], ds)
        if not mgcv_formula:
            raise RuntimeError(
                f"Missing required shared mgcv formula for non-survival scenario '{scenario['name']}'"
            )

    with _workspace_tempdir(prefix="gam_bench_mgcv_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_mgcv_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario["name"],
            "mgcv_formula": mgcv_formula,
            "survival_formula": (
                _coxph_survival_formula_for_scenario(scenario["name"], ds)
                if ds["family"] == "survival"
                else None
            ),
            "use_select": use_select,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

patch_mgcv_gam_fit5_matrix_drop <- function() {
  ns <- asNamespace("mgcv")
  gam_fit5 <- get("gam.fit5", envir=ns)
  body_txt <- paste(deparse(body(gam_fit5)), collapse="\n")
  patched_body_txt <- gsub(
    "\\(D \\*\\s*A\\)\\[piv, \\]",
    "(D * A)[piv, , drop = FALSE]",
    body_txt,
    perl=TRUE
  )
  if (identical(body_txt, patched_body_txt)) {
    stop("failed to patch mgcv::gam.fit5 matrix indexing bug")
  }
  body(gam_fit5) <- parse(text=patched_body_txt)[[1]]
  gam_fit5 <- compiler::cmpfun(gam_fit5)
  unlockBinding("gam.fit5", ns)
  assign("gam.fit5", gam_fit5, envir=ns)
  lockBinding("gam.fit5", ns)
}

patch_mgcv_gam_fit5_matrix_drop()

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
family_name <- as.character(payload$dataset$family)
if (family_name != "survival") {
  if (!nzchar(target_name) || !(target_name %in% colnames(df))) {
    stop(sprintf("invalid or missing dataset target column: %s", target_name))
  }
}
scenario_name <- as.character(payload$scenario_name)
mgcv_formula <- NULL
if (!is.null(payload$mgcv_formula)) {
  mgcv_formula <- as.character(payload$mgcv_formula)
}
survival_formula <- NULL
if (!is.null(payload$survival_formula)) {
  survival_formula <- as.character(payload$survival_formula)
}
use_select <- TRUE
if (!is.null(payload$use_select)) {
  use_select <- isTRUE(payload$use_select)
}

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- NULL
y_train <- NULL
if (family_name != "survival") {
  y_all <- as.numeric(df[[target_name]])
  y_train <- y_all[train_idx]
  y_test <- y_all[test_idx]
}

fam <- if (family_name == "binomial") binomial(link="logit") else gaussian(link="identity")

if (family_name != "survival") {
  feature_cols <- as.character(payload$dataset$features)
  for (cn in feature_cols) {
    mu <- mean(train_df[[cn]])
    sdv <- stats::sd(train_df[[cn]])
    if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
    train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
    test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
  }
}

if (family_name == "survival") {
  suppressPackageStartupMessages(library(survival))
  feature_cols <- as.character(payload$dataset$features)
  for (cn in feature_cols) {
    mu <- mean(train_df[[cn]])
    sdv <- stats::sd(train_df[[cn]])
    if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
    train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
    test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
  }
  if (is.null(survival_formula) || !nzchar(survival_formula)) {
    stop(sprintf("missing survival formula for scenario: %s", scenario_name))
  }
  ftxt <- survival_formula
  t0 <- proc.time()[["elapsed"]]
  fit <- coxph(as.formula(ftxt), data=train_df, ties="efron")
  fit_sec <- proc.time()[["elapsed"]] - t0

  pred_t0 <- proc.time()[["elapsed"]]
  lp_train <- as.numeric(predict(fit, newdata=train_df, type="lp"))
  lp <- as.numeric(predict(fit, newdata=test_df, type="lp"))
  pred_sec <- proc.time()[["elapsed"]] - pred_t0

  risk <- as.numeric(lp)
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    risk=as.numeric(risk),
    train_risk=as.numeric(lp_train),
    auc=NULL,
    brier=NULL,
    logloss=NULL,
    rmse=NULL,
    mae=NULL,
    r2=NULL,
    model_spec=ftxt
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

if (is.null(mgcv_formula) || !nzchar(mgcv_formula)) {
  stop(sprintf("missing shared mgcv formula for scenario: %s", scenario_name))
}
ftxt <- mgcv_formula

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch(
  gam(as.formula(ftxt), family=fam, data=train_df, method="REML", select=use_select),
  error = function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(
    status="failed",
    error=paste0("r_mgcv fit failed: ", conditionMessage(fit))
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, type="response")),
  error = function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(
    status="failed",
    error=paste0("r_mgcv predict failed: ", conditionMessage(p))
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

if (family_name == "binomial") {
  ord <- order(p)
  yy <- y_test[ord]
  n_pos <- sum(yy > 0.5)
  n_neg <- sum(yy <= 0.5)
  if (n_pos == 0 || n_neg == 0) {
    auc <- 0.5
  } else {
    ranks <- seq_along(yy)
    rank_sum_pos <- sum(ranks[yy > 0.5])
    auc <- (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
  }
  brier <- mean((y_test - p)^2)
  p_safe <- pmin(pmax(p, 1e-12), 1 - 1e-12)
  logloss <- mean(-(y_test * log(p_safe) + (1 - y_test) * log(1 - p_safe)))
  p_mean <- mean(y_train)
  if (is.finite(p_mean) && p_mean > 0 && p_mean < 1) {
    ll_null <- sum(y_test * log(p_mean) + (1 - y_test) * log(1 - p_mean))
    ll_model <- sum(y_test * log(p_safe) + (1 - y_test) * log(1 - p_safe))
    n_obs <- length(y_test)
    r2_cs <- 1 - exp((2 / n_obs) * (ll_null - ll_model))
    max_r2_cs <- 1 - exp((2 / n_obs) * ll_null)
    nagelkerke_r2 <- if (max_r2_cs > 0) r2_cs / max_r2_cs else NULL
  } else {
    nagelkerke_r2 <- NULL
  }
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    pred=as.numeric(p),
    auc=auc,
    brier=brier,
    logloss=logloss,
    nagelkerke_r2=nagelkerke_r2,
    rmse=NULL,
    mae=NULL,
    r2=NULL,
    model_spec=ftxt
  )
} else {
  sigma_hat <- NA_real_
  fit_scale <- tryCatch(as.numeric(summary(fit)$scale), error=function(e) NA_real_)
  if (is.finite(fit_scale) && fit_scale > 0) {
    sigma_hat <- sqrt(fit_scale)
  } else {
    p_train <- tryCatch(
      as.numeric(predict(fit, newdata=train_df, type="response")),
      error = function(e) e
    )
    if (inherits(p_train, "error")) {
      out <- list(
        status="failed",
        error=paste0("r_mgcv train-predict failed: ", conditionMessage(p_train))
      )
      write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
      quit(save="no")
    }
    sigma_hat <- sqrt(mean((y_train - p_train)^2))
  }
  sigma_hat <- max(as.numeric(sigma_hat), 1e-12)
  rmse <- sqrt(mean((y_test - p)^2))
  mae <- mean(abs(y_test - p))
  sst <- sum((y_test - mean(y_test))^2)
  if (sst <= 0) {
    r2 <- 0.0
  } else {
    r2 <- 1.0 - sum((y_test - p)^2) / sst
  }
  logloss <- mean(0.5 * log(2 * pi * sigma_hat^2) + ((y_test - p)^2) / (2 * sigma_hat^2))
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    pred=as.numeric(p),
    auc=NULL,
    brier=NULL,
    logloss=logloss,
    rmse=rmse,
    mae=mae,
    r2=r2,
    model_spec=ftxt
  )
}

write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        plot_payload = _init_plot_payload(ds)
        all_df = pd.DataFrame(ds["rows"])
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if ds["family"] == "survival":
                all_df = pd.DataFrame(ds["rows"])
                train_df = all_df.iloc[fold.train_idx].copy()
                test_df = all_df.iloc[fold.test_idx].copy()
                event_times = test_df[ds["time_col"]].to_numpy(dtype=float)
                risk = np.asarray(fold_row.get("risk", []), dtype=float).reshape(-1)
                train_risk = np.asarray(fold_row.get("train_risk", []), dtype=float).reshape(-1)
                if risk.shape[0] != event_times.shape[0]:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario["name"],
                        "status": "failed",
                        "error": (
                            "r_survival_coxph fold output missing/invalid risk vector "
                            f"(got {risk.shape[0]}, expected {event_times.shape[0]})"
                        ),
                    }
                if train_risk.shape[0] != len(fold.train_idx):
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario["name"],
                        "status": "failed",
                        "error": (
                            "r_survival_coxph fold output missing/invalid train risk vector "
                            f"(got {train_risk.shape[0]}, expected {len(fold.train_idx)})"
                        ),
                    }
                surv_metrics = score_survival_fold(
                    train_df,
                    test_df,
                    time_col=ds["time_col"],
                    event_col=ds["event_col"],
                    risk_score=risk,
                    train_risk_score=train_risk,
                )
                _append_survival_plot_fold(
                    plot_payload,
                    test_df,
                    time_col=ds["time_col"],
                    event_col=ds["event_col"],
                    risk_score=risk,
                )
                fold_row["auc"] = surv_metrics["auc"]
                fold_row["brier"] = surv_metrics["brier"]
                fold_row["logloss"] = surv_metrics["logloss"]
                fold_row["nagelkerke_r2"] = surv_metrics["nagelkerke_r2"]
                fold_row.pop("risk", None)
                fold_row.pop("train_risk", None)
            else:
                pred = np.asarray(fold_row.get("pred", []), dtype=float).reshape(-1)
                if pred.shape[0] != len(fold.test_idx):
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario["name"],
                        "status": "failed",
                        "error": (
                            f"{contender_name} fold output missing/invalid prediction vector "
                            f"(got {pred.shape[0]}, expected {len(fold.test_idx)})"
                        ),
                    }
                test_df = all_df.iloc[fold.test_idx].copy()
                _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
                fold_row.pop("pred", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    return _finalize_cv_result(
        contender=contender_name,
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_mgcv_gaulss_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "gaussian":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)
    mu_formula = _mgcv_formula_for_scenario(scenario["name"], ds)
    rust_cfg = _scenario_fit_mapping(scenario["name"])
    use_select = bool((rust_cfg or {}).get("double_penalty", True))

    # For gaulss, cap the mu formula k values too so the combined mu + sigma
    # basis stays within the training-set budget.  The sigma formula already
    # has the same cap (see _sigma_feature_terms).  Only kicks in for small
    # datasets; for large datasets the cap is above the configured k.
    n_smooth_features = sum(
        1 for c in ds.get("features", []) if _feature_should_be_smooth(ds, str(c))
    )
    gaulss_total = 2 * max(1, n_smooth_features)
    gaulss_k_cap = f"max(4L, as.integer(floor(nrow(train_df) * 0.8 / {gaulss_total})))"
    mu_formula = re.sub(
        r"k=min\((\d+),\s*nrow\(train_df\)-1\)",
        rf"k=min(\1, {gaulss_k_cap})",
        mu_formula,
    )

    with _workspace_tempdir(prefix="gam_bench_mgcv_gaulss_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_mgcv_gaulss_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario["name"],
            "mu_formula": mu_formula,
            "sigma_formula": _sigma_feature_formula(ds, scenario_name=scenario["name"], backend="mgcv"),
            "use_select": use_select,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

patch_mgcv_gam_fit5_matrix_drop <- function() {
  ns <- asNamespace("mgcv")
  gam_fit5 <- get("gam.fit5", envir=ns)
  body_txt <- paste(deparse(body(gam_fit5)), collapse="\n")
  patched_body_txt <- gsub(
    "\\(D \\*\\s*A\\)\\[piv, \\]",
    "(D * A)[piv, , drop = FALSE]",
    body_txt,
    perl=TRUE
  )
  if (identical(body_txt, patched_body_txt)) {
    stop("failed to patch mgcv::gam.fit5 matrix indexing bug")
  }
  body(gam_fit5) <- parse(text=patched_body_txt)[[1]]
  gam_fit5 <- compiler::cmpfun(gam_fit5)
  unlockBinding("gam.fit5", ns)
  assign("gam.fit5", gam_fit5, envir=ns)
  lockBinding("gam.fit5", ns)
}

patch_mgcv_gam_fit5_matrix_drop()

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.character(payload$sigma_formula)
use_select <- TRUE
if (!is.null(payload$use_select)) {
  use_select <- isTRUE(payload$use_select)
}

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

rhs_parts <- strsplit(mu_formula, "~", fixed=TRUE)[[1]]
if (length(rhs_parts) < 2) {
  out <- list(status="failed", error=paste0("invalid mu formula: ", mu_formula))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
mu_rhs <- trimws(rhs_parts[[2]])

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch(
  gam(
    list(as.formula(mu_formula), as.formula(sigma_formula)),
    family=gaulss(),
    data=train_df,
    method="REML",
    select=use_select
  ),
  error=function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(status="failed", error=paste0("r_mgcv_gaulss fit failed: ", conditionMessage(fit)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
pred <- tryCatch(as.matrix(predict(fit, newdata=test_df, type="response")), error=function(e) e)
pred_sec <- proc.time()[["elapsed"]] - pred_t0
if (inherits(pred, "error")) {
  out <- list(status="failed", error=paste0("r_mgcv_gaulss predict failed: ", conditionMessage(pred)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

if (ncol(pred) < 1) {
  out <- list(status="failed", error="r_mgcv_gaulss predict returned empty matrix")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
p <- as.numeric(pred[,1])
if (ncol(pred) < 2) {
  out <- list(status="failed", error="r_mgcv_gaulss predict output missing sigma column")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
inv_sigma_hat <- as.numeric(pred[,2])
if (length(inv_sigma_hat) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_mgcv_gaulss inverse-sigma length mismatch (got ", length(inv_sigma_hat), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(inv_sigma_hat) | inv_sigma_hat <= 0)) {
  out <- list(status="failed", error="r_mgcv_gaulss inverse-sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
sigma_hat <- 1.0 / inv_sigma_hat
rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
logloss <- mean(0.5 * log(2 * pi * sigma_hat^2) + ((y_test - p)^2) / (2 * sigma_hat^2))

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  pred=as.numeric(p),
  auc=NULL,
  brier=NULL,
  logloss=logloss,
  rmse=rmse,
  mae=mae,
  r2=r2,
  model_spec=paste0("gam(list(", target_name, " ~ ", mu_rhs, ", ", sigma_formula, "), family=gaulss())")
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        plot_payload = _init_plot_payload(ds)
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_mgcv_gaulss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_mgcv_gaulss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_mgcv_gaulss fold failed")),
                }
            pred = np.asarray(fold_row.get("pred", []), dtype=float).reshape(-1)
            if pred.shape[0] != len(fold.test_idx):
                return {
                    "contender": "r_mgcv_gaulss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (
                        "r_mgcv_gaulss fold output missing/invalid prediction vector "
                        f"(got {pred.shape[0]}, expected {len(fold.test_idx)})"
                    ),
                }
            test_df = pd.DataFrame(ds["rows"]).iloc[fold.test_idx].copy()
            _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
            fold_row.pop("pred", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    return _finalize_cv_result(
        contender="r_mgcv_gaulss",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def _gamboostlss_formulas_for_scenario(scenario_name: str, ds: typing.Any) -> typing.Any:
    cfg = _scenario_fit_mapping(scenario_name)
    if cfg is None:
        return None, None
    if ds["family"] != "gaussian":
        return None, None
    if _requires_joint_spatial_term(cfg):
        return None, None

    knot_count = max(4, int(cfg.get("knots", 8)))
    knot_expr = f"max(1, min({knot_count}, nrow(train_df)-1))"
    mu_terms = [f"bols({c}, intercept=FALSE)" for c in cfg.get("linear_cols", [])]

    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        for col in smooth_cols:
            mu_terms.append(f"bbs({col}, knots={knot_expr})")
    elif cfg.get("smooth_col"):
        col = cfg["smooth_col"]
        mu_terms.append(f"bbs({col}, knots={knot_expr})")

    if not mu_terms:
        mu_terms = ["1"]
    mu_formula = f"{ds['target']} ~ " + " + ".join(mu_terms)

    sigma_formula = _sigma_feature_formula(ds, scenario_name=scenario_name, backend="gamboostlss")
    return mu_formula, sigma_formula


def run_external_r_gamboostlss_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "gaussian":
        return None
    scenario_name = scenario["name"]
    if folds is None:
        folds = folds_for_dataset(ds)
    mu_formula, sigma_formula = _gamboostlss_formulas_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None

    with _workspace_tempdir(prefix="gam_bench_r_gamboostlss_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_gamboostlss_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario_name,
            "mu_formula": mu_formula,
            "sigma_formula": sigma_formula,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(gamboostLSS)
  library(mboost)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.character(payload$sigma_formula)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch({
  fit_full <- gamboostLSS(
    formula=list(mu=as.formula(mu_formula), sigma=as.formula(sigma_formula)),
    data=train_df,
    families=GaussianLSS(),
    control=boost_control(mstop=600)
  )
  aic_obj <- tryCatch(
    AIC(fit_full, method="corrected"),
    error=function(e) e
  )
  if (inherits(aic_obj, "error")) {
    stop(paste0("AIC selection failed: ", conditionMessage(aic_obj)))
  } else {
    selected_mstop <- as.integer(mstop(aic_obj))
    if (!is.finite(selected_mstop) || selected_mstop < 1) {
      stop(paste0("AIC selection returned invalid mstop: ", as.character(selected_mstop)))
    }
  }
  fit_final <- fit_full[selected_mstop]
  attr(fit_final, "selected_mstop") <- selected_mstop
  fit_final
},
error=function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(status="failed", error=paste0("r_gamboostlss fit failed: ", conditionMessage(fit)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, parameter="mu", type="response")),
  error=function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(status="failed", error=paste0("r_gamboostlss predict failed: ", conditionMessage(p)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(p) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_gamboostlss mu length mismatch (got ", length(p), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(p))) {
  out <- list(status="failed", error="r_gamboostlss mu has non-finite values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
sigma_pred <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, parameter="sigma", type="response")),
  error=function(e) e
)
if (inherits(sigma_pred, "error")) {
  out <- list(status="failed", error=paste0("r_gamboostlss sigma predict failed: ", conditionMessage(sigma_pred)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(sigma_pred) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_gamboostlss sigma length mismatch (got ", length(sigma_pred), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(sigma_pred) | sigma_pred <= 0)) {
  out <- list(status="failed", error="r_gamboostlss sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
logloss <- mean(0.5 * log(2 * pi * sigma_pred^2) + ((y_test - p)^2) / (2 * sigma_pred^2))

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  pred=as.numeric(p),
  auc=NULL,
  brier=NULL,
  logloss=logloss,
  rmse=rmse,
  mae=mae,
  r2=r2,
  model_spec=paste0(
    "gamboostLSS(GaussianLSS; AIC-selected mstop=",
    as.integer(attr(fit, "selected_mstop")),
    "): ",
    mu_formula,
    " ; sigma ",
    sigma_formula
  )
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        plot_payload = _init_plot_payload(ds)
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_gamboostlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_gamboostlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_gamboostlss fold failed")),
                }
            pred = np.asarray(fold_row.get("pred", []), dtype=float).reshape(-1)
            if pred.shape[0] != len(fold.test_idx):
                return {
                    "contender": "r_gamboostlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (
                        "r_gamboostlss fold output missing/invalid prediction vector "
                        f"(got {pred.shape[0]}, expected {len(fold.test_idx)})"
                    ),
                }
            test_df = pd.DataFrame(ds["rows"]).iloc[fold.test_idx].copy()
            _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
            fold_row.pop("pred", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    return _finalize_cv_result(
        contender="r_gamboostlss",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def _bamlss_formulas_for_scenario(scenario_name: str, ds: typing.Any) -> typing.Any:
    if ds["family"] != "gaussian":
        return None, None
    mu_formula = _mgcv_formula_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None, None
    # Cap mu formula k for combined location-scale models (same logic as
    # run_external_mgcv_gaulss_cv) so total basis stays within nrow budget.
    n_smooth_features = sum(
        1 for c in ds.get("features", []) if _feature_should_be_smooth(ds, str(c))
    )
    gaulss_total = 2 * max(1, n_smooth_features)
    gaulss_k_cap = f"max(4L, as.integer(floor(nrow(train_df) * 0.8 / {gaulss_total})))"
    mu_formula = re.sub(
        r"k=min\((\d+),\s*nrow\(train_df\)-1\)",
        rf"k=min(\1, {gaulss_k_cap})",
        mu_formula,
    )
    sigma_formula = _sigma_feature_formula(ds, scenario_name=scenario_name, backend="bamlss")
    return mu_formula, sigma_formula


def _brms_formulas_for_scenario(scenario_name: str, ds: typing.Any) -> typing.Any:
    if ds["family"] != "gaussian":
        return None, None
    mu_formula = _mgcv_formula_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None, None
    # Cap mu formula k for location-scale models (same as mgcv gaulss / bamlss).
    n_smooth_features = sum(
        1 for c in ds.get("features", []) if _feature_should_be_smooth(ds, str(c))
    )
    gaulss_total = 2 * max(1, n_smooth_features)
    gaulss_k_cap = f"max(4L, as.integer(floor(nrow(train_df) * 0.8 / {gaulss_total})))"
    mu_formula = re.sub(
        r"k=min\((\d+),\s*nrow\(train_df\)-1\)",
        rf"k=min(\1, {gaulss_k_cap})",
        mu_formula,
    )
    sigma_formula = _sigma_feature_formula(ds, scenario_name=scenario_name, backend="brms")
    return mu_formula, sigma_formula


def run_external_r_bamlss_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "gaussian":
        return None
    scenario_name = scenario["name"]
    if folds is None:
        folds = folds_for_dataset(ds)
    mu_formula, sigma_formula = _bamlss_formulas_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None

    with _workspace_tempdir(prefix="gam_bench_r_bamlss_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_bamlss_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario_name,
            "mu_formula": mu_formula,
            "sigma_formula": sigma_formula,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(bamlss)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.character(payload$sigma_formula)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch(
  bamlss(
    formula = list(mu = as.formula(mu_formula), sigma = as.formula(sigma_formula)),
    family = "gaussian",
    data = train_df,
    optimizer = TRUE,
    sampler = FALSE,
    verbose = FALSE
  ),
  error = function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(status="failed", error=paste0("r_bamlss fit failed: ", conditionMessage(fit)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, model="mu", type="response")),
  error=function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(status="failed", error=paste0("r_bamlss predict failed: ", conditionMessage(p)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(p) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_bamlss mu length mismatch (got ", length(p), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(p))) {
  out <- list(status="failed", error="r_bamlss mu has non-finite values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
sigma_pred <- tryCatch(
  as.numeric(predict(fit, newdata=test_df, model="sigma", type="response")),
  error=function(e) e
)
if (inherits(sigma_pred, "error")) {
  out <- list(status="failed", error=paste0("r_bamlss sigma predict failed: ", conditionMessage(sigma_pred)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(sigma_pred) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_bamlss sigma length mismatch (got ", length(sigma_pred), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(sigma_pred) | sigma_pred <= 0)) {
  out <- list(status="failed", error="r_bamlss sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
logloss <- mean(0.5 * log(2 * pi * sigma_pred^2) + ((y_test - p)^2) / (2 * sigma_pred^2))

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  pred=as.numeric(p),
  auc=NULL,
  brier=NULL,
  logloss=logloss,
  rmse=rmse,
  mae=mae,
  r2=r2,
  model_spec=paste0("bamlss(gaussian; optimizer-only): ", mu_formula, " ; sigma ", sigma_formula)
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        plot_payload = _init_plot_payload(ds)
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_bamlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_bamlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_bamlss fold failed")),
                }
            pred = np.asarray(fold_row.get("pred", []), dtype=float).reshape(-1)
            if pred.shape[0] != len(fold.test_idx):
                return {
                    "contender": "r_bamlss",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (
                        "r_bamlss fold output missing/invalid prediction vector "
                        f"(got {pred.shape[0]}, expected {len(fold.test_idx)})"
                    ),
                }
            test_df = pd.DataFrame(ds["rows"]).iloc[fold.test_idx].copy()
            _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
            fold_row.pop("pred", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    return _finalize_cv_result(
        contender="r_bamlss",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_r_brms_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "gaussian":
        return None
    scenario_name = scenario["name"]
    if folds is None:
        folds = folds_for_dataset(ds)
    mu_formula, sigma_formula = _brms_formulas_for_scenario(scenario_name, ds)
    if not mu_formula:
        return None

    # Per-fold wall-clock cap for the brms MCMC fit (#1390). The brms
    # contender runs full Bayesian sampling (4 chains x 2000 iter) per CV
    # fold; on the larger panels (e.g. us48_demand_31day) the five folds
    # together overran the 42-minute per-shard GNU `timeout`, which then
    # SIGKILLed the whole shard (exit 124) AFTER gam had already finished —
    # discarding every result in the shard with no per-fold attribution.
    # Bounding each fold turns a slow/hung brms fold into a recorded, visible
    # per-fold failure instead. brms is in NON_BLOCKING_FAILURE_CONTENDERS, so
    # a capped fold does NOT fail the shard; the gam result and all other
    # contenders are preserved. Default 1500s (25 min) keeps the worst-case
    # 5-fold serial brms run inside the shard budget while leaving headroom for
    # the gam lane; override with BENCH_BRMS_FOLD_TIMEOUT_SEC (0 disables and
    # falls back to the global BENCH_CMD_TIMEOUT_SEC).
    brms_fold_timeout = _env_int("BENCH_BRMS_FOLD_TIMEOUT_SEC", 1500)

    with _workspace_tempdir(prefix="gam_bench_r_brms_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_brms_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario_name,
            "mu_formula": mu_formula,
            "sigma_formula": sigma_formula,
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(brms)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
target_name <- as.character(payload$dataset$target)
mu_formula <- as.character(payload$mu_formula)
sigma_formula <- as.character(payload$sigma_formula)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- as.numeric(df[[target_name]])[test_idx]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

t0 <- proc.time()[["elapsed"]]
fit <- tryCatch(
  brm(
    formula = bf(as.formula(mu_formula), sigma = as.formula(sigma_formula)),
    data = train_df,
    family = gaussian(),
    chains = 4,
    iter = 2000,
    warmup = 1000,
    cores = min(4L, max(1L, parallel::detectCores(logical=FALSE))),
    seed = 123,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    refresh = 0,
    silent = 2
  ),
  error = function(e) e
)
fit_sec <- proc.time()[["elapsed"]] - t0

if (inherits(fit, "error")) {
  out <- list(status="failed", error=paste0("r_brms fit failed: ", conditionMessage(fit)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

pred_t0 <- proc.time()[["elapsed"]]
p <- tryCatch(
  as.numeric(fitted(fit, newdata=test_df, summary=TRUE)[, "Estimate"]),
  error=function(e) e
)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (inherits(p, "error")) {
  out <- list(status="failed", error=paste0("r_brms predict failed: ", conditionMessage(p)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
sigma_pred <- tryCatch(
  as.numeric(fitted(fit, dpar="sigma", newdata=test_df, summary=TRUE)[, "Estimate"]),
  error=function(e) e
)
if (inherits(sigma_pred, "error")) {
  out <- list(status="failed", error=paste0("r_brms sigma extract failed: ", conditionMessage(sigma_pred)))
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (length(sigma_pred) != length(y_test)) {
  out <- list(
    status="failed",
    error=paste0("r_brms sigma length mismatch (got ", length(sigma_pred), ", expected ", length(y_test), ")")
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}
if (any(!is.finite(sigma_pred) | sigma_pred <= 0)) {
  out <- list(status="failed", error="r_brms sigma has non-finite or non-positive values")
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

rmse <- sqrt(mean((y_test - p)^2))
mae <- mean(abs(y_test - p))
sst <- sum((y_test - mean(y_test))^2)
r2 <- if (sst <= 0) 0.0 else 1.0 - sum((y_test - p)^2) / sst
logloss <- mean(0.5 * log(2 * pi * sigma_pred^2) + ((y_test - p)^2) / (2 * sigma_pred^2))

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  pred=as.numeric(p),
  auc=NULL,
  brier=NULL,
  logloss=logloss,
  rmse=rmse,
  mae=mae,
  r2=r2,
  model_spec=paste0("brms::brm(bf(", mu_formula, ", sigma ", sigma_formula, "); gaussian)")
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        plot_payload = _init_plot_payload(ds)
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
                timeout_sec=brms_fold_timeout if brms_fold_timeout > 0 else None,
            )
            if code != 0:
                # #1390: run_cmd reports rc=124 when a fold exceeded its
                # per-invocation BENCH_BRMS_FOLD_TIMEOUT_SEC budget (as opposed to
                # a genuine brms/Stan modeling error). Tag the recorded failure so
                # the suite can distinguish a budget overrun from a model failure
                # — the overrun is the expected, capped outcome that keeps a slow
                # brms fold from consuming the whole shard, not a regression.
                timed_out = code == 124
                base_error = err.strip() or out.strip()
                error_msg = (
                    f"r_brms fold exceeded the per-fold timeout "
                    f"({brms_fold_timeout:g}s); capped to protect the shard budget"
                    if timed_out
                    else base_error
                )
                return {
                    "contender": "r_brms",
                    "scenario_name": scenario["name"],
                    "status": "timeout" if timed_out else "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": error_msg,
                }
            fold_row = json.loads(out_path.read_text())
            if fold_row.get("status") != "ok":
                return {
                    "contender": "r_brms",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": str(fold_row.get("error", "r_brms fold failed")),
                }
            pred = np.asarray(fold_row.get("pred", []), dtype=float).reshape(-1)
            if pred.shape[0] != len(fold.test_idx):
                return {
                    "contender": "r_brms",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (
                        "r_brms fold output missing/invalid prediction vector "
                        f"(got {pred.shape[0]}, expected {len(fold.test_idx)})"
                    ),
                }
            test_df = pd.DataFrame(ds["rows"]).iloc[fold.test_idx].copy()
            _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
            fold_row.pop("pred", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    return _finalize_cv_result(
        contender="r_brms",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_mgcv_survival_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    with _workspace_tempdir(prefix="gam_bench_mgcv_surv_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_mgcv_survival_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario["name"],
            "survival_formula": _mgcv_survival_formula_for_scenario(scenario["name"], ds),
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(mgcv)
  library(survival)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
scenario_name <- as.character(payload$scenario_name)
time_col <- as.character(payload$dataset$time_col)
event_col <- as.character(payload$dataset$event_col)
survival_formula <- as.character(payload$survival_formula)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]

feature_cols <- as.character(payload$dataset$features)
for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

if (!nzchar(survival_formula)) {
  stop(sprintf("missing mgcv survival formula for scenario: %s", scenario_name))
}
ftxt <- survival_formula

t0 <- proc.time()[["elapsed"]]
fit <- gam(
  as.formula(ftxt),
  family=cox.ph(),
  weights=as.numeric(train_df[[event_col]]),
  data=train_df,
  method="REML",
  select=TRUE
)
fit_sec <- proc.time()[["elapsed"]] - t0

pred_t0 <- proc.time()[["elapsed"]]
lp_train <- as.numeric(predict(fit, newdata=train_df, type="link"))
lp_test <- as.numeric(predict(fit, newdata=test_df, type="link"))
risk <- as.numeric(lp_test)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  auc=NULL,
  train_risk=as.numeric(lp_train),
  risk=risk,
  brier=NULL,
  logloss=NULL,
  rmse=NULL,
  mae=NULL,
  r2=NULL,
  model_spec=ftxt
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        plot_payload = _init_plot_payload(ds)
        all_df = pd.DataFrame(ds["rows"])
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_mgcv_coxph",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": err.strip() or out.strip(),
                }

            fold_row = json.loads(out_path.read_text())
            train_df = all_df.iloc[fold.train_idx].copy()
            test_df = all_df.iloc[fold.test_idx].copy()
            event_times = test_df[ds["time_col"]].to_numpy(dtype=float)
            risk = np.asarray(fold_row.get("risk", []), dtype=float).reshape(-1)
            train_risk = np.asarray(fold_row.get("train_risk", []), dtype=float).reshape(-1)
            if risk.shape[0] != event_times.shape[0]:
                return {
                    "contender": "r_mgcv_coxph",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": (
                        "r_mgcv_coxph fold output missing/invalid risk vector "
                        f"(got {risk.shape[0]}, expected {event_times.shape[0]})"
                    ),
                }
            if train_risk.shape[0] != len(fold.train_idx):
                return {
                    "contender": "r_mgcv_coxph",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": (
                        "r_mgcv_coxph fold output missing/invalid train risk vector "
                        f"(got {train_risk.shape[0]}, expected {len(fold.train_idx)})"
                    ),
                }
            surv_metrics = score_survival_fold(
                train_df,
                test_df,
                time_col=ds["time_col"],
                event_col=ds["event_col"],
                risk_score=risk,
                train_risk_score=train_risk,
            )
            _append_survival_plot_fold(
                plot_payload,
                test_df,
                time_col=ds["time_col"],
                event_col=ds["event_col"],
                risk_score=risk,
            )
            fold_row["auc"] = surv_metrics["auc"]
            fold_row["brier"] = surv_metrics["brier"]
            fold_row["logloss"] = surv_metrics["logloss"]
            fold_row["nagelkerke_r2"] = surv_metrics["nagelkerke_r2"]
            fold_row.pop("risk", None)
            fold_row.pop("train_risk", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    return _finalize_cv_result(
        contender="r_mgcv_coxph",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_sksurv_rsf_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        sksurv_ensemble: typing.Any = importlib.import_module("sksurv.ensemble")
        sksurv_util: typing.Any = importlib.import_module("sksurv.util")
        RandomSurvivalForest = sksurv_ensemble.RandomSurvivalForest
        Surv = sksurv_util.Surv
    except _EXPECTED_OPTIONAL_IMPORT_FAILURES as e:
        return {
            "contender": "python_sksurv_rsf",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"scikit-survival import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]
    cv_rows = []
    plot_payload = _init_plot_payload(ds)

    for fold_id, fold in enumerate(folds):
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_train = Surv.from_arrays(
            event=train_df[event_col].to_numpy(dtype=float) > 0.5,
            time=train_df[time_col].to_numpy(dtype=float),
        )

        fit_start = datetime.now(timezone.utc)
        rsf = RandomSurvivalForest(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=CV_SEED + fold_id,
        )
        rsf.fit(x_train, y_train)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        train_risk = rsf.predict(x_train).astype(float)
        risk = rsf.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()

        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                **score_survival_fold(
                    train_df,
                    test_df,
                    time_col=time_col,
                    event_col=event_col,
                    risk_score=risk,
                    train_risk_score=train_risk,
                ),
                "n_test": int(len(fold.test_idx)),
                "model_spec": (
                    "RandomSurvivalForest("
                    "n_estimators=300,min_samples_split=10,min_samples_leaf=5,max_features='sqrt')"
                ),
            }
        )
        _append_survival_plot_fold(
            plot_payload,
            test_df,
            time_col=time_col,
            event_col=event_col,
            risk_score=risk,
        )

    return _finalize_cv_result(
        contender="python_sksurv_rsf",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_sksurv_coxnet_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        sksurv_linear_model: typing.Any = importlib.import_module("sksurv.linear_model")
        sksurv_util: typing.Any = importlib.import_module("sksurv.util")
        CoxnetSurvivalAnalysis = sksurv_linear_model.CoxnetSurvivalAnalysis
        Surv = sksurv_util.Surv
    except _EXPECTED_OPTIONAL_IMPORT_FAILURES as e:
        return {
            "contender": "python_sksurv_coxnet",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"scikit-survival import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]
    cv_rows = []
    plot_payload = _init_plot_payload(ds)

    for fold_id, fold in enumerate(folds):
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_train = Surv.from_arrays(
            event=train_df[event_col].to_numpy(dtype=float) > 0.5,
            time=train_df[time_col].to_numpy(dtype=float),
        )

        fit_start = datetime.now(timezone.utc)
        model = CoxnetSurvivalAnalysis(
            l1_ratio=0.5,
            alpha_min_ratio=0.01,
            max_iter=100000,
            fit_baseline_model=False,
        )
        model.fit(x_train, y_train)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        train_risk = model.predict(x_train).astype(float)
        risk = model.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                **score_survival_fold(
                    train_df,
                    test_df,
                    time_col=time_col,
                    event_col=event_col,
                    risk_score=risk,
                    train_risk_score=train_risk,
                ),
                "n_test": int(len(fold.test_idx)),
                "model_spec": "CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01)",
            }
        )
        _append_survival_plot_fold(
            plot_payload,
            test_df,
            time_col=time_col,
            event_col=event_col,
            risk_score=risk,
        )

    return _finalize_cv_result(
        contender="python_sksurv_coxnet",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_lifelines_coxph_enet_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        CoxPHFitter, ConvergenceWarning = _require_lifelines_coxph()
    except _EXPECTED_OPTIONAL_IMPORT_FAILURES as e:
        return {
            "contender": "python_lifelines_coxph_enet",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"lifelines import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]
    cv_rows = []
    plot_payload = _init_plot_payload(ds)

    for fold in folds:
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        fit_feature_cols = feature_cols
        model_spec = "CoxPHFitter(linear terms; train-fold z-score; penalizer=0.05; l1_ratio=0.5)"

        fit_start = datetime.now(timezone.utc)
        cph = CoxPHFitter(penalizer=0.05, l1_ratio=0.5)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            cph.fit(
                train_df[[*fit_feature_cols, time_col, event_col]],
                duration_col=time_col,
                event_col=event_col,
            )
        conv_warn = [w for w in wlist if issubclass(w.category, ConvergenceWarning)]
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        train_risk = cph.predict_partial_hazard(train_df[fit_feature_cols]).to_numpy(dtype=float).reshape(-1)
        risk = cph.predict_partial_hazard(test_df[fit_feature_cols]).to_numpy(dtype=float).reshape(-1)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                **score_survival_fold(
                    train_df,
                    test_df,
                    time_col=time_col,
                    event_col=event_col,
                    risk_score=risk,
                    train_risk_score=train_risk,
                ),
                "n_test": int(len(fold.test_idx)),
                "warning": (
                    f"lifelines convergence warning: {str(conv_warn[0].message)}"
                    if conv_warn
                    else None
                ),
                "model_spec": model_spec,
            }
        )
        _append_survival_plot_fold(
            plot_payload,
            test_df,
            time_col=time_col,
            event_col=event_col,
            risk_score=risk,
        )

    return _finalize_cv_result(
        contender="python_lifelines_coxph_enet",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_glmnet_cox_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    with _workspace_tempdir(prefix="gam_bench_glmnet_cox_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_glmnet_cox_cv.R"

        payload = {
            "dataset": ds,
            "scenario_name": scenario["name"],
        }
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(glmnet)
  library(survival)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
time_col <- as.character(payload$dataset$time_col)
event_col <- as.character(payload$dataset$event_col)
feature_cols <- as.character(payload$dataset$features)

coerce_positive_times <- function(x) {
  vals <- as.numeric(x)
  bad <- !is.finite(vals) | vals <= 0
  if (!any(bad)) {
    return(vals)
  }
  pos <- vals[is.finite(vals) & vals > 0]
  if (!length(pos)) {
    stop("No strictly positive survival times available for Cox glmnet")
  }
  replacement <- max(min(pos) * 0.5, 1e-12)
  vals[bad] <- replacement
  vals
}

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L
train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
train_df[[time_col]] <- coerce_positive_times(train_df[[time_col]])
test_df[[time_col]] <- coerce_positive_times(test_df[[time_col]])

for (cn in feature_cols) {
  mu <- mean(train_df[[cn]])
  sdv <- stats::sd(train_df[[cn]])
  if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
  train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
  test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
}

x_train <- as.matrix(train_df[, feature_cols, drop=FALSE])
x_test <- as.matrix(test_df[, feature_cols, drop=FALSE])
y_train <- survival::Surv(
  time=as.numeric(train_df[[time_col]]),
  event=as.numeric(train_df[[event_col]]) > 0.5
)

t0 <- proc.time()[["elapsed"]]
cvfit <- cv.glmnet(
  x=x_train,
  y=y_train,
  family="cox",
  alpha=0.5,
  nfolds=5,
  standardize=FALSE
)
fit_sec <- proc.time()[["elapsed"]] - t0

pred_t0 <- proc.time()[["elapsed"]]
lp_train <- as.numeric(predict(cvfit, newx=x_train, s="lambda.min", type="link"))
lp_test <- as.numeric(predict(cvfit, newx=x_test, s="lambda.min", type="link"))
risk <- as.numeric(lp_test)
pred_sec <- proc.time()[["elapsed"]] - pred_t0

out <- list(
  status="ok",
  fit_sec=fit_sec,
  predict_sec=pred_sec,
  auc=NULL,
  train_risk=as.numeric(lp_train),
  risk=risk,
  brier=NULL,
  logloss=NULL,
  rmse=NULL,
  mae=NULL,
  r2=NULL,
  model_spec="cv.glmnet(family='cox', alpha=0.5, s='lambda.min')"
)
write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        plot_payload = _init_plot_payload(ds)
        all_df = pd.DataFrame(ds["rows"])
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_glmnet_cox",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": err.strip() or out.strip(),
                }

            fold_row = json.loads(out_path.read_text())
            train_df = all_df.iloc[fold.train_idx].copy()
            test_df = all_df.iloc[fold.test_idx].copy()
            event_times = test_df[ds["time_col"]].to_numpy(dtype=float)
            risk = np.asarray(fold_row.get("risk", []), dtype=float).reshape(-1)
            train_risk = np.asarray(fold_row.get("train_risk", []), dtype=float).reshape(-1)
            if risk.shape[0] != event_times.shape[0]:
                return {
                    "contender": "r_glmnet_cox",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": (
                        "r_glmnet_cox fold output missing/invalid risk vector "
                        f"(got {risk.shape[0]}, expected {event_times.shape[0]})"
                    ),
                }
            if train_risk.shape[0] != len(fold.train_idx):
                return {
                    "contender": "r_glmnet_cox",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": (
                        "r_glmnet_cox fold output missing/invalid train risk vector "
                        f"(got {train_risk.shape[0]}, expected {len(fold.train_idx)})"
                    ),
                }
            surv_metrics = score_survival_fold(
                train_df,
                test_df,
                time_col=ds["time_col"],
                event_col=ds["event_col"],
                risk_score=risk,
                train_risk_score=train_risk,
            )
            fold_row["auc"] = surv_metrics["auc"]
            fold_row["brier"] = surv_metrics["brier"]
            fold_row["logloss"] = surv_metrics["logloss"]
            fold_row["nagelkerke_r2"] = surv_metrics["nagelkerke_r2"]
            fold_row.pop("risk", None)
            fold_row.pop("train_risk", None)
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)
            _append_survival_plot_fold(
                plot_payload,
                test_df,
                time_col=ds["time_col"],
                event_col=ds["event_col"],
                risk_score=risk,
            )

    return _finalize_cv_result(
        contender="r_glmnet_cox",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )


def run_external_sksurv_gb_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        sksurv_ensemble: typing.Any = importlib.import_module("sksurv.ensemble")
        sksurv_util: typing.Any = importlib.import_module("sksurv.util")
        ComponentwiseGradientBoostingSurvivalAnalysis = (
            sksurv_ensemble.ComponentwiseGradientBoostingSurvivalAnalysis
        )
        GradientBoostingSurvivalAnalysis = sksurv_ensemble.GradientBoostingSurvivalAnalysis
        Surv = sksurv_util.Surv
    except _EXPECTED_OPTIONAL_IMPORT_FAILURES as e:
        return {
            "contender": "python_sksurv_gb_coxph",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"scikit-survival import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]

    gb_rows = []
    cgb_rows = []
    gb_plot_payload = _init_plot_payload(ds)
    cgb_plot_payload = _init_plot_payload(ds)
    gb_spec = "GradientBoostingSurvivalAnalysis(loss='coxph', n_estimators=300, lr=0.05, max_depth=3)"
    cgb_spec = "ComponentwiseGradientBoostingSurvivalAnalysis(loss='coxph', n_estimators=500, lr=0.05)"
    for fold in folds:
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        x_train = train_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_train = Surv.from_arrays(
            event=train_df[event_col].to_numpy(dtype=float) > 0.5,
            time=train_df[time_col].to_numpy(dtype=float),
        )

        gb_model = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=CV_SEED,
        )
        fit_start = datetime.now(timezone.utc)
        gb_model.fit(x_train, y_train)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()
        pred_start = datetime.now(timezone.utc)
        train_gb_risk = gb_model.predict(x_train).astype(float)
        gb_risk = gb_model.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        gb_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                **score_survival_fold(
                    train_df,
                    test_df,
                    time_col=time_col,
                    event_col=event_col,
                    risk_score=gb_risk,
                    train_risk_score=train_gb_risk,
                ),
                "n_test": int(len(fold.test_idx)),
                "model_spec": gb_spec,
            }
        )
        _append_survival_plot_fold(
            gb_plot_payload,
            test_df,
            time_col=time_col,
            event_col=event_col,
            risk_score=gb_risk,
        )

        cgb_model = ComponentwiseGradientBoostingSurvivalAnalysis(
            loss="coxph",
            n_estimators=500,
            learning_rate=0.05,
            random_state=CV_SEED,
        )
        fit_start = datetime.now(timezone.utc)
        cgb_model.fit(x_train, y_train)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()
        pred_start = datetime.now(timezone.utc)
        train_cgb_risk = cgb_model.predict(x_train).astype(float)
        cgb_risk = cgb_model.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        cgb_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                **score_survival_fold(
                    train_df,
                    test_df,
                    time_col=time_col,
                    event_col=event_col,
                    risk_score=cgb_risk,
                    train_risk_score=train_cgb_risk,
                ),
                "n_test": int(len(fold.test_idx)),
                "model_spec": cgb_spec,
            }
        )
        _append_survival_plot_fold(
            cgb_plot_payload,
            test_df,
            time_col=time_col,
            event_col=event_col,
            risk_score=cgb_risk,
        )

    return [
        _finalize_cv_result(
            contender="python_sksurv_gb_coxph",
            scenario_name=scenario["name"],
            family=ds["family"],
            cv_rows=gb_rows,
            plot_payload=gb_plot_payload,
            model_spec=f"{gb_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
        ),
        _finalize_cv_result(
            contender="python_sksurv_componentwise_gb_coxph",
            scenario_name=scenario["name"],
            family=ds["family"],
            cv_rows=cgb_rows,
            plot_payload=cgb_plot_payload,
            model_spec=f"{cgb_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
        ),
    ]


def run_external_lifelines_aft_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        LogNormalAFTFitter, WeibullAFTFitter, ConvergenceWarning = _require_lifelines_aft_fitters()
    except _EXPECTED_OPTIONAL_IMPORT_FAILURES as e:
        return {
            "contender": "python_lifelines_weibull_aft",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"lifelines import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]

    weibull_rows = []
    lognormal_rows = []
    weibull_plot_payload = _init_plot_payload(ds)
    lognormal_plot_payload = _init_plot_payload(ds)
    weibull_spec = "WeibullAFTFitter(train-fold z-score; penalizer=1e-3; l1_ratio=0.2)"
    lognormal_spec = "LogNormalAFTFitter(train-fold z-score; penalizer=1e-3; l1_ratio=0.2)"
    for fold in folds:
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)

        weibull = WeibullAFTFitter(penalizer=1e-3, l1_ratio=0.2)
        fit_start = datetime.now(timezone.utc)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            weibull.fit(
                train_df[[*feature_cols, time_col, event_col]],
                duration_col=time_col,
                event_col=event_col,
            )
        conv_warn = [w for w in wlist if issubclass(w.category, ConvergenceWarning)]
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()
        pred_start = datetime.now(timezone.utc)
        train_pred_time = weibull.predict_expectation(train_df[feature_cols]).to_numpy(dtype=float).reshape(-1)
        pred_time = weibull.predict_expectation(test_df[feature_cols]).to_numpy(dtype=float).reshape(-1)
        train_risk = -train_pred_time
        risk = -pred_time
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        weibull_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                **score_survival_fold(
                    train_df,
                    test_df,
                    time_col=time_col,
                    event_col=event_col,
                    risk_score=risk,
                    train_risk_score=train_risk,
                ),
                "n_test": int(len(fold.test_idx)),
                "warning": (
                    f"lifelines convergence warning: {str(conv_warn[0].message)}"
                    if conv_warn
                    else None
                ),
                "model_spec": weibull_spec,
            }
        )
        _append_survival_plot_fold(
            weibull_plot_payload,
            test_df,
            time_col=time_col,
            event_col=event_col,
            risk_score=risk,
            pred_time=pred_time,
        )

        lognormal = LogNormalAFTFitter(penalizer=1e-3, l1_ratio=0.2)
        fit_start = datetime.now(timezone.utc)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            lognormal.fit(
                train_df[[*feature_cols, time_col, event_col]],
                duration_col=time_col,
                event_col=event_col,
            )
        conv_warn = [w for w in wlist if issubclass(w.category, ConvergenceWarning)]
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()
        pred_start = datetime.now(timezone.utc)
        train_pred_time = lognormal.predict_expectation(train_df[feature_cols]).to_numpy(dtype=float).reshape(-1)
        pred_time = lognormal.predict_expectation(test_df[feature_cols]).to_numpy(dtype=float).reshape(-1)
        train_risk = -train_pred_time
        risk = -pred_time
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
        lognormal_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                **score_survival_fold(
                    train_df,
                    test_df,
                    time_col=time_col,
                    event_col=event_col,
                    risk_score=risk,
                    train_risk_score=train_risk,
                ),
                "n_test": int(len(fold.test_idx)),
                "warning": (
                    f"lifelines convergence warning: {str(conv_warn[0].message)}"
                    if conv_warn
                    else None
                ),
                "model_spec": lognormal_spec,
            }
        )
        _append_survival_plot_fold(
            lognormal_plot_payload,
            test_df,
            time_col=time_col,
            event_col=event_col,
            risk_score=risk,
            pred_time=pred_time,
        )

    return [
        _finalize_cv_result(
            contender="python_lifelines_weibull_aft",
            scenario_name=scenario["name"],
            family=ds["family"],
            cv_rows=weibull_rows,
            plot_payload=weibull_plot_payload,
            model_spec=f"{weibull_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
        ),
        _finalize_cv_result(
            contender="python_lifelines_lognormal_aft",
            scenario_name=scenario["name"],
            family=ds["family"],
            cv_rows=lognormal_rows,
            plot_payload=lognormal_plot_payload,
            model_spec=f"{lognormal_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
        ),
    ]


def run_external_xgboost_aft_cv(scenario: typing.Any, *, ds: dict[str, typing.Any] | None = None, folds: list[Fold] | None = None) -> typing.Any:
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    try:
        xgb = importlib.import_module("xgboost")
    except _EXPECTED_OPTIONAL_IMPORT_FAILURES as e:
        return {
            "contender": "python_xgboost_aft",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": f"xgboost import failed: {e}",
        }

    if folds is None:
        folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    feature_cols = ds["features"]
    time_col = ds["time_col"]
    event_col = ds["event_col"]
    cv_rows = []
    plot_payload = _init_plot_payload(ds)

    for fold_id, fold in enumerate(folds):
        train_df = df.iloc[fold.train_idx].copy()
        test_df = df.iloc[fold.test_idx].copy()
        train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)

        x_train = train_df[feature_cols].to_numpy(dtype=float)
        x_test = test_df[feature_cols].to_numpy(dtype=float)
        y_time = train_df[time_col].to_numpy(dtype=float)
        y_event = train_df[event_col].to_numpy(dtype=float) > 0.5
        dtest = xgb.DMatrix(x_test)

        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bynode": 0.8,
            "lambda": 1.0,
            "alpha": 0.0,
            "seed": int(CV_SEED + fold_id),
            "nthread": 1,
        }

        fit_start = datetime.now(timezone.utc)
        dtrain = xgb.DMatrix(x_train)
        dtrain.set_float_info("label_lower_bound", y_time.copy())
        dtrain.set_float_info("label_upper_bound", np.where(y_event, y_time, np.inf))
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=300,
            verbose_eval=False,
        )
        selected_rounds = int(booster.num_boosted_rounds())
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        train_pred_time = booster.predict(dtrain).astype(float).reshape(-1)
        pred_time = booster.predict(dtest).astype(float).reshape(-1)
        train_risk = -train_pred_time
        risk = -pred_time
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()

        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                **score_survival_fold(
                    train_df,
                    test_df,
                    time_col=time_col,
                    event_col=event_col,
                    risk_score=risk,
                    train_risk_score=train_risk,
                ),
                "n_test": int(len(fold.test_idx)),
                "model_spec": (
                    "xgboost.train(objective='survival:aft',loss='normal',scale=1.0,"
                    f"max_depth=3,eta=0.05,selected_rounds={selected_rounds})"
                ),
            }
        )
        _append_survival_plot_fold(
            plot_payload,
            test_df,
            time_col=time_col,
            event_col=event_col,
            risk_score=risk,
            pred_time=pred_time,
        )

    return _finalize_cv_result(
        contender="python_xgboost_aft",
        scenario_name=scenario["name"],
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=f"{cv_rows[0]['model_spec']} {_evaluation_suffix(folds)}",
    )



