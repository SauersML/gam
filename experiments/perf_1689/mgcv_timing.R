# #1689 mgcv reference-timing arm for the gam phase-profiling harness
# (examples/perf_1689.rs). Times mgcv on the SAME data-generating processes and
# shapes so the coordinator can pair each gam row with its mgcv baseline and read
# the true slowdown multiple at equal predictive accuracy.
#
#   ps  s(x, bs='ps', k=12)   n in {1e3, 1e4, 1e5}
#   tp  s(x, z, bs='tp')      n in {2e3, 2e4}
#
# All fits: family=gaussian, method="REML", select=TRUE (matching gam's default
# Marra-Wood double penalty). Fit and predict are timed SEPARATELY, best of 3.
# RMSE-vs-truth on the same test grids the Rust harness uses is reported so the
# "equal accuracy" premise is verifiable, not assumed.
#
# Deterministic: fixed set.seed per case. Self-contained: no jsonlite dependency
# (JSON is emitted by hand), so a bare mgcv install runs it. Run with:
#   R_LIBS_USER=/projects/standard/hsiehph/sauer354/Rlib422 Rscript experiments/perf_1689/mgcv_timing.R
#
# Output: a JSON array of per-case objects on stdout (last line), plus a
# human-readable line per case on stderr for live monitoring.

suppressPackageStartupMessages(library(mgcv))

REPS <- 3L

# Best-of-REPS elapsed seconds for fitting `formula` on `df`, plus the fit object
# and separately-timed predict seconds on `newdata`.
time_case <- function(formula, df, newdata) {
  best_fit <- Inf
  best_pred <- Inf
  fit <- NULL
  pr <- NULL
  for (i in seq_len(REPS)) {
    t0 <- proc.time()[["elapsed"]]
    f <- gam(formula, family = gaussian(), data = df, method = "REML", select = TRUE)
    fit_s <- proc.time()[["elapsed"]] - t0
    t1 <- proc.time()[["elapsed"]]
    p <- as.numeric(predict(f, newdata = newdata, type = "response"))
    pred_s <- proc.time()[["elapsed"]] - t1
    if (fit_s < best_fit) { best_fit <- fit_s; fit <- f }
    if (pred_s < best_pred) { best_pred <- pred_s; pr <- p }
  }
  list(fit_s = best_fit, predict_s = best_pred, fit = fit, pred = pr)
}

rmse <- function(a, b) sqrt(mean((a - b)^2))

results <- list()

# --- 1-D P-spline: truth = sin(5x)+0.5x, noise N(0,0.2), x ~ U(0,1) ---
ps_truth <- function(x) sin(5 * x) + 0.5 * x
xt <- seq(0, 1, length.out = 300)
ft <- ps_truth(xt)
te_ps <- data.frame(x = xt)
for (n in c(1000L, 10000L, 100000L)) {
  set.seed(0)
  x <- runif(n, 0, 1)
  y <- ps_truth(x) + rnorm(n, 0, 0.2)
  df <- data.frame(x = x, y = y)
  r <- time_case(y ~ s(x, bs = "ps", k = 12), df, te_ps)
  edf <- sum(r$fit$edf)
  rm <- rmse(r$pred, ft)
  results[[length(results) + 1L]] <- list(
    tag = sprintf("ps-1d-n%d", n), n = n, term = "s(x,bs='ps',k=12)",
    edf = edf, fit_s = r$fit_s, predict_s = r$predict_s, rmse_vs_truth = rm
  )
  cat(sprintf("[mgcv #1689] %-14s edf=%.1f fit=%.3fs predict=%.3fs rmse=%.4f\n",
              sprintf("ps-1d-n%d", n), edf, r$fit_s, r$predict_s, rm), file = stderr())
}

# --- 2-D thin-plate: truth = gaussian bump, noise N(0,0.1), x,z ~ U(0,1) ---
tp_truth <- function(x, z) exp(-((x - 0.5)^2 + (z - 0.5)^2) / 0.1)
g <- seq(0, 1, length.out = 30)
grid <- expand.grid(x = g, z = g)
ft2 <- tp_truth(grid$x, grid$z)
for (n in c(2000L, 20000L)) {
  set.seed(0)
  x <- runif(n, 0, 1)
  z <- runif(n, 0, 1)
  y <- tp_truth(x, z) + rnorm(n, 0, 0.1)
  df <- data.frame(x = x, z = z, y = y)
  r <- time_case(y ~ s(x, z, bs = "tp"), df, grid)
  edf <- sum(r$fit$edf)
  rm <- rmse(r$pred, ft2)
  results[[length(results) + 1L]] <- list(
    tag = sprintf("tp-2d-n%d", n), n = n, term = "s(x,z,bs='tp')",
    edf = edf, fit_s = r$fit_s, predict_s = r$predict_s, rmse_vs_truth = rm
  )
  cat(sprintf("[mgcv #1689] %-14s edf=%.1f fit=%.3fs predict=%.3fs rmse=%.4f\n",
              sprintf("tp-2d-n%d", n), edf, r$fit_s, r$predict_s, rm), file = stderr())
}

# Emit a JSON array by hand (no jsonlite dependency).
esc <- function(s) gsub('"', '\\\\"', s)
obj_json <- function(o) {
  sprintf(
    '{"tag":"%s","n":%d,"term":"%s","edf":%.6g,"fit_s":%.6g,"predict_s":%.6g,"rmse_vs_truth":%.6g}',
    esc(o$tag), o$n, esc(o$term), o$edf, o$fit_s, o$predict_s, o$rmse_vs_truth
  )
}
cat("[", paste(vapply(results, obj_json, character(1)), collapse = ","), "]\n", sep = "")
