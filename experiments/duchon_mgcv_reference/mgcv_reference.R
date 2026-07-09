#!/usr/bin/env Rscript
# mgcv reference numbers for the three RED Duchon λ-selection fixtures
# (#1815 duchon_hilbert_scale::{deselection_recovers_truth, null_recovery},
#  #1818 duchon_sin8_quality::duchon_sin8_max_error_within_budget).
#
# Purpose: hand the gam side the mature-Duchon (mgcv bs="ds", m=c(2,0), REML)
# reference for each fixture — max error vs truth, summed EDF, and the smoothing
# parameter sp (= 1/λ scale mgcv reports) — so the gam over-smoothing / failed
# REML-deselection can be calibrated against a known-good target rather than a
# guessed bound. Emits one JSON blob to stdout.
#
# DATA FIDELITY. The Rust fixtures draw noise from rand's StdRng; R's RNG is a
# different stream, so this script cannot reproduce the exact noise realization
# from the same integer seed. It reproduces the data-generating PROCESS faithfully:
#   * x is DETERMINISTIC in two of the three fixtures (a uniform grid i/(n-1)),
#     reproduced here EXACTLY; only the noise realization differs.
#   * sin8 draws x ~ sort(Unif(0,1)); reproduced as a process, not bit-for-bit.
# The summed EDF / sp / amplitude are stable process-level statistics (they do
# not hinge on the specific noise draw), so they are valid calibration targets.
# For a BIT-EXACT run, dump the Rust fixture columns to CSV (x,y with header) and
# point the matching env-style path below at them; the script uses the CSV verbatim
# when the file exists, else self-generates. Pass the data dir as argv[1]
# (default: alongside this script). CSV names: sin8.csv, deselection.csv, null.csv.
#
#   Rscript mgcv_reference.R [data_dir]
#
# Requires: R + mgcv (MSI: gam_env.sh sets R_LIBS_USER with mgcv 1.8-42).

suppressPackageStartupMessages(library(mgcv))

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "."
}

# ---- helpers ---------------------------------------------------------------
maybe_csv <- function(name) {
  p <- file.path(data_dir, name)
  if (file.exists(p)) {
    d <- read.csv(p)
    stopifnot(all(c("x", "y") %in% names(d)))
    message(sprintf("[mgcv-ref] using exact fixture CSV: %s (n=%d)", p, nrow(d)))
    list(x = as.numeric(d$x), y = as.numeric(d$y), exact = TRUE)
  } else {
    NULL
  }
}

# Fit mgcv Duchon bs="ds" m=c(2,0) REML; return the reference stats on a truth
# grid. `truth_fun` may be NULL (null-recovery: no signal, report flatness only).
fit_ref <- function(x, y, k, x_grid, truth_fun) {
  train <- data.frame(x = x, y = y)
  m <- gam(y ~ s(x, bs = "ds", k = k, m = c(2, 0)), data = train, method = "REML")
  grid <- data.frame(x = x_grid)
  pred <- as.numeric(predict(m, newdata = grid))
  sum_edf <- as.numeric(sum(m$edf))
  sp <- as.numeric(m$sp)               # mgcv smoothing parameter(s)
  amp <- max(pred) - min(pred)         # fitted peak-to-peak on the grid
  mean_fit <- mean(pred)
  max_dev_from_mean <- max(abs(pred - mean_fit))
  out <- list(
    k = k, n_train = length(x), sum_edf = sum_edf, sp = as.list(sp),
    fitted_amplitude = amp, max_dev_from_mean = max_dev_from_mean,
    reml_score = as.numeric(m$gcv.ubre)
  )
  if (!is.null(truth_fun)) {
    truth <- truth_fun(x_grid)
    out$max_err_vs_truth <- max(abs(pred - truth))
    out$truth_rmse <- sqrt(mean((pred - truth)^2))
  }
  out
}

# ---- fixture 1: sin8 (#1818) ----------------------------------------------
# Rust: make_sin_dataset(freq=8, sigma=0.10, n=240, seed=11):
#   x = sort(runif(240)); y = sin(2*pi*8*x) + N(0, 0.10).
# x_test = 0.001 + 0.998*i/399, i=0..399; truth = sin(2*pi*8*x_test).
sin8 <- function() {
  freq <- 8; sigma <- 0.10; n <- 240
  d <- maybe_csv("sin8.csv")
  if (is.null(d)) {
    set.seed(11)                       # process-matched (see DATA FIDELITY note)
    x <- sort(runif(n))
    y <- sin(2 * pi * freq * x) + rnorm(n, 0, sigma)
    d <- list(x = x, y = y, exact = FALSE)
  }
  x_grid <- 0.001 + 0.998 * (0:399) / 399
  tf <- function(t) sin(2 * pi * freq * t)
  # k=40 is the resolved default duchon(x); centers=50 -> k=50; centers=20 -> k=20.
  list(
    exact = d$exact,
    k40 = fit_ref(d$x, d$y, 40, x_grid, tf),
    k50 = fit_ref(d$x, d$y, 50, x_grid, tf),
    k20 = fit_ref(d$x, d$y, 20, x_grid, tf)
  )
}

# ---- fixture 2: deselection_recovers_truth (#1815) -------------------------
# Rust: n=200, x = i/(n-1) (DETERMINISTIC grid), y = sin(2*pi*x) + N(0,0.05),
#       seed=123. x_test = 0.005 + 0.99*i/200, i=0..200 (m=201). truth=sin(2*pi*x).
deselection <- function() {
  n <- 200; sigma <- 0.05
  d <- maybe_csv("deselection.csv")
  if (is.null(d)) {
    x <- (0:(n - 1)) / (n - 1)         # exact deterministic grid
    set.seed(123)
    y <- sin(2 * pi * x) + rnorm(n, 0, sigma)
    d <- list(x = x, y = y, exact = FALSE)
  }
  x_grid <- 0.005 + 0.99 * (0:200) / 200
  tf <- function(t) sin(2 * pi * t)
  list(exact = d$exact, k20 = fit_ref(d$x, d$y, 20, x_grid, tf))
}

# ---- fixture 3: null_recovery (#1815) -------------------------------------
# Rust: n=300, x = i/(n-1) (DETERMINISTIC grid), y = N(0,1) INDEPENDENT of x,
#       seed=98765. x_test = i/200, i=0..200 (m=201). No truth signal: the target
#       is collapse, so report summed EDF (should be ~1-2) and fitted flatness.
null_recovery <- function() {
  n <- 300; sigma <- 1.0
  d <- maybe_csv("null.csv")
  if (is.null(d)) {
    x <- (0:(n - 1)) / (n - 1)         # exact deterministic grid
    set.seed(98765)
    y <- rnorm(n, 0, sigma)
    d <- list(x = x, y = y, exact = FALSE)
  }
  x_grid <- (0:200) / 200
  # truth_fun = NULL: null case reports flatness (max_dev_from_mean) + EDF.
  list(exact = d$exact, sigma = sigma, k20 = fit_ref(d$x, d$y, 20, x_grid, NULL))
}

result <- list(
  meta = list(
    comparator = "mgcv gam s(x, bs='ds', m=c(2,0)) method='REML'",
    note = "sp is mgcv's smoothing parameter; sum_edf and sp are the calibration targets. null.k20.sum_edf collapsing to ~1-2 with large sp is the mgcv baseline gam must match.",
    mgcv_version = as.character(packageVersion("mgcv"))
  ),
  sin8 = sin8(),
  deselection = deselection(),
  null = null_recovery()
)

# Minimal base-R JSON serializer (avoid a jsonlite dependency: only mgcv is
# guaranteed on MSI). Handles nested lists, numeric/logical/character scalars,
# and numeric vectors; that covers everything emitted above.
to_json <- function(x, indent = 0) {
  pad <- strrep("  ", indent)
  pad1 <- strrep("  ", indent + 1)
  if (is.list(x)) {
    nms <- names(x)
    if (is.null(nms)) {                       # unnamed list -> JSON array
      if (length(x) == 0) return("[]")
      items <- vapply(x, function(v) paste0(pad1, to_json(v, indent + 1)), "")
      return(paste0("[\n", paste(items, collapse = ",\n"), "\n", pad, "]"))
    }
    if (length(x) == 0) return("{}")
    items <- vapply(seq_along(x), function(i) {
      paste0(pad1, "\"", nms[i], "\": ", to_json(x[[i]], indent + 1))
    }, "")
    return(paste0("{\n", paste(items, collapse = ",\n"), "\n", pad, "}"))
  }
  if (is.character(x)) {
    esc <- gsub("\"", "\\\\\"", x)
    if (length(x) == 1) return(paste0("\"", esc, "\""))
    return(paste0("[", paste0("\"", esc, "\"", collapse = ", "), "]"))
  }
  if (is.logical(x)) {
    v <- ifelse(is.na(x), "null", tolower(as.character(x)))
    if (length(x) == 1) return(v)
    return(paste0("[", paste(v, collapse = ", "), "]"))
  }
  # numeric
  fmt <- function(v) ifelse(is.na(v) | !is.finite(v), "null", formatC(v, digits = 10, format = "g"))
  if (length(x) == 1) return(fmt(x))
  paste0("[", paste(vapply(x, fmt, ""), collapse = ", "), "]")
}

cat(to_json(result), "\n")
