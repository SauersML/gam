# Usage: Rscript tensor_reference_audit.R DIRECTORY_CONTAINING_EXPORTED_CSVS
# Preserve the original P-spline comparator and add natural-cubic diagnostics.
suppressPackageStartupMessages(library(mgcv))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
interaction <- function(v, nx, nz) {
  m <- matrix(v, nrow = nx, ncol = nz, byrow = TRUE)
  as.numeric(t(sweep(sweep(m, 1, rowMeans(m)), 2, colMeans(m)) + mean(m)))
}
for (family_name in c("poisson", "gaussian")) {
  df <- read.csv(file.path(args[1], paste0(family_name, "-tensor.csv")))
  poisson_case <- family_name == "poisson"
  target <- if (poisson_case) df$truth else interaction(df$truth, 18, 18)
  score <- function(v) sqrt(mean(((if (poisson_case) v else interaction(v, 18, 18)) - target)^2))
  if ("gam" %in% names(df)) {
    cat(sprintf("family=%s GAM_rmse=%.12g\n", family_name, score(df$gam)))
  } else {
    cat(sprintf("family=%s GAM_fit_unavailable\n", family_name))
  }
  for (basis in c("ps", "cr")) {
    formula <- if (poisson_case) y ~ te(x, z, bs = basis, k = c(6, 6)) else y ~ ti(x, z, bs = basis, k = c(6, 6))
    for (null_shrinkage in c(FALSE, TRUE)) {
      m <- gam(formula, data = df, family = if (poisson_case) poisson() else gaussian(),
               method = "REML", select = null_shrinkage)
      cat(sprintf("family=%s basis=%s select=%s rmse=%.12g edf=%.12g sp=%s\n", family_name,
                  basis, null_shrinkage, score(fitted(m)), sum(m$edf),
                  paste(format(m$sp, digits = 12), collapse = ",")))
      if ("gam" %in% names(df)) {
        cat(sprintf("  max_abs_prediction_difference_from_GAM=%.12g\n", max(abs(fitted(m) - df$gam))))
      }
    }
  }
}
cat("R=", R.version.string, " mgcv=", as.character(packageVersion("mgcv")), "\n", sep = "")
