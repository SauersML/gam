# Compare Rust PG(1,z) sampler with BayesLogit reference
options(warn = -1)

suppressPackageStartupMessages({
  library(BayesLogit)
  library(readr)
  library(ggplot2)
  library(dplyr)
  library(e1071)
})

# Parse command line args for z_grid, --seed, and --B
args <- commandArgs(trailingOnly = TRUE)
default_z_grid <- c(0.5, 1, 2, 3.2, 5)
z_grid <- numeric(0)
seed <- 1 # Default seed
B <- 1.0 # Default B value

# Process named arguments
i <- 1
while (i <= length(args)) {
  if (args[i] %in% c("--seed", "-s")) {
    seed_val <- as.numeric(args[i + 1])
    if (!is.na(seed_val)) seed <- seed_val
    i <- i + 2 # Skip the next argument (value)
  } else if (args[i] %in% c("--B", "-b", "--b")) {
    B_val <- as.numeric(args[i + 1])
    if (!is.na(B_val) && B_val > 0) B <- B_val
    i <- i + 2 # Skip the next argument (value)
  } else {
    # Treat as z value
    z_val <- as.numeric(args[i])
    if (!is.na(z_val)) z_grid <- c(z_grid, z_val)
    i <- i + 1
  }
}

if (length(z_grid) == 0) {
  z_grid <- default_z_grid
}

cat(sprintf(
  "Using parameters: B=%.1f, seed=%d, z_grid=c(%s)\n",
  B, seed, paste(z_grid, collapse = ", ")
), file = stderr())

set.seed(seed)
n <- 1e6
indir <- "examples/reference_implementation/data"
outdir <- "examples/reference_implementation/results"
outpdf <- file.path(outdir, "pg_comparison_plots.pdf")
outtxt <- file.path(outdir, "pg_comparison_results.txt")

mean_theory <- function(z) tanh(z / 2) / (2 * z)

# Time only the BayesLogit sampling

# Store diagnostics
all_df_sum <- list()
test_results <- list()
time_taken <- numeric(0)
one_z_check <- function(z, txt_con, csv) {
  rust <- csv[[paste0("z=", sprintf("%.1f", z))]]
  stopifnot(length(rust) == n)
  ref_time <- system.time({
    ref <- BayesLogit::rpg(num = n, h = B, z = z)
  })
  time_taken[[as.character(z)]] <<- ref_time["elapsed"]
  # basic numbers -----------------------------------------------------
  df_sum <- tibble(
    z     = z,
    which = c("Rust", "Reference"),
    mean  = c(mean(rust), mean(ref)),
    var   = c(var(rust), var(ref)),
    sd    = c(sd(rust), sd(ref)),
    skew  = c(skewness(rust), skewness(ref)),
    kurt  = c(kurtosis(rust), kurtosis(ref))
  )
  all_df_sum[[as.character(z)]] <<- df_sum

  cat("\n--------------------------------------------------\n", file = txt_con)
  cat(sprintf("PG(1, %.3g)  —  sample size = %d\n", z, n), file = txt_con)
  capture.output(print(df_sum, digits = 4), file = txt_con, append = TRUE)

  # theoretical mean
  cat(sprintf("Theoretical mean  = %.6f\n", mean_theory(z)), file = txt_con)

  # formal tests ------------------------------------------------------
  ttest <- t.test(rust, ref)
  kstest <- ks.test(rust, ref)
  t_corrected_p_values <- p.adjust(ttest$p.value, method = "bonferroni")
  ks_corrected_p_values <- p.adjust(kstest$p.value, method = "bonferroni")
  test_results[[as.character(z)]] <<- list(
    t_p = t_corrected_p_values,
    ks_p = ks_corrected_p_values
  )


  cat("\nTwo–sample t-test:\n", file = txt_con)
  capture.output(print(ttest), file = txt_con, append = TRUE)

  cat("\nKolmogorov–Smirnov test:\n", file = txt_con)
  capture.output(print(kstest), file = txt_con, append = TRUE)

  # plots -------------------------------------------------------------
  # Downsample for plotting
  plot_n <- min(25000, length(rust), length(ref))
  idx_rust <- sample.int(length(rust), plot_n)
  idx_ref <- sample.int(length(ref), plot_n)
  rust_plot <- rust[idx_rust]
  ref_plot <- ref[idx_ref]

  # QQ-plot
  qqplot(rust_plot, ref_plot,
    pch = 20,
    main = sprintf("QQ plot  PG(1, %.2f): Rust vs BayesLogit", z),
    xlab = "Rust quantiles", ylab = "Reference quantiles"
  )
  abline(0, 1, col = "red")

  # Overlaid kernel densities
  d <- bind_rows(
    data.frame(value = rust_plot, which = "Rust"),
    data.frame(value = ref_plot, which = "Reference")
  )
  plt <- ggplot(d, aes(x = value, color = which)) +
    geom_density(adjust = 1.2, size = 1) +
    theme_minimal() +
    labs(title = sprintf("PG(1, %.2f) – density comparison", z))

  print(plt)
}

# Color helper
color_pass <- function(x) paste0("\033[32m", x, "\033[39m")
color_fail <- function(x) paste0("\033[31m", x, "\033[39m")

# Open output connections
pdf(outpdf, width = 6, height = 4.5)
txt_con <- file(outtxt, open = "wt")
csv <- read_csv(file.path(indir, "pg_samples.csv"), show_col_types = FALSE)
# Run the checks and write output
for (z in z_grid) {
  one_z_check(z, txt_con, csv)
}

# Print summary at end
cat("\n\n========== SUMMARY ==========\n\n")

# Combine all_df_sum into one table
summary_df <- bind_rows(all_df_sum)
print(summary_df, digits = 4)
cat("\n")

# Print formal test results table
cat("Formal test results (Bonferroni-corrected p-values):\n")
cat(sprintf("%-8s  %-12s  %-12s  %-8s  %-8s\n", "z", "t-test p", "KS-test p", "t-test", "KS-test"))
for (z in z_grid) {
  res <- test_results[[as.character(z)]]
  tpass_raw <- if (res$t_p > 0.05) "PASS" else "FAIL"
  kspass_raw <- if (res$ks_p > 0.05) "PASS" else "FAIL"
  tpass <- if (tpass_raw == "PASS") color_pass(sprintf("%-8s", tpass_raw)) else color_fail(sprintf("%-8s", tpass_raw))
  kspass <- if (kspass_raw == "PASS") color_pass(sprintf("%-8s", kspass_raw)) else color_fail(sprintf("%-8s", kspass_raw))
  cat(sprintf("%-8.2f  %-12.3g  %-12.3g  %s  %s\n", z, res$t_p, res$ks_p, tpass, kspass))
}
cat("\nDiagnostics and plots written to results/. See the full output and visualizations there.\n")

cat("\n[BayesLogit] Cumulative sample generation time: ", sum(time_taken), "seconds\n")
