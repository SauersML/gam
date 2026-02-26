#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("mgcv", quietly = TRUE)) {
    install.packages("mgcv", repos = "https://cloud.r-project.org")
  }
  if (!requireNamespace("gamair", quietly = TRUE)) {
    install.packages("gamair", repos = "https://cloud.r-project.org")
  }
  library(mgcv)
  library(gamair)
})

data(prostate, package = "gamair")

out_dir <- file.path(getwd(), "benchmarks")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Panel 1: six spectra traces (2 x 3 layout), like the docs example.
png(
  filename = file.path(out_dir, "prostate_spectra.png"),
  width = 1800,
  height = 1200,
  res = 180
)
par(mfrow = c(2, 3), mar = c(5, 5, 3, 1))
ind <- c(1, 163, 319)
lab <- list("Healthy", "Enlarged", "Cancer")
for (i in 1:3) {
  plot(
    prostate$MZ[ind[i], ],
    prostate$intensity[ind[i], ],
    type = "l",
    ylim = c(0, 60),
    xlab = "Daltons",
    ylab = "Intensity",
    main = lab[[i]],
    cex.axis = 1.2,
    cex.lab = 1.3
  )
  lines(prostate$MZ[ind[i], ], prostate$intensity[ind[i] + 2, ] + 5, col = 2)
  lines(prostate$MZ[ind[i], ], prostate$intensity[ind[i] + 4, ] + 10, col = 4)
}
dev.off()

# Panel 2: ordered categorical GAM fit and diagnostic-style plots.
b <- gam(
  type ~ s(MZ, by = intensity, k = 100),
  family = ocat(R = 3),
  data = prostate,
  method = "ML"
)
pb <- predict(b, type = "response")

png(
  filename = file.path(out_dir, "prostate_gam_panels.png"),
  width = 1800,
  height = 600,
  res = 180
)
par(mfrow = c(1, 3), mar = c(5, 5, 3, 1))
plot(
  b,
  rug = FALSE,
  scheme = 1,
  xlab = "Daltons",
  ylab = "f(D)",
  cex.lab = 1.3,
  cex.axis = 1.2,
  main = "a"
)
plot(
  factor(prostate$type),
  pb[, 3],
  xlab = "Type",
  ylab = "Pr(Cancer)",
  cex.lab = 1.3,
  cex.axis = 1.2,
  main = "b"
)
qq.gam(
  b,
  rep = 100,
  lev = 0.95,
  cex.lab = 1.3,
  cex.axis = 1.2,
  main = "c"
)
dev.off()

cat("Wrote:\n")
cat(file.path(out_dir, "prostate_spectra.png"), "\n")
cat(file.path(out_dir, "prostate_gam_panels.png"), "\n")
