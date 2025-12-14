Start RStudio first.

This will assume you have a file ending in .sscore already. This example uses scores and phenotype for Alzheimer's disease, but other phenotypes could work too.

You might want to upload it e.g.:
```
import os, subprocess

bucket = os.getenv("WORKSPACE_BUCKET")
src = "arrays.sscore"
subprocess.run(["gsutil", "-m", "cp", src, f"{bucket}/arrays.sscore"], check=True)
```

Or download:
```
bucket <- Sys.getenv("WORKSPACE_BUCKET")
remote <- sprintf("%s/arrays.sscore", bucket)
local  <- "arrays.sscore"
system2("gsutil", c("-m", "cp", remote, local))
```

We want to compare a few models.

- 1.a. Prediction from linear PCs alone
- 1.b. Prediction from spline PCs alone
- 2.a  Prediction from linear scores alone
- 2.b  Prediction from spline scores alone
- 3.a  Predicting accuracy with linear PC
- 3.b  Predicting accuracy with PC splines
- 4.a  Predicting with linear scores and linear PCs
- 4.b  Predicting with spline scores and spline PCs

Let's define the cases and controls:
```
# Alzheimer's case definition in BigQuery + cohort counts (cases/controls)

library(bigrquery)

cdr_id  <- Sys.getenv("WORKSPACE_CDR")
project <- Sys.getenv("GOOGLE_PROJECT")

icd10_alz <- c("G30.0", "G30.1", "G30.8", "G30.9")
icd9_alz  <- "331.0"
codes_raw <- c(icd10_alz, icd9_alz)
codes     <- toupper(unique(c(codes_raw, gsub("\\.", "", codes_raw))))

# Case IDs (distinct person_id with AD codes in condition_source_value)
sql_cases <- sprintf(
  "SELECT DISTINCT CAST(person_id AS STRING) AS person_id
   FROM `%s.condition_occurrence`
   WHERE condition_source_value IS NOT NULL
     AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)",
  cdr_id
)

h_cases <- bq_project_query(
  x = project,
  query = sql_cases,
  use_legacy_sql = FALSE,
  parameters = list(codes = codes),
  location = "US"
)

cases <- unique(as.character(bq_table_download(h_cases, bigint = "character")$person_id))

# Counts from BigQuery: total persons, cases, controls (= total - cases)
sql_counts <- sprintf(
  "WITH cases AS (
     SELECT DISTINCT CAST(person_id AS STRING) AS person_id
     FROM `%s.condition_occurrence`
     WHERE condition_source_value IS NOT NULL
       AND UPPER(TRIM(condition_source_value)) IN UNNEST(@codes)
   ),
   tot AS (
     SELECT COUNT(DISTINCT CAST(person_id AS STRING)) AS n_total
     FROM `%s.person`
   )
   SELECT
     (SELECT COUNT(*) FROM cases)                   AS n_cases,
     (SELECT n_total FROM tot)                      AS n_total,
     (SELECT n_total FROM tot) - (SELECT COUNT(*) FROM cases) AS n_controls",
  cdr_id, cdr_id
)

h_counts <- bq_project_query(
  x = project,
  query = sql_counts,
  use_legacy_sql = FALSE,
  parameters = list(codes = codes),
  location = "US"
)

cnt <- bq_table_download(h_counts, bigint = "integer")
n_cases    <- as.integer(cnt$n_cases[1])
n_total    <- as.integer(cnt$n_total[1])
n_controls <- as.integer(cnt$n_controls[1])
prev       <- 100 * n_cases / n_total

cat(sprintf(
  "Total: %s\nCases: %s\nControls: %s\nPrevalence: %.3f%%\n",
  format(n_total, big.mark = ","),
  format(n_cases, big.mark = ","),
  format(n_controls, big.mark = ","),
  prev
))
```
The prevalence is 0.201%.

Let's check per-score accuracies:
```
install.packages("pROC")
# AUROC of each score column in arrays.sscore vs. case/control
library(data.table)
library(pROC)

ss <- fread("arrays.sscore", sep = "\t", showProgress = FALSE)
ids <- as.character(ss[[1]])
y <- as.integer(ids %chin% unique(as.character(cases)))  # 1 = case, 0 = control

score_cols <- grep("_AVG$", names(ss), value = TRUE)

res <- rbindlist(lapply(score_cols, function(cn) {
  x <- suppressWarnings(as.numeric(ss[[cn]]))
  ok <- is.finite(x)
  n1 <- sum(y[ok] == 1L); n0 <- sum(y[ok] == 0L)
  if (n1 == 0L || n0 == 0L) return(data.table())
  r <- roc(response = y[ok], predictor = x[ok], quiet = TRUE)
  data.table(
    score    = sub("_AVG$", "", cn),
    n        = sum(ok),
    cases    = n1,
    controls = n0,
    AUROC    = as.numeric(auc(r))
  )
}))

setorder(res, -AUROC)
print(res)
```

Let's remove all but the top 5 scores:
```
# Keep only the top-5 PGS columns (plus ID) in arrays.sscore

library(data.table)

keep <- c("PGS004146","PGS000015","PGS004898","PGS004869","PGS000332")

dt <- fread("arrays.sscore", sep = "\t", showProgress = FALSE)
idcol <- names(dt)[1]

pat <- paste0("^(", paste(keep, collapse = "|"), ")_")
cols_keep <- c(idcol, grep(pat, names(dt), value = TRUE, perl = TRUE))

dt <- dt[, ..cols_keep]
fwrite(dt, "arrays.sscore", sep = "\t", quote = FALSE)
```

Let's load the PCs:
```
library(data.table)

NUM_PCS <- 16
PCS_URI <- "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
project <- Sys.getenv("GOOGLE_PROJECT")

pcs_raw <- fread(
  cmd = sprintf("gsutil -u %s cat %s", shQuote(project), shQuote(PCS_URI)),
  sep = "\t",
  select = c("research_id", "pca_features"),
  showProgress = FALSE
)

parse_vec <- function(s, k = NUM_PCS) {
  v <- as.numeric(trimws(strsplit(sub("^\\[|\\]$", "", s), ",", fixed = TRUE)[[1]]))
  len <- length(v)
  if (len < k) v <- c(v, rep(NA_real_, k - len))
  v[1:k]
}

pcs_mat <- t(vapply(pcs_raw$pca_features, parse_vec, numeric(NUM_PCS)))
pc_cols <- paste0("PC", seq_len(NUM_PCS))

pc_df <- as.data.table(pcs_mat)
setnames(pc_df, pc_cols)
pc_df[, person_id := as.character(pcs_raw$research_id)]
setcolorder(pc_df, c("person_id", pc_cols))
```

Let's use them to predict Alzheimer's disease:
```
# Logistic model: Alzheimer's ~ 16 ancestry PCs

library(data.table)

pc_cols <- paste0("PC", 1:16)
case_ids <- unique(as.character(cases))

# build modeling frame
dt <- as.data.table(pc_df[, c("person_id", pc_cols), with = FALSE])
for (cn in pc_cols) set(dt, j = cn, value = as.numeric(dt[[cn]]))
dt[, y := as.integer(person_id %chin% case_ids)]

# drop rows with no PC information at all
nz <- rowSums(is.finite(as.matrix(dt[, ..pc_cols]))) > 0
dt <- dt[nz]

# mean-impute PCs (keeps all 16 PCs; drops any column that is entirely NA/Inf)
keep_cols <- pc_cols
for (cn in pc_cols) {
  v <- dt[[cn]]
  if (!any(is.finite(v))) { keep_cols <- setdiff(keep_cols, cn); next }
  m <- mean(v[is.finite(v)])
  v[!is.finite(v)] <- m
  set(dt, j = cn, value = v)
}
dt <- dt[, c("y", keep_cols), with = FALSE]

stopifnot(length(unique(dt$y)) == 2L, nrow(dt) > 0L, length(keep_cols) > 0L)

fit <- glm(y ~ ., data = as.data.frame(dt), family = binomial())
summary(fit)
```

PCs 2 and 4 are significantly associated with Alzheimer's disease.

Let's check the AUROC:
```
library(pROC)

p <- predict(fit, type = "response")
r <- roc(response = dt$y, predictor = p, quiet = TRUE)
cat(sprintf("AUROC: %.6f\n", as.numeric(auc(r))))
```
AUROC: 0.59518.

Let's try with each score:
```
# Logistic model: Alzheimer's (cases) ~ all PGS scores
library(data.table)
library(pROC)

# cases: character vector of case person_ids already loaded from BigQuery
case_ids <- unique(as.character(cases))

ss  <- fread("arrays.sscore", sep = "\t", showProgress = FALSE)
ids <- as.character(ss[[1]])
y   <- as.integer(ids %chin% case_ids)

pgs_cols <- grep("_AVG$", names(ss), value = TRUE)

X <- ss[, ..pgs_cols]
# coerce to numeric
for (j in seq_along(pgs_cols)) set(X, j = j, value = as.numeric(X[[j]]))

# drop columns that are entirely non-finite
keep <- vapply(X, function(v) any(is.finite(v)), logical(1))
X    <- X[, which(keep), with = FALSE]

# mean impute per column
for (j in seq_len(ncol(X))) {
  v <- X[[j]]
  m <- mean(v[is.finite(v)])
  v[!is.finite(v)] <- m
  set(X, j = j, value = v)
}

stopifnot(length(unique(y)) == 2L, nrow(X) == length(y), ncol(X) > 0L)

df       <- data.frame(y = y, X, check.names = FALSE)
fit_pgs  <- glm(y ~ ., data = df, family = binomial())
p_hat    <- as.numeric(predict(fit_pgs, type = "response"))
roc_obj  <- roc(response = y, predictor = p_hat, quiet = TRUE)
auc_val  <- as.numeric(auc(roc_obj))

cat(sprintf("Predictors: %d\nAUROC (PGS-only): %.6f\n", ncol(X), auc_val))
```

AUROC: 0.589798.

Let's check how well we can predict individual-level accuracy with PCs:
```
library(data.table)

pc_cols <- paste0("PC", 1:16)

pred_dt <- data.table(
  person_id = as.character(ids),
  err_abs   = abs(as.integer(y) - as.numeric(predict(fit_pgs, type = "response")))
)

Z <- as.data.table(pc_df[, c("person_id", pc_cols), with = FALSE])
Z[, person_id := as.character(person_id)]
for (cn in pc_cols) set(Z, j = cn, value = as.numeric(Z[[cn]]))

dt <- merge(pred_dt, Z, by = "person_id", all = FALSE)

keep <- vapply(pc_cols, function(cn) any(is.finite(dt[[cn]])), logical(1))
pcs_use <- pc_cols[keep]

for (cn in pcs_use) {
  v <- dt[[cn]]
  m <- mean(v[is.finite(v)])
  v[!is.finite(v)] <- m
  set(dt, j = cn, value = v)
}

dt <- dt[is.finite(err_abs)]
stopifnot(nrow(dt) > 0L, length(pcs_use) > 0L)

fit_err <- lm(err_abs ~ ., data = dt[, c("err_abs", pcs_use), with = FALSE])
cat(sprintf("R^2: %.6f\n", summary(fit_err)$r.squared))
```
R^2: 0.000360. Not great.

Let's predict Alzheimer's with both scores and PCs:
```
# PCs + all PGS → logistic model → AUROC and OR/SD

library(data.table)
library(pROC)

pc_cols  <- paste0("PC", 1:16)
case_ids <- unique(as.character(cases))

# PCs
Z <- as.data.table(pc_df[, c("person_id", pc_cols), with = FALSE])
Z[, person_id := as.character(person_id)]
for (cn in pc_cols) set(Z, j = cn, value = suppressWarnings(as.numeric(Z[[cn]])))
keep_pcs <- vapply(pc_cols, function(cn) any(is.finite(Z[[cn]])), logical(1))
Z <- Z[, c("person_id", pc_cols[keep_pcs]), with = FALSE]
for (cn in setdiff(names(Z), "person_id")) {
  v <- Z[[cn]]; m <- mean(v[is.finite(v)]); v[!is.finite(v)] <- m; set(Z, j = cn, value = v)
}

# PGS
ss <- fread("arrays.sscore", sep = "\t", showProgress = FALSE)
setnames(ss, names(ss)[1], "person_id")
ss[, person_id := as.character(person_id)]
pgs_cols <- grep("_AVG$", names(ss), value = TRUE)
Xpgs <- ss[, c("person_id", pgs_cols), with = FALSE]
for (cn in pgs_cols) set(Xpgs, j = cn, value = suppressWarnings(as.numeric(Xpgs[[cn]])))
keep_pgs <- vapply(pgs_cols, function(cn) any(is.finite(Xpgs[[cn]])), logical(1))
Xpgs <- Xpgs[, c("person_id", pgs_cols[keep_pgs]), with = FALSE]
for (cn in setdiff(names(Xpgs), "person_id")) {
  v <- Xpgs[[cn]]; m <- mean(v[is.finite(v)]); v[!is.finite(v)] <- m; set(Xpgs, j = cn, value = v)
}

# Align & model
M <- merge(Xpgs, Z, by = "person_id", all = FALSE)
y <- as.integer(M$person_id %chin% case_ids)
X <- M[, -1, with = FALSE]

fit   <- glm(y ~ ., data = data.frame(y = y, X, check.names = FALSE), family = binomial())
p_hat <- as.numeric(predict(fit, type = "response"))

auc_val <- as.numeric(auc(roc(response = y, predictor = p_hat, quiet = TRUE)))

mu <- mean(p_hat); sdv <- sd(p_hat)
z  <- (p_hat - mu) / sdv
fit_z <- glm(y ~ z, family = binomial())
or_sd <- unname(exp(coef(fit_z)["z"]))

cat(sprintf(
  "Merged n: %s | Predictors: %d\nAUROC: %.6f\nOR per SD of prediction: %.6f\n",
  format(length(y), big.mark = ","), ncol(X), auc_val, or_sd
))
```
AUROC: 0.630836
OR per SD of prediction: 1.439452

Let's view Alzheimer's risk along PCs 2 and 4:
```
# PC2 / PC4 quantile bins (20 each) vs Alzheimer's prevalence

library(data.table)

dt <- as.data.table(pc_df[, .(person_id, PC2, PC4)])
dt[, person_id := as.character(person_id)]
dt[, y := as.integer(person_id %chin% unique(as.character(cases)))]

qprev <- function(v, y, q = 20L) {
  n <- sum(is.finite(v))
  r <- frank(v, ties.method = "average", na.last = "keep")
  b <- ceiling(r / (n / q))
  b[b < 1L | b > q] <- NA_integer_
  out <- data.table(bin = b, y = y)[!is.na(bin), .(prev = mean(y)), by = bin][order(bin)]
  out[]
}

p2 <- qprev(dt$PC2, dt$y, q = 20L)
p4 <- qprev(dt$PC4, dt$y, q = 20L)

ylim <- c(0, 100 * max(c(p2$prev, p4$prev), na.rm = TRUE))

plot(p2$bin, 100 * p2$prev, type = "o", xlab = "PC2 quantile (1 = low)",
     ylab = "Alzheimer's prevalence (%)", main = "PC2 vs AD prevalence", ylim = ylim)
plot(p4$bin, 100 * p4$prev, type = "o", xlab = "PC4 quantile (1 = low)",
     ylab = "Alzheimer's prevalence (%)", main = "PC4 vs AD prevalence", ylim = ylim)
```

<img width="810" height="519" alt="image" src="https://github.com/user-attachments/assets/2b9d6dc6-40d9-4173-b0a7-38c50ba194d6" />

<img width="810" height="519" alt="image" src="https://github.com/user-attachments/assets/9a30edc4-7b20-4f55-beae-00172105575a" />


Now let's check with splines:
```
# Spline prevalence curves (mgcv): PC2 and PC4 → AD probability

library(data.table)
library(mgcv)

dt <- as.data.table(pc_df[, .(person_id, PC2, PC4)])
dt[, person_id := as.character(person_id)]
dt[, y := as.integer(person_id %chin% unique(as.character(cases)))]
dt <- dt[is.finite(PC2) & is.finite(PC4)]

m2 <- gam(y ~ s(PC2, k = 9), data = dt, family = binomial())
m4 <- gam(y ~ s(PC4, k = 9), data = dt, family = binomial())

ilogit <- function(x) 1/(1 + exp(-x))

g2 <- data.table(PC2 = seq(min(dt$PC2), max(dt$PC2), length.out = 200))
p2 <- predict(m2, newdata = g2, type = "link", se.fit = TRUE)
g2[, `:=`(prev = 100 * ilogit(p2$fit),
          lo   = 100 * ilogit(p2$fit - 1.96 * p2$se.fit),
          hi   = 100 * ilogit(p2$fit + 1.96 * p2$se.fit))]

g4 <- data.table(PC4 = seq(min(dt$PC4), max(dt$PC4), length.out = 200))
p4 <- predict(m4, newdata = g4, type = "link", se.fit = TRUE)
g4[, `:=`(prev = 100 * ilogit(p4$fit),
          lo   = 100 * ilogit(p4$fit - 1.96 * p4$se.fit),
          hi   = 100 * ilogit(p4$fit + 1.96 * p4$se.fit))]

plot(g2$PC2, g2$prev, type = "l", xlab = "PC2", ylab = "Alzheimer's prevalence (%)",
     main = "Spline: PC2 vs prevalence")
lines(g2$PC2, g2$lo, lty = 2); lines(g2$PC2, g2$hi, lty = 2)

plot(g4$PC4, g4$prev, type = "l", xlab = "PC4", ylab = "Alzheimer's prevalence (%)",
     main = "Spline: PC4 vs prevalence")
lines(g4$PC4, g4$lo, lty = 2); lines(g4$PC4, g4$hi, lty = 2)
```

<img width="876" height="519" alt="image" src="https://github.com/user-attachments/assets/ee38a540-0bf8-4733-8636-284069adf64e" />

<img width="876" height="519" alt="image" src="https://github.com/user-attachments/assets/0098ee1c-c815-4bfb-82aa-3dc79c1f32ec" />

The non-linearity makes sense given the complex landscape of population structure proxied by PCs.


Let's check if using REML makes any difference:
```
# Spline prevalence curves with REML: PC2 and PC4 → AD probability

library(data.table)
library(mgcv)

dt <- as.data.table(pc_df[, .(person_id, PC2, PC4)])
dt[, person_id := as.character(person_id)]
dt[, y := as.integer(person_id %chin% unique(as.character(cases)))]
dt <- dt[is.finite(PC2) & is.finite(PC4)]

m2 <- gam(y ~ s(PC2, k = 9), data = dt, family = binomial(), method = "REML")
m4 <- gam(y ~ s(PC4, k = 9), data = dt, family = binomial(), method = "REML")

g2 <- data.table(PC2 = seq(min(dt$PC2), max(dt$PC2), length.out = 200))
p2 <- predict(m2, newdata = g2, type = "link", se.fit = TRUE)
g2[, `:=`(prev = 100 * plogis(p2$fit),
          lo   = 100 * plogis(p2$fit - 1.96 * p2$se.fit),
          hi   = 100 * plogis(p2$fit + 1.96 * p2$se.fit))]

g4 <- data.table(PC4 = seq(min(dt$PC4), max(dt$PC4), length.out = 200))
p4 <- predict(m4, newdata = g4, type = "link", se.fit = TRUE)
g4[, `:=`(prev = 100 * plogis(p4$fit),
          lo   = 100 * plogis(p4$fit - 1.96 * p4$se.fit),
          hi   = 100 * plogis(p4$fit + 1.96 * p4$se.fit))]

plot(g2$PC2, g2$prev, type = "l", xlab = "PC2", ylab = "Alzheimer's prevalence (%)",
     main = "Spline (REML): PC2 vs prevalence")
lines(g2$PC2, g2$lo, lty = 2); lines(g2$PC2, g2$hi, lty = 2)

plot(g4$PC4, g4$prev, type = "l", xlab = "PC4", ylab = "Alzheimer's prevalence (%)",
     main = "Spline (REML): PC4 vs prevalence")
lines(g4$PC4, g4$lo, lty = 2); lines(g4$PC4, g4$hi, lty = 2)
```

<img width="876" height="519" alt="image" src="https://github.com/user-attachments/assets/62cb1ee0-8c9b-48cb-be3e-196898767d03" />
<img width="876" height="519" alt="image" src="https://github.com/user-attachments/assets/61fcbfe4-9ecf-4403-8a6b-3832ad3cd516" />


REML is similar but not exactly the same as the default, GCV.
