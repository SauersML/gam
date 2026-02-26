if (!require(mgcv)) {
  install.packages("mgcv")
  library(mgcv)
}

# --- 1. I/O ---
input_csv_file <- "synthetic_classification_data.csv"
output_rds_file <- "gam_model_fit.rds"

# --- 2. Load Data ---
cat("Loading data from '", input_csv_file, "'...\n\n", sep = "")
data <- read.csv(input_csv_file)

# --- 3. Define and Fit the GAM ---
# Goal:
#   Use k=11 so the effective columns are ~10 after mgcv's identifiability handling.
# - PC1 main effect (variable_two): penalized, but match Rust's 10 cols -> set k=11 (not 12).
# - Interaction: penalized, anisotropic (2 smoothing params) via ti().
#   With k=c(12,12) and penalty order 2, ti() builds the "interaction-only" 100-column block.

gam_formula <- outcome ~
  s(variable_two, bs = "ps", k = 11, m = c(4, 2)) +                         # penalized PC1 main (≈10 cols)
  ti(variable_one, variable_two, bs = c("ps", "ps"), k = c(12, 12), m = c(4, 2))  # penalized interaction (2 λ)

cat("Fitting the GAM (REML)...\n")
gam_fit <- gam(
  gam_formula,
  family = binomial(link = "logit"),
  data = data,
  method = "REML"
)

# --- 4. Inspect the Model Fit ---
cat("\n--- GAM Model Summary ---\n")
print(summary(gam_fit))

# --- 5. Save the Model Object ---
cat("\nSaving the fitted GAM object to '", output_rds_file, "'...\n", sep = "")
saveRDS(gam_fit, file = output_rds_file)
cat("Script finished successfully.\n\n")

