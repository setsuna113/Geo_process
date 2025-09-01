#!/usr/bin/env Rscript
# Test if required packages are installed

packages <- c("vegan", "arrow", "ggplot2", "gridExtra", "moments", 
              "nortest", "speciesgeocodeR", "sp", "dplyr", "jsonlite")

cat("Testing R package installation...\n\n")

for (pkg in packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("✓ %s installed\n", pkg))
  } else {
    cat(sprintf("✗ %s NOT installed\n", pkg))
  }
}

# Test reading arrow file
cat("\n\nTesting arrow file reading...\n")
tryCatch({
  data <- arrow::read_feather("/scratch/yl998/econml_results_5m/homogeneous/residuals.arrow")
  cat(sprintf("✓ Successfully read arrow file with %d rows and %d columns\n", 
              nrow(data), ncol(data)))
  cat("Columns:", paste(names(data), collapse=", "), "\n")
}, error = function(e) {
  cat("✗ Failed to read arrow file:", e$message, "\n")
})