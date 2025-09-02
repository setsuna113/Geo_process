#!/usr/bin/env Rscript

# Single, clean installation script for piecewiseSEM
cat("Installing piecewiseSEM and dependencies\n")
cat("========================================\n\n")

# Use CRAN
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Core packages that MUST work
required <- c("arrow", "feather", "dplyr", "ggplot2", "MASS", "nlme", "lme4")

# Install core packages
for(pkg in required) {
  if(!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg)
  }
}

# Try piecewiseSEM without the problematic dependencies
cat("\nAttempting piecewiseSEM installation...\n")
install.packages("piecewiseSEM")

# Check
if(requireNamespace("piecewiseSEM", quietly = TRUE)) {
  library(piecewiseSEM)
  cat("\n✓ SUCCESS: piecewiseSEM version", as.character(packageVersion("piecewiseSEM")), "installed\n")
} else {
  cat("\n✗ FAILED: Could not install piecewiseSEM\n")
  cat("This is likely due to igraph/car compilation issues on this system.\n")
  cat("Consider using a different R environment or the Python analysis results.\n")
}