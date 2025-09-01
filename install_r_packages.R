#!/usr/bin/env Rscript
# Install required R packages for residue analysis

# Set user library path
user_lib <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_lib)) {
  dir.create(user_lib, recursive = TRUE)
}
.libPaths(c(user_lib, .libPaths()))

# List of required packages from the R scripts
required_packages <- c(
  "vegan",           # For ecological analysis
  "arrow",           # For reading arrow files
  "ggplot2",         # For visualization
  "gridExtra",       # For grid plots
  "moments",         # For skewness and kurtosis
  "nortest",         # For Anderson-Darling test
  "speciesgeocodeR", # For WWF biome data
  "sp",              # For spatial data
  "dplyr",           # For data manipulation
  "jsonlite"         # For JSON config
)

# Function to install packages if not already installed
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", lib = user_lib)
  } else {
    cat(sprintf("%s is already installed\n", pkg))
  }
}

# Install each package
for (pkg in required_packages) {
  tryCatch({
    install_if_missing(pkg)
  }, error = function(e) {
    cat(sprintf("Error installing %s: %s\n", pkg, e$message))
  })
}

cat("\nPackage installation complete!\n")