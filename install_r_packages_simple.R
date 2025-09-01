#!/usr/bin/env Rscript
# Simple R package installation

# Essential packages for residue analysis
install.packages(c("arrow", "vegan", "ggplot2", "dplyr"), 
                 repos = "https://cloud.r-project.org",
                 verbose = TRUE)

cat("\nInstallation complete!\n")