#!/usr/bin/env Rscript
# ============================================================================
# Distribution Analysis of DML Residuals
# ============================================================================
# Analyzes the distribution of residuals from Double ML to check normality,
# correlations, and basic statistical properties
# 
# Author: Biodiversity-Environment Analysis Pipeline
# Date: 2024
# ============================================================================

library(vegan)
library(arrow)
library(ggplot2)
library(gridExtra)

#' Analyze residual distributions
#'
#' @param data_path Path to residuals arrow file
#' @param output_dir Directory to save plots and results
#' @param sample_size Sample size for computationally intensive tests
#' @return List containing correlation matrix and summary statistics
analyze_residual_distributions <- function(
  data_path = "/scratch/yl998/econml_results_5m/homogeneous/residuals.arrow",
  output_dir = "./results/distribution",
  sample_size = 10000
) {
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Load residuals
  cat("Loading residuals from:", data_path, "\n")
  data <- read_feather(data_path)
  
  n_samples <- nrow(data)
  cat("Total samples:", format(n_samples, big.mark=","), "\n\n")
  
  # ============ Distribution Plots ============
  
  # Set up multi-panel plot
  pdf(file.path(output_dir, "distribution_plots.pdf"), width = 12, height = 8)
  par(mfrow = c(2, 3))
  
  # Histograms with density overlay
  taxa <- c("fungi_richness", "plants_richness", "vertebrates_richness")
  taxa_labels <- c("Fungi", "Plants", "Vertebrates")
  
  for (i in 1:3) {
    x <- data[[taxa[i]]]
    hist(x, breaks = 100, 
         main = paste(taxa_labels[i], "Residuals"),
         xlab = "Residual Value",
         ylab = "Frequency",
         col = "lightblue",
         border = "white",
         probability = TRUE)
    
    # Add normal density overlay
    x_seq <- seq(min(x, na.rm = TRUE), max(x, na.rm = TRUE), length.out = 100)
    lines(x_seq, dnorm(x_seq, mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)), 
          col = "red", lwd = 2)
  }
  
  # Q-Q plots for normality assessment
  for (i in 1:3) {
    qqnorm(data[[taxa[i]]], 
           main = paste(taxa_labels[i], "Q-Q Plot"),
           pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.3))
    qqline(data[[taxa[i]]], col = "red", lwd = 2)
  }
  
  dev.off()
  
  # ============ Summary Statistics ============
  
  cat("=" * 60, "\n")
  cat("SUMMARY STATISTICS\n")
  cat("=" * 60, "\n\n")
  
  summary_stats <- list()
  
  for (i in 1:3) {
    x <- data[[taxa[i]]]
    
    stats <- list(
      mean = mean(x, na.rm = TRUE),
      median = median(x, na.rm = TRUE),
      sd = sd(x, na.rm = TRUE),
      min = min(x, na.rm = TRUE),
      max = max(x, na.rm = TRUE),
      skewness = moments::skewness(x, na.rm = TRUE),
      kurtosis = moments::kurtosis(x, na.rm = TRUE),
      n_missing = sum(is.na(x))
    )
    
    summary_stats[[taxa[i]]] <- stats
    
    cat(taxa_labels[i], "Residuals:\n")
    cat("  Mean:", sprintf("%.6f", stats$mean), "\n")
    cat("  Median:", sprintf("%.6f", stats$median), "\n")
    cat("  SD:", sprintf("%.6f", stats$sd), "\n")
    cat("  Range: [", sprintf("%.6f", stats$min), ",", sprintf("%.6f", stats$max), "]\n")
    cat("  Skewness:", sprintf("%.4f", stats$skewness), "\n")
    cat("  Kurtosis:", sprintf("%.4f", stats$kurtosis), "\n")
    cat("  Missing:", stats$n_missing, "\n\n")
  }
  
  # ============ Normality Tests ============
  
  cat("=" * 60, "\n")
  cat("NORMALITY TESTS (on sample)\n")
  cat("=" * 60, "\n\n")
  
  # Sample for Shapiro-Wilk test (max 5000 observations)
  set.seed(42)
  sample_idx <- sample(1:n_samples, min(sample_size, 5000))
  
  normality_tests <- list()
  
  for (i in 1:3) {
    x_sample <- data[[taxa[i]]][sample_idx]
    
    # Shapiro-Wilk test
    shapiro_result <- shapiro.test(x_sample)
    
    # Anderson-Darling test (more sensitive to tails)
    ad_result <- nortest::ad.test(x_sample)
    
    normality_tests[[taxa[i]]] <- list(
      shapiro = shapiro_result,
      anderson_darling = ad_result
    )
    
    cat(taxa_labels[i], ":\n")
    cat("  Shapiro-Wilk: W =", sprintf("%.6f", shapiro_result$statistic), 
        ", p-value =", sprintf("%.6e", shapiro_result$p.value), "\n")
    cat("  Anderson-Darling: A =", sprintf("%.6f", ad_result$statistic),
        ", p-value =", sprintf("%.6e", ad_result$p.value), "\n\n")
  }
  
  # ============ Correlation Analysis ============
  
  cat("=" * 60, "\n")
  cat("CORRELATION MATRIX\n")
  cat("=" * 60, "\n\n")
  
  # Compute correlation matrix
  cor_matrix <- cor(data[, taxa], use = "complete.obs")
  
  # Print correlation matrix
  print(round(cor_matrix, 4))
  
  # Test significance of correlations
  cat("\nCorrelation significance tests:\n")
  
  cor_tests <- list()
  for (i in 1:2) {
    for (j in (i+1):3) {
      test_result <- cor.test(data[[taxa[i]]], data[[taxa[j]]])
      cor_tests[[paste(taxa[i], taxa[j], sep = "_vs_")]] <- test_result
      
      cat(sprintf("  %s vs %s: r = %.4f, p-value = %.4e\n",
                  taxa_labels[i], taxa_labels[j],
                  test_result$estimate, test_result$p.value))
    }
  }
  
  # ============ Scatterplot Matrix ============
  
  cat("\nGenerating scatterplot matrix...\n")
  
  # Sample for visualization
  viz_idx <- sample(1:n_samples, min(sample_size, n_samples))
  
  pdf(file.path(output_dir, "scatterplot_matrix.pdf"), width = 10, height = 10)
  
  # Custom panel function for pairs plot
  panel.cor <- function(x, y, ...) {
    usr <- par("usr")
    par(usr = c(0, 1, 0, 1))
    r <- cor(x, y, use = "complete.obs")
    txt <- sprintf("r = %.3f", r)
    text(0.5, 0.5, txt, cex = 1.5)
  }
  
  panel.hist <- function(x, ...) {
    usr <- par("usr")
    par(usr = c(usr[1:2], 0, 1.5))
    h <- hist(x, plot = FALSE, breaks = 50)
    breaks <- h$breaks
    nB <- length(breaks)
    y <- h$counts
    y <- y / max(y)
    rect(breaks[-nB], 0, breaks[-1], y, col = "lightblue", border = "white")
  }
  
  pairs(data[viz_idx, taxa],
        labels = taxa_labels,
        diag.panel = panel.hist,
        upper.panel = panel.cor,
        lower.panel = panel.smooth,
        pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.1),
        main = "Residual Correlations (sampled)")
  
  dev.off()
  
  # ============ Save Results ============
  
  results <- list(
    summary_stats = summary_stats,
    correlation_matrix = cor_matrix,
    normality_tests = normality_tests,
    correlation_tests = cor_tests,
    n_samples = n_samples
  )
  
  # Save as RDS for later use
  saveRDS(results, file.path(output_dir, "distribution_results.rds"))
  
  # Save correlation matrix as CSV
  write.csv(cor_matrix, file.path(output_dir, "correlation_matrix.csv"))
  
  cat("\nResults saved to:", output_dir, "\n")
  
  return(results)
}

# Run if executed directly
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) > 0) {
    data_path <- args[1]
  } else {
    data_path <- "/scratch/yl998/econml_results_5m/homogeneous/residuals.arrow"
  }
  
  if (length(args) > 1) {
    output_dir <- args[2]
  } else {
    output_dir <- "./results/distribution"
  }
  
  results <- analyze_residual_distributions(data_path, output_dir)
}