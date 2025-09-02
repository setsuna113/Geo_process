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
  data,
  data_viz = NULL,
  output_dir = "./results/distribution"
) {
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  n_samples <- nrow(data)
  cat("Analyzing", format(n_samples, big.mark=","), "samples\n")
  
  # Use visualization sample if provided
  if (is.null(data_viz)) {
    data_viz <- data
  }
  cat("Using", format(nrow(data_viz), big.mark=","), "samples for visualization\n\n")
  
  # ============ Distribution Plots ============
  
  # Set up multi-panel plot
  png(file.path(output_dir, "distribution_plots.png"), width = 1200, height = 800)
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
  
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("SUMMARY STATISTICS\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
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
  
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("NORMALITY TESTS (on sample)\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Sample for Shapiro-Wilk test (max 5000 observations)
  set.seed(42)
  # Use subset of visualization sample for normality tests (max 5000 for Shapiro)
  if (nrow(data_viz) > 5000) {
    test_sample <- data_viz[sample(nrow(data_viz), 5000), ]
  } else {
    test_sample <- data_viz
  }
  
  normality_tests <- list()
  
  for (i in 1:3) {
    x_sample <- test_sample[[taxa[i]]]
    
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
  
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("CORRELATION MATRIX\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
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
  
  png(file.path(output_dir, "scatterplot_matrix.png"), width = 1000, height = 1000)
  
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
  
  pairs(data_viz[, taxa],
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

# This file should be sourced from main_analysis.R
# Do not run if sourced