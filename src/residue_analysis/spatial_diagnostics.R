#!/usr/bin/env Rscript
# ============================================================================
# Spatial Diagnostics for DML Residuals
# ============================================================================
# Tests for spatial autocorrelation in residuals to verify that DML
# successfully removed spatial/environmental effects
#
# Author: Biodiversity-Environment Analysis Pipeline
# Date: 2024
# ============================================================================

library(vegan)
library(arrow)
library(ggplot2)
library(sp)

#' Perform spatial diagnostics on residuals
#'
#' @param data_path Path to residuals arrow file
#' @param output_dir Directory to save results
#' @param sample_size Sample size for Mantel tests
#' @param distance_classes Number of distance classes for correlogram
#' @return List containing spatial diagnostic results
spatial_diagnostics <- function(
  data,
  data_sample = NULL,
  output_dir = "./results/spatial_diagnostics",
  distance_classes = 10
) {
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  n_samples <- nrow(data)
  cat("Analyzing", format(n_samples, big.mark=","), "samples\n")
  
  # Check if we have coordinates
  if (!all(c("longitude", "latitude") %in% names(data))) {
    stop("Longitude and latitude columns are required for spatial analysis")
  }
  
  # Use provided sample or full data
  if (is.null(data_sample)) {
    data_sample <- data
  }
  cat("Using", format(nrow(data_sample), big.mark=","), "samples for analysis\n\n")
  
  
  # ============ Create Distance Matrices ============
  
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("CREATING DISTANCE MATRICES\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Spatial distance matrix
  coords <- cbind(data_sample$longitude, data_sample$latitude)
  spatial_dist <- dist(coords)
  
  # Residual distance matrix (multivariate)
  residual_mat <- cbind(
    data_sample$fungi_richness,
    data_sample$plants_richness,
    data_sample$vertebrates_richness
  )
  residual_dist <- dist(residual_mat, method = "euclidean")
  
  cat("Spatial distance matrix created\n")
  cat("Residual distance matrix created\n\n")
  
  # ============ Mantel Test ============
  
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("MANTEL TEST FOR SPATIAL AUTOCORRELATION\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  mantel_result <- mantel(
    residual_dist, 
    spatial_dist,
    method = "pearson",
    permutations = 999,
    na.rm = TRUE
  )
  
  print(mantel_result)
  
  # Interpretation
  cat("\nInterpretation:\n")
  if (mantel_result$signif < 0.05) {
    cat("  ⚠ SIGNIFICANT spatial autocorrelation detected (p < 0.05)\n")
    cat("  This suggests DML may not have fully removed spatial/environmental effects\n")
    cat(sprintf("  Mantel r = %.4f\n", mantel_result$statistic))
  } else {
    cat("  ✓ No significant spatial autocorrelation (p >= 0.05)\n")
    cat("  Residuals appear spatially independent\n")
    cat(sprintf("  Mantel r = %.4f\n", mantel_result$statistic))
  }
  
  # ============ Individual Taxa Mantel Tests ============
  
  cat("\n\n")
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("MANTEL TESTS BY TAXON\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  taxa <- c("fungi_richness", "plants_richness", "vertebrates_richness")
  taxa_labels <- c("Fungi", "Plants", "Vertebrates")
  mantel_by_taxon <- list()
  
  for (i in 1:3) {
    cat(sprintf("%s residuals:\n", taxa_labels[i]))
    
    # Create distance matrix for single taxon
    taxon_dist <- dist(data_sample[[taxa[i]]])
    
    # Mantel test
    mantel_taxon <- mantel(
      taxon_dist,
      spatial_dist,
      method = "pearson",
      permutations = 999,
      na.rm = TRUE
    )
    
    mantel_by_taxon[[taxa[i]]] <- mantel_taxon
    
    cat(sprintf("  Mantel r = %.4f, p-value = %.4f %s\n",
                mantel_taxon$statistic,
                mantel_taxon$signif,
                ifelse(mantel_taxon$signif < 0.05, "***", "")))
  }
  
  # ============ Mantel Correlogram ============
  
  cat("\n\n")
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("MANTEL CORRELOGRAM\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  cat("Computing correlogram (this may take several minutes)...\n")
  
  # For smaller sample to make computation feasible
  if (nrow(data_sample) > 2000) {
    cat("Using 2000 samples for correlogram...\n")
    corr_idx <- sample(1:nrow(data_sample), 2000)
    corr_spatial <- dist(coords[corr_idx, ])
    corr_residual <- dist(residual_mat[corr_idx, ])
  } else {
    corr_spatial <- spatial_dist
    corr_residual <- residual_dist
  }
  
  # Compute correlogram
  correlogram <- mantel.correlog(
    corr_residual,
    corr_spatial,
    n.class = distance_classes,
    nperm = 99  # Reduced for speed
  )
  
  print(correlogram)
  
  # Plot correlogram
  pdf(file.path(output_dir, "mantel_correlogram.pdf"), width = 10, height = 6)
  plot(correlogram, main = "Spatial Correlogram of Residuals")
  abline(h = 0, lty = 2, col = "gray")
  dev.off()
  
  # ============ Test Against Environmental Variables ============
  
  cat("\n\n")
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("TESTING RESIDUAL INDEPENDENCE FROM ENVIRONMENT\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Check if we have environmental variables in the data
  env_vars <- c("temperature", "precipitation", "elevation", "soil_pH")
  available_env <- env_vars[env_vars %in% names(data_sample)]
  
  if (length(available_env) > 0) {
    cat("Testing residuals against available environmental variables:\n\n")
    
    env_correlations <- data.frame()
    
    for (env_var in available_env) {
      cat(sprintf("%s:\n", env_var))
      
      for (i in 1:3) {
        cor_test <- cor.test(
          data_sample[[taxa[i]]],
          data_sample[[env_var]],
          method = "pearson"
        )
        
        cat(sprintf("  %s: r = %.4f, p = %.4f %s\n",
                    taxa_labels[i],
                    cor_test$estimate,
                    cor_test$p.value,
                    ifelse(cor_test$p.value < 0.05, "***", "")))
        
        env_correlations <- rbind(env_correlations, data.frame(
          taxon = taxa_labels[i],
          env_variable = env_var,
          correlation = cor_test$estimate,
          p_value = cor_test$p.value,
          significant = cor_test$p.value < 0.05
        ))
      }
      cat("\n")
    }
    
    # Save environmental correlations
    write.csv(env_correlations,
              file.path(output_dir, "residual_environment_correlations.csv"),
              row.names = FALSE)
    
    # Check overall independence
    n_significant <- sum(env_correlations$significant)
    n_tests <- nrow(env_correlations)
    
    cat("Summary:\n")
    cat(sprintf("  %d/%d correlations significant at p < 0.05\n",
                n_significant, n_tests))
    
    if (n_significant == 0) {
      cat("  ✓ Residuals appear independent of environmental variables\n")
    } else {
      cat("  ⚠ Some residual correlations with environment detected\n")
      cat("  This may indicate incomplete removal of environmental effects\n")
    }
  } else {
    cat("No environmental variables found in dataset for testing\n")
    env_correlations <- NULL
  }
  
  # ============ Spatial Visualization ============
  
  cat("\n\n")
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("SPATIAL VISUALIZATION\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Create spatial plots
  pdf(file.path(output_dir, "spatial_residuals.pdf"), width = 15, height = 5)
  par(mfrow = c(1, 3))
  
  # Sample for visualization if too many points
  if (nrow(data_sample) > 10000) {
    viz_idx <- sample(1:nrow(data_sample), 10000)
    viz_data <- data_sample[viz_idx, ]
  } else {
    viz_data <- data_sample
  }
  
  # Create color palette
  col_palette <- colorRampPalette(c("blue", "white", "red"))(100)
  
  for (i in 1:3) {
    # Normalize residuals to 0-1 for coloring
    residuals <- viz_data[[taxa[i]]]
    residuals_norm <- (residuals - min(residuals)) / (max(residuals) - min(residuals))
    colors <- col_palette[ceiling(residuals_norm * 99) + 1]
    
    plot(viz_data$longitude, viz_data$latitude,
         col = colors, pch = 16, cex = 0.5,
         main = paste(taxa_labels[i], "Residuals"),
         xlab = "Longitude", ylab = "Latitude")
    
    # Add legend
    legend_vals <- seq(min(residuals), max(residuals), length.out = 5)
    legend("bottomleft", 
           legend = sprintf("%.2f", legend_vals),
           col = col_palette[c(1, 25, 50, 75, 100)],
           pch = 16, cex = 0.8, title = "Residual")
  }
  
  dev.off()
  
  # ============ Summary Report ============
  
  cat("\n\n")
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("SPATIAL DIAGNOSTICS SUMMARY\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  summary_text <- c(
    "SPATIAL DIAGNOSTICS SUMMARY",
    paste(rep("=", 60), collapse=""),
    sprintf("Samples analyzed: %d", n_samples),
    "",
    "MANTEL TEST RESULTS:",
    sprintf("  Overall: r = %.4f, p = %.4f %s",
            mantel_result$statistic,
            mantel_result$signif,
            ifelse(mantel_result$signif < 0.05, "(SIGNIFICANT)", "")),
    "",
    "BY TAXON:"
  )
  
  for (i in 1:3) {
    m <- mantel_by_taxon[[taxa[i]]]
    summary_text <- c(summary_text,
                     sprintf("  %s: r = %.4f, p = %.4f %s",
                             taxa_labels[i],
                             m$statistic,
                             m$signif,
                             ifelse(m$signif < 0.05, "(SIGNIFICANT)", "")))
  }
  
  summary_text <- c(summary_text, "", "INTERPRETATION:")
  
  if (mantel_result$signif >= 0.05) {
    summary_text <- c(summary_text,
                     "  ✓ Residuals appear spatially independent",
                     "  ✓ DML successfully removed spatial/environmental effects")
  } else {
    summary_text <- c(summary_text,
                     "  ⚠ Significant spatial autocorrelation detected",
                     "  ⚠ Consider additional spatial covariates in DML")
  }
  
  # Write summary
  writeLines(summary_text, file.path(output_dir, "spatial_diagnostics_summary.txt"))
  
  # ============ Save Results ============
  
  results <- list(
    mantel_overall = mantel_result,
    mantel_by_taxon = mantel_by_taxon,
    correlogram = correlogram,
    env_correlations = env_correlations,
    n_samples = n_samples
  )
  
  saveRDS(results, file.path(output_dir, "spatial_diagnostics_results.rds"))
  
  cat("\nResults saved to:", output_dir, "\n")
  
  return(results)
}

# Run if executed directly
# This file should be sourced from main_analysis.R
# Do not run if sourced
