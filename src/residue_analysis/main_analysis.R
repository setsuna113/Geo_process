#!/usr/bin/env Rscript
# ============================================================================
# Main Analysis Runner for DML Residuals
# ============================================================================
# Orchestrates all residual analyses with centralized configuration
#
# Author: Biodiversity-Environment Analysis Pipeline
# Date: 2024
# ============================================================================

library(jsonlite)

#' Parse command line arguments or use defaults
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  
  config <- list(
    # Data paths
    data_path = "/scratch/yl998/econml_results_5m/homogeneous/residuals.arrow",
    output_base = "./results",
    
    # Sampling parameters
    sample_size = 10000,
    permanova_sample = 10000,
    mantel_sample = 5000,
    correlogram_sample = 2000,
    nmds_sample = 5000,
    viz_sample = 10000,
    
    # Analysis parameters
    min_samples_per_biome = 1000,
    distance_classes = 10,
    permutations = 999,
    
    # Analysis flags
    run_distribution = TRUE,
    run_variance_partitioning = TRUE,
    run_biome = TRUE,
    run_spatial = TRUE
  )
  
  # Override with command line arguments if provided
  if (length(args) > 0) {
    if (file.exists(args[1])) {
      # First arg is config file
      user_config <- fromJSON(args[1])
      config <- modifyList(config, user_config)
    } else {
      # First arg is data path
      config$data_path <- args[1]
    }
  }
  
  if (length(args) > 1) {
    config$output_base <- args[2]
  }
  
  return(config)
}

#' Run distribution analysis
run_distribution_analysis <- function(config) {
  cat("\n" * 2)
  cat("=" * 70, "\n")
  cat("RUNNING DISTRIBUTION ANALYSIS\n")
  cat("=" * 70, "\n\n")
  
  source("src/residue_analysis/distribution_analysis.R")
  
  output_dir <- file.path(config$output_base, "distribution")
  
  results <- analyze_residual_distributions(
    data_path = config$data_path,
    output_dir = output_dir,
    sample_size = config$sample_size
  )
  
  cat("✓ Distribution analysis complete\n")
  return(results)
}

#' Run variance partitioning
run_variance_partitioning <- function(config) {
  cat("\n" * 2)
  cat("=" * 70, "\n")
  cat("RUNNING VARIANCE PARTITIONING\n")
  cat("=" * 70, "\n\n")
  
  source("src/residue_analysis/variance_partitioning.R")
  
  output_dir <- file.path(config$output_base, "variance_partitioning")
  
  results <- variance_partitioning_analysis(
    data_path = config$data_path,
    output_dir = output_dir
  )
  
  cat("✓ Variance partitioning complete\n")
  return(results)
}

#' Run biome analysis
run_biome_analysis <- function(config) {
  cat("\n" * 2)
  cat("=" * 70, "\n")
  cat("RUNNING BIOME ANALYSIS\n")
  cat("=" * 70, "\n\n")
  
  source("src/residue_analysis/biome_analysis.R")
  
  output_dir <- file.path(config$output_base, "biome_analysis")
  
  results <- biome_analysis(
    data_path = config$data_path,
    output_dir = output_dir,
    sample_size = config$permanova_sample,
    min_samples_per_biome = config$min_samples_per_biome
  )
  
  cat("✓ Biome analysis complete\n")
  return(results)
}

#' Run spatial diagnostics
run_spatial_diagnostics <- function(config) {
  cat("\n" * 2)
  cat("=" * 70, "\n")
  cat("RUNNING SPATIAL DIAGNOSTICS\n")
  cat("=" * 70, "\n\n")
  
  source("src/residue_analysis/spatial_diagnostics.R")
  
  output_dir <- file.path(config$output_base, "spatial_diagnostics")
  
  results <- spatial_diagnostics(
    data_path = config$data_path,
    output_dir = output_dir,
    sample_size = config$mantel_sample,
    distance_classes = config$distance_classes
  )
  
  cat("✓ Spatial diagnostics complete\n")
  return(results)
}

#' Main analysis pipeline
main <- function() {
  # Parse configuration
  config <- parse_args()
  
  cat("\n")
  cat("╔═══════════════════════════════════════════════════════════════════╗\n")
  cat("║         DML RESIDUAL ANALYSIS PIPELINE                           ║\n")
  cat("╚═══════════════════════════════════════════════════════════════════╝\n")
  cat("\n")
  
  # Print configuration
  cat("Configuration:\n")
  cat("  Data path:", config$data_path, "\n")
  cat("  Output base:", config$output_base, "\n")
  cat("  Sample sizes:\n")
  cat("    - General:", config$sample_size, "\n")
  cat("    - PERMANOVA:", config$permanova_sample, "\n")
  cat("    - Mantel:", config$mantel_sample, "\n")
  cat("    - NMDS:", config$nmds_sample, "\n")
  cat("\n")
  
  # Check data file exists
  if (!file.exists(config$data_path)) {
    stop("Data file not found: ", config$data_path)
  }
  
  # Create main output directory
  dir.create(config$output_base, recursive = TRUE, showWarnings = FALSE)
  
  # Save configuration
  config_file <- file.path(config$output_base, "analysis_config.json")
  write(toJSON(config, pretty = TRUE), config_file)
  cat("Configuration saved to:", config_file, "\n")
  
  # Initialize results list
  all_results <- list()
  
  # Track timing
  start_time <- Sys.time()
  
  # Run analyses based on flags
  tryCatch({
    
    if (config$run_distribution) {
      all_results$distribution <- run_distribution_analysis(config)
    }
    
    if (config$run_variance_partitioning) {
      all_results$variance_partitioning <- run_variance_partitioning(config)
    }
    
    if (config$run_biome) {
      all_results$biome <- run_biome_analysis(config)
    }
    
    if (config$run_spatial) {
      all_results$spatial <- run_spatial_diagnostics(config)
    }
    
  }, error = function(e) {
    cat("\n⚠ Error in analysis pipeline:\n")
    cat("  ", e$message, "\n")
    stop(e)
  })
  
  # Calculate runtime
  end_time <- Sys.time()
  runtime <- difftime(end_time, start_time, units = "mins")
  
  # Generate summary report
  cat("\n" * 2)
  cat("=" * 70, "\n")
  cat("GENERATING SUMMARY REPORT\n")
  cat("=" * 70, "\n\n")
  
  summary_file <- file.path(config$output_base, "analysis_summary.txt")
  sink(summary_file)
  
  cat("DML RESIDUAL ANALYSIS SUMMARY\n")
  cat("=" * 70, "\n")
  cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
  cat("Runtime:", round(runtime, 2), "minutes\n")
  cat("Data:", config$data_path, "\n")
  cat("\n")
  
  # Distribution summary
  if (!is.null(all_results$distribution)) {
    cat("DISTRIBUTION ANALYSIS:\n")
    cat("  Samples analyzed:", all_results$distribution$n_samples, "\n")
    cat("  Correlation matrix:\n")
    print(round(all_results$distribution$correlation_matrix, 3))
    cat("\n")
  }
  
  # Variance partitioning summary
  if (!is.null(all_results$variance_partitioning)) {
    cat("VARIANCE PARTITIONING:\n")
    vp <- all_results$variance_partitioning
    cat("  Fungi explained by others:", 
        sprintf("%.1f%%", vp$vp_fungi$part$fract$Adj.R.squared[3] * 100), "\n")
    cat("  Plants explained by others:",
        sprintf("%.1f%%", vp$vp_plants$part$fract$Adj.R.squared[3] * 100), "\n")
    cat("  Vertebrates explained by others:",
        sprintf("%.1f%%", vp$vp_vertebrates$part$fract$Adj.R.squared[3] * 100), "\n")
    cat("\n")
  }
  
  # Biome summary
  if (!is.null(all_results$biome)) {
    cat("BIOME ANALYSIS:\n")
    cat("  Biomes analyzed:", nrow(all_results$biome$biome_summary), "\n")
    cat("  PERMANOVA R²:", 
        sprintf("%.3f", all_results$biome$adonis_result$R2[1]), "\n")
    cat("  PERMANOVA p-value:",
        sprintf("%.4f", all_results$biome$adonis_result$`Pr(>F)`[1]), "\n")
    cat("\n")
  }
  
  # Spatial summary
  if (!is.null(all_results$spatial)) {
    cat("SPATIAL DIAGNOSTICS:\n")
    mantel <- all_results$spatial$mantel_overall
    cat("  Mantel r:", sprintf("%.4f", mantel$statistic), "\n")
    cat("  Mantel p-value:", sprintf("%.4f", mantel$signif), "\n")
    if (mantel$signif < 0.05) {
      cat("  ⚠ Significant spatial autocorrelation detected\n")
    } else {
      cat("  ✓ No significant spatial autocorrelation\n")
    }
    cat("\n")
  }
  
  sink()
  
  # Save combined results
  saveRDS(all_results, file.path(config$output_base, "all_results.rds"))
  
  cat("Summary report saved to:", summary_file, "\n")
  cat("Combined results saved to:", file.path(config$output_base, "all_results.rds"), "\n")
  
  cat("\n")
  cat("╔═══════════════════════════════════════════════════════════════════╗\n")
  cat("║         ANALYSIS COMPLETE                                        ║\n")
  cat("╚═══════════════════════════════════════════════════════════════════╝\n")
  cat("\n")
  
  return(all_results)
}

# Run main if executed directly
if (!interactive()) {
  results <- main()
}