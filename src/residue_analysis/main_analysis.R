#!/usr/bin/env Rscript
# ============================================================================
# Main Analysis Runner for Residual Analysis
# ============================================================================
# Orchestrates residual analyses after environmental correction (e.g., DML)
# For raw data heterogeneity analysis, use src/heterogeneity_analysis/
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
    mantel_sample = 5000,
    correlogram_sample = 2000,
    viz_sample = 10000,
    
    # Analysis parameters
    distance_classes = 10,
    permutations = 999,
    
    # Analysis flags for residual data
    run_distribution = TRUE,
    run_residual_analysis = TRUE,  # Correlation and RDA on residuals
    run_spatial = TRUE  # Test for remaining spatial autocorrelation
  )
  
  # Override with command line arguments if provided
  if (length(args) > 0) {
    # First arg should be config file
    if (file.exists(args[1])) {
      user_config <- fromJSON(args[1])
      config <- modifyList(config, user_config)
    }
  }
  
  return(config)
}

#' Run distribution analysis
run_distribution_analysis <- function(config, data, data_samples) {
  cat("\n\n")
  cat(paste(rep("=", 70), collapse=""), "\n")
  cat("RUNNING DISTRIBUTION ANALYSIS\n")
  cat(paste(rep("=", 70), collapse=""), "\n\n")
  
  source("src/residue_analysis/distribution_analysis.R")
  
  output_dir <- file.path(config$output_base, "distribution")
  
  results <- analyze_residual_distributions(
    data = data,
    data_viz = data_samples$viz,
    output_dir = output_dir
  )
  
  cat("✓ Distribution analysis complete\n")
  return(results)
}

#' Run residual biodiversity relationship analysis
run_residual_analysis <- function(config, data, data_samples) {
  cat("\n\n")
  cat(paste(rep("=", 70), collapse=""), "\n")
  cat("RUNNING RESIDUAL BIODIVERSITY RELATIONSHIP ANALYSIS\n")
  cat(paste(rep("=", 70), collapse=""), "\n\n")
  
  source("src/residue_analysis/residual_biodiversity_analysis.R")
  
  output_dir <- file.path(config$output_base, "residual_analysis")
  
  results <- variance_partitioning_analysis(
    data = data,
    data_rda = data_samples$rda,
    output_dir = output_dir
  )
  
  cat("✓ Residual biodiversity analysis complete\n")
  return(results)
}


#' Run spatial diagnostics
run_spatial_diagnostics <- function(config, data, data_samples) {
  cat("\n\n")
  cat(paste(rep("=", 70), collapse=""), "\n")
  cat("RUNNING SPATIAL DIAGNOSTICS\n")
  cat(paste(rep("=", 70), collapse=""), "\n\n")
  
  source("src/residue_analysis/spatial_diagnostics.R")
  
  output_dir <- file.path(config$output_base, "spatial_diagnostics")
  
  results <- spatial_diagnostics(
    data = data,
    data_sample = data_samples$spatial,
    output_dir = output_dir,
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
  cat("║         RESIDUAL BIODIVERSITY ANALYSIS PIPELINE                  ║\n")
  cat("╚═══════════════════════════════════════════════════════════════════╝\n")
  cat("\n")
  
  # Print configuration
  cat("Configuration:\n")
  cat("  Data path:", config$data_path, "\n")
  cat("  Output base:", config$output_base, "\n")
  cat("  Sample sizes:\n")
  cat("    - General:", config$sample_size, "\n")
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
  
  # Load data ONCE
  cat("\nLoading data from:", config$data_path, "\n")
  library(arrow)
  data <- read_feather(config$data_path)
  n_samples <- nrow(data)
  cat("Total samples:", format(n_samples, big.mark=","), "\n\n")
  
  # Create samples for different analyses
  cat("Creating analysis samples...\n")
  set.seed(42)  # Reproducibility
  data_samples <- list()
  
  # Sample for visualization
  if (n_samples > config$viz_sample) {
    data_samples$viz <- data[sample(n_samples, config$viz_sample), ]
  } else {
    data_samples$viz <- data
  }
  
  # Sample for RDA (computationally intensive)
  if (n_samples > config$sample_size) {
    data_samples$rda <- data[sample(n_samples, config$sample_size), ]
  } else {
    data_samples$rda <- data
  }
  
  # Sample for spatial analysis
  if (n_samples > config$mantel_sample) {
    data_samples$spatial <- data[sample(n_samples, config$mantel_sample), ]
  } else {
    data_samples$spatial <- data
  }
  
  cat("  Visualization sample:", format(nrow(data_samples$viz), big.mark=","), "\n")
  cat("  RDA sample:", format(nrow(data_samples$rda), big.mark=","), "\n")
  cat("  Spatial sample:", format(nrow(data_samples$spatial), big.mark=","), "\n\n")
  
  # Initialize results list
  all_results <- list()
  
  # Track timing
  start_time <- Sys.time()
  
  # Run analyses based on flags
  tryCatch({
    
    if (config$run_distribution) {
      all_results$distribution <- run_distribution_analysis(config, data, data_samples)
    }
    
    if (config$run_residual_analysis) {
      all_results$residual_analysis <- run_residual_analysis(config, data, data_samples)
    }
    
    if (config$run_spatial) {
      all_results$spatial <- run_spatial_diagnostics(config, data, data_samples)
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
  cat("\n\n")
  cat(paste(rep("=", 70), collapse=""), "\n")
  cat("GENERATING SUMMARY REPORT\n")
  cat(paste(rep("=", 70), collapse=""), "\n\n")
  
  summary_file <- file.path(config$output_base, "analysis_summary.txt")
  sink(summary_file)
  
  cat("DML RESIDUAL ANALYSIS SUMMARY\n")
  cat(paste(rep("=", 70), collapse=""), "\n")
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
  
  # Residual analysis summary
  if (!is.null(all_results$residual_analysis)) {
    cat("RESIDUAL BIODIVERSITY RELATIONSHIPS:\n")
    vp <- all_results$residual_analysis
    
    # Report correlations
    cat("  Partial Correlations (after removing environment):\n")
    if (!is.null(vp$cor_summary)) {
      for(i in 1:nrow(vp$cor_summary)) {
        cat(sprintf("    %s: r=%.3f %s\n", 
                   vp$cor_summary$Pair[i],
                   vp$cor_summary$Correlation[i],
                   ifelse(vp$cor_summary$Significant[i], "(p<0.05)", "")))
      }
    }
    
    # Report RDA results
    cat("  RDA Models:\n")
    if (!is.null(vp$rda_summary)) {
      for(i in 1:nrow(vp$rda_summary)) {
        cat(sprintf("    %s: R²=%.3f %s\n",
                   vp$rda_summary$Response[i],
                   vp$rda_summary$Adj_R2[i],
                   ifelse(vp$rda_summary$Significant[i], "(p<0.05)", "")))
      }
    }
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