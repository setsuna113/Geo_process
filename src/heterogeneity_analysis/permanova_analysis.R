#!/usr/bin/env Rscript
# ============================================================================
# PERMANOVA and Beta Dispersion Analysis
# Config-driven analysis - no hardcoded fallbacks
# Tests for biome differences in community composition and variance heterogeneity
# ============================================================================

library(vegan)
library(arrow)
library(dplyr)
library(ggplot2)
library(yaml)

# Function to load and prepare data
load_and_prepare_data <- function(data_path, config) {
  if (!file.exists(data_path)) {
    stop("ERROR: Data file not found: ", data_path)
  }
  
  cat("Loading data from:", data_path, "\n")
  data <- read_feather(data_path)
  
  n_samples <- nrow(data)
  cat("Total samples:", format(n_samples, big.mark=","), "\n")
  
  # Get column names from config
  fungi_col <- config$columns$fungi
  plants_col <- config$columns$plants
  vertebrates_col <- config$columns$vertebrates
  biome_code_col <- config$columns$biome_code
  biome_name_col <- config$columns$biome_name
  
  # Check for required columns
  required_cols <- c(fungi_col, plants_col, vertebrates_col, biome_code_col, biome_name_col)
  missing_cols <- setdiff(required_cols, names(data))
  
  if (length(missing_cols) > 0) {
    stop("ERROR: Required columns missing from data: ", paste(missing_cols, collapse=", "))
  }
  
  # HARD LIMIT for PERMANOVA: 46,340 samples maximum
  # Reason: Distance matrix n×n must fit in .Fortran 32-bit integer limit (2^31-1)
  max_samples <- config$parameters$max_samples_permanova
  if (max_samples > 46340) {
    stop("ERROR: max_samples_permanova cannot exceed 46,340 (distance matrix limit)")
  }
  
  if (n_samples > max_samples) {
    cat("Sampling", format(max_samples, big.mark=","), "rows for PERMANOVA analysis...\n")
    cat("(.Fortran limit: distance matrix must have <2^31 elements)\n")
    set.seed(config$parameters$random_seed)
    data <- data[sample(n_samples, max_samples), ]
  }
  
  return(data)
}

# Run PERMANOVA analysis
run_permanova_analysis <- function(data, config) {
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("1. PERMANOVA: Testing for biome differences in composition\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Get column names
  fungi_col <- config$columns$fungi
  plants_col <- config$columns$plants
  vertebrates_col <- config$columns$vertebrates
  biome_code_col <- config$columns$biome_code
  
  # Create community matrix
  community <- cbind(
    data[[fungi_col]],
    data[[plants_col]],
    data[[vertebrates_col]]
  )
  colnames(community) <- c("Fungi", "Plants", "Vertebrates")
  
  # Run PERMANOVA
  permutations <- config$parameters$permutations
  adonis_result <- adonis2(community ~ data[[biome_code_col]], 
                          method = "euclidean", 
                          permutations = permutations)
  
  print(adonis_result)
  
  # Calculate R-squared
  r_squared <- adonis_result$R2[1]
  cat(sprintf("\nR² = %.3f (%.1f%% of variance explained by biomes)\n", 
              r_squared, r_squared * 100))
  
  if (adonis_result$`Pr(>F)`[1] < 0.001) {
    cat("✓ Highly significant differences between biomes (p < 0.001)\n")
  } else if (adonis_result$`Pr(>F)`[1] < 0.05) {
    cat("✓ Significant differences between biomes (p < 0.05)\n")
  } else {
    cat("✗ No significant differences between biomes\n")
  }
  
  return(adonis_result)
}

# Run beta dispersion analysis
run_beta_dispersion <- function(data, config, output_dir) {
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("2. BETA DISPERSION: Testing variance heterogeneity\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Get column names
  fungi_col <- config$columns$fungi
  plants_col <- config$columns$plants
  vertebrates_col <- config$columns$vertebrates
  biome_code_col <- config$columns$biome_code
  
  # Create community matrix
  community <- cbind(
    data[[fungi_col]],
    data[[plants_col]],
    data[[vertebrates_col]]
  )
  
  # Calculate distances and dispersion
  dist_comm <- dist(community, method = "euclidean")
  beta_disp <- betadisper(dist_comm, data[[biome_code_col]])
  
  # Test for differences
  permutations <- config$parameters$permutations
  permutest_result <- permutest(beta_disp, permutations = permutations)
  print(permutest_result)
  
  if (permutest_result$tab$`Pr(>F)`[1] < 0.05) {
    cat("\n✓ Significant heterogeneity in variance between biomes\n")
  } else {
    cat("\n✗ Homogeneous variance across biomes\n")
  }
  
  # Save plot
  fig_width <- config$visualization$figure_width
  fig_height <- config$visualization$figure_height
  fig_dpi <- config$visualization$dpi
  
  png(file.path(output_dir, "beta_dispersion.png"), 
      width = fig_width, height = fig_height, res = fig_dpi)
  plot(beta_disp, main = "Beta Dispersion by Biome")
  dev.off()
  
  return(list(beta_disp = beta_disp, permutest = permutest_result))
}

# Run NMDS ordination
run_nmds <- function(data, config, output_dir) {
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("3. NMDS ORDINATION: Visualizing community structure\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Get column names
  fungi_col <- config$columns$fungi
  plants_col <- config$columns$plants
  vertebrates_col <- config$columns$vertebrates
  biome_code_col <- config$columns$biome_code
  biome_name_col <- config$columns$biome_name
  
  # Create community matrix
  community <- cbind(
    data[[fungi_col]],
    data[[plants_col]],
    data[[vertebrates_col]]
  )
  
  # Run NMDS
  set.seed(config$parameters$random_seed)
  nmds <- metaMDS(community, distance = "euclidean", k = 2, trymax = 100)
  
  cat(sprintf("NMDS stress: %.4f\n", nmds$stress))
  if (nmds$stress < 0.05) {
    cat("Excellent representation\n")
  } else if (nmds$stress < 0.1) {
    cat("Good representation\n")
  } else if (nmds$stress < 0.2) {
    cat("Reasonable representation\n")
  } else {
    cat("Poor representation (high stress)\n")
  }
  
  # Create plot
  nmds_scores <- as.data.frame(scores(nmds, display = "sites"))
  nmds_scores$biome <- data[[biome_code_col]]
  nmds_scores$biome_name <- data[[biome_name_col]]
  
  # Create NMDS plot with ggplot2
  p <- ggplot(nmds_scores, aes(x = NMDS1, y = NMDS2, color = biome_name)) +
    geom_point(alpha = 0.6, size = 1) +
    stat_ellipse(level = 0.95, show.legend = FALSE) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom",
          legend.title = element_blank(),
          legend.text = element_text(size = 9)) +
    labs(title = "NMDS Ordination of Biodiversity by Biome",
         subtitle = sprintf("Stress = %.3f", nmds$stress)) +
    guides(color = guide_legend(ncol = 2))
  
  # Save plot
  fig_width <- config$visualization$figure_width
  fig_height <- config$visualization$figure_height
  fig_dpi <- config$visualization$dpi
  
  png(file.path(output_dir, "nmds_by_biome.png"), 
      width = fig_width + 200, height = fig_height + 200, res = fig_dpi)
  print(p)
  dev.off()
  
  return(nmds)
}

# Main analysis function
permanova_analysis <- function(config) {
  
  # Get paths from config
  data_path <- config$data$biodiversity_with_biomes
  output_dir <- file.path(config$output$base_dir, config$output$permanova_analysis)
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Load and prepare data
  data <- load_and_prepare_data(data_path, config)
  
  # Filter valid biomes
  biome_code_col <- config$columns$biome_code
  biome_name_col <- config$columns$biome_name
  min_samples <- config$parameters$min_samples_per_biome
  
  cat("\nFiltering biomes...\n")
  biome_counts <- table(data[[biome_code_col]])
  valid_biomes <- names(biome_counts)[biome_counts >= min_samples]
  
  data <- data[data[[biome_code_col]] %in% valid_biomes, ]
  
  cat("Using", length(valid_biomes), "biomes with >=", min_samples, "samples each\n")
  cat("Total samples for analysis:", format(nrow(data), big.mark=","), "\n")
  
  # Print biome distribution
  cat("\nBiome distribution:\n")
  biome_summary <- data %>%
    group_by(!!sym(biome_code_col), !!sym(biome_name_col)) %>%
    summarise(n = n(), .groups = "drop") %>%
    arrange(desc(n))
  
  for (i in 1:nrow(biome_summary)) {
    cat(sprintf("  %s: %s sites\n", 
                biome_summary[[biome_name_col]][i],
                format(biome_summary$n[i], big.mark=",")))
  }
  
  # Run analyses
  adonis_result <- run_permanova_analysis(data, config)
  beta_results <- run_beta_dispersion(data, config, output_dir)
  nmds <- run_nmds(data, config, output_dir)
  
  # Save results
  results <- list(
    adonis_result = adonis_result,
    beta_dispersion = beta_results$beta_disp,
    permutest_result = beta_results$permutest,
    nmds = nmds,
    n_samples = nrow(data),
    n_biomes = length(valid_biomes),
    biome_summary = biome_summary
  )
  
  saveRDS(results, file.path(output_dir, "permanova_results.rds"))
  
  # Save summary text
  sink(file.path(output_dir, "permanova_summary.txt"))
  cat("PERMANOVA AND BETA DISPERSION ANALYSIS SUMMARY\n")
  cat("=" , rep("=", 59), "\n", sep="")
  cat("Date:", Sys.Date(), "\n")
  cat("Samples analyzed:", format(nrow(data), big.mark=","), "\n")
  cat("Biomes included:", length(valid_biomes), "\n")
  cat("Permutations:", config$parameters$permutations, "\n\n")
  
  cat("PERMANOVA Results:\n")
  print(adonis_result)
  cat("\n")
  
  cat("Beta Dispersion Results:\n")
  print(beta_results$permutest)
  cat("\n")
  
  cat("NMDS Stress:", nmds$stress, "\n")
  sink()
  
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("ANALYSIS COMPLETE\n")
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("Results saved to:", output_dir, "\n")
  cat("  - permanova_results.rds (R data file)\n")
  cat("  - permanova_summary.txt (text summary)\n")
  cat("  - beta_dispersion.png\n")
  cat("  - nmds_by_biome.png\n")
  
  return(results)
}

# Main execution
if (!interactive()) {
  # Determine script directory
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", args[grep("--file=", args)])
  if (length(script_path) == 0) {
    script_path <- sys.frames()[[1]]$ofile
  }
  script_dir <- dirname(normalizePath(script_path))
  
  # Load config from script directory
  config_path <- file.path(script_dir, "config.yaml")
  if (!file.exists(config_path)) {
    stop("ERROR: config.yaml not found in ", script_dir)
  }
  
  config <- read_yaml(config_path)
  
  # Run analysis
  results <- permanova_analysis(config)
}