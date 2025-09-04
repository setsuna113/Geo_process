#!/usr/bin/env Rscript
# ============================================================================
# Spearman Correlation Analysis for Biodiversity Relationships Across Biomes
# Config-driven analysis - no hardcoded fallbacks
# ============================================================================

library(arrow)
library(dplyr)
library(ggplot2)
library(reshape2)
library(yaml)

# Load configuration
load_config <- function() {
  config_path <- file.path(dirname(sys.frame(1)$ofile), "config.yaml")
  if (!file.exists(config_path)) {
    stop("ERROR: config.yaml not found in ", dirname(sys.frame(1)$ofile))
  }
  config <- read_yaml(config_path)
  return(config)
}

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
  
  # Check for required columns
  required_cols <- c(fungi_col, plants_col, vertebrates_col, 
                     config$columns$biome_code, config$columns$biome_name)
  missing_cols <- setdiff(required_cols, names(data))
  
  if (length(missing_cols) > 0) {
    stop("ERROR: Required columns missing from data: ", paste(missing_cols, collapse=", "))
  }
  
  # Sample if needed
  max_samples <- config$parameters$max_samples_spearman
  if (n_samples > max_samples) {
    cat("Sampling", format(max_samples, big.mark=","), "rows for analysis...\n")
    set.seed(config$parameters$random_seed)
    data <- data[sample(n_samples, max_samples), ]
  }
  
  return(data)
}

# Main analysis function
spearman_correlation_analysis <- function(config) {
  
  # Get paths from config
  data_path <- config$data$biodiversity_with_biomes
  output_dir <- file.path(config$output$base_dir, config$output$spearman_correlations)
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Load data
  data <- load_and_prepare_data(data_path, config)
  
  # Get column names
  fungi_col <- config$columns$fungi
  plants_col <- config$columns$plants
  vertebrates_col <- config$columns$vertebrates
  biome_code_col <- config$columns$biome_code
  biome_name_col <- config$columns$biome_name
  
  # Filter valid biomes
  cat("\nAnalyzing biomes...\n")
  biome_counts <- table(data[[biome_code_col]])
  min_samples <- config$parameters$min_samples_per_biome
  valid_biomes <- names(biome_counts)[biome_counts >= min_samples]
  
  data <- data[data[[biome_code_col]] %in% valid_biomes, ]
  cat("Using", length(valid_biomes), "biomes with >=", min_samples, "samples\n")
  cat("Total samples for analysis:", format(nrow(data), big.mark=","), "\n\n")
  
  # Calculate correlations by biome
  cat("Calculating Spearman correlations by biome...\n")
  correlation_results <- list()
  
  for (biome in valid_biomes) {
    biome_data <- data[data[[biome_code_col]] == biome, ]
    biome_name <- unique(biome_data[[biome_name_col]])[1]
    
    # Calculate Spearman correlations
    cor_matrix <- cor(biome_data[, c(fungi_col, plants_col, vertebrates_col)],
                     use = "complete.obs", method = "spearman")
    
    correlation_results[[biome]] <- data.frame(
      biome_code = biome,
      biome_name = biome_name,
      n_sites = nrow(biome_data),
      cor_fungi_plants = cor_matrix[fungi_col, plants_col],
      cor_fungi_vertebrates = cor_matrix[fungi_col, vertebrates_col],
      cor_plants_vertebrates = cor_matrix[plants_col, vertebrates_col],
      stringsAsFactors = FALSE
    )
    
    if (config$processing$verbosity >= 1) {
      cat(sprintf("  %s (n=%d): F-P=%.3f, F-V=%.3f, P-V=%.3f\n",
                  biome_name, nrow(biome_data),
                  cor_matrix[fungi_col, plants_col],
                  cor_matrix[fungi_col, vertebrates_col],
                  cor_matrix[plants_col, vertebrates_col]))
    }
  }
  
  # Combine results
  correlation_summary <- do.call(rbind, correlation_results)
  correlation_summary$mean_abs_cor <- rowMeans(abs(correlation_summary[, c("cor_fungi_plants", 
                                                                           "cor_fungi_vertebrates", 
                                                                           "cor_plants_vertebrates")]))
  
  # Order by mean absolute correlation
  correlation_summary <- correlation_summary[order(correlation_summary$mean_abs_cor, decreasing = TRUE), ]
  
  # Save CSV
  csv_path <- file.path(output_dir, "spearman_correlations_by_biome.csv")
  write.csv(correlation_summary, csv_path, row.names = FALSE)
  cat("\nSaved correlation summary to:", csv_path, "\n")
  
  # ============ GENERATE VISUALIZATIONS ============
  cat("\nGenerating visualizations...\n")
  
  # Get visualization settings
  fig_width <- config$visualization$figure_width
  fig_height <- config$visualization$figure_height
  fig_dpi <- config$visualization$dpi
  
  # Prepare for plotting - shorten long biome names
  correlation_summary$biome_short <- gsub("Tropical & Subtropical ", "Trop. ", correlation_summary$biome_name)
  correlation_summary$biome_short <- gsub("Temperate ", "Temp. ", correlation_summary$biome_short)
  correlation_summary$biome_short <- gsub("Forests, Woodlands & Scrub", "Forest/Scrub", correlation_summary$biome_short)
  correlation_summary$biome_short <- gsub("Grasslands, Savannas & Shrublands", "Grass/Savanna", correlation_summary$biome_short)
  correlation_summary$biome_short <- gsub("Broadleaf & Mixed Forests", "Broadleaf", correlation_summary$biome_short)
  
  # 1. Main bar chart - all correlations by biome
  cat("  Creating main correlation bar chart...\n")
  cor_long <- melt(correlation_summary[, c("biome_short", "cor_fungi_plants", 
                                          "cor_fungi_vertebrates", "cor_plants_vertebrates")],
                  id.vars = "biome_short")
  
  levels(cor_long$variable) <- c("Fungi-Plants", "Fungi-Vertebrates", "Plants-Vertebrates")
  
  p1 <- ggplot(cor_long, aes(x = reorder(biome_short, value), y = value, fill = variable)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.8) +
    coord_flip() +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top",
          plot.margin = unit(c(1,1,1,1), "cm"),
          panel.grid.minor = element_blank()) +
    labs(title = "Spearman Correlations of Biodiversity Across Biomes",
         subtitle = paste("Based on", format(nrow(data), big.mark=","), "samples"),
         x = "",
         y = "Spearman Correlation Coefficient",
         fill = "Relationship") +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    scale_fill_brewer(palette = "Set1") +
    scale_y_continuous(limits = c(-0.5, 1.0), breaks = seq(-0.5, 1.0, 0.25))
  
  png(file.path(output_dir, "spearman_correlations_all.png"), 
      width = fig_width + 200, height = fig_height, res = fig_dpi)
  print(p1)
  dev.off()
  
  # 2. Fungi-Plants correlation comparison
  cat("  Creating Fungi-Plants correlation plot...\n")
  p2 <- ggplot(correlation_summary, aes(x = reorder(biome_short, cor_fungi_plants), 
                                        y = cor_fungi_plants, fill = cor_fungi_plants)) +
    geom_bar(stat = "identity", width = 0.8) +
    geom_text(aes(label = sprintf("%.2f", cor_fungi_plants)), 
              hjust = ifelse(correlation_summary$cor_fungi_plants < 0, 1.2, -0.2),
              size = 3.5) +
    coord_flip() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0,
                        limits = c(-0.5, 1), name = "Correlation") +
    theme_minimal(base_size = 12) +
    theme(plot.margin = unit(c(1,1,1,1), "cm"),
          panel.grid.minor = element_blank()) +
    labs(title = "Fungi-Plants Spearman Correlations Across Biomes",
         subtitle = paste("Sample sizes:", min(correlation_summary$n_sites), "-", 
                         max(correlation_summary$n_sites), "per biome"),
         x = "",
         y = "Spearman Correlation Coefficient") +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    scale_y_continuous(limits = c(-0.5, 1.0), breaks = seq(-0.5, 1.0, 0.25))
  
  png(file.path(output_dir, "spearman_fungi_plants.png"), 
      width = fig_width, height = fig_height, res = fig_dpi)
  print(p2)
  dev.off()
  
  # 3. Fungi-Vertebrates correlation comparison
  cat("  Creating Fungi-Vertebrates correlation plot...\n")
  p3 <- ggplot(correlation_summary, aes(x = reorder(biome_short, cor_fungi_vertebrates), 
                                        y = cor_fungi_vertebrates, fill = cor_fungi_vertebrates)) +
    geom_bar(stat = "identity", width = 0.8) +
    geom_text(aes(label = sprintf("%.2f", cor_fungi_vertebrates)), 
              hjust = ifelse(correlation_summary$cor_fungi_vertebrates < 0, 1.2, -0.2),
              size = 3.5) +
    coord_flip() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0,
                        limits = c(-0.5, 1), name = "Correlation") +
    theme_minimal(base_size = 12) +
    theme(plot.margin = unit(c(1,1,1,1), "cm"),
          panel.grid.minor = element_blank()) +
    labs(title = "Fungi-Vertebrates Spearman Correlations Across Biomes",
         subtitle = paste("Sample sizes:", min(correlation_summary$n_sites), "-", 
                         max(correlation_summary$n_sites), "per biome"),
         x = "",
         y = "Spearman Correlation Coefficient") +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    scale_y_continuous(limits = c(-0.5, 1.0), breaks = seq(-0.5, 1.0, 0.25))
  
  png(file.path(output_dir, "spearman_fungi_vertebrates.png"), 
      width = fig_width, height = fig_height, res = fig_dpi)
  print(p3)
  dev.off()
  
  # 4. Plants-Vertebrates correlation comparison
  cat("  Creating Plants-Vertebrates correlation plot...\n")
  p4 <- ggplot(correlation_summary, aes(x = reorder(biome_short, cor_plants_vertebrates), 
                                        y = cor_plants_vertebrates, fill = cor_plants_vertebrates)) +
    geom_bar(stat = "identity", width = 0.8) +
    geom_text(aes(label = sprintf("%.2f", cor_plants_vertebrates)), 
              hjust = ifelse(correlation_summary$cor_plants_vertebrates < 0, 1.2, -0.2),
              size = 3.5) +
    coord_flip() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0,
                        limits = c(-0.5, 1), name = "Correlation") +
    theme_minimal(base_size = 12) +
    theme(plot.margin = unit(c(1,1,1,1), "cm"),
          panel.grid.minor = element_blank()) +
    labs(title = "Plants-Vertebrates Spearman Correlations Across Biomes",
         subtitle = paste("Sample sizes:", min(correlation_summary$n_sites), "-", 
                         max(correlation_summary$n_sites), "per biome"),
         x = "",
         y = "Spearman Correlation Coefficient") +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    scale_y_continuous(limits = c(-0.5, 1.0), breaks = seq(-0.5, 1.0, 0.25))
  
  png(file.path(output_dir, "spearman_plants_vertebrates.png"), 
      width = fig_width, height = fig_height, res = fig_dpi)
  print(p4)
  dev.off()
  
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("ANALYSIS COMPLETE\n")
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("Results saved to:", output_dir, "\n")
  cat("  - spearman_correlations_by_biome.csv\n")
  cat("  - spearman_correlations_all.png (main bar chart)\n")
  cat("  - spearman_fungi_plants.png\n")
  cat("  - spearman_fungi_vertebrates.png\n")
  cat("  - spearman_plants_vertebrates.png\n")
  
  return(correlation_summary)
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
  results <- spearman_correlation_analysis(config)
}