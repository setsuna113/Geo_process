#!/usr/bin/env Rscript
# ============================================================================
# Biome-Based Analysis of Biological Networks
# ============================================================================
# Analyzes how biological interdependencies vary across Earth's biomes
# using WWF ecoregions classification
#
# Author: Biodiversity-Environment Analysis Pipeline  
# Date: 2024
# ============================================================================

library(vegan)
library(arrow)
library(speciesgeocodeR)
library(sp)
library(ggplot2)
library(dplyr)

#' Perform biome-based analysis of biological networks
#'
#' @param data_path Path to residuals arrow file
#' @param output_dir Directory to save results
#' @param sample_size Sample size for computationally intensive analyses
#' @param min_samples_per_biome Minimum samples required for biome analysis
#' @return List containing biome analysis results
biome_analysis <- function(
  data_path = "/scratch/yl998/econml_results_5m/homogeneous/residuals.arrow",
  output_dir = "./results/biome_analysis",
  sample_size = 10000,
  min_samples_per_biome = 1000
) {
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # ============ Load Data and WWF Biomes ============
  
  cat("Loading residuals from:", data_path, "\n")
  data <- read_feather(data_path)
  
  n_samples <- nrow(data)
  cat("Total samples:", format(n_samples, big.mark=","), "\n\n")
  
  # Check if we have coordinates
  if (!all(c("longitude", "latitude") %in% names(data))) {
    stop("Longitude and latitude columns are required for biome analysis")
  }
  
  cat("Loading WWF Terrestrial Ecoregions...\n")
  tryCatch({
    wwf <- WwfLoad()
    cat("WWF ecoregions loaded successfully\n\n")
  }, error = function(e) {
    cat("Error loading WWF data:", e$message, "\n")
    cat("Attempting to download WWF data...\n")
    # WwfLoad will automatically download if needed
    wwf <- WwfLoad()
  })
  
  # ============ Assign Biomes to Sites ============
  
  cat("Assigning biomes to", format(n_samples, big.mark=","), "sites...\n")
  cat("This may take several minutes for large datasets...\n")
  
  # Create SpatialPoints object
  coords_sp <- SpatialPoints(
    cbind(data$longitude, data$latitude),
    proj4string = CRS("+proj=longlat +datum=WGS84")
  )
  
  # Extract biome information
  biome_info <- over(coords_sp, wwf)
  
  # Add biome information to data
  data$biome_code <- as.factor(biome_info$BIOME)
  data$biome_name <- biome_info$BIOME_NAME
  data$ecoregion <- biome_info$ECO_NAME
  data$realm <- biome_info$REALM
  
  # Handle missing biomes (e.g., marine sites)
  n_missing <- sum(is.na(data$biome_code))
  if (n_missing > 0) {
    cat(sprintf("Warning: %d sites (%.1f%%) could not be assigned to terrestrial biomes\n",
                n_missing, n_missing / n_samples * 100))
    cat("These may be marine or unclassified locations\n\n")
    
    # Remove sites without biome assignment
    data <- data[!is.na(data$biome_code), ]
    n_samples <- nrow(data)
    cat("Continuing with", format(n_samples, big.mark=","), "terrestrial sites\n\n")
  }
  
  # ============ Biome Summary ============
  
  cat("=" * 60, "\n")
  cat("BIOME DISTRIBUTION\n")
  cat("=" * 60, "\n\n")
  
  biome_summary <- data %>%
    group_by(biome_code, biome_name) %>%
    summarise(
      n_sites = n(),
      percent = n() / n_samples * 100,
      .groups = 'drop'
    ) %>%
    arrange(desc(n_sites))
  
  print(as.data.frame(biome_summary))
  
  # Save biome summary
  write.csv(biome_summary, 
            file.path(output_dir, "biome_distribution.csv"),
            row.names = FALSE)
  
  # ============ Create Community Matrix ============
  
  community <- cbind(
    data$fungi_richness,
    data$plants_richness,
    data$vertebrates_richness
  )
  colnames(community) <- c("fungi", "plants", "vertebrates")
  
  # ============ PERMANOVA: Test Biome Differences ============
  
  cat("\n" * 2)
  cat("=" * 60, "\n")
  cat("PERMANOVA: Testing for biome differences\n")
  cat("=" * 60, "\n\n")
  
  # Sample for PERMANOVA if dataset is large
  if (n_samples > sample_size) {
    cat(sprintf("Sampling %d sites for PERMANOVA analysis...\n", sample_size))
    set.seed(42)
    perm_idx <- sample(1:n_samples, sample_size)
    perm_data <- data[perm_idx, ]
    perm_community <- community[perm_idx, ]
  } else {
    perm_data <- data
    perm_community <- community
  }
  
  # Run PERMANOVA
  adonis_result <- adonis2(
    perm_community ~ biome_code,
    data = perm_data,
    permutations = 999,
    method = "euclidean"
  )
  
  print(adonis_result)
  
  # Save PERMANOVA results
  capture.output(adonis_result, 
                 file = file.path(output_dir, "permanova_results.txt"))
  
  # ============ Beta Dispersion Test ============
  
  cat("\n" * 2)
  cat("=" * 60, "\n")
  cat("BETA DISPERSION: Testing variance homogeneity\n")
  cat("=" * 60, "\n\n")
  
  # Calculate distances
  dist_comm <- dist(perm_community, method = "euclidean")
  
  # Beta dispersion analysis
  beta_disp <- betadisper(dist_comm, perm_data$biome_code)
  
  # Test for differences in dispersion
  permutest_result <- permutest(beta_disp, permutations = 999)
  print(permutest_result)
  
  # Save beta dispersion plot
  pdf(file.path(output_dir, "beta_dispersion.pdf"), width = 10, height = 8)
  plot(beta_disp, main = "Beta Dispersion by Biome")
  dev.off()
  
  # ============ Variance Partitioning by Biome ============
  
  cat("\n" * 2)
  cat("=" * 60, "\n")
  cat("VARIANCE PARTITIONING BY BIOME\n")
  cat("=" * 60, "\n\n")
  
  # Get biomes with sufficient samples
  biome_counts <- table(data$biome_code)
  valid_biomes <- names(biome_counts)[biome_counts >= min_samples_per_biome]
  
  cat(sprintf("Analyzing %d biomes with >= %d samples\n\n",
              length(valid_biomes), min_samples_per_biome))
  
  vp_by_biome <- list()
  biome_vp_summary <- data.frame()
  
  for (biome in valid_biomes) {
    idx <- which(data$biome_code == biome)
    biome_name <- unique(data$biome_name[idx])[1]
    n_biome <- length(idx)
    
    cat(sprintf("Biome %s: %s (n=%d)\n",
                biome, biome_name, n_biome))
    
    # Variance partitioning for fungi
    vp <- varpart(
      data$fungi_richness[idx],
      data$plants_richness[idx],
      data$vertebrates_richness[idx]
    )
    
    vp_by_biome[[biome]] <- vp
    
    # Extract key metrics
    total_r2 <- vp$part$fract$Adj.R.squared[3]
    shared_frac <- vp$part$indfract$Adj.R.squared[3]
    
    cat(sprintf("  Fungi R²: %.4f (%.1f%% shared)\n",
                total_r2, 
                ifelse(total_r2 > 0, shared_frac / total_r2 * 100, 0)))
    
    # Add to summary
    biome_vp_summary <- rbind(biome_vp_summary, data.frame(
      biome_code = biome,
      biome_name = biome_name,
      n_sites = n_biome,
      fungi_r2 = total_r2,
      plants_unique = vp$part$indfract$Adj.R.squared[1],
      vertebrates_unique = vp$part$indfract$Adj.R.squared[2],
      shared = shared_frac
    ))
  }
  
  # Save biome variance partitioning summary
  write.csv(biome_vp_summary,
            file.path(output_dir, "biome_variance_partitioning.csv"),
            row.names = FALSE)
  
  # ============ Compare Tropical vs Temperate ============
  
  cat("\n" * 2)
  cat("=" * 60, "\n")
  cat("TROPICAL VS TEMPERATE COMPARISON\n")
  cat("=" * 60, "\n\n")
  
  # Identify tropical and temperate biomes
  tropical_idx <- grep("Tropical", data$biome_name)
  temperate_idx <- grep("Temperate", data$biome_name)
  
  if (length(tropical_idx) > 0 && length(temperate_idx) > 0) {
    # Calculate correlations
    cor_tropical <- cor(community[tropical_idx, ])
    cor_temperate <- cor(community[temperate_idx, ])
    
    cat("Tropical correlations:\n")
    print(round(cor_tropical, 4))
    
    cat("\nTemperate correlations:\n")
    print(round(cor_temperate, 4))
    
    # Test difference in correlations
    cat("\nDifferences (Tropical - Temperate):\n")
    cor_diff <- cor_tropical - cor_temperate
    print(round(cor_diff, 4))
    
    # Save correlation comparison
    save(cor_tropical, cor_temperate, cor_diff,
         file = file.path(output_dir, "tropical_temperate_comparison.RData"))
  } else {
    cat("Insufficient tropical or temperate sites for comparison\n")
  }
  
  # ============ NMDS Ordination by Biome ============
  
  cat("\n" * 2)
  cat("=" * 60, "\n")
  cat("NMDS ORDINATION\n")
  cat("=" * 60, "\n\n")
  
  # Sample for visualization
  if (n_samples > 5000) {
    cat("Sampling 5000 sites for NMDS visualization...\n")
    set.seed(42)
    nmds_idx <- sample(1:n_samples, 5000)
    nmds_data <- data[nmds_idx, ]
    nmds_community <- community[nmds_idx, ]
  } else {
    nmds_data <- data
    nmds_community <- community
  }
  
  # Perform NMDS
  cat("Running NMDS (this may take a few minutes)...\n")
  nmds <- metaMDS(nmds_community, distance = "euclidean", 
                  k = 2, trymax = 100, autotransform = FALSE)
  
  cat(sprintf("NMDS stress: %.4f\n", nmds$stress))
  
  # Create NMDS plot
  pdf(file.path(output_dir, "nmds_by_biome.pdf"), width = 12, height = 10)
  
  # Extract scores
  sites_scores <- scores(nmds, display = "sites")
  
  # Create data frame for plotting
  plot_data <- data.frame(
    NMDS1 = sites_scores[, 1],
    NMDS2 = sites_scores[, 2],
    Biome = nmds_data$biome_name
  )
  
  # Create ggplot
  p <- ggplot(plot_data, aes(x = NMDS1, y = NMDS2, color = Biome)) +
    geom_point(alpha = 0.5, size = 1) +
    stat_ellipse(level = 0.95, linetype = 2) +
    theme_minimal() +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 8)) +
    labs(title = "Biological Network Patterns by Biome",
         subtitle = sprintf("NMDS ordination (stress = %.3f)", nmds$stress),
         x = "NMDS Axis 1",
         y = "NMDS Axis 2")
  
  print(p)
  dev.off()
  
  # ============ Save Results ============
  
  results <- list(
    biome_summary = biome_summary,
    adonis_result = adonis_result,
    beta_dispersion = beta_disp,
    vp_by_biome = vp_by_biome,
    biome_vp_summary = biome_vp_summary,
    nmds = nmds,
    n_samples = n_samples
  )
  
  saveRDS(results, file.path(output_dir, "biome_analysis_results.rds"))
  
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
    output_dir <- "./results/biome_analysis"
  }
  
  results <- biome_analysis(data_path, output_dir)
}