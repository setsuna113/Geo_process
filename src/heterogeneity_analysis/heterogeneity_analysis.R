#!/usr/bin/env Rscript
# ============================================================================
# Heterogeneity Analysis of Biodiversity Relationships
# ============================================================================
# Tests whether biological relationships (correlations) vary across environmental
# contexts (biomes). This is fundamentally about testing context-dependency of
# ecological relationships.
#
# Author: Biodiversity-Environment Analysis Pipeline
# Date: 2024
# ============================================================================

library(vegan)
library(arrow)
library(dplyr)
library(ggplot2)

# Try to load optional packages
has_cocor <- requireNamespace("cocor", quietly = TRUE)
if (has_cocor) library(cocor)

#' Perform heterogeneity analysis across biomes
#'
#' @param data_path Path to biodiversity data (RAW data, not residuals)
#' @param output_dir Directory to save results
#' @param min_samples_per_biome Minimum samples required for biome analysis
#' @param permutations Number of permutations for tests
#' @return List containing heterogeneity analysis results
heterogeneity_analysis <- function(
  data_path,
  output_dir = "./results/heterogeneity_analysis",
  min_samples_per_biome = 100,
  permutations = 999,
  use_preassigned_biomes = FALSE
) {
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Load data
  cat("Loading data from:", data_path, "\n")
  data <- read_feather(data_path)
  
  n_samples <- nrow(data)
  cat("Total samples:", format(n_samples, big.mark=","), "\n")
  
  # Handle column name variations
  cat("Checking and standardizing column names...\n")
  
  # Fungi - could be separate AM and ECM or combined
  if ("fungi_richness" %in% names(data)) {
    # Already has combined fungi column
  } else if (all(c("am_fungi_richness", "ecm_fungi_richness") %in% names(data))) {
    # Combine AM and ECM fungi
    data$fungi_richness <- data$am_fungi_richness + data$ecm_fungi_richness
    cat("  Combined AM and ECM fungi into total fungi richness\n")
  } else {
    stop("Could not find fungi richness data (need 'fungi_richness' or both 'am_fungi_richness' and 'ecm_fungi_richness')")
  }
  
  # Plants
  if (!"plants_richness" %in% names(data)) {
    stop("Missing required column: plants_richness")
  }
  
  # Vertebrates - might be called terrestrial_richness
  if ("vertebrates_richness" %in% names(data)) {
    # Already has vertebrates column
  } else if ("terrestrial_richness" %in% names(data)) {
    data$vertebrates_richness <- data$terrestrial_richness
    cat("  Using terrestrial_richness as vertebrates_richness\n")
  } else {
    stop("Could not find vertebrates data (need 'vertebrates_richness' or 'terrestrial_richness')")
  }
  
  # Coordinates - might be x/y or longitude/latitude
  if (!all(c("longitude", "latitude") %in% names(data))) {
    if (all(c("x", "y") %in% names(data))) {
      data$longitude <- data$x
      data$latitude <- data$y
      cat("  Using x/y as longitude/latitude\n")
    } else {
      stop("Could not find coordinates (need 'longitude'/'latitude' or 'x'/'y')")
    }
  }
  
  # Sample if dataset is too large
  max_samples <- 50000  # Reduced for computational feasibility
  if (n_samples > max_samples) {
    cat("\nDataset is very large (", format(n_samples, big.mark=","), " rows)\n", sep="")
    cat("Sampling ", format(max_samples, big.mark=","), " rows for analysis...\n", sep="")
    set.seed(42)
    sample_idx <- sample(n_samples, max_samples)
    data <- data[sample_idx, ]
    n_samples <- nrow(data)
  }
  
  cat("\nProceeding with", format(n_samples, big.mark=","), "samples\n\n")
  
  # ============ Assign Biomes ============
  
  # Check if biomes are already assigned
  if ("biome_code" %in% names(data) && "biome_name" %in% names(data)) {
    cat("Using pre-assigned WWF biomes from data file\n")
    use_preassigned_biomes <- TRUE
  }
  
  if (use_preassigned_biomes && "biome_code" %in% names(data)) {
    cat("Using pre-assigned biomes...\n")
    # Biomes already in the data, just ensure they're factors
    data$biome_code <- as.factor(data$biome_code)
    
    # Remove any NA biomes
    n_missing <- sum(is.na(data$biome_code))
    if (n_missing > 0) {
      cat(sprintf("Removing %d sites with missing biome assignments\n", n_missing))
      data <- data[!is.na(data$biome_code), ]
    }
  } else {
    cat("Loading WWF Terrestrial Ecoregions (REQUIRED)...\n")
  
  # WWF data is MANDATORY - no fallbacks allowed
  wwf_dir <- "/scratch/yl998/data/wwf_ecoregions"
  wwf_file <- NULL
  
  # Check for existing WWF shapefiles
  possible_paths <- c(
    file.path(wwf_dir, "Ecoregions2017.shp"),
    file.path(wwf_dir, "wwf_terr_ecos.shp"),
    file.path(wwf_dir, "official/wwf_terr_ecos.shp"),
    file.path(wwf_dir, "WWF_ecoregions/wwf_terr_ecos.shp"),
    file.path(wwf_dir, "official_teow/wwf_terr_ecos.shp")
  )
  
  for (path in possible_paths) {
    if (file.exists(path)) {
      wwf_file <- path
      cat("Found WWF shapefile at:", wwf_file, "\n")
      break
    }
  }
  
  # If not found, try to find any shapefile in the directory
  if (is.null(wwf_file) && dir.exists(wwf_dir)) {
    shp_files <- list.files(wwf_dir, pattern = "\\.shp$", recursive = TRUE, full.names = TRUE)
    if (length(shp_files) > 0) {
      wwf_file <- shp_files[1]
      cat("Found shapefile at:", wwf_file, "\n")
    }
  }
  
  # Download if not found - WITH SAFETY LIMITS
  max_attempts <- 2  # Reduced from 3
  attempt <- 1
  max_download_size <- 200 * 1024 * 1024  # 200MB limit
  temp_files <- c()  # Track temp files for cleanup
  
  while (is.null(wwf_file) || !file.exists(wwf_file)) {
    cat(sprintf("\nAttempt %d/%d: Downloading WWF ecoregions...\n", attempt, max_attempts))
    
    dir.create(wwf_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Only use verified URL
    url <- "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
    zip_file <- file.path(wwf_dir, paste0("wwf_attempt_", attempt, ".zip"))
    temp_files <- c(temp_files, zip_file)
    
    # Clean up any previous failed attempts
    if (file.exists(zip_file)) {
      file.remove(zip_file)
    }
    
    success <- FALSE
    tryCatch({
      # Download with size limit and timeout
      cat("Downloading from:", url, "\n")
      cat("Max size: 200MB, Timeout: 60s\n")
      
      # Use curl with size limit instead of wget
      download_result <- download.file(
        url = url,
        destfile = zip_file,
        method = "curl",
        quiet = FALSE,
        mode = "wb",
        extra = c("--max-filesize", as.character(max_download_size),
                  "--connect-timeout", "30",
                  "--max-time", "120")
      )
      
      # Verify download
      if (file.exists(zip_file)) {
        file_info <- file.info(zip_file)
        
        # Check size is reasonable (WWF data is ~150MB)
        if (file_info$size > 100000000 && file_info$size < max_download_size) {
          cat("Download complete:", round(file_info$size/1024/1024, 1), "MB\n")
          
          # Extract with error handling
          unzip_result <- tryCatch({
            unzip(zip_file, exdir = wwf_dir, overwrite = FALSE)
            TRUE
          }, error = function(e) {
            cat("Extraction failed:", e$message, "\n")
            FALSE
          })
          
          if (unzip_result) {
            # Find the shapefile
            shp_files <- list.files(wwf_dir, pattern = "\\.shp$", recursive = TRUE, full.names = TRUE)
            if (length(shp_files) > 0) {
              wwf_file <- shp_files[1]
              cat("Found shapefile:", wwf_file, "\n")
              success <- TRUE
              
              # Clean up zip file after successful extraction
              file.remove(zip_file)
            }
          }
        } else {
          cat("Invalid file size:", round(file_info$size/1024/1024, 1), "MB\n")
        }
      }
    }, error = function(e) {
      cat("Download error:", e$message, "\n")
    }, finally = {
      # Always clean up failed downloads
      if (!success && file.exists(zip_file)) {
        file.remove(zip_file)
      }
    })
    
    if (success) break
    
    attempt <- attempt + 1
    if (attempt > max_attempts) {
      # Clean up all temp files before stopping
      for (temp_file in temp_files) {
        if (file.exists(temp_file)) file.remove(temp_file)
      }
      
      stop("FATAL: Could not download WWF ecoregions after ", max_attempts, 
           " attempts. WWF data is REQUIRED for heterogeneity analysis.\n",
           "The WWF shapefile is already downloaded at: /scratch/yl998/data/wwf_ecoregions/Ecoregions2017.shp\n",
           "Or download manually (150MB) from: https://storage.googleapis.com/teow2016/Ecoregions2017.zip")
    }
    
    Sys.sleep(1)  # Shorter wait
  }
  
  # Load sf package (required for WWF data)
  if (!requireNamespace("sf", quietly = TRUE)) {
    cat("Installing sf package (required for WWF data)...\n")
    install.packages("sf", repos = "https://cloud.r-project.org/")
  }
  library(sf)
  
  # Read WWF shapefile
  cat("\nReading WWF shapefile from:", wwf_file, "\n")
  wwf <- st_read(wwf_file, quiet = TRUE)
  
  cat("Successfully loaded WWF ecoregions with", nrow(wwf), "polygons\n")
  cat("Assigning sites to WWF biomes...\n")
  
  # Create spatial points
  sites_sf <- st_as_sf(data, 
                       coords = c("longitude", "latitude"),
                       crs = 4326,
                       remove = FALSE)
  
  # Spatial join with WWF ecoregions
  sites_with_biomes <- st_join(sites_sf, wwf[c("BIOME", "BIOME_NAME")], 
                               join = st_intersects)
  
  # Extract biome information
  data$biome_code <- as.factor(sites_with_biomes$BIOME)
  data$biome_name <- sites_with_biomes$BIOME_NAME
  
  # Check for missing assignments
  n_missing <- sum(is.na(data$biome_code))
  if (n_missing > 0) {
    cat(sprintf("\nWarning: %d sites (%.1f%%) could not be assigned to terrestrial biomes\n",
                n_missing, n_missing / nrow(data) * 100))
    cat("These are likely marine or lake sites\n")
    cat("Removing unassigned sites from analysis...\n")
    data <- data[!is.na(data$biome_code), ]
  }
  
  cat("\nSuccessfully assigned all sites to WWF biomes\n")
  } # End of else block for WWF assignment
  
  # Print biome distribution
  biome_table <- table(data$biome_name)
  cat("\nBiome distribution:\n")
  for (i in 1:length(biome_table)) {
    cat(sprintf("  %s: %s sites\n", 
                names(biome_table)[i], 
                format(biome_table[i], big.mark=",")))
  }
  
  # Filter out biomes with too few samples
  biome_counts <- as.data.frame(table(data$biome_code))
  valid_biomes <- biome_counts$Var1[biome_counts$Freq >= min_samples_per_biome]
  
  if (length(valid_biomes) < 2) {
    stop("Not enough biomes with sufficient samples for heterogeneity analysis")
  }
  
  data <- data[data$biome_code %in% valid_biomes, ]
  n_samples <- nrow(data)
  
  cat(sprintf("\nUsing %d biomes with >= %d samples each\n", 
              length(valid_biomes), min_samples_per_biome))
  cat(sprintf("Total samples for analysis: %s\n\n", 
              format(n_samples, big.mark=",")))
  
  # Remove any rows with NA values in biodiversity columns
  complete_rows <- complete.cases(data[, c("fungi_richness", "plants_richness", "vertebrates_richness")])
  if (sum(!complete_rows) > 0) {
    cat(sprintf("Removing %d rows with missing biodiversity data\n", sum(!complete_rows)))
    data <- data[complete_rows, ]
    n_samples <- nrow(data)
    cat(sprintf("Continuing with %s complete samples\n\n", format(n_samples, big.mark=",")))
  }
  
  # ============ 1. PERMANOVA: Test Composition Differences ============
  
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("1. PERMANOVA: Testing for biome differences in composition\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Create community matrix
  community <- cbind(
    data$fungi_richness,
    data$plants_richness,
    data$vertebrates_richness
  )
  colnames(community) <- c("fungi", "plants", "vertebrates")
  
  # Run PERMANOVA with fewer permutations for speed
  adonis_result <- adonis2(
    community ~ biome_code,
    data = data,
    permutations = 99,  # Reduced for speed
    method = "euclidean"
  )
  
  print(adonis_result)
  
  if (adonis_result$`Pr(>F)`[1] < 0.05) {
    cat("\n✓ Significant differences in biodiversity composition between biomes\n")
  } else {
    cat("\n✗ No significant differences between biomes\n")
  }
  
  # ============ 2. BETA DISPERSION: Test Variance Heterogeneity ============
  
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("2. BETA DISPERSION: Testing variance heterogeneity\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Calculate distances
  dist_comm <- dist(community, method = "euclidean")
  
  # Beta dispersion analysis
  beta_disp <- betadisper(dist_comm, data$biome_code)
  
  # Test for differences in dispersion
  permutest_result <- permutest(beta_disp, permutations = 99)
  print(permutest_result)
  
  if (permutest_result$tab$`Pr(>F)`[1] < 0.05) {
    cat("\n✓ Significant heterogeneity in variance between biomes\n")
  } else {
    cat("\n✗ Homogeneous variance across biomes\n")
  }
  
  # Save beta dispersion plot
  pdf(file.path(output_dir, "beta_dispersion.pdf"), width = 10, height = 8)
  plot(beta_disp, main = "Beta Dispersion by Biome")
  dev.off()
  
  # ============ 3. CORRELATION ANALYSIS BY BIOME (Core Analysis) ============
  
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("3. CORRELATION ANALYSIS BY BIOME\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Get biomes with sufficient samples
  biome_counts <- table(data$biome_code)
  valid_biomes <- names(biome_counts)[biome_counts >= min_samples_per_biome]
  
  cat(sprintf("Analyzing %d biomes with >= %d samples\n\n",
              length(valid_biomes), min_samples_per_biome))
  
  # Store correlations for each biome
  biome_correlations <- list()
  correlation_summary <- data.frame()
  
  for (biome in valid_biomes) {
    idx <- which(data$biome_code == biome)
    biome_name <- unique(data$biome_name[idx])[1]
    n_biome <- length(idx)
    
    cat(sprintf("Biome %s: %s (n=%d)\n", biome, biome_name, n_biome))
    
    # Calculate correlation matrix
    biome_data <- data[idx, c("fungi_richness", "plants_richness", "vertebrates_richness")]
    cor_mat <- cor(biome_data, use = "complete.obs")
    
    # Store full correlation matrix
    biome_correlations[[biome]] <- list(
      cor_matrix = cor_mat,
      n = n_biome,
      biome_name = biome_name
    )
    
    # Extract pairwise correlations
    fungi_plants <- cor_mat[1, 2]
    fungi_vertebrates <- cor_mat[1, 3]
    plants_vertebrates <- cor_mat[2, 3]
    
    cat(sprintf("  Fungi-Plants: r = %.3f\n", fungi_plants))
    cat(sprintf("  Fungi-Vertebrates: r = %.3f\n", fungi_vertebrates))
    cat(sprintf("  Plants-Vertebrates: r = %.3f\n", plants_vertebrates))
    
    # Add to summary
    correlation_summary <- rbind(correlation_summary, data.frame(
      biome_code = biome,
      biome_name = biome_name,
      n_sites = n_biome,
      cor_fungi_plants = fungi_plants,
      cor_fungi_vertebrates = fungi_vertebrates,
      cor_plants_vertebrates = plants_vertebrates,
      mean_abs_cor = mean(abs(c(fungi_plants, fungi_vertebrates, plants_vertebrates)))
    ))
  }
  
  # Save correlation summary
  write.csv(correlation_summary,
            file.path(output_dir, "correlation_summary_by_biome.csv"),
            row.names = FALSE)
  
  # ============ 4. TEST CORRELATION DIFFERENCES ============
  
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("4. TESTING CORRELATION DIFFERENCES BETWEEN BIOMES\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Compare correlations between major biome types
  # Example: Tropical vs Temperate
  tropical_idx <- grep("Tropical", data$biome_name)
  temperate_idx <- grep("Temperate", data$biome_name)
  
  comparison_results <- list()
  
  if (length(tropical_idx) >= min_samples_per_biome && 
      length(temperate_idx) >= min_samples_per_biome) {
    
    cat("Comparing Tropical vs Temperate biomes:\n\n")
    
    # Calculate correlations
    trop_data <- data[tropical_idx, c("fungi_richness", "plants_richness", "vertebrates_richness")]
    temp_data <- data[temperate_idx, c("fungi_richness", "plants_richness", "vertebrates_richness")]
    
    cor_tropical <- cor(trop_data, use = "complete.obs")
    cor_temperate <- cor(temp_data, use = "complete.obs")
    
    # Test each pairwise correlation
    pairs <- list(
      c("Fungi-Plants", 1, 2),
      c("Fungi-Vertebrates", 1, 3),
      c("Plants-Vertebrates", 2, 3)
    )
    
    for (pair in pairs) {
      r1 <- cor_tropical[pair[[2]], pair[[3]]]
      r2 <- cor_temperate[pair[[2]], pair[[3]]]
      n1 <- length(tropical_idx)
      n2 <- length(temperate_idx)
      
      # Fisher's Z transformation
      z1 <- 0.5 * log((1 + r1) / (1 - r1))
      z2 <- 0.5 * log((1 + r2) / (1 - r2))
      
      # Test statistic
      se_diff <- sqrt(1/(n1-3) + 1/(n2-3))
      z_stat <- (z1 - z2) / se_diff
      p_value <- 2 * (1 - pnorm(abs(z_stat)))
      
      cat(sprintf("%s:\n", pair[[1]]))
      cat(sprintf("  Tropical: r = %.3f (n = %d)\n", r1, n1))
      cat(sprintf("  Temperate: r = %.3f (n = %d)\n", r2, n2))
      cat(sprintf("  Difference: %.3f, p = %.4f %s\n\n", 
                  r1 - r2, p_value, ifelse(p_value < 0.05, "***", "")))
      
      comparison_results[[pair[[1]]]] <- list(
        r_tropical = r1,
        r_temperate = r2,
        difference = r1 - r2,
        p_value = p_value,
        n_tropical = n1,
        n_temperate = n2
      )
    }
  }
  
  # ============ 5. VISUALIZATION: NMDS ============
  
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("5. NMDS ORDINATION\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Sample for NMDS if too many points
  nmds_sample_size <- min(5000, n_samples)
  if (n_samples > nmds_sample_size) {
    cat(sprintf("Sampling %d sites for NMDS visualization...\n", nmds_sample_size))
    set.seed(42)
    nmds_idx <- sample(1:n_samples, nmds_sample_size)
    nmds_data <- data[nmds_idx, ]
    nmds_community <- community[nmds_idx, ]
  } else {
    nmds_data <- data
    nmds_community <- community
  }
  
  # Perform NMDS
  cat("Running NMDS...\n")
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
    labs(title = "Biodiversity Patterns by Biome",
         subtitle = sprintf("NMDS ordination (stress = %.3f)", nmds$stress),
         x = "NMDS Axis 1",
         y = "NMDS Axis 2")
  
  print(p)
  dev.off()
  
  # ============ 6. SUMMARY VISUALIZATION ============
  
  cat("\nGenerating summary plots...\n")
  
  # Plot correlation heatmap by biome
  pdf(file.path(output_dir, "correlation_heterogeneity.pdf"), width = 12, height = 8)
  
  # Prepare data for heatmap
  cor_long <- reshape2::melt(correlation_summary[, c("biome_name", "cor_fungi_plants", 
                                                     "cor_fungi_vertebrates", "cor_plants_vertebrates")])
  names(cor_long) <- c("Biome", "Pair", "Correlation")
  
  ggplot(cor_long, aes(x = Pair, y = Biome, fill = Correlation)) +
    geom_tile() +
    geom_text(aes(label = round(Correlation, 2)), color = "white", size = 3) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                        midpoint = 0, limit = c(-1, 1)) +
    labs(title = "Heterogeneity of Biodiversity Correlations Across Biomes",
         x = "Taxon Pair", y = "Biome") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  dev.off()
  
  # ============ SAVE RESULTS ============
  
  results <- list(
    biome_correlations = biome_correlations,
    correlation_summary = correlation_summary,
    adonis_result = adonis_result,
    beta_dispersion = beta_disp,
    permutest_result = permutest_result,
    comparison_results = comparison_results,
    nmds = nmds,
    n_samples = n_samples,
    n_biomes_analyzed = length(valid_biomes)
  )
  
  saveRDS(results, file.path(output_dir, "heterogeneity_analysis_results.rds"))
  
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("ANALYSIS COMPLETE\n")
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("Results saved to:", output_dir, "\n")
  
  return(results)
}

# Run if executed directly as script (not when sourced)
if (!interactive() && sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) > 0) {
    data_path <- args[1]
  } else {
    stop("Please provide data path as argument")
  }
  
  if (length(args) > 1) {
    output_dir <- args[2]
  } else {
    output_dir <- "./results/heterogeneity_analysis"
  }
  
  results <- heterogeneity_analysis(data_path, output_dir)
}