#!/usr/bin/env Rscript

# Enhanced Mycorrhizal SEM Analysis with Vegetation Structure
# Incorporates MODIS vegetation cover as mediating variables
# Model: Environment -> Vegetation Structure -> Mycorrhizae <-> Plants -> Animals

library(piecewiseSEM)
library(dplyr)
library(ggplot2)

# Set up paths
data_path <- Sys.getenv("BIODIV_DATA_PATH", 
                        default = "/scratch/yl998/biodiversity_climate_production_WITH_MOD44B.arrow")
if (!file.exists(data_path)) {
  stop("Data file not found. Set BIODIV_DATA_PATH or check /scratch/yl998/")
}

output_dir <- "src/sem/outputs_vegetation"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Enhanced Mycorrhizal SEM Analysis with Vegetation Structure ===\n")
cat("Loading data from:", data_path, "\n")

# Load data
if (requireNamespace("arrow", quietly = TRUE)) {
  df <- arrow::read_feather(data_path)
} else if (requireNamespace("feather", quietly = TRUE)) {
  df <- feather::read_feather(data_path)
} else {
  stop("Neither arrow nor feather package available")
}

cat("Data dimensions:", nrow(df), "rows x", ncol(df), "columns\n\n")

# Prepare enhanced SEM data with vegetation structure
sem_data <- df %>%
  rename(
    # Standardize naming
    plant_richness = plants_richness,
    animal_richness = terrestrial_richness,
    temp = annual_temp,
    precip = annual_precip,
    tree_cover = modis_tree_cover_pct,
    nontree_veg = modis_nontree_veg_pct,
    bare_ground = modis_bare_ground_pct
  ) %>%
  mutate(
    # Combine mycorrhizal richness
    mycorrhizal_richness = case_when(
      !is.na(am_fungi_richness) & !is.na(ecm_fungi_richness) ~ am_fungi_richness + ecm_fungi_richness,
      !is.na(am_fungi_richness) ~ am_fungi_richness,
      !is.na(ecm_fungi_richness) ~ ecm_fungi_richness,
      TRUE ~ NA_real_
    ),
    # Create vegetation complexity index
    vegetation_complexity = tree_cover + (nontree_veg * 0.5),  # Trees weighted more than grasses
    # Habitat heterogeneity from vegetation diversity
    habitat_heterogeneity = sqrt(tree_cover * nontree_veg)
  ) %>%
  select(
    # Coordinates
    x, y,
    # Biodiversity
    mycorrhizal_richness, am_fungi_richness, ecm_fungi_richness,
    plant_richness, animal_richness,
    # Climate
    temp, temp_seasonality, precip, precip_seasonality,
    # Vegetation structure (NEW)
    tree_cover, nontree_veg, bare_ground,
    vegetation_complexity, habitat_heterogeneity,
    # Topography
    elevation, slope, tpi,
    # Soil
    soil_organic_carbon, soil_ph, sand_content_pct,
    # Energy
    pet
  )

# Remove NA values in key variables
cat("Cleaning data...\n")
sem_data_clean <- sem_data %>%
  filter(
    !is.na(mycorrhizal_richness) &
    !is.na(plant_richness) &
    !is.na(animal_richness) &
    !is.na(tree_cover) &
    !is.na(nontree_veg) &
    !is.na(temp) &
    !is.na(precip)
  )

cat("Clean data:", nrow(sem_data_clean), "rows\n")

# Sample if needed
if(nrow(sem_data_clean) > 100000) {
  set.seed(42)
  cat("Stratified sampling by vegetation cover...\n")
  
  sem_data_clean <- sem_data_clean %>%
    mutate(
      veg_strata = cut(vegetation_complexity, breaks = 5),
      spatial_bin = paste(cut(x, 10), cut(y, 10), sep = "_")
    ) %>%
    group_by(veg_strata) %>%
    sample_n(size = 20000, replace = FALSE) %>%
    ungroup() %>%
    select(-veg_strata, -spatial_bin)
}

# Validation
cat("\n=== Data Validation ===\n")
cat("Vegetation cover ranges:\n")
cat("  Tree cover:", range(sem_data_clean$tree_cover, na.rm = TRUE), "\n")
cat("  Non-tree vegetation:", range(sem_data_clean$nontree_veg, na.rm = TRUE), "\n")
cat("  Bare ground:", range(sem_data_clean$bare_ground, na.rm = TRUE), "\n")

# Check correlations between vegetation and biodiversity
veg_bio_cor <- cor(sem_data_clean[c("tree_cover", "nontree_veg", 
                                     "mycorrhizal_richness", "plant_richness")], 
                   use = "complete.obs")
cat("\nVegetation-Biodiversity Correlations:\n")
print(round(veg_bio_cor, 2))

# Log-transform and standardize
cat("\nTransforming and standardizing variables...\n")
sem_data_scaled <- sem_data_clean %>%
  mutate(
    # Log-transform richness
    mycorrhizal_richness = log(mycorrhizal_richness + 1),
    plant_richness = log(plant_richness + 1),
    animal_richness = log(animal_richness + 1),
    # Logit transform percentages (bounded 0-100)
    tree_cover_logit = log((tree_cover + 0.01) / (100.01 - tree_cover)),
    nontree_veg_logit = log((nontree_veg + 0.01) / (100.01 - nontree_veg))
  ) %>%
  mutate(across(c(mycorrhizal_richness, plant_richness, animal_richness,
                  tree_cover_logit, nontree_veg_logit, vegetation_complexity,
                  temp, precip, elevation, soil_organic_carbon, soil_ph),
                scale))

# Convert to numeric
sem_data_scaled <- as.data.frame(lapply(sem_data_scaled, as.numeric))

# Save data
write.csv(sem_data_clean, file.path(output_dir, "sem_vegetation_data.csv"), row.names = FALSE)

cat("\n=== Building Enhanced SEM Model ===\n")

# Model 1: Full model with vegetation structure
model_vegetation <- psem(
  # Climate and soil drive vegetation structure
  lm(tree_cover_logit ~ temp + precip + temp_seasonality + 
     soil_organic_carbon + elevation,
     data = sem_data_scaled),
  
  lm(nontree_veg_logit ~ temp + precip + precip_seasonality + 
     soil_ph + elevation,
     data = sem_data_scaled),
  
  # Vegetation structure affects mycorrhizae
  # Tree cover strongly affects ECM, non-tree affects AM
  lm(mycorrhizal_richness ~ tree_cover_logit + nontree_veg_logit + 
     soil_organic_carbon + soil_ph + temp + precip,
     data = sem_data_scaled),
  
  # Plants respond to mycorrhizae and vegetation structure
  lm(plant_richness ~ mycorrhizal_richness + tree_cover_logit + nontree_veg_logit +
     temp + precip + elevation,
     data = sem_data_scaled),
  
  # Animals respond to plants and habitat complexity
  lm(animal_richness ~ plant_richness + mycorrhizal_richness + 
     vegetation_complexity + elevation,
     data = sem_data_scaled),
  
  # Bidirectional relationships
  plant_richness %~~% mycorrhizal_richness,
  animal_richness %~~% mycorrhizal_richness,
  
  # Vegetation types may covary
  tree_cover_logit %~~% nontree_veg_logit,
  
  data = sem_data_scaled
)

cat("Enhanced model created.\n")

# Summary
cat("\n=== Enhanced Model Summary ===\n")
tryCatch({
  summary_veg <- summary(model_vegetation)
  print(summary_veg)
  
  cat("\nKey statistics:\n")
  cat("Fisher's C:", summary_veg$Cstat$Fisher.C, "\n")
  cat("P-value:", summary_veg$Cstat$P, "\n")
  cat("AIC:", summary_veg$AIC$AIC, "\n")
  
  sink(file.path(output_dir, "sem_vegetation_summary.txt"))
  print(summary_veg)
  sink()
}, error = function(e) {
  cat("Error:", e$message, "\n")
})

# Coefficients
cat("\n=== Path Coefficients ===\n")
tryCatch({
  coefs_veg <- coefs(model_vegetation, standardize = "scale")
  print(coefs_veg)
  write.csv(coefs_veg, file.path(output_dir, "sem_vegetation_coefficients.csv"), 
            row.names = FALSE)
  
  # Extract key paths
  key_paths <- coefs_veg %>%
    filter(
      (Response == "mycorrhizal_richness" & Predictor %in% c("tree_cover_logit", "nontree_veg_logit")) |
      (Response == "plant_richness" & Predictor == "mycorrhizal_richness") |
      (Response == "animal_richness" & Predictor == "vegetation_complexity")
    )
  
  cat("\nKey vegetation-mediated paths:\n")
  print(key_paths)
}, error = function(e) {
  cat("Error:", e$message, "\n")
})

# R-squared
cat("\n=== Variance Explained ===\n")
tryCatch({
  r2 <- rsquared(model_vegetation)
  print(r2)
  write.csv(r2, file.path(output_dir, "sem_vegetation_rsquared.csv"), 
            row.names = FALSE)
}, error = function(e) {
  cat("Error:", e$message, "\n")
})

# Test AM vs ECM with vegetation
cat("\n=== Testing AM vs ECM Responses to Vegetation ===\n")
if(!is.na(sem_data_clean$am_fungi_richness[1])) {
  am_ecm_veg <- sem_data_clean %>%
    filter(!is.na(am_fungi_richness) & !is.na(ecm_fungi_richness)) %>%
    mutate(
      am_fungi_richness = log(am_fungi_richness + 1),
      ecm_fungi_richness = log(ecm_fungi_richness + 1),
      tree_cover_logit = log((tree_cover + 0.01) / (100.01 - tree_cover)),
      nontree_veg_logit = log((nontree_veg + 0.01) / (100.01 - nontree_veg))
    ) %>%
    mutate(across(everything(), scale))
  
  am_ecm_veg <- as.data.frame(lapply(am_ecm_veg, as.numeric))
  
  if(nrow(am_ecm_veg) > 1000) {
    # AM model - should respond more to non-tree vegetation
    model_am_veg <- lm(am_fungi_richness ~ tree_cover_logit + nontree_veg_logit + 
                      temp + precip + soil_ph,
                      data = am_ecm_veg)
    
    # ECM model - should respond more to tree cover
    model_ecm_veg <- lm(ecm_fungi_richness ~ tree_cover_logit + nontree_veg_logit + 
                       temp + precip + soil_organic_carbon,
                       data = am_ecm_veg)
    
    cat("\nAM fungi response to vegetation:\n")
    print(summary(model_am_veg)$coefficients[c("tree_cover_logit", "nontree_veg_logit"), ])
    
    cat("\nECM fungi response to vegetation:\n")
    print(summary(model_ecm_veg)$coefficients[c("tree_cover_logit", "nontree_veg_logit"), ])
    
    cat("\nHypothesis test:\n")
    cat("AM should respond more to non-tree vegetation: ",
        abs(coef(model_am_veg)["nontree_veg_logit"]) > abs(coef(model_am_veg)["tree_cover_logit"]), "\n")
    cat("ECM should respond more to tree cover: ",
        abs(coef(model_ecm_veg)["tree_cover_logit"]) > abs(coef(model_ecm_veg)["nontree_veg_logit"]), "\n")
  }
}

# Visualization
cat("\n=== Creating Visualizations ===\n")
tryCatch({
  library(ggplot2)
  library(gridExtra)
  
  pdf(file.path(output_dir, "vegetation_biodiversity_relationships.pdf"), 
      width = 12, height = 8)
  
  # Sample for plotting
  plot_data <- sem_data_clean[sample(nrow(sem_data_clean), min(5000, nrow(sem_data_clean))), ]
  
  p1 <- ggplot(plot_data, aes(x = tree_cover, y = log(mycorrhizal_richness + 1))) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", color = "darkgreen") +
    labs(title = "Tree Cover → Mycorrhizal Richness",
         x = "Tree Cover (%)", y = "Log(Mycorrhizal Richness)") +
    theme_minimal()
  
  p2 <- ggplot(plot_data, aes(x = nontree_veg, y = log(plant_richness + 1))) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", color = "orange") +
    labs(title = "Non-tree Vegetation → Plant Richness",
         x = "Non-tree Vegetation (%)", y = "Log(Plant Richness)") +
    theme_minimal()
  
  p3 <- ggplot(plot_data, aes(x = vegetation_complexity, y = log(animal_richness + 1))) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", color = "blue") +
    labs(title = "Vegetation Complexity → Animal Richness",
         x = "Vegetation Complexity Index", y = "Log(Animal Richness)") +
    theme_minimal()
  
  p4 <- ggplot(plot_data, aes(x = tree_cover, y = nontree_veg)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "loess", color = "purple") +
    labs(title = "Tree vs Non-tree Vegetation Trade-off",
         x = "Tree Cover (%)", y = "Non-tree Vegetation (%)") +
    theme_minimal()
  
  grid.arrange(p1, p2, p3, p4, ncol = 2)
  dev.off()
  
  cat("Plots saved to:", file.path(output_dir, "vegetation_biodiversity_relationships.pdf"), "\n")
}, error = function(e) {
  cat("Error creating plots:", e$message, "\n")
})

cat("\n=== Analysis Complete ===\n")
cat("Outputs saved to:", output_dir, "\n")

# Summary statistics
cat("\n=== Vegetation Structure Summary ===\n")
veg_summary <- sem_data_clean %>%
  select(tree_cover, nontree_veg, bare_ground, vegetation_complexity) %>%
  summary()
print(veg_summary)