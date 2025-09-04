#!/usr/bin/env Rscript

# Mycorrhizal SEM Analysis
# Model: Environmental drivers -> Mycorrhizal fungi -> Plants -> Animals
# With bidirectional relationships for fungi-plant and fungi-animal interactions

# library(arrow)  # Commented out - will load conditionally later
library(piecewiseSEM)
library(dplyr)
library(ggplot2)

# Set up paths
data_path <- Sys.getenv("BIODIV_DATA_PATH")
if (data_path == "") {
  data_path <- "/scratch/yl998/biodiv_with_coverage_sample.csv"
}
if (!file.exists(data_path)) {
  stop("Data file not found. Please run the Python conversion script first or set BIODIV_DATA_PATH")
}
output_dir <- "src/sem/outputs"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Mycorrhizal SEM Analysis ===\n")
cat("Loading data from:", data_path, "\n")

# Load CSV data
df <- read.csv(data_path)

cat("Data dimensions:", nrow(df), "rows x", ncol(df), "columns\n\n")

# Prepare data for SEM
# Check if we have vegetation coverage data
has_coverage <- "tree_cover" %in% names(df)

if (!has_coverage) {
  # Original data without coverage
  sem_data <- df %>%
    mutate(
      mycorrhizal_richness = case_when(
        !is.na(am_fungi_richness) & !is.na(ecm_fungi_richness) ~ am_fungi_richness + ecm_fungi_richness,
        !is.na(am_fungi_richness) & is.na(ecm_fungi_richness) ~ am_fungi_richness,
        is.na(am_fungi_richness) & !is.na(ecm_fungi_richness) ~ ecm_fungi_richness,
        TRUE ~ NA_real_
      )
    ) %>%
    select(
      x, y,
      mycorrhizal_richness, am_fungi_richness, ecm_fungi_richness,
      plant_richness, animal_richness,
      temp = bio01, temp_seasonality = bio04,
      precip = bio12, precip_seasonality = bio15,
      elevation, slope, tpi,
      soil_carbon = soil_organic_carbon, soil_ph,
      pet_mean, pet_range
    )
} else {
  # Data with vegetation coverage
  sem_data <- df %>%
    select(
      x, y,
      mycorrhizal_richness, am_fungi_richness, ecm_fungi_richness,
      plant_richness, animal_richness,
      tree_cover, nontree_veg, bare_ground,
      temp = bio01, temp_seasonality = bio04,
      precip = bio12, precip_seasonality = bio15,
      elevation, slope, tpi,
      soil_carbon = soil_organic_carbon, soil_ph,
      pet_mean, pet_range
    )
}

# Remove rows with NA in any variables we'll use
cat("Removing NA values...\n")
if (!has_coverage) {
  sem_data_clean <- sem_data %>%
    filter(
      !is.na(mycorrhizal_richness) &
      !is.na(plant_richness) &
      !is.na(animal_richness) &
      !is.na(temp) &
      !is.na(temp_seasonality) &
      !is.na(precip) &
      !is.na(precip_seasonality) &
      !is.na(soil_carbon) &
      !is.na(soil_ph) &
      !is.na(elevation) &
      !is.na(slope)
    )
} else {
  sem_data_clean <- sem_data %>%
    filter(
      !is.na(mycorrhizal_richness) &
      !is.na(plant_richness) &
      !is.na(animal_richness) &
      !is.na(tree_cover) &
      !is.na(nontree_veg) &
      !is.na(temp) &
      !is.na(temp_seasonality) &
      !is.na(precip) &
      !is.na(precip_seasonality) &
      !is.na(soil_carbon) &
      !is.na(soil_ph) &
      !is.na(elevation)
    )
}

cat("Clean data dimensions:", nrow(sem_data_clean), "rows\n")

# Data is already sampled (500k rows with pure random sampling from Python conversion)

# Data validation
cat("\n=== Data Validation ===\n")

# Check for invalid values
invalid_checks <- list(
  negative_richness = sum(sem_data_clean$mycorrhizal_richness < 0, na.rm = TRUE),
  negative_plants = sum(sem_data_clean$plant_richness < 0, na.rm = TRUE),
  negative_animals = sum(sem_data_clean$animal_richness < 0, na.rm = TRUE),
  extreme_temp = sum(abs(sem_data_clean$temp) > 100, na.rm = TRUE),
  invalid_ph = sum(sem_data_clean$soil_ph < 0 | sem_data_clean$soil_ph > 14, na.rm = TRUE)
)

for(check in names(invalid_checks)) {
  if(invalid_checks[[check]] > 0) {
    warning(paste(check, ":", invalid_checks[[check]], "invalid values detected"))
  }
}

# Check for multicollinearity
cat("Checking multicollinearity in environmental variables...\n")
env_vars <- sem_data_clean %>%
  select(temp, precip, soil_carbon, soil_ph, elevation) %>%
  na.omit()

if(nrow(env_vars) > 10) {
  cor_matrix <- cor(env_vars)
  high_cor <- which(abs(cor_matrix) > 0.8 & cor_matrix < 1, arr.ind = TRUE)
  if(nrow(high_cor) > 0) {
    warning("High correlation detected between environmental variables")
    print(high_cor)
  }
}

# Outlier detection using IQR method
detect_outliers <- function(x) {
  q <- quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)
  iqr <- q[2] - q[1]
  outliers <- x < (q[1] - 3 * iqr) | x > (q[2] + 3 * iqr)
  return(sum(outliers, na.rm = TRUE))
}

outlier_counts <- sem_data_clean %>%
  select(mycorrhizal_richness, plant_richness, animal_richness) %>%
  summarise(across(everything(), detect_outliers))

cat("Extreme outliers detected:\n")
print(outlier_counts)

# Log-transform richness values first, then standardize all variables
cat("\nLog-transforming richness values and standardizing variables...\n")

# First log-transform the richness variables (log(x+1) to handle any zeros)
sem_data_transformed <- sem_data_clean %>%
  mutate(
    mycorrhizal_richness = log(mycorrhizal_richness + 1),
    plant_richness = log(plant_richness + 1),
    animal_richness = log(animal_richness + 1)
  )

# Report transformation
cat("Richness variables log-transformed using log(x+1)\n")

# Now standardize all variables (including log-transformed richness)
if (!has_coverage) {
  sem_data_scaled <- sem_data_transformed %>%
    mutate(across(c(mycorrhizal_richness, plant_richness, animal_richness,
                    temp, temp_seasonality, precip, precip_seasonality,
                    elevation, slope, soil_carbon, soil_ph, pet_mean), scale))
} else {
  sem_data_scaled <- sem_data_transformed %>%
    mutate(across(c(mycorrhizal_richness, plant_richness, animal_richness,
                    tree_cover, nontree_veg,
                    temp, temp_seasonality, precip, precip_seasonality,
                    elevation, slope, soil_carbon, soil_ph, pet_mean), scale))
}

# Convert scaled columns to numeric vectors (fix for piecewiseSEM)
sem_data_scaled <- as.data.frame(lapply(sem_data_scaled, as.numeric))

# Save prepared data
write.csv(sem_data_clean, file.path(output_dir, "mycorrhizal_sem_data_raw.csv"), row.names = FALSE)

cat("\n=== Building Mycorrhizal SEM Model ===\n")

if (!has_coverage) {
  cat("Building model WITHOUT vegetation coverage\n")
  
  model_full <- psem(
    # Environmental drivers of mycorrhizal fungi
    lm(mycorrhizal_richness ~ temp + precip + temp_seasonality + precip_seasonality + 
       soil_ph + soil_carbon + elevation, 
       data = sem_data_scaled),
    
    # Plants respond to mycorrhizae and environment  
    lm(plant_richness ~ mycorrhizal_richness + precip + slope, 
       data = sem_data_scaled),
    
    # Animals respond to biological predictors
    lm(animal_richness ~ plant_richness + mycorrhizal_richness, 
       data = sem_data_scaled),
    
    # Bidirectional relationships
    plant_richness %~~% mycorrhizal_richness,
    animal_richness %~~% mycorrhizal_richness,
    
    data = sem_data_scaled
  )
} else {
  cat("Building model WITH vegetation coverage (refined - weak paths removed)\n")
  
  model_full <- psem(
    # Environment → Mycorrhizae (INCLUDING PET - water-energy balance)
    lm(mycorrhizal_richness ~ temp + precip + temp_seasonality + 
       precip_seasonality + soil_ph + soil_carbon + elevation + pet_mean,
       data = sem_data_scaled),
    
    # Environment + FUNGI → Tree Coverage (THEORY: seasonality drives vegetation structure/phenology)
    lm(tree_cover ~ mycorrhizal_richness + temp + precip + pet_mean +
       temp_seasonality + precip_seasonality + soil_ph + soil_carbon,
       data = sem_data_scaled),
    
    # Environment + FUNGI → Non-tree Coverage (THEORY: seasonality drives vegetation structure/phenology)
    lm(nontree_veg ~ mycorrhizal_richness + temp + precip + pet_mean +
       temp_seasonality + precip_seasonality + soil_ph + soil_carbon,
       data = sem_data_scaled),
    
    # Plant richness - biological (fungi) AND direct environmental effects
    # SIMPLIFIED: Removed weak coverage mediators, added direct precipitation effects
    lm(plant_richness ~ mycorrhizal_richness + 
       temp + temp_seasonality + precip + precip_seasonality + 
       elevation + pet_mean,
       data = sem_data_scaled),
    
    # Animal richness - biological, habitat structure, AND climate
    # Includes vegetation coverage for habitat structure (not just plant diversity)
    lm(animal_richness ~ plant_richness + mycorrhizal_richness + 
       tree_cover + nontree_veg +
       temp + temp_seasonality,
       data = sem_data_scaled),
    
    # Bidirectional relationships
    plant_richness %~~% mycorrhizal_richness,
    animal_richness %~~% mycorrhizal_richness,
    tree_cover %~~% nontree_veg,  # Competition/trade-off between vegetation types
    
    data = sem_data_scaled
  )
}

cat("Full model created.\n")

# Get model summary
cat("\n=== Full Model Summary ===\n")
tryCatch({
  summary_full <- summary(model_full)
  print(summary_full)
  
  # Extract key statistics
  cat("\nFisher's C:", summary_full$Cstat$Fisher.C, "\n")
  cat("C p-value:", summary_full$Cstat$P, "\n")
  cat("AIC:", summary_full$AIC$AIC, "\n")
  
  # Save summary
  sink(file.path(output_dir, "mycorrhizal_sem_summary_full.txt"))
  print(summary_full)
  sink()
}, error = function(e) {
  cat("Error in full model:", e$message, "\n")
  cat("Trying alternative Fisher's C computation...\n")
  tryCatch({
    fc <- fisherC(model_full)
    cat("Fisher's C (alternative):", fc$Fisher.C, "\n")
    cat("Fisher's C p-value:", fc$P, "\n")
  }, error = function(e2) {
    cat("Alternative also failed:", e2$message, "\n")
  })
})

# Extract and save coefficients from full model
cat("\n=== Path Coefficients (Full Model) ===\n")
tryCatch({
  coef_std <- coefs(model_full, standardize = "scale")
  print(coef_std)
  write.csv(coef_std, file.path(output_dir, "mycorrhizal_sem_coefficients.csv"), row.names = FALSE)
}, error = function(e) {
  cat("Error extracting coefficients:", e$message, "\n")
})

# Get R-squared values
cat("\n=== R-squared Values ===\n")
tryCatch({
  r2 <- rsquared(model_full)
  print(r2)
  write.csv(r2, file.path(output_dir, "mycorrhizal_sem_rsquared.csv"), row.names = FALSE)
}, error = function(e) {
  cat("Error calculating R-squared:", e$message, "\n")
})

# Create path diagram
cat("\n=== Creating Path Diagram ===\n")
tryCatch({
  pdf(file.path(output_dir, "mycorrhizal_sem_path_diagram.pdf"), width = 12, height = 10)
  plot(model_full, show = "std", digits = 2)
  dev.off()
  cat("Path diagram saved.\n")
}, error = function(e) {
  cat("Could not create path diagram:", e$message, "\n")
})

# Test separate models for AM and ECM fungi
cat("\n=== Testing AM vs ECM Fungi Models ===\n")

# Prepare data for AM/ECM comparison
am_ecm_data <- sem_data_clean %>%
  filter(!is.na(am_fungi_richness) & !is.na(ecm_fungi_richness)) %>%
  mutate(
    # Log-transform richness first
    am_fungi_richness = log(am_fungi_richness + 1),
    ecm_fungi_richness = log(ecm_fungi_richness + 1),
    plant_richness = log(plant_richness + 1),
    animal_richness = log(animal_richness + 1)
  ) %>%
  mutate(across(c(am_fungi_richness, ecm_fungi_richness, plant_richness, animal_richness,
                  temp, precip, soil_carbon, elevation), scale))

# Convert to numeric vectors for piecewiseSEM
am_ecm_data <- as.data.frame(lapply(am_ecm_data, as.numeric))

if(nrow(am_ecm_data) > 1000) {
  cat("Testing AM fungi model...\n")
  
  model_am <- psem(
    lm(am_fungi_richness ~ temp + precip + soil_carbon + elevation, 
       data = am_ecm_data),
    lm(plant_richness ~ am_fungi_richness + temp + precip + soil_carbon, 
       data = am_ecm_data),
    lm(animal_richness ~ plant_richness + am_fungi_richness, 
       data = am_ecm_data),
    plant_richness %~~% am_fungi_richness,
    data = am_ecm_data
  )
  
  cat("Testing ECM fungi model...\n")
  
  model_ecm <- psem(
    lm(ecm_fungi_richness ~ temp + precip + soil_carbon + elevation, 
       data = am_ecm_data),
    lm(plant_richness ~ ecm_fungi_richness + temp + precip + soil_carbon, 
       data = am_ecm_data),
    lm(animal_richness ~ plant_richness + ecm_fungi_richness, 
       data = am_ecm_data),
    plant_richness %~~% ecm_fungi_richness,
    data = am_ecm_data
  )
  
  # Compare AM vs ECM effects
  tryCatch({
    coef_am <- coefs(model_am, standardize = "scale")
    coef_ecm <- coefs(model_ecm, standardize = "scale")
    
    write.csv(coef_am, file.path(output_dir, "am_fungi_coefficients.csv"), row.names = FALSE)
    write.csv(coef_ecm, file.path(output_dir, "ecm_fungi_coefficients.csv"), row.names = FALSE)
    
    cat("\nAM fungi -> Plant effect:", 
        coef_am$Std.Estimate[coef_am$Predictor == "am_fungi_richness" & 
                             coef_am$Response == "plant_richness"], "\n")
    cat("ECM fungi -> Plant effect:", 
        coef_ecm$Std.Estimate[coef_ecm$Predictor == "ecm_fungi_richness" & 
                              coef_ecm$Response == "plant_richness"], "\n")
  }, error = function(e) {
    cat("Error comparing AM/ECM models:", e$message, "\n")
  })
}

# Create diagnostic plots
cat("\n=== Creating Diagnostic Plots ===\n")
tryCatch({
  library(ggplot2)
  
  # Scatter plots of key relationships
  pdf(file.path(output_dir, "mycorrhizal_relationships.pdf"), width = 12, height = 8)
  
  # Fungi-Plant relationship
  p1 <- ggplot(sem_data_clean[sample(nrow(sem_data_clean), min(5000, nrow(sem_data_clean))), ], 
               aes(x = mycorrhizal_richness, y = plant_richness)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", color = "blue") +
    labs(title = "Mycorrhizal-Plant Relationship",
         x = "Mycorrhizal Richness", y = "Plant Richness") +
    theme_minimal()
  
  # Plant-Animal relationship  
  p2 <- ggplot(sem_data_clean[sample(nrow(sem_data_clean), min(5000, nrow(sem_data_clean))), ],
               aes(x = plant_richness, y = animal_richness)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", color = "green") +
    labs(title = "Plant-Animal Relationship",
         x = "Plant Richness", y = "Animal Richness") +
    theme_minimal()
  
  # Fungi-Animal relationship
  p3 <- ggplot(sem_data_clean[sample(nrow(sem_data_clean), min(5000, nrow(sem_data_clean))), ],
               aes(x = mycorrhizal_richness, y = animal_richness)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "lm", color = "red") +
    labs(title = "Mycorrhizal-Animal Relationship",
         x = "Mycorrhizal Richness", y = "Animal Richness") +
    theme_minimal()
  
  # Combine plots
  library(gridExtra)
  grid.arrange(p1, p2, p3, ncol = 2)
  
  dev.off()
  cat("Diagnostic plots saved.\n")
}, error = function(e) {
  cat("Error creating diagnostic plots:", e$message, "\n")
})

# Test for missing paths (d-separation)
cat("\n=== Testing for Missing Paths ===\n")
tryCatch({
  # Check if there are missing paths in the full model
  dsep <- dSep(model_full)
  if (nrow(dsep) > 0) {
    cat("Potential missing paths detected:\n")
    print(dsep)
  } else {
    cat("No missing paths detected - model is complete\n")
  }
}, error = function(e) {
  cat("Error testing missing paths:", e$message, "\n")
})

cat("\n=== Analysis Complete ===\n")
cat("All outputs saved to:", output_dir, "\n")

# Print summary statistics
cat("\n=== Data Summary Statistics ===\n")
summary_stats <- sem_data_clean %>%
  select(mycorrhizal_richness, plant_richness, animal_richness) %>%
  summary()
print(summary_stats)