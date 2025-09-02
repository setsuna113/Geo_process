#!/usr/bin/env Rscript

# Mycorrhizal SEM Analysis
# Model: Environmental drivers -> Mycorrhizal fungi -> Plants -> Animals
# With bidirectional relationships for fungi-plant and fungi-animal interactions

library(arrow)
library(piecewiseSEM)
library(dplyr)
library(ggplot2)

# Set up paths - use environment variable or relative path
data_path <- Sys.getenv("BIODIV_DATA_PATH", 
                        default = file.path(dirname(getwd()), "data", "biodiv_1x_renamed.arrow"))
if (!file.exists(data_path)) {
  # Try alternate location
  data_path <- "/scratch/yl998/biodiv_1x_renamed.arrow"
  if (!file.exists(data_path)) {
    stop("Data file not found. Set BIODIV_DATA_PATH environment variable or place data in ../data/")
  }
}
output_dir <- "src/sem/outputs"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Mycorrhizal SEM Analysis ===\n")
cat("Loading data from:", data_path, "\n")

# Load Arrow data - use appropriate function based on package
if (requireNamespace("arrow", quietly = TRUE)) {
  df <- arrow::read_feather(data_path)
} else if (requireNamespace("feather", quietly = TRUE)) {
  df <- feather::read_feather(data_path)
} else {
  stop("Neither arrow nor feather package available for reading data")
}

cat("Data dimensions:", nrow(df), "rows x", ncol(df), "columns\n\n")

# Prepare data for SEM
# We have both AM and ECM fungi - create total mycorrhizal richness
# Note: Only combine when both are non-NA, otherwise keep as NA
sem_data <- df %>%
  mutate(
    # Only combine fungi richness when we have actual data
    mycorrhizal_richness = case_when(
      !is.na(am_fungi_richness) & !is.na(ecm_fungi_richness) ~ am_fungi_richness + ecm_fungi_richness,
      !is.na(am_fungi_richness) & is.na(ecm_fungi_richness) ~ am_fungi_richness,
      is.na(am_fungi_richness) & !is.na(ecm_fungi_richness) ~ ecm_fungi_richness,
      TRUE ~ NA_real_
    )
  ) %>%
  select(
    # Coordinates
    x, y,
    
    # Biodiversity variables
    mycorrhizal_richness,
    am_fungi_richness,
    ecm_fungi_richness,
    plant_richness,
    animal_richness,
    
    # Climate variables (WorldClim BioClim)
    temp = bio01,           # Annual mean temperature
    temp_seasonality = bio04,  # Temperature seasonality
    precip = bio12,         # Annual precipitation  
    precip_seasonality = bio15, # Precipitation seasonality
    
    # Topography
    elevation,
    slope,
    tpi,                    # Topographic position index
    
    # Soil
    soil_carbon = soil_organic_carbon,
    soil_ph,
    
    # Other environmental
    pet_mean,               # Potential evapotranspiration
    pet_range
  )

# Remove rows with NA in key variables
cat("Removing NA values...\n")
sem_data_clean <- sem_data %>%
  filter(
    !is.na(mycorrhizal_richness) &
    !is.na(plant_richness) &
    !is.na(animal_richness) &
    !is.na(temp) &
    !is.na(precip) &
    !is.na(soil_carbon)
  )

cat("Clean data dimensions:", nrow(sem_data_clean), "rows\n")

# If data is too large, use stratified sampling to maintain spatial representation
if(nrow(sem_data_clean) > 100000) {
  cat("Data is large. Using stratified spatial sampling...\n")
  set.seed(42)
  
  # Create spatial bins for stratified sampling
  sem_data_clean <- sem_data_clean %>%
    mutate(
      x_bin = cut(x, breaks = 10),
      y_bin = cut(y, breaks = 10),
      strata = paste(x_bin, y_bin, sep = "_")
    ) %>%
    group_by(strata) %>%
    sample_n(size = min(n(), ceiling(100000 / n_distinct(.$strata))), replace = FALSE) %>%
    ungroup() %>%
    select(-x_bin, -y_bin, -strata) %>%
    slice_sample(n = min(100000, n()))  # Ensure exactly 100k rows
  
  cat("Sampled", nrow(sem_data_clean), "rows maintaining spatial distribution\n")
}

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

# Standardize variables for better convergence
cat("\nStandardizing variables...\n")
sem_data_scaled <- sem_data_clean %>%
  mutate(across(c(mycorrhizal_richness, plant_richness, animal_richness,
                  temp, temp_seasonality, precip, precip_seasonality,
                  elevation, slope, soil_carbon, soil_ph), scale))

# Save prepared data
write.csv(sem_data_clean, file.path(output_dir, "mycorrhizal_sem_data_raw.csv"), row.names = FALSE)

cat("\n=== Building Mycorrhizal SEM Model ===\n")

# Model 1: Full model with all environmental predictors
model_full <- psem(
  # Environmental drivers of mycorrhizal fungi
  lm(mycorrhizal_richness ~ temp + precip + temp_seasonality + precip_seasonality + 
     soil_carbon + soil_ph + elevation + slope, 
     data = sem_data_scaled),
  
  # Plants respond to mycorrhizae and environment
  lm(plant_richness ~ mycorrhizal_richness + temp + precip + 
     soil_carbon + soil_ph + elevation, 
     data = sem_data_scaled),
  
  # Animals respond to plants and mycorrhizae
  lm(animal_richness ~ plant_richness + mycorrhizal_richness, 
     data = sem_data_scaled),
  
  # Bidirectional relationships
  plant_richness %~~% mycorrhizal_richness,    # Nutrient-carbon exchange
  animal_richness %~~% mycorrhizal_richness,   # Consumption-dispersal cycle
  
  data = sem_data_scaled
)

cat("Full model created.\n")

# Get model summary
cat("\n=== Full Model Summary ===\n")
tryCatch({
  summary_full <- summary(model_full)
  print(summary_full)
  
  # Save summary
  sink(file.path(output_dir, "mycorrhizal_sem_summary_full.txt"))
  print(summary_full)
  sink()
}, error = function(e) {
  cat("Error in full model:", e$message, "\n")
})

# Model 2: Simplified model with key predictors only
cat("\n=== Building Simplified Model ===\n")

model_simple <- psem(
  # Environmental drivers of mycorrhizal fungi (simplified)
  lm(mycorrhizal_richness ~ temp + precip + soil_carbon + elevation, 
     data = sem_data_scaled),
  
  # Plants respond to mycorrhizae and key environment
  lm(plant_richness ~ mycorrhizal_richness + temp + precip + soil_carbon, 
     data = sem_data_scaled),
  
  # Animals respond to resources
  lm(animal_richness ~ plant_richness + mycorrhizal_richness, 
     data = sem_data_scaled),
  
  # Bidirectional relationships
  plant_richness %~~% mycorrhizal_richness,
  animal_richness %~~% mycorrhizal_richness,
  
  data = sem_data_scaled
)

cat("Simplified model created.\n")

# Get simplified model summary
cat("\n=== Simplified Model Summary ===\n")
tryCatch({
  summary_simple <- summary(model_simple)
  print(summary_simple)
  
  # Save summary
  sink(file.path(output_dir, "mycorrhizal_sem_summary_simple.txt"))
  print(summary_simple)
  sink()
}, error = function(e) {
  cat("Error in simplified model:", e$message, "\n")
})

# Extract and save coefficients
cat("\n=== Path Coefficients (Simplified Model) ===\n")
tryCatch({
  coef_std <- coefs(model_simple, standardize = "scale")
  print(coef_std)
  write.csv(coef_std, file.path(output_dir, "mycorrhizal_sem_coefficients.csv"), row.names = FALSE)
}, error = function(e) {
  cat("Error extracting coefficients:", e$message, "\n")
})

# Get R-squared values
cat("\n=== R-squared Values ===\n")
tryCatch({
  r2 <- rsquared(model_simple)
  print(r2)
  write.csv(r2, file.path(output_dir, "mycorrhizal_sem_rsquared.csv"), row.names = FALSE)
}, error = function(e) {
  cat("Error calculating R-squared:", e$message, "\n")
})

# Model comparison if both models worked
cat("\n=== Model Comparison ===\n")
tryCatch({
  aic_full <- AIC(model_full)
  aic_simple <- AIC(model_simple)
  cat("Full model AIC:", aic_full, "\n")
  cat("Simplified model AIC:", aic_simple, "\n")
  cat("Better model:", ifelse(aic_full < aic_simple, "Full", "Simplified"), "\n")
}, error = function(e) {
  cat("Error comparing models:", e$message, "\n")
})

# Create path diagram
cat("\n=== Creating Path Diagram ===\n")
tryCatch({
  pdf(file.path(output_dir, "mycorrhizal_sem_path_diagram.pdf"), width = 12, height = 10)
  plot(model_simple, show = "std", digits = 2)
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
  mutate(across(c(am_fungi_richness, ecm_fungi_richness, plant_richness, animal_richness,
                  temp, precip, soil_carbon, elevation), scale))

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

cat("\n=== Analysis Complete ===\n")
cat("All outputs saved to:", output_dir, "\n")

# Print summary statistics
cat("\n=== Data Summary Statistics ===\n")
summary_stats <- sem_data_clean %>%
  select(mycorrhizal_richness, plant_richness, animal_richness) %>%
  summary()
print(summary_stats)