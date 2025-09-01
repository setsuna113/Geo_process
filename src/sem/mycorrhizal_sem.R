#!/usr/bin/env Rscript

# Mycorrhizal SEM Analysis
# Model: Environmental drivers -> Mycorrhizal fungi -> Plants -> Animals
# With bidirectional relationships for fungi-plant and fungi-animal interactions

library(arrow)
library(piecewiseSEM)
library(dplyr)
library(ggplot2)

# Set up paths
data_path <- "/scratch/yl998/biodiv_1x_renamed.arrow"
output_dir <- "src/sem/outputs"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

cat("=== Mycorrhizal SEM Analysis ===\n")
cat("Loading data from:", data_path, "\n")

# Load Arrow data
df <- read_feather(data_path)

cat("Data dimensions:", nrow(df), "rows x", ncol(df), "columns\n\n")

# Prepare data for SEM
# We have both AM and ECM fungi - create total mycorrhizal richness
sem_data <- df %>%
  mutate(
    # Combine AM and ECM fungi richness
    mycorrhizal_richness = coalesce(am_fungi_richness, 0) + coalesce(ecm_fungi_richness, 0)
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

# If data is too large, sample for initial analysis
if(nrow(sem_data_clean) > 100000) {
  cat("Data is large. Sampling 100,000 rows for analysis...\n")
  set.seed(42)
  sem_data_clean <- sem_data_clean %>%
    sample_n(100000)
}

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