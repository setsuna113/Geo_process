#!/usr/bin/env Rscript
# Conditional Subgroup PFI - Correct Implementation
# Based on Molnar et al. (2023) conditional subgroup approach
# ONLY reads from Julia results JSON which contains the complete model

.libPaths("~/R")
library(jsonlite)
library(arrow)
library(ggplot2)
library(dplyr)

# Configuration
CONFIG <- list(
  data_path = "/scratch/yl998/biodiversity_climate_production_WITH_MOD44B.arrow",  # Use original data for unbiased PFI
  julia_results_dir = "/scratch/yl998/julia/",
  n_samples = 5000,
  n_permutations = 10,
  n_subgroups = 5,
  correlation_threshold = 0.7,
  response_map = list(
    fungi = "am_fungi_richness",
    plants = "plants_richness", 
    vertebrates = "terrestrial_richness"
  )
)

#' Load Julia GLM results and create predict function
load_glm_model <- function(response_type) {
  # Find the latest julia_results file for this response type
  pattern <- sprintf("julia_results_%s_.*\\.json$", response_type)
  
  # Get all matching files in the results directory
  json_files <- list.files(CONFIG$julia_results_dir, pattern = pattern, full.names = TRUE)
  
  if (length(json_files) == 0) {
    stop(sprintf("No julia_results files found for %s in %s", response_type, CONFIG$julia_results_dir))
  }
  
  # Sort by modification time and get the most recent
  file_info <- file.info(json_files)
  julia_results_file <- json_files[which.max(file_info$mtime)]
  
  message(sprintf("Using latest results file: %s", basename(julia_results_file)))
  model_name <- "method1_full"  # These files use generic names
  
  if (!file.exists(julia_results_file)) {
    stop("Julia results file not found at: ", julia_results_file)
  }
  
  # Load Julia results
  julia_results <- fromJSON(julia_results_file)
  
  # Extract model coefficients and names
  if (is.null(julia_results$models[[model_name]])) {
    stop("Model '", model_name, "' not found in Julia results")
  }
  
  model <- julia_results$models[[model_name]]
  coef_values <- model$coefficients
  coef_names <- model$coef_names
  
  # Create named vector of coefficients
  names(coef_values) <- coef_names
  
  # Verify intercept is present
  if (!("(Intercept)" %in% names(coef_values))) {
    stop("No intercept found in Julia model - this should not happen!")
  }
  
  cat("Loaded model from Julia results:\n")
  cat("  Intercept:", coef_values["(Intercept)"], "\n")
  cat("  Number of coefficients:", length(coef_values), "\n")
  cat("  Response will be calculated using fungi_richness = AM + ECM\n")
  
  # Get feature names (excluding intercept and MESF - they're not in original data)
  feature_names <- names(coef_values)[!names(coef_values) %in% c("(Intercept)")]
  feature_names <- feature_names[!grepl("^mesf_", feature_names)]  # Skip MESF - not in original data
  
  # Map standardized names back to original names for PFI computation
  feature_names_original <- gsub("_std$", "", feature_names)
  feature_names_original <- gsub("_sq$", "", feature_names_original)  # Remove squared suffix
  
  # Map vertebrates_richness back to terrestrial_richness (the actual column name)
  feature_names_original <- gsub("^vertebrates_richness$", "terrestrial_richness", feature_names_original)
  
  # Create predict function for GLM with log link
  predict_fn <- function(X) {
    # We always have an intercept from Julia results
    intercept <- coef_values["(Intercept)"]
    beta <- coef_values[names(coef_values) != "(Intercept)"]
    
    # Create feature matrix matching model expectations
    X_matrix <- matrix(0, nrow = nrow(X), ncol = length(beta))
    colnames(X_matrix) <- names(beta)
    
    for (i in seq_along(beta)) {
      feat_name <- names(beta)[i]
      
      if (feat_name == "elevation_std_sq") {
        # Special case: squared elevation term
        if ("elevation" %in% names(X)) {
          X_matrix[, i] <- X[["elevation"]]^2
        }
      } else if (grepl("^ME", feat_name)) {
        # MESF spatial features - set to mean (0 for standardized)
        X_matrix[, i] <- 0
      } else if (grepl("_std$", feat_name)) {
        # Standardized features - map from unstandardized names
        clean_name <- gsub("_std$", "", feat_name)
        # Map vertebrates_richness to terrestrial_richness
        if (clean_name == "vertebrates_richness") {
          clean_name <- "terrestrial_richness"
        }
        if (clean_name %in% names(X)) {
          # X is already standardized in load_and_prepare_data
          X_matrix[, i] <- X[[clean_name]]
        } else {
          X_matrix[, i] <- 0
        }
      } else {
        # Direct mapping
        if (feat_name %in% names(X)) {
          X_matrix[, i] <- X[[feat_name]]
        }
      }
    }
    
    # Negative binomial GLM with log link
    linear_pred <- as.vector(X_matrix %*% beta + intercept)
    predictions <- exp(linear_pred)
    
    # Ensure reasonable predictions
    predictions[!is.finite(predictions)] <- 1e6
    predictions[predictions <= 0] <- 0.001
    predictions[predictions > 1e6] <- 1e6
    
    return(predictions)
  }
  
  return(list(
    model_name = model_name,
    coefficients = coef_values,
    predict_fn = predict_fn,
    feature_names = feature_names,
    feature_names_original = unique(feature_names_original)  # Unique to handle duplicates like elevation
  ))
}

#' Load and prepare data for analysis
load_and_prepare_data <- function(response_type) {
  # Load data
  df <- arrow::read_feather(CONFIG$data_path)
  
  # Convert to data frame if needed
  if (!is.data.frame(df)) {
    df <- as.data.frame(df)
  }
  
  # Sample if too large
  if (nrow(df) > CONFIG$n_samples) {
    set.seed(123)
    df <- df[sample(nrow(df), CONFIG$n_samples), ]
  }
  
  # Get response - MUST use fungi_richness = AM + ECM
  if (response_type == "fungi") {
    y <- df$am_fungi_richness + df$ecm_fungi_richness
    cat("Using fungi_richness = AM + ECM as response\n")
  } else {
    y <- df[[CONFIG$response_map[[response_type]]]]
  }
  
  # Get feature columns - ONLY use features that are in the model!
  # Exclude response variables and identifiers based on response type
  # But KEEP the biodiversity predictors that are used as features!
  exclude_cols <- c("x", "y", "cell_id", "row_id", "longitude", "latitude")
  
  # Only exclude the response variable, not the predictors
  if (response_type == "fungi") {
    exclude_cols <- c(exclude_cols, "am_fungi_richness", "ecm_fungi_richness")
    # Keep plants_richness and terrestrial_richness as predictors!
  } else if (response_type == "plants") {
    exclude_cols <- c(exclude_cols, "plants_richness")
    # Keep fungi and vertebrates as predictors!
  } else if (response_type == "vertebrates") {
    exclude_cols <- c(exclude_cols, "terrestrial_richness")
    # Keep fungi and plants as predictors!
  }
  
  feature_cols <- setdiff(names(df), exclude_cols)
  
  # Transform and standardize features to match Julia pipeline
  X <- df[, feature_cols]
  
  # First log-transform biodiversity predictors (Julia uses log_biodiversity: true)
  biodiv_cols <- c("plants_richness", "terrestrial_richness", "vertebrates_richness",
                    "am_fungi_richness", "ecm_fungi_richness", "fungi_richness")
  for (col in biodiv_cols) {
    if (col %in% names(X)) {
      X[[col]] <- log(X[[col]] + 1)  # log1p transformation
    }
  }
  
  # Then standardize ALL features using 2SD (Julia pipeline uses 2SD!)
  for (col in names(X)) {
    if (is.numeric(X[[col]])) {
      mean_val <- mean(X[[col]], na.rm = TRUE)
      sd_val <- sd(X[[col]], na.rm = TRUE)
      if (sd_val > 0) {
        X[[col]] <- (X[[col]] - mean_val) / (2 * sd_val)  # 2SD standardization
      }
    }
  }
  
  # Remove any remaining NA
  complete_cases <- complete.cases(X, y)
  X <- X[complete_cases, ]
  y <- y[complete_cases]
  
  cat(sprintf("Data prepared: %d observations, %d features\n", nrow(X), ncol(X)))
  
  return(list(X = X, y = y))
}

#' Compute loss (using appropriate metric for count data)
compute_loss <- function(y_true, y_pred) {
  # Simple MSE - what DALEX uses by default
  # This is more stable and interpretable for PFI
  mse <- mean((y_true - y_pred)^2)
  return(mse)
}

#' Standard (marginal) PFI
compute_marginal_pfi <- function(X, y, predict_fn, n_perms = 10, feature_names = NULL) {
  baseline_pred <- predict_fn(X)
  baseline_loss <- compute_loss(y, baseline_pred)
  
  # Only compute PFI for features that are in the model (excluding MESF)
  if (!is.null(feature_names)) {
    # Only test features that exist in both X and feature_names
    vars_to_test <- intersect(names(X), feature_names)
  } else {
    vars_to_test <- names(X)
  }
  
  importance <- numeric(length(vars_to_test))
  names(importance) <- vars_to_test
  
  for (var in vars_to_test) {
    perm_losses <- numeric(n_perms)
    
    for (p in 1:n_perms) {
      X_perm <- X
      # Marginal permutation - shuffle across all observations
      X_perm[[var]] <- sample(X[[var]])
      
      perm_pred <- predict_fn(X_perm)
      perm_losses[p] <- compute_loss(y, perm_pred)
    }
    
    # Importance is the increase in loss
    importance[var] <- mean(perm_losses) - baseline_loss
  }
  
  return(importance)
}

#' Identify correlated feature pairs
identify_correlated_pairs <- function(X, threshold = 0.7) {
  # Compute correlation matrix for numeric features
  numeric_cols <- sapply(X, is.numeric)
  if (sum(numeric_cols) < 2) {
    return(list())
  }
  
  cor_matrix <- cor(X[, numeric_cols], use = "complete.obs")
  
  # Find pairs above threshold
  pairs <- list()
  for (i in 1:(ncol(cor_matrix) - 1)) {
    for (j in (i + 1):ncol(cor_matrix)) {
      if (abs(cor_matrix[i, j]) > threshold) {
        pairs[[length(pairs) + 1]] <- list(
          var1 = colnames(cor_matrix)[i],
          var2 = colnames(cor_matrix)[j],
          correlation = cor_matrix[i, j]
        )
      }
    }
  }
  
  return(pairs)
}

#' Create subgroups based on correlated features
create_subgroups <- function(X, conditioning_vars, n_groups = 5) {
  # Use quantiles to create subgroups
  if (length(conditioning_vars) == 1) {
    # Single variable: use quantiles
    var <- conditioning_vars[1]
    cuts <- quantile(X[[var]], probs = seq(0, 1, length.out = n_groups + 1), na.rm = TRUE)
    # Handle non-unique breaks by using unique values
    cuts <- unique(cuts)
    if (length(cuts) < 2) {
      # If all values are the same, return all observations in one group
      groups <- rep(1, nrow(X))
    } else {
      groups <- cut(X[[var]], breaks = cuts, include.lowest = TRUE, labels = FALSE)
    }
  } else {
    # Multiple variables: use k-means clustering
    X_subset <- X[, conditioning_vars]
    # Handle any remaining NA
    complete_rows <- complete.cases(X_subset)
    groups <- rep(NA, nrow(X))
    
    if (sum(complete_rows) > n_groups) {
      kmeans_result <- kmeans(X_subset[complete_rows, ], centers = n_groups, nstart = 10)
      groups[complete_rows] <- kmeans_result$cluster
    }
  }
  
  # Handle NA groups
  groups[is.na(groups)] <- sample(1:n_groups, sum(is.na(groups)), replace = TRUE)
  
  return(groups)
}

#' Conditional subgroup PFI (Molnar et al. 2023)
compute_conditional_pfi <- function(X, y, predict_fn, corr_pairs, n_perms = 10, n_groups = 5, feature_names = NULL) {
  baseline_pred <- predict_fn(X)
  baseline_loss <- compute_loss(y, baseline_pred)
  
  # Only compute PFI for features in the model (excluding MESF)
  if (!is.null(feature_names)) {
    vars_to_test <- intersect(names(X), feature_names)
  } else {
    vars_to_test <- names(X)
  }
  
  importance <- numeric(length(vars_to_test))
  names(importance) <- vars_to_test
  
  # Track which variables need conditional permutation
  conditional_vars <- unique(unlist(lapply(corr_pairs, function(p) c(p$var1, p$var2))))
  
  for (var in vars_to_test) {
    if (var %in% conditional_vars) {
      # Find correlated variables for conditioning
      conditioning_vars <- c()
      for (pair in corr_pairs) {
        if (pair$var1 == var) {
          conditioning_vars <- c(conditioning_vars, pair$var2)
        } else if (pair$var2 == var) {
          conditioning_vars <- c(conditioning_vars, pair$var1)
        }
      }
      conditioning_vars <- unique(conditioning_vars)
      
      if (length(conditioning_vars) > 0) {
        # Create subgroups based on correlated features
        groups <- create_subgroups(X, conditioning_vars, n_groups)
        
        perm_losses <- numeric(n_perms)
        for (p in 1:n_perms) {
          X_perm <- X
          
          # Permute within subgroups
          for (g in unique(groups)) {
            group_idx <- which(groups == g)
            if (length(group_idx) > 1) {
              X_perm[group_idx, var] <- sample(X[group_idx, var])
            }
          }
          
          perm_pred <- predict_fn(X_perm)
          perm_losses[p] <- compute_loss(y, perm_pred)
        }
        
        importance[var] <- mean(perm_losses) - baseline_loss
      } else {
        # No conditioning needed (shouldn't happen if var is in conditional_vars)
        importance[var] <- NA
      }
    } else {
      # Standard marginal permutation for uncorrelated variables
      perm_losses <- numeric(n_perms)
      
      for (p in 1:n_perms) {
        X_perm <- X
        X_perm[[var]] <- sample(X[[var]])
        perm_pred <- predict_fn(X_perm)
        perm_losses[p] <- compute_loss(y, perm_pred)
      }
      
      importance[var] <- mean(perm_losses) - baseline_loss
    }
  }
  
  return(importance)
}

#' Create visualization comparing marginal and conditional PFI
create_pfi_plot <- function(marginal_imp, conditional_imp, corr_pairs, title) {
  # Keep negative values - they're meaningful!
  # Negative PFI means permutation improved predictions (often due to correlation)
  
  # Prepare data for plotting
  df_plot <- data.frame(
    feature = names(marginal_imp),
    marginal = marginal_imp,
    conditional = conditional_imp[names(marginal_imp)],
    stringsAsFactors = FALSE
  )
  
  # Mark correlated features
  correlated_features <- unique(unlist(lapply(corr_pairs, function(p) c(p$var1, p$var2))))
  df_plot$is_correlated <- df_plot$feature %in% correlated_features
  
  # Sort by marginal importance - use absolute value for sorting to handle any remaining negatives
  df_plot <- df_plot[order(abs(df_plot$marginal), decreasing = TRUE), ]
  df_plot <- df_plot[1:min(20, nrow(df_plot)), ]  # Top 20 features
  
  # Reshape for plotting
  df_long <- reshape2::melt(df_plot, id.vars = c("feature", "is_correlated"),
                            variable.name = "method", value.name = "importance")
  
  # Create plot
  p <- ggplot(df_long, aes(x = reorder(feature, importance), y = importance)) +
    geom_bar(aes(fill = method), stat = "identity", position = "dodge") +
    coord_flip() +
    scale_fill_manual(values = c("marginal" = "#2E86AB", "conditional" = "#A23B72"),
                      labels = c("Marginal PFI", "Conditional PFI")) +
    labs(
      title = title,
      subtitle = sprintf("Correlation threshold: %.2f | %d correlated pairs identified",
                        CONFIG$correlation_threshold, length(corr_pairs)),
      x = "Feature",
      y = "Permutation Importance (MSE increase)",
      fill = "Method"
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    )
  
  # Add markers for correlated features
  if (any(df_plot$is_correlated)) {
    corr_features <- df_plot$feature[df_plot$is_correlated]
    p <- p + 
      annotate("text", x = which(levels(factor(df_long$feature)) %in% corr_features),
               y = max(df_long$importance) * 1.1, label = "*", size = 5, color = "red")
  }
  
  return(p)
}

#' Main analysis function
analyze_response <- function(response_type) {
  message(sprintf("\n=== Analyzing %s ===", toupper(response_type)))
  
  # Load response-specific model from Julia results
  model <- load_glm_model(response_type)
  
  # Load and prepare data
  data_list <- load_and_prepare_data(response_type)
  X <- data_list$X
  y <- data_list$y
  
  # Verify predictions
  test_pred <- model$predict_fn(X[1:10, ])
  message(sprintf("Sample predictions: min=%.2f, mean=%.2f, max=%.2f",
                  min(test_pred), mean(test_pred), max(test_pred)))
  message(sprintf("Actual values: min=%.2f, mean=%.2f, max=%.2f",
                  min(y), mean(y), max(y)))
  
  # Identify correlated pairs
  corr_pairs <- identify_correlated_pairs(X, CONFIG$correlation_threshold)
  message(sprintf("Found %d correlated feature pairs (r > %.2f)", 
                  length(corr_pairs), CONFIG$correlation_threshold))
  
  # Show top correlations
  if (length(corr_pairs) > 0) {
    message("\nTop correlated pairs:")
    for (i in 1:min(5, length(corr_pairs))) {
      p <- corr_pairs[[i]]
      # Rename for display
      var1_display <- ifelse(p$var1 == "terrestrial_richness", "vertebrates_richness", p$var1)
      var2_display <- ifelse(p$var2 == "terrestrial_richness", "vertebrates_richness", p$var2)
      message(sprintf("  %s <-> %s: r = %.3f", var1_display, var2_display, p$correlation))
    }
  }
  
  # Compute marginal PFI (only for features in the model, excluding MESF)
  message("\nComputing marginal PFI...")
  marginal_imp <- compute_marginal_pfi(X, y, model$predict_fn, CONFIG$n_permutations,
                                       feature_names = model$feature_names_original)
  
  # Rename terrestrial_richness to vertebrates_richness in results for clarity
  if ("terrestrial_richness" %in% names(marginal_imp)) {
    names(marginal_imp)[names(marginal_imp) == "terrestrial_richness"] <- "vertebrates_richness"
  }
  
  # Debug: Check if all features have importance values
  message(sprintf("Features in X: %d", ncol(X)))
  message(sprintf("Features with importance: %d", length(marginal_imp)))
  message(sprintf("Feature names in X: %s", paste(names(X), collapse=", ")))
  
  # Show ALL marginal importance values
  message("\nALL Marginal PFI values:")
  for (feat in names(marginal_imp)) {
    message(sprintf("  %s: %.4f", feat, marginal_imp[feat]))
  }
  
  # Compute conditional PFI
  message("Computing conditional PFI...")
  conditional_imp <- compute_conditional_pfi(X, y, model$predict_fn, corr_pairs, 
                                            CONFIG$n_permutations, CONFIG$n_subgroups,
                                            feature_names = model$feature_names_original)
  
  # Rename terrestrial_richness to vertebrates_richness in results for clarity
  if ("terrestrial_richness" %in% names(conditional_imp)) {
    names(conditional_imp)[names(conditional_imp) == "terrestrial_richness"] <- "vertebrates_richness"
  }
  
  # Show top features
  top_n <- 10
  marginal_sorted <- sort(marginal_imp, decreasing = TRUE)[1:min(top_n, length(marginal_imp))]
  conditional_sorted <- sort(conditional_imp, decreasing = TRUE)[1:min(top_n, length(conditional_imp))]
  
  message(sprintf("\nTop %d features by marginal importance:", top_n))
  for (i in 1:length(marginal_sorted)) {
    message(sprintf("  %2d. %s: %.4f", i, names(marginal_sorted)[i], marginal_sorted[i]))
  }
  
  message(sprintf("\nTop %d features by conditional importance:", top_n))
  for (i in 1:length(conditional_sorted)) {
    message(sprintf("  %2d. %s: %.4f", i, names(conditional_sorted)[i], conditional_sorted[i]))
  }
  
  # Create visualization
  p <- create_pfi_plot(marginal_imp, conditional_imp, corr_pairs,
                      sprintf("Conditional vs Marginal PFI: %s", toupper(response_type)))
  
  # Save plot - explicitly control device to prevent stray output
  filename <- sprintf("results/%s/conditional_pfi_comparison.png", response_type)
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  # Use explicit device to ensure no stray plots
  suppressMessages({
    ggsave(filename, p, width = 12, height = 8, dpi = 150, device = "png")
  })
  message(sprintf("\nSaved: %s", filename))
  
  return(list(
    marginal = marginal_imp,
    conditional = conditional_imp,
    correlations = corr_pairs
  ))
}

# Main execution
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  response_types <- if (length(args) > 0 && args[1] != "all") {
    args[1]
  } else {
    c("fungi", "plants", "vertebrates")
  }
  
  results <- list()
  for (response in response_types) {
    tryCatch({
      results[[response]] <- analyze_response(response)
    }, error = function(e) {
      message(sprintf("Error processing %s: %s", response, e$message))
    })
  }
  
  message("\n=== Analysis complete ===")
}