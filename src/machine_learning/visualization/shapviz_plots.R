#!/usr/bin/env Rscript
# shapviz visualization for SHAP values from Julia GLM models
# Creates beautiful visualizations using ggplot2-based shapviz package

library(shapviz)
library(ggplot2)
library(jsonlite)
library(readr)
library(dplyr)
library(patchwork)  # For combining plots

# Set theme
theme_set(theme_minimal())

#' Load SHAP data from Python output
#'
#' @param model_dir Directory containing SHAP outputs for a model
#' @param model_name Name of the model
#' @return shapviz object
load_shap_data <- function(model_dir, model_name) {
  message(sprintf("Loading SHAP data for model: %s", model_name))
  
  # Try new single Arrow file first
  arrow_file <- file.path(model_dir, paste0(model_name, "_shap_results.arrow"))
  
  if (file.exists(arrow_file)) {
    # Load from single Arrow file
    library(arrow)
    data <- read_feather(arrow_file)
    
    # Extract metadata from Arrow schema
    arrow_table <- arrow::read_feather(arrow_file, as_data_frame = FALSE)
    metadata <- arrow_table$schema$metadata
    
    # Separate features and SHAP values based on column names
    feature_cols <- grep("^shap_", names(data), value = TRUE, invert = TRUE)
    shap_cols <- grep("^shap_", names(data), value = TRUE)
    
    # Extract data
    X <- data[, feature_cols]
    shap_values <- as.matrix(data[, shap_cols])
    colnames(shap_values) <- gsub("^shap_", "", shap_cols)  # Remove prefix
    
    # Parse baseline from metadata
    baseline <- as.numeric(metadata$baseline)
    
    message(sprintf("  Loaded from single Arrow file: %d samples with %d features", 
                    nrow(shap_values), ncol(shap_values)))
  } else {
    # Fallback to old CSV format for backward compatibility
    shap_file <- file.path(model_dir, paste0(model_name, "_shap_values.csv"))
    shap_values <- as.matrix(read_csv(shap_file, show_col_types = FALSE))
    
    feature_file <- file.path(model_dir, paste0(model_name, "_features.csv"))
    X <- read_csv(feature_file, show_col_types = FALSE)
    
    metadata_file <- file.path(model_dir, paste0(model_name, "_shap_metadata.json"))
    metadata <- fromJSON(metadata_file)
    baseline <- metadata$baseline
    
    message(sprintf("  Loaded from CSV files (legacy): %d samples with %d features", 
                    nrow(shap_values), ncol(shap_values)))
  }
  
  # Create shapviz object
  sv <- shapviz(
    object = shap_values,
    X = X,
    baseline = baseline
  )
  
  return(sv)
}

#' Create comprehensive SHAP visualizations
#'
#' @param sv shapviz object
#' @param output_dir Directory to save plots
#' @param model_name Name of the model
#' @param save_plots Whether to save plots to files
create_shapviz_plots <- function(sv, output_dir, model_name, save_plots = TRUE) {
  
  if (save_plots) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  # 1. Feature importance plot (bar and beeswarm)
  message("Creating feature importance plots...")
  
  p_importance_bar <- sv_importance(sv, kind = "bar", show_numbers = TRUE) +
    labs(title = paste("Feature Importance -", model_name),
         subtitle = "Mean absolute SHAP values") +
    theme(plot.title = element_text(size = 14, face = "bold"),
          plot.subtitle = element_text(size = 11))
  
  p_importance_bee <- sv_importance(sv, kind = "beeswarm", max_display = 15) +
    labs(title = paste("SHAP Summary -", model_name),
         subtitle = "Distribution of SHAP values") +
    theme(plot.title = element_text(size = 14, face = "bold"),
          plot.subtitle = element_text(size = 11))
  
  if (save_plots) {
    ggsave(file.path(output_dir, paste0(model_name, "_importance_bar.png")),
           p_importance_bar, width = 10, height = 8, dpi = 150)
    ggsave(file.path(output_dir, paste0(model_name, "_importance_beeswarm.png")),
           p_importance_bee, width = 12, height = 8, dpi = 150)
  }
  
  # 2. Dependence plots for top features
  message("Creating dependence plots...")
  
  # Get top 6 features by importance
  importance <- sv_importance(sv, kind = "no_plot")
  top_features <- head(rownames(importance), 6)
  
  dep_plots <- list()
  for (i in seq_along(top_features)) {
    feat <- top_features[i]
    dep_plots[[i]] <- sv_dependence(sv, v = feat, color_var = "auto") +
      labs(title = paste("SHAP Dependence:", feat)) +
      theme(plot.title = element_text(size = 11))
  }
  
  # Combine dependence plots in a grid
  p_dependence_grid <- wrap_plots(dep_plots, ncol = 3, nrow = 2)
  
  if (save_plots) {
    ggsave(file.path(output_dir, paste0(model_name, "_dependence_grid.png")),
           p_dependence_grid, width = 15, height = 10, dpi = 150)
  }
  
  # 3. 2D dependence plots (unique shapviz feature!)
  message("Creating 2D dependence plots...")
  
  if (length(top_features) >= 2) {
    # Create 2D plots for pairs of top features
    p_2d_1 <- sv_dependence2D(sv, x = top_features[1], y = top_features[2]) +
      labs(title = paste("2D SHAP Interaction:", 
                         top_features[1], "vs", top_features[2]))
    
    if (length(top_features) >= 4) {
      p_2d_2 <- sv_dependence2D(sv, x = top_features[3], y = top_features[4]) +
        labs(title = paste("2D SHAP Interaction:", 
                           top_features[3], "vs", top_features[4]))
      
      p_2d_combined <- p_2d_1 + p_2d_2
      
      if (save_plots) {
        ggsave(file.path(output_dir, paste0(model_name, "_dependence2D.png")),
               p_2d_combined, width = 14, height = 6, dpi = 150)
      }
    } else {
      if (save_plots) {
        ggsave(file.path(output_dir, paste0(model_name, "_dependence2D.png")),
               p_2d_1, width = 7, height = 6, dpi = 150)
      }
    }
  }
  
  # 4. Waterfall plots for specific observations
  message("Creating waterfall plots...")
  
  # Select interesting observations (highest/lowest predictions)
  pred_order <- order(sv$baseline + rowSums(sv$S))
  interesting_obs <- c(pred_order[1], pred_order[length(pred_order)/2], 
                      pred_order[length(pred_order)])
  
  waterfall_plots <- list()
  for (i in seq_along(interesting_obs)) {
    obs_idx <- interesting_obs[i]
    waterfall_plots[[i]] <- sv_waterfall(sv, row_id = obs_idx) +
      labs(title = paste("SHAP Waterfall - Observation", obs_idx)) +
      theme(plot.title = element_text(size = 11))
  }
  
  p_waterfall_grid <- wrap_plots(waterfall_plots, ncol = 1)
  
  if (save_plots) {
    ggsave(file.path(output_dir, paste0(model_name, "_waterfall.png")),
           p_waterfall_grid, width = 10, height = 12, dpi = 150)
  }
  
  # 5. Force plot (if available)
  message("Creating force plot...")
  
  tryCatch({
    p_force <- sv_force(sv, row_id = 1) +
      labs(title = paste("SHAP Force Plot -", model_name))
    
    if (save_plots) {
      ggsave(file.path(output_dir, paste0(model_name, "_force.png")),
             p_force, width = 12, height = 4, dpi = 150)
    }
  }, error = function(e) {
    message("  Note: Force plot not available or failed")
  })
  
  # 6. Create combined summary plot
  message("Creating combined summary plot...")
  
  p_combined <- (p_importance_bar | p_importance_bee) / p_dependence_grid
  
  if (save_plots) {
    ggsave(file.path(output_dir, paste0(model_name, "_summary_combined.png")),
           p_combined, width = 16, height = 14, dpi = 150)
  }
  
  message(sprintf("All plots saved to: %s", output_dir))
  
  return(list(
    importance_bar = p_importance_bar,
    importance_bee = p_importance_bee,
    dependence_grid = p_dependence_grid,
    waterfall_grid = p_waterfall_grid,
    combined = p_combined
  ))
}

#' Process multiple models
#'
#' @param shap_base_dir Base directory containing model subdirectories
#' @param output_base_dir Base directory for outputs
#' @param models Vector of model names to process (NULL for all)
process_models <- function(shap_base_dir, output_base_dir = "results/shapviz", 
                          models = NULL) {
  
  # Find all model directories
  if (is.null(models)) {
    model_dirs <- list.dirs(shap_base_dir, full.names = FALSE, recursive = FALSE)
    models <- model_dirs[model_dirs != ""]
  }
  
  message(sprintf("Processing %d model(s)", length(models)))
  
  results <- list()
  
  for (model_name in models) {
    message(sprintf("\n=== Processing model: %s ===", model_name))
    
    model_dir <- file.path(shap_base_dir, model_name)
    
    if (!dir.exists(model_dir)) {
      warning(sprintf("Model directory not found: %s", model_dir))
      next
    }
    
    tryCatch({
      # Load SHAP data
      sv <- load_shap_data(model_dir, model_name)
      
      # Create output directory
      output_dir <- file.path(output_base_dir, model_name)
      
      # Create visualizations
      plots <- create_shapviz_plots(sv, output_dir, model_name, save_plots = TRUE)
      
      results[[model_name]] <- list(
        shapviz_object = sv,
        plots = plots
      )
      
      message(sprintf("✓ Successfully processed: %s", model_name))
      
    }, error = function(e) {
      warning(sprintf("Failed to process %s: %s", model_name, e$message))
    })
  }
  
  message(sprintf("\n=== Completed processing %d/%d models ===", 
                  length(results), length(models)))
  
  return(results)
}

#' Create comparative plots across models
#'
#' @param results List of results from process_models
#' @param output_dir Directory to save comparative plots
create_comparative_plots <- function(results, output_dir = "results/shapviz/comparative") {
  
  if (length(results) < 2) {
    message("Need at least 2 models for comparative plots")
    return(NULL)
  }
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  message("Creating comparative plots...")
  
  # Extract importance from each model
  importance_data <- list()
  for (model_name in names(results)) {
    sv <- results[[model_name]]$shapviz_object
    imp <- sv_importance(sv, kind = "no_plot")
    imp_df <- data.frame(
      feature = rownames(imp),
      importance = imp$value,
      model = model_name
    )
    importance_data[[model_name]] <- imp_df
  }
  
  # Combine importance data
  importance_combined <- bind_rows(importance_data)
  
  # Create comparative importance plot
  p_comparative <- ggplot(importance_combined, 
                          aes(x = reorder(feature, importance), 
                              y = importance, 
                              fill = model)) +
    geom_bar(stat = "identity", position = "dodge") +
    coord_flip() +
    labs(title = "Comparative Feature Importance Across Models",
         x = "Feature",
         y = "Mean |SHAP value|",
         fill = "Model") +
    theme_minimal() +
    theme(plot.title = element_text(size = 14, face = "bold"),
          legend.position = "bottom")
  
  ggsave(file.path(output_dir, "comparative_importance.png"),
         p_comparative, width = 12, height = 8, dpi = 150)
  
  message(sprintf("Comparative plots saved to: %s", output_dir))
  
  return(p_comparative)
}

# Main function for command-line usage
main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    cat("Usage: Rscript shapviz_plots.R <shap_data_dir> [output_dir] [model1,model2,...]\n")
    cat("  shap_data_dir: Directory containing Python SHAP outputs\n")
    cat("  output_dir: Directory for shapviz plots (default: results/shapviz)\n")
    cat("  models: Comma-separated list of models (default: all)\n")
    quit(status = 1)
  }
  
  shap_dir <- args[1]
  output_dir <- ifelse(length(args) >= 2, args[2], "results/shapviz")
  
  models <- NULL
  if (length(args) >= 3) {
    models <- strsplit(args[3], ",")[[1]]
  }
  
  # Process models
  results <- process_models(shap_dir, output_dir, models)
  
  # Create comparative plots if multiple models
  if (length(results) > 1) {
    create_comparative_plots(results, file.path(output_dir, "comparative"))
  }
  
  message("\n=== shapviz visualization complete ===")
}

# Run main function if script is executed directly
if (!interactive()) {
  main()
}