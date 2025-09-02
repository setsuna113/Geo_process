#!/usr/bin/env Rscript
# ============================================================================
# Residual Biodiversity Relationship Analysis
# ============================================================================
# Analyzes biodiversity relationships after environmental effects removed:
# 1. Partial correlation analysis (symmetric relationships)
# 2. Three RDA models (directional relationships)
#
# Note: Since data are already residualized by DML/ML, we do NOT use
# partial RDA but regular RDA on residuals
#
# Author: Biodiversity-Environment Analysis Pipeline
# Date: 2024
# ============================================================================

library(vegan)
library(arrow)
library(ggplot2)
library(ppcor)  # For partial correlations

#' Analyze residual biodiversity relationships
#'
#' @param data_path Path to residuals arrow file
#' @param output_dir Directory to save results
#' @return List containing partial correlation and RDA results
variance_partitioning_analysis <- function(
  data,
  data_rda = NULL,
  output_dir = "./results/variance_partitioning"
) {
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  n_samples <- nrow(data)
  cat("Analyzing", format(n_samples, big.mark=","), "samples\n")
  cat("Note: These are ML-residualized data (environmental effects already removed)\n\n")
  
  # Use provided RDA sample or full data
  if (is.null(data_rda)) {
    data_rda <- data
  }
  cat("Using", format(nrow(data_rda), big.mark=","), "samples for RDA analysis\n\n")
  
  # ============ PART 1: Partial Correlation Analysis ============
  
  cat(paste(rep("=", 60), collapse=""), "\n")
  cat("PARTIAL CORRELATION ANALYSIS (Symmetric Relationships)\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Since data are already residualized, this is just correlation
  residual_matrix <- as.matrix(data[, c("fungi_richness", "plants_richness", "vertebrates_richness")])
  
  # Compute correlation matrix
  cor_matrix <- cor(residual_matrix)
  
  cat("Correlation Matrix (environmental effects already removed):\n")
  print(round(cor_matrix, 3))
  
  # Compute p-values for correlations
  n <- nrow(residual_matrix)
  p_values <- matrix(NA, 3, 3)
  rownames(p_values) <- colnames(p_values) <- c("fungi", "plants", "vertebrates")
  
  for(i in 1:3) {
    for(j in 1:3) {
      if(i != j) {
        r <- cor_matrix[i, j]
        t_stat <- r * sqrt(n - 2) / sqrt(1 - r^2)
        p_values[i, j] <- 2 * (1 - pt(abs(t_stat), n - 2))
      } else {
        p_values[i, j] <- 0
      }
    }
  }
  
  cat("\nP-values:\n")
  print(round(p_values, 4))
  
  # Identify significant relationships
  cat("\nSignificant Relationships (p < 0.05):\n")
  for(i in 1:2) {
    for(j in (i+1):3) {
      if(p_values[i, j] < 0.05) {
        taxa <- c("fungi", "plants", "vertebrates")
        cat(sprintf("  %s ↔ %s: r = %.3f, p = %.4f\n",
                    taxa[i], taxa[j], cor_matrix[i, j], p_values[i, j]))
      }
    }
  }
  
  # ============ PART 2: Three RDA Models ============
  
  cat("\n", paste(rep("=", 60), collapse=""), "\n")
  cat("RDA ANALYSIS (Directional Relationships)\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  rda_results <- list()
  
  # Model 1: Fungi explained by Plants and Vertebrates
  cat("Model 1: fungi_residuals ~ plants_residuals + vertebrates_residuals\n")
  cat(paste(rep("-", 40), collapse=""), "\n")
  
  rda1 <- rda(data_rda$fungi_richness ~ data_rda$plants_richness + data_rda$vertebrates_richness)
  rda_results$fungi <- rda1
  
  r2_1 <- RsquareAdj(rda1)
  cat(sprintf("  R²: %.4f\n", r2_1$r.squared))
  cat(sprintf("  Adjusted R²: %.4f\n", r2_1$adj.r.squared))
  
  # Permutation test
  anova1 <- anova(rda1, permutations = 999)
  cat(sprintf("  F-statistic: %.2f\n", anova1$F[1]))
  cat(sprintf("  P-value: %.4f\n", anova1$`Pr(>F)`[1]))
  
  # Model 2: Plants explained by Fungi and Vertebrates
  cat("\nModel 2: plants_residuals ~ fungi_residuals + vertebrates_residuals\n")
  cat(paste(rep("-", 40), collapse=""), "\n")
  
  rda2 <- rda(data_rda$plants_richness ~ data_rda$fungi_richness + data_rda$vertebrates_richness)
  rda_results$plants <- rda2
  
  r2_2 <- RsquareAdj(rda2)
  cat(sprintf("  R²: %.4f\n", r2_2$r.squared))
  cat(sprintf("  Adjusted R²: %.4f\n", r2_2$adj.r.squared))
  
  anova2 <- anova(rda2, permutations = 999)
  cat(sprintf("  F-statistic: %.2f\n", anova2$F[1]))
  cat(sprintf("  P-value: %.4f\n", anova2$`Pr(>F)`[1]))
  
  # Model 3: Vertebrates explained by Fungi and Plants
  cat("\nModel 3: vertebrates_residuals ~ fungi_residuals + plants_residuals\n")
  cat(paste(rep("-", 40), collapse=""), "\n")
  
  rda3 <- rda(data_rda$vertebrates_richness ~ data_rda$fungi_richness + data_rda$plants_richness)
  rda_results$vertebrates <- rda3
  
  r2_3 <- RsquareAdj(rda3)
  cat(sprintf("  R²: %.4f\n", r2_3$r.squared))
  cat(sprintf("  Adjusted R²: %.4f\n", r2_3$adj.r.squared))
  
  anova3 <- anova(rda3, permutations = 999)
  cat(sprintf("  F-statistic: %.2f\n", anova3$F[1]))
  cat(sprintf("  P-value: %.4f\n", anova3$`Pr(>F)`[1]))
  
  # ============ Create Visualizations ============
  
  cat("\n\nGenerating visualizations...\n")
  
  # Create RDA biplots using vegan's native plot function
  png(file.path(output_dir, "rda_biplot_fungi.png"), width = 800, height = 800)
  plot(rda1, main = "RDA: Fungi ~ Plants + Vertebrates")
  dev.off()
  
  png(file.path(output_dir, "rda_biplot_plants.png"), width = 800, height = 800)
  plot(rda2, main = "RDA: Plants ~ Fungi + Vertebrates")
  dev.off()
  
  png(file.path(output_dir, "rda_biplot_vertebrates.png"), width = 800, height = 800)
  plot(rda3, main = "RDA: Vertebrates ~ Fungi + Plants")
  dev.off()
  
  # Create correlation heatmap
  png(file.path(output_dir, "correlation_heatmap.png"), width = 800, height = 600)
  
  # Prepare data for heatmap
  cor_df <- reshape2::melt(cor_matrix)
  names(cor_df) <- c("Taxa1", "Taxa2", "Correlation")
  
  print(ggplot(cor_df, aes(x = Taxa1, y = Taxa2, fill = Correlation)) +
    geom_tile() +
    geom_text(aes(label = round(Correlation, 2)), color = "white", size = 5) +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                        midpoint = 0, limit = c(-1, 1)) +
    labs(title = "Residual Correlations Between Taxa",
         subtitle = "After removing environmental effects via ML",
         x = "", y = "") +
    theme_minimal() +
    theme(axis.text = element_text(size = 12),
          plot.title = element_text(size = 14, face = "bold")))
  
  dev.off()
  
  # Create RDA comparison plot
  png(file.path(output_dir, "rda_comparison.png"), width = 1000, height = 400)
  
  r2_data <- data.frame(
    Model = c("Fungi ~ P+V", "Plants ~ F+V", "Vertebrates ~ F+P"),
    R2 = c(r2_1$r.squared, r2_2$r.squared, r2_3$r.squared),
    Adj_R2 = c(r2_1$adj.r.squared, r2_2$adj.r.squared, r2_3$adj.r.squared),
    P_value = c(anova1$`Pr(>F)`[1], anova2$`Pr(>F)`[1], anova3$`Pr(>F)`[1])
  )
  
  print(ggplot(r2_data, aes(x = Model, y = R2)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    geom_bar(aes(y = Adj_R2), stat = "identity", fill = "darkblue", alpha = 0.9) +
    geom_text(aes(label = sprintf("p=%.3f", P_value)), 
              vjust = -0.5, size = 4) +
    ylim(0, max(r2_data$R2) * 1.2) +
    labs(title = "RDA Model Comparison",
         subtitle = "Variance explained in residuals (dark = adjusted R²)",
         x = "Model", y = "R²") +
    theme_minimal())
  
  dev.off()
  
  # ============ Summary Tables ============
  
  cat("\n\n", paste(rep("=", 60), collapse=""), "\n")
  cat("SUMMARY OF ANALYSES\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Correlation summary
  cat("1. PARTIAL CORRELATIONS (symmetric relationships):\n")
  cat(paste(rep("-", 40), collapse=""), "\n")
  cor_summary <- data.frame(
    Pair = c("Fungi-Plants", "Fungi-Vertebrates", "Plants-Vertebrates"),
    Correlation = c(cor_matrix[1,2], cor_matrix[1,3], cor_matrix[2,3]),
    P_value = c(p_values[1,2], p_values[1,3], p_values[2,3]),
    Significant = c(p_values[1,2] < 0.05, p_values[1,3] < 0.05, p_values[2,3] < 0.05)
  )
  print(cor_summary)
  
  # RDA summary
  cat("\n2. RDA MODELS (directional relationships):\n")
  cat(paste(rep("-", 40), collapse=""), "\n")
  rda_summary <- data.frame(
    Response = c("Fungi", "Plants", "Vertebrates"),
    Predictors = c("Plants + Vertebrates", "Fungi + Vertebrates", "Fungi + Plants"),
    R2 = c(r2_1$r.squared, r2_2$r.squared, r2_3$r.squared),
    Adj_R2 = c(r2_1$adj.r.squared, r2_2$adj.r.squared, r2_3$adj.r.squared),
    F_stat = c(anova1$F[1], anova2$F[1], anova3$F[1]),
    P_value = c(anova1$`Pr(>F)`[1], anova2$`Pr(>F)`[1], anova3$`Pr(>F)`[1]),
    Significant = c(anova1$`Pr(>F)`[1] < 0.05, anova2$`Pr(>F)`[1] < 0.05, anova3$`Pr(>F)`[1] < 0.05)
  )
  print(rda_summary)
  
  # Save summary tables
  write.csv(cor_summary, 
            file.path(output_dir, "correlation_summary.csv"),
            row.names = FALSE)
  write.csv(rda_summary,
            file.path(output_dir, "rda_summary.csv"),
            row.names = FALSE)
  
  # ============ Network Strength Analysis ============
  
  cat("\n\n", paste(rep("=", 60), collapse=""), "\n")
  cat("BIOLOGICAL NETWORK STRENGTH\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  # Calculate average absolute correlation as network strength
  cor_abs <- abs(cor_matrix)
  diag(cor_abs) <- 0  # Remove self-correlations
  
  network_strength <- data.frame(
    Taxon = c("Fungi", "Plants", "Vertebrates"),
    Mean_Correlation = c(
      mean(cor_abs[1, -1]),  # Fungi's mean correlation with others
      mean(cor_abs[2, -2]),  # Plants's mean correlation with others
      mean(cor_abs[3, -3])   # Vertebrates's mean correlation with others
    ),
    RDA_R2 = c(r2_1$adj.r.squared, r2_2$adj.r.squared, r2_3$adj.r.squared),
    RDA_Significant = c(
      anova1$`Pr(>F)`[1] < 0.05,
      anova2$`Pr(>F)`[1] < 0.05,
      anova3$`Pr(>F)`[1] < 0.05
    )
  )
  
  cat("Network connectivity (after removing environmental effects):\n")
  for (i in 1:3) {
    cat(sprintf("  %s: Mean |r| = %.3f, RDA R² = %.3f %s\n",
                network_strength$Taxon[i],
                network_strength$Mean_Correlation[i],
                network_strength$RDA_R2[i],
                ifelse(network_strength$RDA_Significant[i], "(significant)", "")))
  }
  
  # ============ Create Network Visualization ============
  
  cat("\nGenerating network visualization...\n")
  
  png(file.path(output_dir, "network_strength.png"), width = 1000, height = 800)
  
  # Bar plot of RDA R²
  print(ggplot(network_strength, aes(x = Taxon, y = RDA_R2 * 100)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label = sprintf("%.1f%%", RDA_R2 * 100)), 
              vjust = -0.5, size = 5) +
    ylim(0, max(network_strength$RDA_R2 * 100) * 1.2) +
    labs(title = "Biological Network Strength",
         subtitle = "Percentage of variance explained by other taxa (after removing environment)",
         x = "Taxon",
         y = "Variance Explained (%)") +
    theme_minimal() +
    theme(text = element_text(size = 14)))
  
  dev.off()
  
  # ============ Interpretation Guidelines ============
  
  cat("\n\n", paste(rep("=", 60), collapse=""), "\n")
  cat("INTERPRETATION GUIDELINES\n")
  cat(paste(rep("=", 60), collapse=""), "\n\n")
  
  cat("1. PARTIAL CORRELATIONS:\n")
  cat("   - Show symmetric relationships between taxa\n")
  cat("   - Values represent linear associations after removing environmental effects\n")
  cat("   - Significant correlations suggest biological relationships\n\n")
  
  cat("2. RDA MODELS:\n")
  cat("   - Show directional predictive relationships\n")
  cat("   - R² indicates variance in response explained by predictors\n")
  cat("   - Different from correlations (directional vs symmetric)\n\n")
  
  cat("3. KEY INSIGHT:\n")
  cat("   - These are residual relationships AFTER ML removed environmental effects\n")
  cat("   - Represent potential biological interactions, not environmental co-responses\n")
  cat("   - Could include: mutualism, competition, facilitation, trophic relationships\n")
  
  # ============ Save Results ============
  
  # Save all results as RDS
  results <- list(
    correlation_matrix = cor_matrix,
    p_values = p_values,
    cor_summary = cor_summary,
    rda_fungi = rda1,
    rda_plants = rda2,
    rda_vertebrates = rda3,
    rda_summary = rda_summary,
    network_strength = network_strength,
    n_samples = n_samples
  )
  
  saveRDS(results, file.path(output_dir, "residual_analysis_results.rds"))
  
  cat("\nResults saved to:", output_dir, "\n")
  cat("Files generated:\n")
  cat("  - rda_biplot_fungi.png\n")
  cat("  - rda_biplot_plants.png\n")
  cat("  - rda_biplot_vertebrates.png\n")
  cat("  - correlation_heatmap.png\n")
  cat("  - rda_comparison.png\n")
  cat("  - network_strength.png\n")
  cat("  - correlation_summary.csv\n")
  cat("  - rda_summary.csv\n")
  cat("  - residual_analysis_results.rds\n")
  
  return(results)
}


# Run if executed directly
# This file should be sourced from main_analysis.R
# Do not run if sourced
