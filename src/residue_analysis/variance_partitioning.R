#!/usr/bin/env Rscript
# ============================================================================
# Core Variance Partitioning Analysis
# ============================================================================
# Performs variance partitioning on DML residuals to understand biological
# interdependencies after removing environmental effects
#
# Author: Biodiversity-Environment Analysis Pipeline
# Date: 2024
# ============================================================================

library(vegan)
library(arrow)
library(ggplot2)

#' Perform three-way variance partitioning
#'
#' @param data_path Path to residuals arrow file
#' @param output_dir Directory to save results
#' @return List containing variance partitioning results for all three taxa
variance_partitioning_analysis <- function(
  data_path = "/scratch/yl998/econml_results_5m/homogeneous/residuals.arrow",
  output_dir = "./results/variance_partitioning"
) {
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Load residuals
  cat("Loading residuals from:", data_path, "\n")
  data <- read_feather(data_path)
  
  n_samples <- nrow(data)
  cat("Total samples:", format(n_samples, big.mark=","), "\n\n")
  
  # ============ Three-way Variance Partitioning ============
  
  cat("=" * 60, "\n")
  cat("VARIANCE PARTITIONING ANALYSIS\n")
  cat("=" * 60, "\n\n")
  
  # Perform variance partitioning for each taxon
  vp_results <- list()
  
  # 1. Fungi explained by Plants and Vertebrates
  cat("1. Fungi ~ Plants + Vertebrates\n")
  cat("-" * 40, "\n")
  
  vp_fungi <- varpart(
    Y = data$fungi_richness,
    X = data$plants_richness,
    data$vertebrates_richness
  )
  
  vp_results$fungi <- vp_fungi
  print_vp_results(vp_fungi, "Fungi", c("Plants", "Vertebrates"))
  
  # 2. Plants explained by Fungi and Vertebrates
  cat("\n2. Plants ~ Fungi + Vertebrates\n")
  cat("-" * 40, "\n")
  
  vp_plants <- varpart(
    Y = data$plants_richness,
    X = data$fungi_richness,
    data$vertebrates_richness
  )
  
  vp_results$plants <- vp_plants
  print_vp_results(vp_plants, "Plants", c("Fungi", "Vertebrates"))
  
  # 3. Vertebrates explained by Fungi and Plants
  cat("\n3. Vertebrates ~ Fungi + Plants\n")
  cat("-" * 40, "\n")
  
  vp_vertebrates <- varpart(
    Y = data$vertebrates_richness,
    X = data$fungi_richness,
    data$plants_richness
  )
  
  vp_results$vertebrates <- vp_vertebrates
  print_vp_results(vp_vertebrates, "Vertebrates", c("Fungi", "Plants"))
  
  # ============ Create Venn Diagrams ============
  
  cat("\n" * 2)
  cat("Generating Venn diagrams...\n")
  
  pdf(file.path(output_dir, "venn_diagrams.pdf"), width = 15, height = 5)
  par(mfrow = c(1, 3))
  
  # Plot each Venn diagram
  plot(vp_fungi, digits = 3, Xnames = c("Plants", "Vertebrates"),
       main = "Fungi Variance Explained", cex = 1.2)
  
  plot(vp_plants, digits = 3, Xnames = c("Fungi", "Vertebrates"),
       main = "Plants Variance Explained", cex = 1.2)
  
  plot(vp_vertebrates, digits = 3, Xnames = c("Fungi", "Plants"),
       main = "Vertebrates Variance Explained", cex = 1.2)
  
  dev.off()
  
  # ============ Summary Table ============
  
  cat("\n" * 2)
  cat("=" * 60, "\n")
  cat("SUMMARY OF VARIANCE PARTITIONING\n")
  cat("=" * 60, "\n\n")
  
  # Create summary table
  summary_table <- data.frame(
    Response = c("Fungi", "Plants", "Vertebrates"),
    Predictor1_unique = c(
      vp_fungi$part$indfract$Adj.R.squared[1],
      vp_plants$part$indfract$Adj.R.squared[1],
      vp_vertebrates$part$indfract$Adj.R.squared[1]
    ),
    Predictor2_unique = c(
      vp_fungi$part$indfract$Adj.R.squared[2],
      vp_plants$part$indfract$Adj.R.squared[2],
      vp_vertebrates$part$indfract$Adj.R.squared[2]
    ),
    Shared = c(
      vp_fungi$part$indfract$Adj.R.squared[3],
      vp_plants$part$indfract$Adj.R.squared[3],
      vp_vertebrates$part$indfract$Adj.R.squared[3]
    ),
    Total_explained = c(
      vp_fungi$part$fract$Adj.R.squared[3],
      vp_plants$part$fract$Adj.R.squared[3],
      vp_vertebrates$part$fract$Adj.R.squared[3]
    ),
    Unexplained = c(
      vp_fungi$part$indfract$Adj.R.squared[4],
      vp_plants$part$indfract$Adj.R.squared[4],
      vp_vertebrates$part$indfract$Adj.R.squared[4]
    )
  )
  
  # Set proper names
  names(summary_table)[2] <- c("Unique_Pred1")
  names(summary_table)[3] <- c("Unique_Pred2")
  
  print(summary_table)
  
  # Save summary table
  write.csv(summary_table, 
            file.path(output_dir, "variance_partitioning_summary.csv"),
            row.names = FALSE)
  
  # ============ Network Strength Analysis ============
  
  cat("\n" * 2)
  cat("=" * 60, "\n")
  cat("BIOLOGICAL NETWORK STRENGTH\n")
  cat("=" * 60, "\n\n")
  
  # Calculate network metrics
  network_strength <- data.frame(
    Taxon = c("Fungi", "Plants", "Vertebrates"),
    Total_R2 = c(
      vp_fungi$part$fract$Adj.R.squared[3],
      vp_plants$part$fract$Adj.R.squared[3],
      vp_vertebrates$part$fract$Adj.R.squared[3]
    ),
    Shared_fraction = c(
      vp_fungi$part$indfract$Adj.R.squared[3] / vp_fungi$part$fract$Adj.R.squared[3],
      vp_plants$part$indfract$Adj.R.squared[3] / vp_plants$part$fract$Adj.R.squared[3],
      vp_vertebrates$part$indfract$Adj.R.squared[3] / vp_vertebrates$part$fract$Adj.R.squared[3]
    )
  )
  
  # Replace NaN with 0 for shared fraction
  network_strength$Shared_fraction[is.nan(network_strength$Shared_fraction)] <- 0
  
  cat("Network connectivity (R² explained by other taxa):\n")
  for (i in 1:3) {
    cat(sprintf("  %s: %.1f%% (%.1f%% shared)\n",
                network_strength$Taxon[i],
                network_strength$Total_R2[i] * 100,
                network_strength$Shared_fraction[i] * 100))
  }
  
  # ============ Create Network Visualization ============
  
  cat("\nGenerating network visualization...\n")
  
  pdf(file.path(output_dir, "network_strength.pdf"), width = 10, height = 8)
  
  # Bar plot of total R²
  ggplot(network_strength, aes(x = Taxon, y = Total_R2 * 100)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label = sprintf("%.1f%%", Total_R2 * 100)), 
              vjust = -0.5, size = 5) +
    ylim(0, max(network_strength$Total_R2 * 100) * 1.2) +
    labs(title = "Biological Network Strength",
         subtitle = "Percentage of variance explained by other taxa (after removing environment)",
         x = "Taxon",
         y = "Variance Explained (%)") +
    theme_minimal() +
    theme(text = element_text(size = 14))
  
  dev.off()
  
  # ============ Testable Fractions ============
  
  cat("\n" * 2)
  cat("=" * 60, "\n")
  cat("TESTABLE FRACTIONS\n")
  cat("=" * 60, "\n\n")
  
  cat("Note: The following fractions can be tested for significance using RDA:\n\n")
  
  for (taxon in c("fungi", "plants", "vertebrates")) {
    vp <- vp_results[[taxon]]
    testable <- which(vp$part$fract$Testable)
    
    cat(toupper(substring(taxon, 1, 1)), substring(taxon, 2), ":\n", sep = "")
    for (i in testable) {
      cat("  Fraction", rownames(vp$part$fract)[i], ":",
          sprintf("Adj.R² = %.4f", vp$part$fract$Adj.R.squared[i]), "\n")
    }
    cat("\n")
  }
  
  # ============ Save Results ============
  
  # Save all results as RDS
  results <- list(
    vp_fungi = vp_fungi,
    vp_plants = vp_plants,
    vp_vertebrates = vp_vertebrates,
    summary_table = summary_table,
    network_strength = network_strength,
    n_samples = n_samples
  )
  
  saveRDS(results, file.path(output_dir, "variance_partitioning_results.rds"))
  
  cat("Results saved to:", output_dir, "\n")
  
  return(results)
}

#' Helper function to print variance partitioning results
#'
#' @param vp Variance partitioning object
#' @param response Name of response variable
#' @param predictors Names of predictor variables
print_vp_results <- function(vp, response, predictors) {
  # Extract values
  adj_r2 <- vp$part$indfract$Adj.R.squared
  
  # Individual fractions
  cat(sprintf("  [a] %s alone: %.4f (%.1f%%)\n", 
              predictors[1], adj_r2[1], adj_r2[1] * 100))
  cat(sprintf("  [b] %s alone: %.4f (%.1f%%)\n",
              predictors[2], adj_r2[2], adj_r2[2] * 100))
  cat(sprintf("  [c] Shared: %.4f (%.1f%%)\n",
              adj_r2[3], adj_r2[3] * 100))
  cat(sprintf("  [d] Unexplained: %.4f (%.1f%%)\n",
              adj_r2[4], adj_r2[4] * 100))
  
  # Total
  total_r2 <- vp$part$fract$Adj.R.squared[3]
  cat(sprintf("  Total explained: %.4f (%.1f%%)\n",
              total_r2, total_r2 * 100))
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
    output_dir <- "./results/variance_partitioning"
  }
  
  results <- variance_partitioning_analysis(data_path, output_dir)
}