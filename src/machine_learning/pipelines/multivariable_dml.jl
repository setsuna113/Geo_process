#!/usr/bin/env julia
"""
Multivariable Double Machine Learning (DML) with NEYMAN ORTHOGONAL SCORE

This implements TRUE DML using the Neyman orthogonal score functions from:
- Xiang et al. (2025) equations 11-12 for multiple treatments

Key Features:
1. Cross-fitting with LightGBM for nuisance functions
2. Neyman orthogonal score: ψ(W;Θ,η) = [A - m(X)]^T * (Y - l(X) - [A - m(X)]^T * θ)
3. Parameter estimation via Equation 11: Θ = -E[ψa]^{-1} * E[ψb]
4. Variance via score function: Var(Θ) = E[ψa]^{-1} * Var(ψ) * E[ψa]^{-1}

This achieves:
- Orthogonality between nuisance parameters and causal effects
- Robustness to first-stage ML estimation errors
- Valid inference even with flexible ML methods

Nuisance functions estimated with LightGBM (Gradient Boosting)
"""

using DataFrames, Statistics, LinearAlgebra, Random
using LightGBM
using StatsBase: sample
using MLBase: Kfold  # For cross-validation folds
using Distributions: Normal, cdf
using JSON, Dates
using YAML  # For loading config
using LinearAlgebra: Diagonal, det, inv  # Explicit imports for clarity

# ============================================================================
# MULTIVARIABLE DML IMPLEMENTATION
# ============================================================================

"""
    MultivariableDML

True implementation of Multivariable Double Machine Learning with cross-fitting.
Handles multiple continuous treatments and their interactions.
"""
mutable struct MultivariableDML
    n_folds::Int
    n_threads::Int
    lgb_params_outcome::String  # Parameters for outcome model g(X)
    lgb_params_treatment::String  # Parameters for treatment models m(X)
    lgb_num_iterations_outcome::Int
    lgb_num_iterations_treatment::Int
    verbose::Bool
    
    # Results storage
    coefficients::Union{Nothing, Vector{Float32}}
    std_errors::Union{Nothing, Vector{Float32}}
    vcov::Union{Nothing, Matrix{Float32}}
    treatment_names::Vector{String}
    n_obs::Int
end

"""
    MultivariableDML(; n_folds=5, config_path="", verbose=true)

Initialize Multivariable DML estimator with config file parameters.

Args:
    n_folds: Number of cross-fitting folds (default: 5)
    config_path: Path to ML config file (default: uses built-in config)
    verbose: Print progress messages (default: true)
"""
function MultivariableDML(; n_folds::Int=5, config_path::String="", verbose::Bool=true)
    # Load config or use defaults
    if !isempty(config_path) && isfile(config_path)
        config = YAML.load_file(config_path)
        lgb_outcome = config["regression_analysis"]["models"]["lightgbm"]["outcome_model"]
        lgb_treatment = config["regression_analysis"]["models"]["lightgbm"]["treatment_model"]
    else
        # Default parameters optimized for large datasets with spatial sampling
        lgb_outcome = Dict{String, Any}(
            "objective" => "regression",
            "metric" => "rmse",
            "num_leaves" => 31,
            "max_depth" => 6,
            "learning_rate" => 0.1,
            "feature_fraction" => 0.9,
            "bagging_fraction" => 0.5,  # Aggressive subsampling
            "bagging_freq" => 1,
            "min_data_in_leaf" => 100,  # Conservative for sampled data
            "num_threads" => 80,
            "verbose" => -1,
            "seed" => 42,
            "num_iterations" => 100,
            "force_col_wise" => true
        )
        lgb_treatment = lgb_outcome  # Same params by default
    end
    
    # Override num_threads if specified
    lgb_outcome["num_threads"] = get(lgb_outcome, "num_threads", 80)
    lgb_treatment["num_threads"] = get(lgb_treatment, "num_threads", 80)
    
    # Convert to parameter strings for LightGBM API
    lgb_params_outcome = join(["$k=$v" for (k,v) in lgb_outcome], " ")
    lgb_params_treatment = join(["$k=$v" for (k,v) in lgb_treatment], " ")
    
    return MultivariableDML(
        n_folds, 
        lgb_outcome["num_threads"],
        lgb_params_outcome, 
        lgb_params_treatment,
        lgb_outcome["num_iterations"], 
        lgb_treatment["num_iterations"],
        verbose,
        nothing, nothing, nothing, String[], 0
    )
end

"""
    fit_dml!(dml::MultivariableDML, data::DataFrame, confounders, treatments, outcome)

Fit the Multivariable DML model using cross-fitting.

Args:
    dml: MultivariableDML object
    data: DataFrame with all variables
    confounders: Vector of confounder column names (climate, elevation, MESF)
    treatments: Dict of treatment names => column names
    outcome: Outcome column name

Returns:
    The fitted DML object with results
"""
function fit_dml!(dml::MultivariableDML, 
              data::DataFrame,
              confounders::Vector{String},
              treatments::Dict{String, String},
              outcome::String)
    
    n = nrow(data)
    dml.n_obs = n
    n_treatments = length(treatments)
    
    if dml.verbose
        println("\n" * "="^70)
        println("MULTIVARIABLE DOUBLE MACHINE LEARNING")
        println("="^70)
        println("Data: $n observations")
        println("Treatments: $(join(keys(treatments), ", "))")
        println("Confounders: $(length(confounders)) variables")
        println("Cross-fitting with $(dml.n_folds) folds")
    end
    
    # ========================================================================
    # STEP 1: SETUP AND INTERACTION HANDLING
    # ========================================================================
    
    # If we have interactions, need to compute residuals for interaction term DIRECTLY
    has_interaction = (n_treatments == 2)
    if has_interaction
        # Create interaction column in data
        t_names = collect(keys(treatments))
        interaction_col = "$(t_names[1])_x_$(t_names[2])"
        
        # Get the actual column names for the treatments
        col1 = treatments[t_names[1]]
        col2 = treatments[t_names[2]]
        
        # Create interaction as product of actual treatment columns
        data[!, Symbol(interaction_col)] = data[!, Symbol(col1)] .* data[!, Symbol(col2)]
        
        # Add to treatments dict
        treatments[interaction_col] = interaction_col
        n_treatments = length(treatments)  # Update count
    end
    
    # Initialize residual storage AFTER adding interaction
    Y_residuals = zeros(Float32, n)
    A_residuals = Dict{String, Vector{Float32}}()
    for t in keys(treatments)
        A_residuals[t] = zeros(Float32, n)
    end
    
    # Create fold indices using MLBase's Kfold
    Random.seed!(42)
    kf = Kfold(n, dml.n_folds)
    fold_indices = collect(kf)  # Convert to array of fold indices
    
    for (fold_idx, test_idx) in enumerate(fold_indices)
        if dml.verbose
            println("\nProcessing fold $fold_idx/$(dml.n_folds)...")
        end
        
        # Get training indices (all except current fold)
        train_idx = setdiff(1:n, test_idx)
        
        # Prepare data splits
        X_train = Matrix(data[train_idx, confounders])
        X_test = Matrix(data[test_idx, confounders])
        
        # ====================================================================
        # 1a. Fit outcome nuisance function: g(X) = E[Y|X]
        # ====================================================================
        y_train = data[train_idx, outcome]  # Already Float32
        
        # Create LightGBM dataset
        train_data = LightGBM.LGBM_DatasetCreateFromMat(X_train, "")
        LightGBM.LGBM_DatasetSetField(train_data, "label", y_train)
        
        # Train outcome model with specific parameters
        booster_y = LightGBM.LGBM_BoosterCreate(train_data, dml.lgb_params_outcome)
        
        # Train without early stopping
        if dml.verbose && fold_idx == 1
            println("    Training outcome model g(X) with $(dml.lgb_num_iterations_outcome) iterations...")
        end
        for iter in 1:dml.lgb_num_iterations_outcome
            LightGBM.LGBM_BoosterUpdateOneIter(booster_y)
        end
        
        # Predict on test fold (workaround for API issue)
        n_test = size(X_test, 1)
        y_pred = zeros(n_test)
        for i in 1:n_test
            pred_result = LightGBM.LGBM_BoosterPredictForMat(
                booster_y, reshape(X_test[i,:], 1, :), 0, 0, 1
            )
            y_pred[i] = pred_result[1]
        end
        
        # Compute residuals
        Y_residuals[test_idx] = data[test_idx, outcome] - y_pred
        
        # Debug: Check if model is learning (first fold only)
        if dml.verbose && fold_idx == 1
            actual_y = data[test_idx, outcome]
            baseline_pred = mean(y_train) * ones(n_test)
            baseline_mse = mean((actual_y - baseline_pred).^2)
            model_mse = mean((actual_y - y_pred).^2)
            println("    Outcome model MSE: $(round(model_mse, digits=2)) (baseline: $(round(baseline_mse, digits=2)))")
            println("    Improvement: $(round(100*(1 - model_mse/baseline_mse), digits=1))%")
        end
        
        # Clean up
        LightGBM.LGBM_DatasetFree(train_data)
        LightGBM.LGBM_BoosterFree(booster_y)
        
        # ====================================================================
        # 1b. Fit treatment nuisance functions: m_j(X) = E[A_j|X]
        # INCLUDING INTERACTION TERM AS ITS OWN TREATMENT!
        # ====================================================================
        for (t_name, t_col) in treatments
            a_train = data[train_idx, t_col]  # Already Float32
            
            # Create dataset
            train_data_a = LightGBM.LGBM_DatasetCreateFromMat(X_train, "")
            LightGBM.LGBM_DatasetSetField(train_data_a, "label", a_train)
            
            # Train treatment model with specific parameters
            booster_a = LightGBM.LGBM_BoosterCreate(train_data_a, dml.lgb_params_treatment)
            
            # Train without early stopping
            if dml.verbose && fold_idx == 1 && t_name == collect(keys(treatments))[1]
                println("    Training treatment models m(X) with $(dml.lgb_num_iterations_treatment) iterations...")
            end
            for iter in 1:dml.lgb_num_iterations_treatment
                LightGBM.LGBM_BoosterUpdateOneIter(booster_a)
            end
            
            # Predict (workaround for API issue)
            n_test = size(X_test, 1)
            a_pred = zeros(n_test)
            for i in 1:n_test
                pred_result = LightGBM.LGBM_BoosterPredictForMat(
                    booster_a, reshape(X_test[i,:], 1, :), 0, 0, 1
                )
                a_pred[i] = pred_result[1]
            end
            
            # Compute residuals (including for interaction term!)
            A_residuals[t_name][test_idx] = data[test_idx, t_col] - a_pred
            
            # Clean up
            LightGBM.LGBM_DatasetFree(train_data_a)
            LightGBM.LGBM_BoosterFree(booster_a)
        end
        
        if dml.verbose
            println("  ✓ Fold $fold_idx complete")
        end
    end
    
    # ========================================================================
    # STEP 2: COMPUTE θ USING NEYMAN ORTHOGONAL SCORE (Equations 11-12)
    # ========================================================================
    
    if dml.verbose
        println("\nComputing causal effects using Neyman orthogonal score...")
    end
    
    # Get treatment names (now includes interaction if it was added)
    treatment_names = collect(keys(treatments))
    
    # Stack ALL residuals across ALL folds
    A_matrix = hcat([A_residuals[t] for t in treatment_names]...)
    
    # ====================================================================
    # NEYMAN ORTHOGONAL SCORE IMPLEMENTATION (Xiang et al. 2025)
    # Equation 12: ψ(W;Θ,η) = [A - m(X)]^T * (Y - l(X) - [A - m(X)]^T * θ)
    # 
    # The score has two components:
    # ψa(W;η) = -[A - m(X)][A - m(X)]^T  (linear in θ)
    # ψb(W;η) = [A - m(X)](Y - l(X))     (independent of θ)
    # ====================================================================
    
    # Compute ψa component: E[-AA^T] where A are treatment residuals
    psi_a = -(A_matrix' * A_matrix) / n  # E[ψa] = -E[AA^T]
    
    # Compute ψb component: E[A*Y] where Y are outcome residuals
    psi_b = (A_matrix' * Y_residuals) / n  # E[ψb] = E[A(Y-l(X))]
    
    # Check for singularity of ψa
    if det(psi_a) ≈ 0
        error("Singular matrix in Neyman score. Check for multicollinearity.")
    end
    
    # Equation 11: Θ = -E[ψa(W;η)]^{-1} * E[ψb(W;η)]
    # Since ψa is negative, this becomes: θ = (E[AA^T])^{-1} * E[AY]
    theta = -inv(psi_a) * psi_b
    
    # ========================================================================
    # STEP 3: VARIANCE ESTIMATION USING NEYMAN ORTHOGONAL SCORE
    # ========================================================================
    
    if dml.verbose
        println("Computing variance using Neyman orthogonal score formula...")
    end
    
    # ====================================================================
    # Variance Formula from Neyman Orthogonal Score:
    # Var(Θ) = E[ψa]^{-1} * Var(ψ(W;Θ,η)) * E[ψa]^{-1}
    # 
    # Where ψ(W;Θ,η) = [A - m(X)]^T * (Y - l(X) - [A - m(X)]^T * θ)
    # ====================================================================
    
    # Compute final residuals after applying estimated θ
    residuals_final = Y_residuals - A_matrix * theta
    
    # Compute the score function values for each observation
    # ψ_i = A_i * (Y_i - A_i^T * θ) for each observation i
    psi_values = zeros(Float32, n, length(treatment_names))
    for i in 1:n
        A_i = A_matrix[i, :]
        psi_values[i, :] = A_i * residuals_final[i]
    end
    
    # Compute variance of the score function
    # Var(ψ) = E[ψψ^T] - E[ψ]E[ψ]^T
    psi_mean = mean(psi_values, dims=1)[:]
    psi_centered = psi_values .- psi_mean'
    var_psi = (psi_centered' * psi_centered) / n
    
    # Apply the variance formula: Var(Θ) = E[ψa]^{-1} * Var(ψ) * E[ψa]^{-1}
    psi_a_inv = inv(psi_a)
    vcov = psi_a_inv * var_psi * psi_a_inv / n
    
    # Extract standard errors
    se = sqrt.(diag(vcov))
    
    # Store results
    dml.coefficients = theta
    dml.std_errors = se
    dml.vcov = vcov
    dml.treatment_names = treatment_names
    
    # ========================================================================
    # STEP 4: DISPLAY RESULTS
    # ========================================================================
    
    if dml.verbose
        println("\n" * "="^70)
        println("RESULTS (DML with Neyman Orthogonal Score)")
        println("="^70)
        println("\nMethod: Cross-fitted DML with $(dml.n_folds) folds")
        println("Estimation: Neyman orthogonal score (Equations 11-12)")
        println("Nuisance functions: LightGBM with Tweedie objective")
        println("Variance: Score-based formula for robust inference")
        
        for (i, name) in enumerate(treatment_names)
            z_stat = theta[i] / se[i]
            p_val = 2 * (1 - cdf(Normal(), abs(z_stat)))
            
            println("\n$name:")
            println("  Coefficient: $(round(theta[i], digits=4))")
            println("  Std. Error:  $(round(se[i], digits=4))")
            println("  Z-statistic: $(round(z_stat, digits=2))")
            println("  P-value:     $(round(p_val, digits=4))")
            
            # Interpretation for main effects
            if !occursin("_x_", name)
                ci_lower = theta[i] - 1.96 * se[i]
                ci_upper = theta[i] + 1.96 * se[i]
                println("  95% CI:      [$(round(ci_lower, digits=4)), $(round(ci_upper, digits=4))]")
                
                if p_val < 0.05
                    effect_dir = theta[i] > 0 ? "increases" : "decreases"
                    println("  ⭐ Causal: One additional $(name) species $effect_dir fungi by $(abs(round(theta[i], digits=2))) species")
                end
            elseif occursin("_x_", name)
                println("  Interpretation: Interaction effect between treatments")
                if p_val < 0.05
                    println("  ⭐ Significant interaction detected (p < 0.05)")
                end
            end
        end
        
        println("\n" * "="^70)
    end
    
    return dml
end

"""
    save_results(dml::MultivariableDML, filepath::String)

Save DML results to JSON file for integration with other pipelines.
"""
function save_results(dml::MultivariableDML, filepath::String)
    results = Dict(
        "method" => "Multivariable DML",
        "coefficients" => dml.coefficients,
        "std_errors" => dml.std_errors,
        "vcov" => dml.vcov,
        "treatment_names" => dml.treatment_names,
        "n_obs" => dml.n_obs,
        "n_folds" => dml.n_folds,
        "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    )
    
    open(filepath, "w") do f
        JSON.print(f, results, 4)
    end
    
    println("Results saved to: $filepath")
end

# ============================================================================
# INTEGRATION WITH EXISTING PIPELINE
# ============================================================================

"""
    fit_true_dml_from_pipeline(data::DataFrame, mesf_columns::Vector{String})

Convenience function to run DML from the existing pipeline.
"""
function fit_true_dml_from_pipeline(data::DataFrame, mesf_columns::Vector{String})
    # Define variables
    confounders = vcat(
        ["bio01_std", "bio04_std", "bio12_std", "bio15_std"],
        ["elevation_std", "elevation_std_sq"],
        mesf_columns
    )
    
    treatments = Dict(
        "plants" => "plants_richness",      # Use raw counts
        "vertebrates" => "vertebrates_richness"  # Use raw counts
    )
    
    outcome = "fungi_richness"  # Raw counts
    
    # Initialize and fit
    dml = MultivariableDML(n_folds=5, n_threads=80)
    fit_dml!(dml, data, confounders, treatments, outcome)
    
    # Save results
    save_results(dml, "results/julia_dml/multivariable_dml_results.json")
    
    return dml
end

# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

# Command-line execution disabled - use julia_hybrid_pipeline.jl instead
# if length(ARGS) > 0
#     data_path = ARGS[1]
#     println("To run DML, use: julia julia_hybrid_pipeline.jl method3 $data_path")
# end