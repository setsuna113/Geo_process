#!/usr/bin/env julia
"""
Julia-Python Hybrid DML Pipeline
Step 1: Julia computes nuisance PREDICTIONS (not residuals!)
Step 2: Python DoubleML uses these for diagnostics
"""

using DataFrames, NPZ, JSON, LightGBM

function compute_nuisance_predictions(
    data::DataFrame,
    confounders::Vector{String},
    treatments::Vector{String},
    outcome::String;
    n_folds::Int=5,
    lgb_params::Dict=Dict()
)
    """
    Compute nuisance function PREDICTIONS for DoubleML
    
    CRITICAL: DoubleML needs predictions, not residuals!
    We compute: ℓ(X), m(X), g(X) for each fold
    """
    
    n = nrow(data)
    
    # Storage for predictions (what DoubleML expects)
    l_hat = zeros(Float64, n)  # E[Y|X] predictions
    m_hats = Dict(t => zeros(Float64, n) for t in treatments)  # E[T|X] predictions
    g_hat = zeros(Float64, n)  # Alternative outcome model if needed
    
    # Also store residuals for our own diagnostics
    y_residuals = zeros(Float64, n)
    t_residuals = Dict(t => zeros(Float64, n) for t in treatments)
    
    # Track out-of-sample RMSE for diagnostics
    rmse_scores = Dict{String, Vector{Float64}}()
    
    # Cross-fitting
    fold_size = n ÷ n_folds
    for fold in 1:n_folds
        # Define test/train splits
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == n_folds ? n : fold * fold_size
        test_idx = test_start:test_end
        train_idx = setdiff(1:n, test_idx)
        
        # Prepare data
        X_train = Matrix(data[train_idx, Symbol.(confounders)])
        X_test = Matrix(data[test_idx, Symbol.(confounders)])
        Y_train = data[train_idx, Symbol(outcome)]
        Y_test = data[test_idx, Symbol(outcome)]
        
        # OUTCOME MODEL: Compute ℓ(X) = E[Y|X]
        outcome_model = LGBMRegression(;
            num_iterations = get(lgb_params, "num_iterations", 100),
            num_threads = get(lgb_params, "num_threads", 1),
            verbosity = -1
        )
        fit!(outcome_model, X_train, Y_train)
        
        # Store PREDICTIONS (not residuals!)
        l_hat[test_idx] = predict(outcome_model, X_test)
        g_hat[test_idx] = l_hat[test_idx]  # For compatibility
        
        # Calculate residuals for diagnostics
        y_residuals[test_idx] = Y_test .- l_hat[test_idx]
        
        # Track RMSE
        fold_rmse = sqrt(mean((Y_test .- l_hat[test_idx]).^2))
        push!(get!(rmse_scores, "outcome", Float64[]), fold_rmse)
        
        # TREATMENT MODELS: Compute m(X) = E[T|X] for each treatment
        for t in treatments
            T_train = data[train_idx, Symbol(t)]
            T_test = data[test_idx, Symbol(t)]
            
            treatment_model = LGBMRegression(;
                num_iterations = get(lgb_params, "num_iterations", 100),
                num_threads = get(lgb_params, "num_threads", 1),
                verbosity = -1
            )
            fit!(treatment_model, X_train, T_train)
            
            # Store PREDICTIONS
            m_hats[t][test_idx] = predict(treatment_model, X_test)
            
            # Calculate residuals
            t_residuals[t][test_idx] = T_test .- m_hats[t][test_idx]
            
            # Track RMSE
            fold_rmse = sqrt(mean((T_test .- m_hats[t][test_idx]).^2))
            push!(get!(rmse_scores, "treatment_$t", Float64[]), fold_rmse)
        end
    end
    
    # Print diagnostic summary
    println("\n=== NUISANCE FUNCTION DIAGNOSTICS ===")
    for (name, scores) in rmse_scores
        println("$name: mean RMSE = $(round(mean(scores), digits=4)), std = $(round(std(scores), digits=4))")
    end
    
    return (
        predictions = Dict(
            "l_hat" => l_hat,
            "m_hats" => m_hats,
            "g_hat" => g_hat
        ),
        residuals = Dict(
            "y_residuals" => y_residuals,
            "t_residuals" => t_residuals
        ),
        diagnostics = Dict(
            "rmse_scores" => rmse_scores,
            "n_obs" => n,
            "n_folds" => n_folds
        )
    )
end

function save_for_doubleml(results, filepath::String)
    """
    Save predictions in format DoubleML expects
    """
    
    # DoubleML expects numpy arrays
    npzwrite(filepath * "_predictions.npz",
        Dict(
            "l_hat" => results.predictions["l_hat"],
            "g_hat" => results.predictions["g_hat"],
            # Flatten treatment predictions
            [string("m_hat_", k) => v for (k, v) in results.predictions["m_hats"]]...
        )
    )
    
    # Save residuals for our own analysis
    npzwrite(filepath * "_residuals.npz",
        Dict(
            "y_residuals" => results.residuals["y_residuals"],
            [string("t_residuals_", k) => v for (k, v) in results.residuals["t_residuals"]]...
        )
    )
    
    # Save diagnostics as JSON
    open(filepath * "_diagnostics.json", "w") do f
        JSON.print(f, results.diagnostics, 4)
    end
    
    println("\nSaved to:")
    println("  - $(filepath)_predictions.npz (for DoubleML)")
    println("  - $(filepath)_residuals.npz (for custom analysis)")
    println("  - $(filepath)_diagnostics.json (diagnostic metrics)")
end