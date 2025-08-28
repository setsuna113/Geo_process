#!/usr/bin/env julia
"""
RidgeNegBin.jl - Ridge-regularized Negative Binomial GLM

Drop-in replacement for GLM.jl's negbin() with optional ridge regularization.
When λ=0, behaves exactly like standard negbin().

Author: Linus (via Claude)
Philosophy: Simple, direct, no special cases. Just add a penalty term.
"""

module RidgeNegBin

using GLM
using StatsModels  
using LinearAlgebra
using DataFrames
using Distributions: Chisq, ccdf
using Distributions
using StatsBase
using Random
using StatsBase: sample
import StatsBase: coef, loglikelihood, nobs, deviance, aic, bic
using Base.Threads  # For parallel computation

export negbin  # Same name as GLM.jl for drop-in replacement
export select_lambda_cv  # Cross-validation for lambda selection
export RidgeNegBinModel  # Wrapper for ridge model

# Wrapper struct to hold ridge-regularized model with lambda
struct RidgeNegBinModel
    model::Any  # The GLM model
    lambda::Float64  # Ridge penalty parameter
end

# Forward key methods to the underlying model
coef(m::RidgeNegBinModel) = coef(m.model)
loglikelihood(m::RidgeNegBinModel) = loglikelihood(m.model)
nobs(m::RidgeNegBinModel) = nobs(m.model)
deviance(m::RidgeNegBinModel) = deviance(m.model)
aic(m::RidgeNegBinModel) = aic_ridge(m)
bic(m::RidgeNegBinModel) = bic_ridge(m)
GLM.dispersion(m::RidgeNegBinModel) = m.model.model.rr.d.r
GLM.dof(m::RidgeNegBinModel) = effective_df(m)  # Use effective df for ridge
GLM.dof_residual(m::RidgeNegBinModel) = nobs(m) - effective_df(m)

# Implement lrtest for RidgeNegBinModel
"""
    lrtest(m0::RidgeNegBinModel, m1::RidgeNegBinModel)

Likelihood ratio test between two ridge negative binomial models.
Uses effective degrees of freedom for ridge models.
"""
function GLM.lrtest(m0::RidgeNegBinModel, m1::RidgeNegBinModel)
    # Extract log-likelihoods
    ll0 = loglikelihood(m0)
    ll1 = loglikelihood(m1)
    
    # Use effective df for ridge models
    df0 = effective_df(m0)
    df1 = effective_df(m1)
    
    # LR statistic
    lr_stat = 2 * (ll1 - ll0)
    df_diff = df1 - df0
    
    # Create LRT result (matches GLM.lrtest return type)
    return (
        deviance = [deviance(m0), lr_stat],
        dof = [Int(round(df0)), Int(round(df_diff))],  # Round effective df to int
        pval = [NaN, ccdf(Chisq(max(1, round(df_diff))), lr_stat)]
    )
end

# Also support mixed model comparisons (RidgeNegBin vs regular GLM)
function GLM.lrtest(m0::GLM.GeneralizedLinearModel, m1::RidgeNegBinModel)
    ll0 = loglikelihood(m0)
    ll1 = loglikelihood(m1)
    
    df0 = GLM.dof(m0) 
    df1 = effective_df(m1)
    
    lr_stat = 2 * (ll1 - ll0)
    df_diff = df1 - df0
    
    return (
        deviance = [deviance(m0), lr_stat],
        dof = [Int(round(df0)), Int(round(df_diff))],
        pval = [NaN, ccdf(Chisq(max(1, round(df_diff))), lr_stat)]
    )
end

function GLM.lrtest(m0::RidgeNegBinModel, m1::GLM.GeneralizedLinearModel)
    ll0 = loglikelihood(m0)
    ll1 = loglikelihood(m1)
    
    df0 = effective_df(m0)
    df1 = GLM.dof(m1)
    
    lr_stat = 2 * (ll1 - ll0)
    df_diff = df1 - df0
    
    return (
        deviance = [deviance(m0), lr_stat],
        dof = [Int(round(df0)), Int(round(df_diff))],
        pval = [NaN, ccdf(Chisq(max(1, round(df_diff))), lr_stat)]
    )
end

"""
    negbin(formula, data, link; ridge=false)

Fit a negative binomial GLM with optional ridge regularization.

# Arguments
- `formula`: StatsModels formula
- `data`: DataFrame
- `link`: Link function (typically LogLink())
- `ridge`: If true, use ridge with automatic lambda selection

# Returns
GLM model (standard if ridge=false, ridge if ridge=true)
"""
function negbin(formula::FormulaTerm, data::DataFrame, link; ridge::Bool=false)
    
    # No ridge - return standard model
    if !ridge
        return GLM.negbin(formula, data, link)
    end
    
    # Ridge - ALWAYS subsample and select lambda
    n_rows = nrow(data)
    subsample_size = min(100000, n_rows)  # Max 100k rows for CV
    
    # Generate 100 lambdas on log scale from 10000 to 100000
    # MUCH higher range for big data with millions of observations
    # This properly scales the penalty to match the likelihood magnitude
    λ_path = exp.(range(log(10000), log(100000), length=100))
    
    # Subsample for CV
    println("Selecting λ via CV on $subsample_size rows (data has $n_rows rows)")
    sample_idx = sample(1:n_rows, subsample_size, replace=false)
    data_cv = data[sample_idx, :]
    
    # Fit base model on subsample
    base_model_cv = GLM.negbin(formula, data_cv, link)
    X_cv = modelmatrix(base_model_cv.mf)
    y_cv = response(base_model_cv.mf)
    θ_cv = base_model_cv.model.rr.d.r
    
    # Select lambda via CV
    λ = select_lambda_cv(X_cv, y_cv, θ_cv, λ_path, 5)
    
    # Fit on FULL data with selected lambda
    println("\nFitting final ridge model with λ=$(round(λ, digits=4)) on full dataset ($n_rows rows)")
    base_model = GLM.negbin(formula, data, link)
    X = modelmatrix(base_model.mf)
    y = response(base_model.mf)
    θ = base_model.model.rr.d.r
    
    # Apply ridge regularization (verbose for final fit)
    β_ridge = apply_ridge(X, y, θ, coef(base_model), λ, verbose=true)
    
    # Create modified model (preserve all GLM structure)
    ridge_model = deepcopy(base_model)
    
    # Update coefficients - GLM stores them in model.model.pp.beta0
    ridge_model.model.pp.beta0 = β_ridge
    
    # Return wrapped model with lambda
    return RidgeNegBinModel(ridge_model, λ)
end

"""
    apply_ridge(X, y, θ, β_init, λ)

Properly fit ridge-penalized negative binomial GLM by minimizing:
-log L(β|y,X) + λ/2 ||β||²

Uses penalized IRLS algorithm with proper iterations.
"""
function apply_ridge(X::Matrix, y::Vector, θ::Float64, β_init::Vector, λ::Float64; 
                    maxiter::Int=30, tol::Float64=1e-6, verbose::Bool=false)
    p = length(β_init)
    β = copy(β_init)
    
    # Ridge penalty matrix (don't penalize intercept)
    Λ = λ * I(p)
    Λ[1,1] = 0.0  # Intercept is always first in GLM.jl
    
    # Penalized IRLS iterations
    for iter in 1:maxiter
        # Current predictions
        η = X * β
        μ = exp.(η)
        
        # NB variance and weights
        V = μ .+ μ.^2 / θ  # Variance function for NB
        W = Diagonal(μ.^2 ./ V)  # Weight matrix
        
        # Working response (IRLS pseudodata)
        z = η + (y - μ) ./ μ
        
        # Penalized normal equations: (X'WX + λI)β = X'Wz
        # This is the correct update for penalized likelihood
        XtWX = X' * W * X
        XtWz = X' * W * z
        
        # Solve for new β
        β_new = (XtWX + Λ) \ XtWz
        
        # Check convergence
        if norm(β_new - β, Inf) < tol
            if verbose
                println("  Ridge IRLS converged in $iter iterations")
            end
            return β_new
        end
        
        β = β_new
    end
    
    @warn "Ridge IRLS did not converge in $maxiter iterations"
    return β
end

"""
    select_lambda_cv(X, y, θ, λ_path, n_folds)

Select optimal λ via cross-validation minimizing deviance.
PARALLEL VERSION: Each lambda computed independently on separate thread.
"""
function select_lambda_cv(X::Matrix, y::Vector, θ::Float64, 
                         λ_path::Vector, n_folds::Int=5)
    n = length(y)
    fold_size = n ÷ n_folds
    n_lambdas = length(λ_path)
    
    # Store CV scores
    cv_scores = Vector{Float64}(undef, n_lambdas)
    
    # Save current BLAS threads and set to 1 to avoid oversubscription
    original_blas_threads = BLAS.get_num_threads()
    if Threads.nthreads() > 10  # Only adjust if using many Julia threads
        BLAS.set_num_threads(1)
    end
    
    println("Testing $n_lambdas lambdas via $n_folds-fold CV (parallel)...")
    
    # Pre-compute fold indices (same for all lambdas)
    fold_indices = []
    for fold in 1:n_folds
        test_start = (fold - 1) * fold_size + 1
        test_end = fold == n_folds ? n : fold * fold_size
        test_idx = test_start:test_end
        train_idx = setdiff(1:n, test_idx)
        push!(fold_indices, (train_idx, test_idx))
    end
    
    # PARALLEL: Each lambda gets its own thread (up to 100 threads)
    Threads.@threads for i in 1:n_lambdas
        λ = λ_path[i]
        fold_deviances = Float64[]
        
        # For this lambda, compute all 5 folds (sequentially within thread)
        for (train_idx, test_idx) in fold_indices
            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fresh MLE init for each lambda (no warm starts between lambdas)
            β_init = zeros(size(X, 2))
            β_init[1] = log(mean(y_train))
            
            # More iterations since no warm start from previous lambda
            β_ridge = apply_ridge(X_train, y_train, θ, β_init, λ, maxiter=30)
            
            # Predict on test
            η_test = X_test * β_ridge
            μ_test = exp.(η_test)
            
            # Compute deviance
            dev = 2 * sum(
                y_test .* log.((y_test .+ 1e-10) ./ μ_test) .- 
                (y_test .+ θ) .* log.((y_test .+ θ) ./ (μ_test .+ θ))
            )
            push!(fold_deviances, dev)
        end
        
        cv_scores[i] = mean(fold_deviances)
    end
    
    # Find best lambda after all parallel computation
    best_idx = argmin(cv_scores)
    best_λ = λ_path[best_idx]
    best_score = cv_scores[best_idx]
    
    # Print summary
    println("\nCV Summary:")
    println("  Best λ = $(round(best_λ, digits=4)) (deviance = $(round(best_score, digits=2)))")
    println("  λ range tested: [$(round(λ_path[1], digits=4)) ... $(round(λ_path[end], digits=4))]")
    
    # Restore original BLAS threads
    if Threads.nthreads() > 10
        BLAS.set_num_threads(original_blas_threads)
    end
    
    return best_λ
end

"""
    fit_mle_simple(X, y, θ)

Quick MLE fit using standard IRLS (no penalty).
Used internally for CV initialization.
"""
function fit_mle_simple(X::Matrix, y::Vector, θ::Float64; maxiter::Int=25, tol::Float64=1e-6)
    n, p = size(X)
    β = zeros(p)
    β[1] = log(mean(y))  # Initialize intercept
    
    for iter in 1:maxiter
        η = X * β
        μ = exp.(η)
        
        # NB variance and weights
        V = μ .+ μ.^2 / θ
        W = Diagonal(μ.^2 ./ V)
        z = η + (y - μ) ./ μ
        
        # Standard IRLS update (no penalty)
        β_new = (X' * W * X) \ (X' * W * z)
        
        # Check convergence
        if norm(β_new - β, Inf) < tol
            break
        end
        β = β_new
    end
    
    return β
end

"""
    effective_df(model)

Calculate effective degrees of freedom for ridge regression.
For ridge: tr(X(X'X + λI)^(-1)X')

MEMORY EFFICIENT: Don't compute the full hat matrix H = X * inv(XtX + λI) * X'
Instead use: tr(H) = tr(inv(XtX + λI) * XtX)
"""
function effective_df(m::RidgeNegBinModel)
    if m.lambda == 0
        return length(coef(m))
    end
    
    X = GLM.modelmatrix(m.model.mf)
    λ = m.lambda
    p = size(X, 2)
    
    Λ = λ * I(p)
    Λ[1,1] = 0.0  # Don't penalize intercept
    
    # Memory efficient: tr(X * inv(X'X + Λ) * X') = tr(inv(X'X + Λ) * X'X)
    XtX = X' * X
    AtA_inv = inv(XtX + Λ)
    
    # Effective df = trace of inv(X'X + λI) * X'X
    return tr(AtA_inv * XtX)
end

# AIC/BIC for ridge models
function aic_ridge(m::RidgeNegBinModel)
    ll = loglikelihood(m)
    df = effective_df(m)
    return -2ll + 2df
end

function bic_ridge(m::RidgeNegBinModel)
    ll = loglikelihood(m)
    df = effective_df(m)
    n = nobs(m)
    return -2ll + df * log(n)
end

end # module