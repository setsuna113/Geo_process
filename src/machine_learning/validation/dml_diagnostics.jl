#!/usr/bin/env julia
"""
DML Post-Estimation Diagnostics
Critical validation procedures that MUST run after DML fitting
"""

using Statistics, LinearAlgebra

struct DMLDiagnostics
    orthogonality_score::Float64
    nuisance_rmse::Dict{String, Float64}
    stability_coefficient::Float64
    overlap_violations::Int
    score_residual_correlation::Float64
end

function validate_dml(dml_result, residuals_dict, n_repetitions=10)
    """
    Run CRITICAL post-DML diagnostics
    
    THIS IS NOT OPTIONAL - without these checks, your DML is garbage
    """
    
    # 1. NEYMAN ORTHOGONALITY TEST
    # The score function ψ should have mean zero
    psi_mean = mean(dml_result.score_residuals)
    orthogonality_score = abs(psi_mean) / std(dml_result.score_residuals)
    
    if orthogonality_score > 0.01
        @warn "ORTHOGONALITY VIOLATION: Score mean = $psi_mean (should be ≈ 0)"
    end
    
    # 2. NUISANCE FUNCTION RMSE
    nuisance_rmse = Dict{String, Float64}()
    for (name, resids) in residuals_dict
        rmse = sqrt(mean(resids .^ 2))
        nuisance_rmse[name] = rmse
        
        # Check if RMSE is reasonable (< 2 standard deviations of outcome)
        if rmse > 2.0 * std(resids)
            @warn "Poor nuisance fit for $name: RMSE = $rmse"
        end
    end
    
    # 3. STABILITY CHECK (requires multiple runs)
    # This is a placeholder - you need to actually run multiple repetitions
    stability_coefficient = NaN  # Should be std(coefficients) / mean(coefficients)
    
    # 4. OVERLAP CHECK (for continuous treatments)
    # Count observations with extreme residuals (poor overlap)
    overlap_violations = 0
    for (_, resids) in residuals_dict
        extreme_resids = abs.(resids) .> (3 * std(resids))
        overlap_violations += sum(extreme_resids)
    end
    
    # 5. SCORE RESIDUAL PATTERNS
    # Check if score residuals are correlated with treatments (BAD)
    score_residual_correlation = 0.0
    if haskey(dml_result, :score_residuals) && haskey(residuals_dict, "treatment")
        score_residual_correlation = cor(
            vec(mean(dml_result.score_residuals, dims=2)),
            residuals_dict["treatment"]
        )[1]
        
        if abs(score_residual_correlation) > 0.1
            @warn "Score residuals correlated with treatment: r = $score_residual_correlation"
        end
    end
    
    return DMLDiagnostics(
        orthogonality_score,
        nuisance_rmse,
        stability_coefficient,
        overlap_violations,
        score_residual_correlation
    )
end

function run_repeated_dml(data, config; n_reps=10)
    """
    Run DML multiple times to assess stability
    THIS IS MANDATORY for production use
    """
    
    coefficients = []
    std_errors = []
    
    for rep in 1:n_reps
        # Run your DML with different random seeds
        dml = fit_dml(data, config, seed=rep*42)
        push!(coefficients, dml.coefficients)
        push!(std_errors, dml.std_errors)
    end
    
    # Compute stability metrics
    coef_matrix = hcat(coefficients...)
    coef_means = mean(coef_matrix, dims=2)
    coef_stds = std(coef_matrix, dims=2)
    
    stability_ratio = coef_stds ./ abs.(coef_means)
    
    if any(stability_ratio .> 0.1)
        @error "UNSTABLE ESTIMATES: Coefficient variation > 10%"
        @info "Stability ratios: $stability_ratio"
    end
    
    return (
        mean_coef = coef_means,
        std_coef = coef_stds,
        stability = stability_ratio,
        all_coefs = coef_matrix
    )
end

# EXPORT FOR USE
export DMLDiagnostics, validate_dml, run_repeated_dml