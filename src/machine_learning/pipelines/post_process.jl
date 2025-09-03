#!/usr/bin/env julia
# Post-Processing Module - Calculation functions calculate. I/O functions do I/O. Period.

using DataFrames, JSON, JLD2, NPZ, Arrow, Dates, Random
using GLM: coef, coefnames, stderror, vcov, aic, bic, loglikelihood, deviance
using GLM: dof_residual, nobs, response, predict, nulldeviance, nullloglikelihood
using GLM: dispersion, coeftable, confint, gvif
using StatsBase: skewness, kurtosis, mean, std, sample
using HypothesisTests: JarqueBeraTest, pvalue
using LinearAlgebra: cond, inv, dot
using Distributions: Normal, Chisq, cdf

# ============================================================================
# PURE CALCULATION FUNCTIONS - No I/O, just compute and return
# ============================================================================

"""
Extract diagnostics from a model. Returns data, doesn't save anything.
"""
function extract_diagnostics(model, model_name::String)
    # Handle both standard GLM and RidgeNegBinModel wrapper
    if isdefined(Main, :RidgeNegBinModel) && isa(model, Main.RidgeNegBinModel)
        actual_model = model.model
        θ = actual_model.model.rr.d.r
    else
        actual_model = model
        θ = model.model.rr.d.r
    end
    
    # Calculate residuals
    y = response(actual_model)
    μ = predict(actual_model)
    response_resid = y .- μ
    deviance_resid = sign.(y .- μ) .* sqrt.(2 .* (
        y .* log.((y .+ 1e-10) ./ μ) .- (y .+ θ) .* log.((y .+ θ) ./ (μ .+ θ))
    ))
    
    # Build diagnostics dictionary
    diagnostics = Dict{String, Any}(
        # Core statistics
        "coefficients" => coef(actual_model),
        "coef_names" => coefnames(actual_model),
        "std_errors" => stderror(actual_model),
        "vcov" => vcov(actual_model),
        "aic" => aic(model),  # Use wrapper for AIC (handles ridge)
        "bic" => bic(model),  # Use wrapper for BIC (handles ridge)
        "loglik" => loglikelihood(actual_model),
        "deviance" => deviance(actual_model),
        "df_residual" => dof_residual(actual_model),
        "n_obs" => nobs(actual_model),
        "theta" => θ,
        
        # Residuals as data, not file references
        "residuals" => Dict(
            "y_values" => Float32.(y),
            "fitted_values" => Float32.(μ),
            "response_residuals" => Float32.(response_resid),
            "deviance_residuals" => Float32.(deviance_resid)
        ),
        
        # Residual summaries
        "residuals_summary" => Dict(
            "mean_response" => mean(response_resid),
            "std_response" => std(response_resid),
            "mean_deviance" => mean(deviance_resid),
            "std_deviance" => std(deviance_resid),
            "min_deviance" => minimum(deviance_resid),
            "max_deviance" => maximum(deviance_resid)
        ),
        
        # Extended statistics
        "null_deviance" => nulldeviance(actual_model),
        "null_loglik" => nullloglikelihood(actual_model),
        "dispersion" => dispersion(actual_model.model)
    )
    
    # Inference statistics (may fail for some models)
    try
        ct = coeftable(actual_model)
        diagnostics["coef_z_values"] = ct.cols[3]
        diagnostics["coef_p_values"] = ct.cols[4]
        diagnostics["confint_95"] = confint(actual_model, 0.95)
    catch e
        # Skip if not available
    end
    
    # Multicollinearity check
    try
        vif_values = gvif(actual_model)
        diagnostics["gvif"] = vif_values
        diagnostics["gvif_max"] = maximum(vif_values)
    catch e
        # Skip if not available
    end
    
    # Diagnostic tests
    try
        dev_resid = deviance_resid
        
        # Jarque-Bera normality test
        jb = JarqueBeraTest(dev_resid)
        diagnostics["test_jarque_bera_stat"] = jb.JB
        diagnostics["test_jarque_bera_p"] = pvalue(jb)
        
        # Outlier detection
        outlier_threshold = 3.0
        n_outliers = sum(abs.(dev_resid) .> outlier_threshold)
        diagnostics["test_outliers_count"] = n_outliers
        diagnostics["test_outliers_percent"] = 100.0 * n_outliers / length(dev_resid)
        diagnostics["test_outliers_threshold"] = outlier_threshold
        
        # Heteroscedasticity check
        n_bins = min(10, Int(ceil(sqrt(length(μ)))))
        sorted_idx = sortperm(μ)
        bin_size = length(μ) ÷ n_bins
        bin_variances = Float64[]
        for i in 1:n_bins
            start_idx = (i-1) * bin_size + 1
            end_idx = i == n_bins ? length(μ) : i * bin_size
            bin_resids = dev_resid[sorted_idx[start_idx:end_idx]]
            push!(bin_variances, var(bin_resids))
        end
        heterosced_cv = std(bin_variances) / mean(bin_variances)
        diagnostics["test_heteroscedasticity_cv"] = heterosced_cv
        diagnostics["test_heteroscedasticity_high"] = heterosced_cv > 0.5
        
        # Skewness and Kurtosis
        resid_skewness = skewness(dev_resid)
        resid_kurtosis = kurtosis(dev_resid)
        diagnostics["test_skewness"] = resid_skewness
        diagnostics["test_kurtosis_excess"] = resid_kurtosis
        diagnostics["test_skewness_high"] = abs(resid_skewness) > 1.0
        diagnostics["test_kurtosis_high"] = abs(resid_kurtosis) > 3.0
    catch e
        # Skip if tests fail
    end
    
    # SHAP parameters - always include for Python/R analysis
    diagnostics["shap_params"] = Dict(
        "coefficients" => coef(actual_model),
        "coef_names" => coefnames(actual_model),
        "intercept_idx" => 1,  # Julia uses 1-based indexing
        "feature_names" => coefnames(actual_model)[2:end],  # Exclude intercept
        "theta" => θ,  # Negative binomial dispersion
        "link" => "log",  # Always log for NegativeBinomial
        "family" => "negative_binomial",
        "n_features" => length(coef(actual_model)) - 1
    )
    
    return diagnostics
end

"""
Process DML results for output. Returns data, doesn't save anything.
"""
function process_dml_results(dml)
    result = Dict{String, Any}(
        "coefficients" => dml.coefficients,
        "coef_names" => dml.treatment_names,
        "std_errors" => dml.std_errors,
        "vcov" => dml.vcov,
        "n_obs" => dml.n_obs,
        "method_note" => "DML with cross-fitting and LightGBM",
        "treatments" => Dict{String, Any}()
    )
    
    # Compute detailed statistics for each treatment
    for (i, name) in enumerate(dml.treatment_names)
        coef = dml.coefficients[i]
        se = dml.std_errors[i]
        z = coef / se
        p = 2 * (1 - cdf(Normal(), abs(z)))
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        
        result["treatments"][name] = Dict(
            "coefficient" => coef,
            "std_error" => se,
            "z_statistic" => z,
            "p_value" => p,
            "ci_95" => [ci_lower, ci_upper],
            "statistically_significant" => p < 0.05
        )
    end
    
    # Overall model statistics
    result["model_stats"] = Dict(
        "condition_number" => cond(dml.vcov),
        "joint_test" => Dict(
            "chi2_statistic" => dot(dml.coefficients, inv(dml.vcov) * dml.coefficients),
            "df" => length(dml.coefficients),
            "p_value" => 1 - cdf(Chisq(length(dml.coefficients)), 
                                 dot(dml.coefficients, inv(dml.vcov) * dml.coefficients))
        )
    )
    
    # Include BDML stats if available
    if isdefined(dml, :bdml_stats) && dml.bdml_stats !== nothing
        result["bdml_stats"] = dml.bdml_stats
    end
    
    return result
end

# ============================================================================
# WRAPPER FUNCTIONS - Bridge between calculation and I/O
# ============================================================================

"""
Extract diagnostics and save residuals - wrapper for backward compatibility.
This is what the main pipeline calls.
"""
function extract_diagnostics_and_save(model, model_name::String, state)
    # Step 1: Pure calculation
    diagnostics = extract_diagnostics(model, model_name)
    
    # Step 2: Save residuals to file
    if haskey(diagnostics, "residuals")
        residuals_file = joinpath(state.output_dir, "$(model_name)_residuals.npz")
        npzwrite(residuals_file, diagnostics["residuals"])
        
        # Replace residuals data with file reference (like original)
        delete!(diagnostics, "residuals")
        diagnostics["residuals_file"] = residuals_file
    end
    
    return diagnostics
end

# ============================================================================
# PURE I/O FUNCTIONS - Each does ONE thing for ONE purpose
# ============================================================================

"""
Save GLM results (Method 1 & 2). Residuals in NPZ, metadata in JSON.
Now includes state for metadata compatibility with original.
"""
function save_glm_results(results::Dict, state)
    mkpath(state.output_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    # Separate residuals from other diagnostics
    residuals_dict = Dict{String, Any}()
    clean_results = Dict{String, Any}()
    
    for (model_name, diagnostics) in results
        if isa(diagnostics, Dict) && haskey(diagnostics, "residuals")
            # Extract residuals for NPZ
            residuals_dict[model_name] = diagnostics["residuals"]
            
            # Copy diagnostics without residuals for JSON
            clean_diag = copy(diagnostics)
            delete!(clean_diag, "residuals")
            clean_diag["residuals_file"] = "$(model_name)_residuals.npz"
            clean_results[model_name] = clean_diag
        else
            # Non-model results (like LRT tests)
            clean_results[model_name] = diagnostics
        end
    end
    
    # Save residuals to NPZ files
    for (model_name, residuals) in residuals_dict
        npzwrite(joinpath(state.output_dir, "$(model_name)_residuals.npz"), residuals)
    end
    
    # Extract response type from config ONLY
    method1_config = get(state.config, "method1", Dict())
    response_type = get(method1_config, "response", nothing)
    
    if isnothing(response_type)
        error("Response type MUST be specified in config under method1.response")
    end
    
    # Prepare final output with full metadata (matching original)
    output = Dict(
        "models" => clean_results,
        "metadata" => Dict(
            "n_rows" => nrow(state.data),
            "n_cols" => ncol(state.data),
            "mesf_count" => length(state.mesf_columns),
            "timestamp" => timestamp,
            "julia_version" => string(VERSION),
            "response_type" => response_type  # ADD THIS!
        )
    )
    
    # ALWAYS save with response type in filename
    json_file = joinpath(state.output_dir, "julia_results_$(response_type)_$(timestamp).json")
    open(json_file, "w") do f
        JSON.print(f, output, 4)
    end
    
    # Save JLD2 for Julia users
    jld2_file = joinpath(state.output_dir, "julia_results_$(timestamp).jld2")
    @save jld2_file output
    
    # Note: Analysis data (for SHAP/DHARMa) is dumped earlier in pipeline after preprocessing
    # via dump_analysis_data() which handles both SHAP background and DHARMa data in one place
    
    # Create response-specific symlink
    response_latest = joinpath(state.output_dir, "julia_results_$(response_type)_latest.json")
    rm(response_latest, force=true)
    symlink(basename(json_file), response_latest)
    println("Created symlink: julia_results_$(response_type)_latest.json -> $(basename(json_file))")
    
    println("GLM results saved: $json_file")
    return json_file
end

"""
Save bootstrap results. Everything in NPZ for Python visualization.
"""
function save_bootstrap_results(results::Dict, output_dir::String, method::String)
    mkpath(output_dir)
    npz_path = joinpath(output_dir, "bootstrap_$(method).npz")
    
    # Prepare data for NPZ (numeric only)
    npz_data = Dict{String, Any}()
    sample_sizes = results["sample_sizes"]
    
    # Get biotic names and response from first sample result
    first_sr = results["sample_results"][sample_sizes[1]]
    
    # Metadata
    npz_data["sample_sizes"] = sample_sizes
    npz_data["optimal_n"] = results["optimal_n"]
    npz_data["N_total"] = results["N"]
    npz_data["n_iterations"] = results["n_iterations"]
    npz_data["biotic1_name"] = first_sr["biotic1_name"]
    npz_data["biotic2_name"] = first_sr["biotic2_name"]
    npz_data["response_var"] = first_sr["response_var"]
    
    # Results for each sample size - ONE TRUE FORMAT
    for n in sample_sizes
        sr = results["sample_results"][n]
        npz_data["n$(n)_biotic1_effects"] = sr["biotic1_effects"]
        npz_data["n$(n)_biotic2_effects"] = sr["biotic2_effects"]
        npz_data["n$(n)_p_values"] = sr["p_values"]
        npz_data["n$(n)_r2_improvements"] = sr["r2_improvements"]
        npz_data["n$(n)_power"] = sr["power"]
        npz_data["n$(n)_eco_power"] = sr["ecological_power"]
        npz_data["n$(n)_biotic1_mean"] = sr["biotic1_effect"]["mean"]
        npz_data["n$(n)_biotic1_std"] = sr["biotic1_effect"]["std"]
        npz_data["n$(n)_biotic2_mean"] = sr["biotic2_effect"]["mean"]
        npz_data["n$(n)_biotic2_std"] = sr["biotic2_effect"]["std"]
    end
    
    npzwrite(npz_path, npz_data)
    println("Bootstrap results saved: $npz_path")
    return npz_path
end

"""
Unified data dump for analysis (SHAP, DHARMa, etc).
Always dumps the same preprocessed data sample - consumers use what they need.
Returns early termination signal if dharma_only=true.
"""
function dump_analysis_data(state, method::String; dharma_only::Bool=false)
    # Use consistent sampling regardless of purpose
    n_sample = min(10000, nrow(state.data))  # Up to 10k rows
    Random.seed!(42)
    sample_idx = sort(sample(1:nrow(state.data), n_sample, replace=false))
    df_sample = state.data[sample_idx, :]
    
    # Determine output location
    if dharma_only
        output_dir = "/scratch/yl998/.cache/dharma"
        filename_base = "dharma_data"
    else
        output_dir = state.output_dir
        filename_base = "analysis_data"
    end
    
    mkpath(output_dir)
    
    # Save Arrow file with ALL columns (consumers filter as needed)
    arrow_file = joinpath(output_dir, "$(filename_base).arrow")
    Arrow.write(arrow_file, df_sample)
    
    # Also save SHAP-specific background file for compatibility
    # Exclude response variables and MESF columns as per remote's intent
    if !dharma_only
        response_vars = ["fungi_richness", "fungi_richness_int", "plants_richness", "plants_richness_int", 
                        "vertebrates_richness", "vertebrates_richness_int"]
        feature_cols = setdiff(names(df_sample), vcat(response_vars, state.mesf_columns))
        if length(feature_cols) > 0
            X_background = df_sample[:, feature_cols]
            shap_file = joinpath(output_dir, "shap_background.arrow")
            Arrow.write(shap_file, X_background)
            println("SHAP background data saved: $shap_file ($(n_sample) samples, $(length(feature_cols)) features)")
        end
    end
    
    # Save metadata
    metadata = Dict(
        "data_file" => arrow_file,
        "n_sampled" => n_sample,
        "n_total" => nrow(state.data),
        "sample_frac" => n_sample / nrow(state.data),
        "mesf_columns" => state.mesf_columns,
        "standardized_columns" => [String(col) for col in names(state.data) if occursin("_std", String(col))],
        "timestamp" => Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"),
        "method" => method
    )
    
    json_file = joinpath(output_dir, "$(filename_base)_meta.json")
    open(json_file, "w") do f
        JSON.print(f, metadata, 2)
    end
    
    println("Analysis data saved: $arrow_file ($(round(filesize(arrow_file)/1e6, digits=1)) MB, $n_sample samples)")
    
    # Early termination for DHARMa-only mode
    if dharma_only
        println("Data dumped for DHARMa diagnostics. Exiting.")
        return Dict("status" => "dharma_cache_saved", "file" => arrow_file)
    end
    
    return nothing  # Continue with normal pipeline
end

# Keep alias for backward compatibility if needed
const dump_for_r_dharma = (state, method; kwargs...) -> dump_analysis_data(state, method; dharma_only=true)

"""
Save DML results. BDML stats in NPZ, rest in JSON.
"""
function save_dml_results(results::Dict, output_dir::String)
    mkpath(output_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    # Handle BDML stats separately if present
    if haskey(results, "bdml_stats") && results["bdml_stats"] !== nothing
        bdml_stats = results["bdml_stats"]
        
        # Save numeric arrays to NPZ
        bdml_file = joinpath(output_dir, "bdml_stats_$(timestamp).npz")
        bdml_numeric = Dict(
            "covariance_matrix" => bdml_stats["covariance_matrix"],
            "n_obs" => bdml_stats["n_obs"],
            "n_treatments" => bdml_stats["n_treatments"],
            "residual_means" => bdml_stats["residual_means"],
            "residual_stds" => bdml_stats["residual_stds"],
            "correlation_matrix" => bdml_stats["correlation_matrix"],
            "condition_number" => [bdml_stats["condition_number"]]
        )
        npzwrite(bdml_file, bdml_numeric)
        
        # Save metadata separately
        bdml_meta_file = joinpath(output_dir, "bdml_meta_$(timestamp).json")
        open(bdml_meta_file, "w") do f
            JSON.print(f, Dict(
                "treatment_names" => bdml_stats["treatment_names"],
                "is_positive_definite" => bdml_stats["is_positive_definite"],
                "bdml_stats_file" => bdml_file
            ), 2)
        end
        
        println("BDML statistics saved: $bdml_file")
        
        # Update results to reference files instead of data
        results = copy(results)
        results["bdml_stats_file"] = bdml_file
        results["bdml_meta_file"] = bdml_meta_file
        delete!(results, "bdml_stats")
    end
    
    # Prepare output
    output = Dict(
        "method3_dml" => results,
        "metadata" => Dict(
            "timestamp" => timestamp,
            "julia_version" => string(VERSION)
        )
    )
    
    # Save JSON
    json_file = joinpath(output_dir, "julia_dml_results_$(timestamp).json")
    open(json_file, "w") do f
        JSON.print(f, output, 4)
    end
    
    # Save JLD2
    jld2_file = joinpath(output_dir, "julia_dml_results_$(timestamp).jld2")
    @save jld2_file output
    
    # Update symlink
    latest = joinpath(output_dir, "julia_dml_results_latest.json")
    rm(latest, force=true)
    symlink(basename(json_file), latest)
    
    println("DML results saved: $json_file")
    return json_file
end

# ============================================================================
# EXPORT
# ============================================================================

# For backward compatibility, export the wrapper as extract_diagnostics
export extract_diagnostics_and_save
export process_dml_results
export save_glm_results, save_bootstrap_results, dump_analysis_data, save_dml_results